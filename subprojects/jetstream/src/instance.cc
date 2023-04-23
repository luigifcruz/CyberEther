#include "jetstream/instance.hh"
#include "jetstream/memory/types.hh"

namespace Jetstream {

const Result Instance::create() {
    JST_DEBUG("Creating compute graph.");

    // Check minimum requirements.

    if (commited) {
        JST_FATAL("The instance was already commited.");
        return Result::ERROR;
    }

    if (blocks.size() < 1) {
        JST_ERROR("No blocks were added to this instance.");
        return Result::ERROR;
    }

    // Register Blocks as Compute or Present.

    for (const auto& moduleBlock : blocks) {
        if (auto computeBlock = std::dynamic_pointer_cast<Compute>(moduleBlock)) {
            computeBlocks[moduleBlock->id()] = computeBlock;
        }
    }

    for (const auto& moduleBlock : blocks) {
        if (auto presentBlock = std::dynamic_pointer_cast<Present>(moduleBlock)) {
            if (!_window) {
                JST_ERROR("A window is required since a present block was added.");
                return Result::ERROR;
            }
            presentBlocks[moduleBlock->id()] = presentBlock;
        }
    }

    // Print summary of blocks.

    for (const auto& block : blocks) {
        JST_INFO("———————————————————————————————————————————————————————————————————————————————————————————————————");
        JST_INFO("[{:03}] [{}] [Device::{}] [C: {}, P: {}]", block->id(),
                                                             block->name(),
                                                             block->device(),
                                                             (computeBlocks.count(block->id())) ? "YES" : "NO",
                                                             (presentBlocks.count(block->id())) ? "YES" : "NO");
        JST_INFO("———————————————————————————————————————————————————————————————————————————————————————————————————");

        {
            JST_INFO("Configuration:");
            block->summary();
        }

        {
            JST_INFO("Block I/O:");

            const auto& inputs = block->getInputs();
            JST_INFO("    Inputs:");
            if (inputs.size() == 0) {
                JST_INFO("        None")
            }
            for (int i = 0; i < inputs.size(); i++) {
                JST_INFO("        {}: [{:>4s}] {} | [Device::{}] | Pointer: {} | Hash: 0x{:016x}", i,
                                                                                                   inputs[i].type,
                                                                                                   inputs[i].shape,
                                                                                                   inputs[i].device,
                                                                                                   fmt::ptr(inputs[i].ptr),
                                                                                                   inputs[i].hash);
            }

            const auto& outputs = block->getOutputs();
            JST_INFO("    Outputs:");
            if (outputs.size() == 0) {
                JST_INFO("        None")
            }
            for (int i = 0; i < outputs.size(); i++) {
                JST_INFO("        {}: [{:>4s}] {} | [Device::{}] | Pointer: {} | Hash: 0x{:016x}", i,
                                                                                                   outputs[i].type,
                                                                                                   outputs[i].shape,
                                                                                                   outputs[i].device,
                                                                                                   fmt::ptr(outputs[i].ptr),
                                                                                                   outputs[i].hash);
            }
        }
    }
    JST_INFO("———————————————————————————————————————————————————————————————————————————————————————————————————");

    // Run topological sort.

    JST_DEBUG("Filtering stale I/O.");
    std::unordered_map<U64, U64> ioValid;
    for (const auto& [id, _] : computeBlocks) {
        for (auto& input : blocks[id]->getInputs()) {
            ioValid[input.hash]++;
        }
        for (auto& output : blocks[id]->getOutputs()) {
            ioValid[output.hash]++;
        }
    }
    for (auto& [key, val] : ioValid) {
        if (val <= 1) {
            JST_TRACE("Nulling stale I/O: {:#016x}", key);
            ioValid[key] = 0;
        }
    }

    JST_DEBUG("Generating I/O map for each block.");
    std::unordered_map<U64, std::vector<U64>> blockInputs, blockOutputs;
    for (const auto& [id, _] : computeBlocks) {
        for (auto input : blocks[id]->getInputs()) {
            if (ioValid[input.hash] != 0) {
                 blockInputs[id].push_back(input.hash);
            }
        }
        for (auto output : blocks[id]->getOutputs()) {
            if (ioValid[output.hash] != 0) {
                blockOutputs[id].push_back(output.hash);
            }
        }
    }

    JST_DEBUG("Calculating block degrees.");
    std::unordered_map<U64, U64> blockDegrees;
    for (const auto& [id, _] : computeBlocks) {
        blockDegrees[id] = blockInputs[id].size();
    }
    JST_TRACE("Block degrees: {}", blockDegrees);

    JST_DEBUG("Populating sorting.");
    std::vector<U64> queue;
    for (const auto& [id, _] : computeBlocks) {
        if (blockDegrees[id] == 0) {
            queue.push_back(id);
        }
    }
    JST_TRACE("Initial sorting queue: {}", queue);

    JST_DEBUG("Calculating primitive execution order.");
    Device lastDevice = Device::None; 
    std::vector<U64> executionOrder;
    while (!queue.empty()) {
        U64 nextIndex = 0;
        for (U64 i = 0; i < queue.size(); i++) {
            if (blocks[queue[i]]->device() == lastDevice) {
                nextIndex = i;
                break;
            }
        }
        U64 nextId = blocks[queue[nextIndex]]->id();
        lastDevice = blocks[queue[nextIndex]]->device();
        queue.erase(queue.begin() + nextIndex);
        executionOrder.push_back(nextId);

        for (const auto& nextOutput : blockOutputs[nextId]) {
            for (const auto& [id, _] : computeBlocks) {
                if (std::count_if(blocks[id]->getInputs().begin(),
                                  blocks[id]->getInputs().end(), 
                                  [&](const auto& input) {
                                      return input.hash == nextOutput;
                                  })) {
                    if (--blockDegrees[id] == 0) {
                        queue.push_back(id);
                    }
                }
            }
        }
    }
    const auto actualSize = executionOrder.size();
    const auto expectedSize = std::max(blockInputs.size(), blockOutputs.size());
    if (actualSize != expectedSize) {
        JST_FATAL("Dependency cycle detected. Expected ({}) and actual "
                  "({}) execution order size mismatch.", expectedSize, actualSize);
        return Result::ERROR;       
    }
    JST_TRACE("Naive execution order: {}", executionOrder);

    JST_DEBUG("Calculating graph execution order.");
    lastDevice = Device::None; 
    std::vector<std::pair<Device, std::vector<U64>>> deviceExecutionOrder;
    for (const auto& id : executionOrder) {
        const auto& currentDevice = blocks[id]->device();
        if ((currentDevice & lastDevice) != lastDevice) {
            deviceExecutionOrder.push_back({currentDevice, {}});
        }
        lastDevice = currentDevice;
        deviceExecutionOrder.back().second.push_back(id);
    }

    // Print execution order of blocks.
    JST_INFO("———————————————————————————————————————————————————");
    JST_INFO("Device execution order:", executionOrder);
    JST_INFO("———————————————————————————————————————————————————");
    for (U64 i = 0; i < deviceExecutionOrder.size(); i++) {
        const auto& [device, blocksId] = deviceExecutionOrder[i];
        JST_INFO("    [{:02}] [Device::{}]:", i, device);

        for (U64 j = 0; j < blocksId.size(); j++) {
            const auto& block = blocks[blocksId[j]];
            JST_INFO("        - {}: [{:03}] [{}]", j, block->id(), block->name());
        }
    }
    JST_INFO("———————————————————————————————————————————————————");
   
    // Create graph from execution order.
    for (auto& [device, blocksId] : deviceExecutionOrder) {
        auto graph = NewGraph(device);
        for (auto& blockId : blocksId) {
            graph->schedule(computeBlocks[blockId]);
        }    
        graphs.push_back(std::move(graph));
    }
 
    // Initialize compute logic from modules.
    for (const auto& graph : graphs) {
        JST_CHECK(graph->createCompute());
    }

    // Initialize present logic from modules.
    for (const auto& [id, block] : presentBlocks) {
        JST_CHECK(block->createPresent(*_window));
    }

    // Initialize instance window.
    if (_window) {
        JST_CHECK(_window->create());
    }

    // Lock instance after initialization.
    commited = true;

    return Result::SUCCESS;
}

const Result Instance::destroy() {
    JST_INFO("Destroy compute graph.");

    if (!commited) {
        JST_FATAL("Can't create instance that wasn't created.");
        return Result::ERROR;
    }

    // Destroy instance window.
    if (_window) {
        JST_CHECK(_window->destroy());
    }

    // Destroy present logic from modules.
    for (const auto& [id, block] : presentBlocks) {
        JST_CHECK(block->destroyPresent(*_window));
    }

    // Destroy compute logic from modules.
    for (const auto& graph : graphs) {
        JST_CHECK(graph->destroyCompute());
    }

    return Result::SUCCESS;
}

const Result Instance::compute() {
    Result err = Result::SUCCESS;

    if (!commited) {
        JST_FATAL("The instance wasn't created.");
        return Result::ERROR;
    }

    presentSync.wait(true);
    computeSync.test_and_set();

    for (const auto& graph : graphs) {
        JST_CHECK(graph->compute());
    }

    computeSync.clear();
    computeSync.notify_one();

    return err;
}

const Result Instance::begin() {
    if (_window) {
        return _window->begin();
    }

    return Result::SUCCESS;
}

const Result Instance::present() {
    Result err = Result::SUCCESS;

    if (!commited) {
        JST_FATAL("The instance wasn't commited.");
        return Result::ERROR;
    }

    presentSync.test_and_set();
    computeSync.wait(true);

    for (const auto& [id, block] : presentBlocks) {
        JST_CHECK(block->present(*_window));
    }

    presentSync.clear();
    presentSync.notify_one();

    return err;
}

const Result Instance::end() {
    if (_window) {
        return _window->end();
    }

    return Result::SUCCESS;
}

}  // namespace Jetstream
