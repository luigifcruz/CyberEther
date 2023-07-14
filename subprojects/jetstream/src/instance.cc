#include "jetstream/instance.hh"
#include "jetstream/memory/types.hh"

namespace Jetstream {

Result Instance::printGraphSummary() {
    for (const auto& block : blocks) {
        JST_INFO("——————————————————————————————————————————————————————————————————————————————————————————————————————————");
        JST_INFO("[{:03}] [{}] [Device::{}] [C: {}, P: {}]", block->id(),
                                                             block->name(),
                                                             block->device(),
                                                             (computeBlocks.count(block->id())) ? "YES" : "NO",
                                                             (presentBlocks.count(block->id())) ? "YES" : "NO");
        JST_INFO("——————————————————————————————————————————————————————————————————————————————————————————————————————————");

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
            for (U64 i = 0; i < inputs.size(); i++) {
                JST_INFO("        {}: [{:>4s}] {} | [Device::{}] | Pointer: {} | Hash: 0x{:016x} | Pos: {}", i,
                                                                                                             inputs[i].type,
                                                                                                             inputs[i].shape,
                                                                                                             inputs[i].device,
                                                                                                             fmt::ptr(inputs[i].ptr),
                                                                                                             inputs[i].hash, 
                                                                                                             inputs[i].phash - inputs[i].hash);
            }

            const auto& outputs = block->getOutputs();
            JST_INFO("    Outputs:");
            if (outputs.size() == 0) {
                JST_INFO("        None")
            }
            for (U64 i = 0; i < outputs.size(); i++) {
                JST_INFO("        {}: [{:>4s}] {} | [Device::{}] | Pointer: {} | Hash: 0x{:016x} | Pos: {}", i,
                                                                                                             outputs[i].type,
                                                                                                             outputs[i].shape,
                                                                                                             outputs[i].device,
                                                                                                             fmt::ptr(outputs[i].ptr),
                                                                                                             outputs[i].hash,
                                                                                                             outputs[i].phash - outputs[i].hash);
            }
        }
    }
    JST_INFO("——————————————————————————————————————————————————————————————————————————————————————————————————————————");

    return Result::SUCCESS;
}

Result Instance::filterStaleIo() {
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
    for (const auto& [id, _] : computeBlocks) {
        for (auto input : blocks[id]->getInputs()) {
            if (ioValid[input.hash] != 0) {
                 blockInputs[id].push_back(input.hash);
                 blockInputsPos[id].push_back(input.phash);
            }
        }
        for (auto output : blocks[id]->getOutputs()) {
            if (ioValid[output.hash] != 0) {
                blockOutputs[id].push_back(output.hash);
                blockOutputsPos[id].push_back(output.phash);
            }
        }
    }

    return Result::SUCCESS;
}

Result Instance::applyTopologicalSort() {
    JST_DEBUG("Calculating block degrees.");
    std::unordered_map<U64, U64> blockDegrees;
    for (const auto& [id, _] : computeBlocks) {
        blockDegrees[id] = blockInputsPos[id].size();
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

        for (const auto& nextOutput : blockOutputsPos[nextId]) {
            for (const auto& [id, _] : computeBlocks) {
                if (std::count_if(blocks[id]->getInputs().begin(),
                                  blocks[id]->getInputs().end(), 
                                  [&](const auto& input) {
                                      return input.phash == nextOutput;
                                  })) {
                    if (--blockDegrees[id] == 0) {
                        queue.push_back(id);
                    }
                }
            }
        }
    }
    const auto actualSize = executionOrder.size();
    const auto expectedSize = std::max(blockInputsPos.size(), blockOutputsPos.size());
    if (actualSize != expectedSize) {
        JST_FATAL("Dependency cycle detected. Expected ({}) and actual "
                  "({}) execution order size mismatch.", expectedSize, actualSize);
        return Result::ERROR;       
    }
    JST_TRACE("Naive execution order: {}", executionOrder);

    JST_DEBUG("Calculating graph execution order.");
    lastDevice = Device::None; 
    for (const auto& id : executionOrder) {
        const auto& currentDevice = blocks[id]->device();
        if ((currentDevice & lastDevice) != lastDevice) {
            deviceExecutionOrder.push_back({currentDevice, {}});
        }
        lastDevice = currentDevice;
        deviceExecutionOrder.back().second.push_back(id);
    }

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

    return Result::SUCCESS;
}

Result Instance::createComputeGraphs() {
    JST_DEBUG("Instantiating compute graphs and adding wired Vectors.");
    for (const auto& [device, blocksId] : deviceExecutionOrder) {
        auto graph = NewGraph(device);
        for (const auto& blockId : blocksId) {
            graph->setWiredInputs(blockInputs[blockId]);
            graph->setWiredOutputs(blockOutputs[blockId]);
            graph->setModule(computeBlocks[blockId]);
        }
        graphs.push_back(std::move(graph));
    }

    JST_DEBUG("Creating dependency list between graphs.");
    std::shared_ptr<Graph> previousGraph;
    for (auto& currentGraph : graphs) {
        if (!previousGraph) {
            previousGraph = currentGraph;       
            continue;
        }

        std::vector<U64> commonItems;
        std::ranges::set_intersection(previousGraph->getWiredOutputs(),
                                      currentGraph->getWiredInputs(),
                                      std::back_inserter(commonItems));

        previousGraph->setExternallyWiredOutputs(commonItems);
        currentGraph->setExternallyWiredInputs(commonItems);
    }
 
    return Result::SUCCESS;
}

Result Instance::assertInplaceCorrectness() {
    JST_DEBUG("Gathering modules with inplace operations.");
    std::unordered_map<U64, std::vector<U64>> inplaceVectorsMap;
    for (U64 i = 0; i < executionOrder.size(); i++) {
        const auto& blockId = executionOrder[i];
            
        const std::set<U64> inputs(blockInputs[blockId].begin(), 
                                   blockInputs[blockId].end());
        const std::set<U64> outputs(blockOutputs[blockId].begin(),
                                    blockOutputs[blockId].end());

        std::vector<U64> inplaceVectors;
        std::ranges::set_intersection(inputs, outputs, std::back_inserter(inplaceVectors));

        for (auto& inplaceVector : inplaceVectors) {
            inplaceVectorsMap[inplaceVector].push_back(i);
        }
    }
    JST_TRACE("In-place module map: {}", inplaceVectorsMap)

    JST_DEBUG("Gathering positional memory layout.");
    std::map<std::pair<U64, U64>, std::vector<U64>> pMap;
    for (U64 i = 0; i < executionOrder.size(); i++) {
        const auto& blockId = executionOrder[i];
        for (U64 j = 0; j < blockInputs[blockId].size(); j++) {
            pMap[{blockInputs[blockId][j], blockInputsPos[blockId][j]}].push_back(i);
        }
    }
    JST_TRACE("In-place vector map: {}", pMap)

    JST_DEBUG("Asserting that positional memory layout meets in-place requirements.");
    for (const auto& [hashes, blocks] : pMap) {
        const auto& [hash, phash] = hashes;

        if (blocks.size() <= 1) {
            continue;
        }

        if (inplaceVectorsMap.count(hash) > 0) {
            std::vector<U64> inplaceModules; 
            std::ranges::set_intersection(blocks,
                                          inplaceVectorsMap[hash],
                                          std::back_inserter(inplaceModules));
            if (inplaceModules.size() > 0) {
                JST_FATAL("Vector is being shared by at least two modules after a branch "
                          "and at least one of them is a In-Place Module.");
                JST_FATAL("    Hash: 0x{:016x} | Pos: {} | Modules: {}", hash,
                                                                         phash - hash,
                                                                         blocks);
                return Result::ERROR;
            }
        }
    }

    return Result::SUCCESS;
}

Result Instance::create() {
    JST_DEBUG("Creating compute graph.");

    if (commited) {
        JST_FATAL("The instance was already commited.");
        return Result::ERROR;
    }

    // This code will take the raw graph defined by the user and break into execution graphs.
    // 1. Identify which blocks are compute and graphical.
    // 2. Filter Vectors that aren't connected inside graph (external, constants, etc).
    // 3. Register all input and output Vectors for each block.
    // 4. Create execution order governed by the Vector wiring.
    // 5. Break the primitive execution order into final governed by the Vector Device locale.
    // 6. Create compute graphs and assign Vectors.
    //    - Wired: When a Vector is connected within or externally the graph.
    // 7. Calculate and assign Externally Wired Vectors to Graph.
    //    - Externally Wired: When a Vector is connected with another graph.
    // 8. Assert that an In-Place Module is not sharing a branched input Vector.

    // TODO: Write a test suite.
    
    for (const auto& moduleBlock : blocks) {
        if (auto computeBlock = std::dynamic_pointer_cast<Compute>(moduleBlock)) {
            computeBlocks[moduleBlock->id()] = computeBlock;
        }
    }

    for (const auto& moduleBlock : blocks) {
        if (auto presentBlock = std::dynamic_pointer_cast<Present>(moduleBlock)) {
            if (!_window || !_viewport) {
                JST_ERROR("A window and viewport are required because a graphical block was added.");
                return Result::ERROR;
            }
            presentBlocks[moduleBlock->id()] = presentBlock;
        }
    }

    JST_CHECK(printGraphSummary());
    JST_CHECK(filterStaleIo());
    JST_CHECK(applyTopologicalSort());
    JST_CHECK(createComputeGraphs());
    JST_CHECK(assertInplaceCorrectness());

    // Initialize compute logic from modules.
    for (const auto& graph : graphs) {
        JST_CHECK(graph->createCompute());
    }

    // Initialize present logic from modules.
    for (const auto& [id, block] : presentBlocks) {
        JST_CHECK(block->createPresent(*_window));
    }

    // Initialize instance window.
    if (_window && _viewport) {
        JST_CHECK(_viewport->create());
        JST_CHECK(_window->create());
    }

    // Lock instance after initialization.
    commited = true;

    return Result::SUCCESS;
}

Result Instance::destroy() {
    JST_INFO("Destroy compute graph.");

    if (!commited) {
        JST_FATAL("Can't create instance that wasn't created.");
        return Result::ERROR;
    }

    // Destroy instance window.
    if (_window) {
        JST_CHECK(_window->destroy());
        JST_CHECK(_viewport->destroy());
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

Result Instance::compute() {
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

Result Instance::begin() {
    if (_window) {
        return _window->begin();
    }

    return Result::SUCCESS;
}

Result Instance::present() {
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

Result Instance::end() {
    if (_window) {
        return _window->end();
    }

    return Result::SUCCESS;
}

}  // namespace Jetstream
