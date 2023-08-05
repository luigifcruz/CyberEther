#include "jetstream/compute/scheduler.hh"

namespace Jetstream {

// This class will take the raw graph defined by the user and break into execution graphs.
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

Scheduler::Scheduler(std::shared_ptr<Render::Window>& window,
                     std::unordered_map<std::string, BlockState>& blockStates,
                     std::unordered_map<U64, std::string>& blockStateMap)
     : blockStates(blockStates),
       blockStateMap(blockStateMap), 
       window(window),
       _computeBlockCount(0),
       _presentBlockCount(0),
       _graphCount(0) {
    JST_DEBUG("[SCHEDULER] Creating compute graph.");

    
    JST_CHECK_THROW(printGraphSummary());
    JST_CHECK_THROW(filterStaleIo());
    JST_CHECK_THROW(applyTopologicalSort());
    JST_CHECK_THROW(createComputeGraphs());
    JST_CHECK_THROW(assertInplaceCorrectness());

    // Initialize compute logic from modules.
    for (const auto& graph : graphs) {
        JST_CHECK_THROW(graph->createCompute());
    }

    // Initialize present logic from modules.
    for (const auto& [_, block] : blockStates) {
        if (block.present) {
            JST_CHECK_THROW(block.present->createPresent(*window));
            _presentBlockCount += 1;
        }
    }
}

Scheduler::~Scheduler() {
    JST_DEBUG("[SCHEDULER] Destroying compute graph.");

    // Destroy present logic from modules.
    for (const auto& [_, block] : blockStates) {
        if (block.present) {
            block.present->destroyPresent(*window);
        }
    }

    // Destroy compute logic from modules.
    for (const auto& graph : graphs) {
        graph->destroyCompute();
    }
}

Result Scheduler::compute() {
    Result err = Result::SUCCESS;

    for (const auto& graph : graphs) {
        JST_CHECK(graph->computeReady());
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

Result Scheduler::present() {
    Result err = Result::SUCCESS;

    presentSync.test_and_set();
    computeSync.wait(true);

    for (const auto& [_, block] : blockStates) {
        if (block.present) {
            JST_CHECK(block.present->present(*window));
        }
    }

    presentSync.clear();
    presentSync.notify_one();

    return err;
}

Result Scheduler::printGraphSummary() {
    JST_INFO("——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————");
    JST_INFO("|                                         JETSTREAM INTERMEDIATE COMPUTE GRAPH                                       |")
    for (U64 i = 0; i < blockStateMap.size(); i++) {
        const auto& name = blockStateMap[i];
        const auto& block = blockStates[name];

        if (!block.module) {
            continue;
        }
                
        JST_INFO("——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————");
        JST_INFO("[{:03}] [{}] [{}] [Device::{}] [C: {}, P: {}] [Internal: {}]", i,
                                                                                 block.module->prettyName(),
                                                                                 name,
                                                                                 block.module->device(),
                                                                                 (block.compute) ? "YES" : "NO",
                                                                                 (block.present) ? "YES" : "NO",
                                                                                 (!block.interface) ? "YES" : "NO");
        JST_INFO("——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————");

        {
            JST_INFO("Configuration:");
            block.module->summary();
        }

        {
            JST_INFO("Block I/O:");

            U64 inputCount = 0;
            const auto& inputs = block.record.data.inputMap;
            JST_INFO("  Inputs:");
            if (inputs.empty()) {
                JST_INFO("    None")
            }
            for (const auto& [name, meta] : inputs) {
                JST_INFO("    {}: [{:>4s}] {} | [Device::{}] | Pointer: 0x{:016X} | Hash: 0x{:016X} | Pos: {:02} | [{}]", inputCount++,
                                                                                                                          meta.vector.type,
                                                                                                                          meta.vector.shape,
                                                                                                                          meta.vector.device,
                                                                                                                          reinterpret_cast<uintptr_t>(meta.vector.ptr),
                                                                                                                          meta.vector.hash, 
                                                                                                                          meta.vector.phash - meta.vector.hash,
                                                                                                                          name);
            }

            U64 outputCount = 0;
            const auto& outputs = block.record.data.outputMap;
            JST_INFO("  Outputs:");
            if (outputs.empty()) {
                JST_INFO("    None")
            }
            for (const auto& [name, meta] : outputs) {
                JST_INFO("    {}: [{:>4s}] {} | [Device::{}] | Pointer: 0x{:016X} | Hash: 0x{:016X} | Pos: {:02} | [{}]", outputCount++,
                                                                                                                          meta.vector.type,
                                                                                                                          meta.vector.shape,
                                                                                                                          meta.vector.device,
                                                                                                                          reinterpret_cast<uintptr_t>(meta.vector.ptr),
                                                                                                                          meta.vector.hash, 
                                                                                                                          meta.vector.phash - meta.vector.hash,
                                                                                                                          name);
            }
        }
    }
    JST_INFO("——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————");

    return Result::SUCCESS;
}

Result Scheduler::filterStaleIo() {
    JST_DEBUG("[SCHEDULER] Filtering stale I/O.");

    std::unordered_map<U64, U64> valid;
    for (const auto& [name, block] : blockStates) {
        if (!block.compute) {
            continue;
        }

        for (auto& [_, meta] : block.record.data.inputMap) {
            valid[meta.vector.hash]++;
        }
        for (auto& [_, meta] : block.record.data.outputMap) {
            valid[meta.vector.hash]++;
        }

        _computeBlockCount += 1;
    }
    for (auto& [key, val] : valid) {
        if (val <= 1) {
            JST_TRACE("Nulling stale I/O: {:#016x}", key);
            valid[key] = 0;
        }
    }

    JST_DEBUG("[SCHEDULER] Generating I/O map for each block.");
    for (auto& [name, block] : blockStates) {
        if (!block.compute) {
            continue;
        }

        for (auto& [key, meta] : block.record.data.inputMap) {
            if (valid[meta.vector.hash] != 0) {
                block.activeInputs.emplace(key);
            }
        }
        for (auto& [key, meta] : block.record.data.outputMap) {
            if (valid[meta.vector.hash] != 0) {
                block.activeOutputs.emplace(key);
            }
        }
    }

    return Result::SUCCESS;
}

Result Scheduler::applyTopologicalSort() {
    JST_DEBUG("[SCHEDULER] Calculating block degrees.");
    std::unordered_set<std::string> queue;
    std::unordered_map<std::string, U64> degrees;
    for (auto& [name, block] : blockStates) {
        if (!block.compute) {
            continue;
        }

        const auto& degree = block.activeInputs.size();
        if (degree == 0) {
            queue.emplace(name);
        }
        degrees[name] = degree;
    }
    JST_TRACE("Block degrees: {}", degrees);
    JST_TRACE("Initial sorting queue: {}", queue);

    if (queue.empty()) {
        JST_FATAL("[SCHEDULER] No initial block found. Graph must be directed acyclic.");
        return Result::ERROR;
    }
    
    JST_DEBUG("[SCHEDULER] Calculating primitive execution order.");
    Device lastDevice = Device::None; 
    while (!queue.empty()) {
        std::string nextName;
        for (const auto& name : queue) {
            const auto& module = blockStates[name].module;
            if (lastDevice == Device::None) {
                lastDevice = module->device();
            }
            if (module->device() == lastDevice) {
                lastDevice = module->device();
                nextName = name;
                break;
            }
        }
        JST_ASSERT(!nextName.empty());
        queue.erase(queue.find(nextName));
        executionOrder.push_back(nextName);

        for (const auto& [nextOutputName, nextOutputMeta] : blockStates[nextName].record.data.outputMap) {
            if (!blockStates[nextName].activeOutputs.contains(nextOutputName)) {
                continue;
            }

            for (const auto& [name, block] : blockStates) {
                if (!block.compute) {
                    continue;
                }

                bool commit = false;
                for (const auto& [inputName, inputMeta] : blockStates[name].record.data.inputMap) {
                    if (!blockStates[name].activeInputs.contains(inputName)) {
                        continue;
                    }

                    if (nextOutputMeta.vector.phash == inputMeta.vector.phash) {
                        commit = true;
                        break;
                    }
                }
                
                if (commit && --degrees[name] == 0) {
                    queue.emplace(name);
                }
            }
        }
    }
    U64 expectedSize = 0;
    for (const auto& [_, state] : blockStates) {
        if (!state.activeInputs.empty() || !state.activeOutputs.empty()) {
            expectedSize += 1;
        }
    }
    const auto actualSize = executionOrder.size();
    if (actualSize != expectedSize) {
        JST_FATAL("[SCHEDULER] Dependency cycle detected. Expected ({}) and actual "
                  "({}) execution order size mismatch.", expectedSize, actualSize);
        return Result::ERROR;       
    }
    JST_TRACE("Naive execution order: {}", executionOrder);

    JST_DEBUG("[SCHEDULER] Calculating graph execution order.");
    lastDevice = Device::None; 
    for (const auto& name : executionOrder) {
        const auto& currentDevice = blockStates[name].module->device();
        if ((currentDevice & lastDevice) != lastDevice) {
            deviceExecutionOrder.push_back({currentDevice, {}});
        }
        lastDevice = currentDevice;
        deviceExecutionOrder.back().second.push_back(name);
    }

    JST_INFO("———————————————————————————————————————————————————");
    JST_INFO("Device execution order:");
    JST_INFO("———————————————————————————————————————————————————");
    for (U64 i = 0; i < deviceExecutionOrder.size(); i++) {
        const auto& [device, blocksNames] = deviceExecutionOrder[i];
        JST_INFO("  [{:02}] [Device::{}]: {}", i, device, blocksNames);
    }
    JST_INFO("———————————————————————————————————————————————————");

    return Result::SUCCESS;
}

Result Scheduler::createComputeGraphs() {
    JST_DEBUG("[SCHEDULER] Instantiating compute graphs and adding wired Vectors.");
    for (const auto& [device, blocksNames] : deviceExecutionOrder) {
        auto graph = NewGraph(device);
        for (const auto& blockName : blocksNames) {
            auto& block = blockStates[blockName];
            for (const auto& inputName : block.activeInputs) {
                graph->setWiredInput(block.record.data.inputMap[inputName].vector.phash);
            }
            for (const auto& outputsName : block.activeOutputs) {
                graph->setWiredOutput(block.record.data.outputMap[outputsName].vector.phash);
            }
            graph->setModule(block.compute);
        }
        graphs.push_back(std::move(graph));

        _computeDevices.emplace(device);
        _graphCount += 1;
    }

    JST_DEBUG("[SCHEDULER] Creating dependency list between graphs.");
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

        for (const auto& item : commonItems) {
            previousGraph->setExternallyWiredOutput(item);
            currentGraph->setExternallyWiredInput(item);
        }
    }
 
    return Result::SUCCESS;
}

Result Scheduler::assertInplaceCorrectness() {
    JST_DEBUG("[SCHEDULER] Gathering modules with inplace operations.");
    std::unordered_map<U64, std::vector<std::string>> inplaceVectorsMap;
    for (const auto& name : executionOrder) {
        auto& block = blockStates[name];
            
        std::unordered_set<U64> inputs; 
        for (const auto& inputName : block.activeInputs) {
            inputs.emplace(block.record.data.inputMap[inputName].vector.hash);
        }

        std::unordered_set<U64> outputs;
        for (const auto& outputName : block.activeOutputs) {
            outputs.emplace(block.record.data.outputMap[outputName].vector.hash);
        }

        std::vector<U64> inplaceVectors;
        std::ranges::set_intersection(inputs, outputs, std::back_inserter(inplaceVectors));

        for (auto& inplaceVector : inplaceVectors) {
            inplaceVectorsMap[inplaceVector].push_back(name);
        }
    }
    JST_TRACE("In-place module map: {}", inplaceVectorsMap)

    JST_DEBUG("[SCHEDULER] Gathering positional memory layout.");
    std::map<std::pair<U64, U64>, std::vector<std::string>> pMap;
    for (const auto& name : executionOrder) {
        auto& block = blockStates[name];
        for (const auto& inputName : block.activeInputs) {
            const auto& vectorMeta = block.record.data.inputMap[inputName].vector;
            pMap[{vectorMeta.hash, vectorMeta.phash}].push_back(name);
        }
    }
    JST_TRACE("In-place vector map: {}", pMap)

    JST_DEBUG("[SCHEDULER] Asserting that positional memory layout meets in-place requirements.");
    for (const auto& [hashes, blocks] : pMap) {
        const auto& [hash, phash] = hashes;

        if (blocks.size() <= 1) {
            continue;
        }

        if (inplaceVectorsMap.count(hash) > 0) {
            std::vector<std::string> inplaceModules; 
            std::ranges::set_intersection(blocks,
                                          inplaceVectorsMap[hash],
                                          std::back_inserter(inplaceModules));
            if (inplaceModules.size() > 0) {
                JST_FATAL("[SCHEDULER] Vector is being shared by at least two modules after a branch "
                          "and at least one of them is an in-place module.");
                JST_FATAL("    Hash: 0x{:016x} | Pos: {} | Modules: {}", hash,
                                                                         phash - hash,
                                                                         blocks);
                return Result::ERROR;
            }
        }
    }

    return Result::SUCCESS;
}

}  // namespace Jetstream
