#include "jetstream/compute/scheduler.hh"

namespace Jetstream {

// This class will take the raw graph defined by the user and break into execution graphs.
// 1. Identify which modules are compute and graphical.
// 2. Filter Vectors that aren't connected inside graph (external, constants, etc).
// 3. Register all input and output Vectors for each module.
// 4. Break original graph into multiple sub-graphs if there is no dependency between them.
// 5. Create execution order governed by the Vector wiring.
// 6. Break the primitive execution order into final governed by the Vector Device locale.
// 7. Create compute graphs and assign Vectors.
//    - Wired: When a Vector is connected within or externally the graph.
// 8. Calculate and assign Externally Wired Vectors to Graph.
//    - Externally Wired: When a Vector is connected with another graph.
// 9. Assert that an In-Place Module is not sharing a branched input Vector.

// TODO: Automatically add copy module if in-place check fails.

Scheduler::Scheduler(std::shared_ptr<Render::Window>& window,
                     const std::unordered_map<std::string, BlockState>& blockStates,
                     const std::vector<std::string>& blockStateOrder) : window(window) {
    JST_DEBUG("[SCHEDULER] Creating compute graph.");

    // Pre-filter module into present and compute.
    for (const auto& [name, state] : blockStates) {
        if (state.present) {
            presentModuleStates[name].block = &state;
        }
        if (state.compute) {
            computeModuleStates[name].block = &state;
        }
    }

    // Print modules without ordering.
    JST_INFO("——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————");
    JST_INFO("|                                         JETSTREAM INTERMEDIATE COMPUTE GRAPH                                       |")
    for (const auto& name : blockStateOrder) {
        const auto& block = blockStates.at(name);

        if (!block.module) {
            continue;
        }
                
        JST_INFO("——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————");
        JST_INFO("[{}] [{}] [Device::{}] [C: {}, P: {}] [Internal: {}]", block.module->prettyName(),
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
    
    // Process modules into graph.
    ExecutionOrder executionOrder;
    
    JST_CHECK_THROW(removeInactive());
    JST_CHECK_THROW(arrangeDependencyOrder(executionOrder));
    JST_CHECK_THROW(checkSequenceValidity(executionOrder));
    JST_CHECK_THROW(createExecutionGraphs());

    // Initialize compute logic from modules.
    for (const auto& graph : graphs) {
        JST_CHECK_THROW(graph->createCompute());
    }

    // Initialize present logic from modules.
    for (const auto& [_, state] : presentModuleStates) {
        JST_CHECK_THROW(state.block->present->createPresent(*window));
    }
}

Scheduler::~Scheduler() {
    JST_DEBUG("[SCHEDULER] Destroying compute graph.");

    // Destroy present logic from modules.
    for (const auto& [_, state] : presentModuleStates) {
        state.block->present->destroyPresent(*window);
    }

    // Destroy compute logic from modules.
    for (const auto& graph : graphs) {
        graph->destroyCompute();
    }
}

Result Scheduler::compute() {
    Result err = Result::SUCCESS;

    // TODO: Move this graph class internals. Remove blocking.
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

    for (const auto& [_, state] : presentModuleStates) {
        JST_CHECK(state.block->present->present(*window));
    }

    presentSync.clear();
    presentSync.notify_one();

    return err;
}

Result Scheduler::removeInactive() {
    JST_DEBUG("[SCHEDULER] Removing inactive I/O and modules.");
    std::unordered_map<U64, U64> valid;
    for (auto& [name, state] : computeModuleStates) {
        for (const auto& [_, meta] : state.block->record.data.inputMap) {
            valid[meta.vector.hash]++;
        }
        for (const auto& [_, meta] : state.block->record.data.outputMap) {
            valid[meta.vector.hash]++;
        }
    }

    JST_DEBUG("[SCHEDULER] Generating I/O map for each module.");
    for (auto& [name, state] : computeModuleStates) {
        for (const auto& [inputName, meta] : state.block->record.data.inputMap) {
            if (valid[meta.vector.hash] > 1) {
                state.activeInputs[inputName] = &meta.vector;
            } else {
                JST_TRACE("Nulling '{}' input from '{}' module ({:#016x}).", inputName, name, meta.vector.hash);
            }
        }
        for (const auto& [outputName, meta] : state.block->record.data.outputMap) {
            if (valid[meta.vector.hash] > 1) {
                state.activeOutputs[outputName] = &meta.vector;
            } else {
                JST_TRACE("Nulling '{}' output from '{}' module ({:#016x}).", outputName, name, meta.vector.hash);
            }
        }
    }

    JST_DEBUG("[SCHEDULER] Removing stale modules.");
    std::vector<std::string> staleModules;
    for (const auto& [name, state] : computeModuleStates) {
        if (state.activeInputs.empty() && state.activeOutputs.empty()) {
            JST_TRACE("Removing stale module '{}'.", name);
            staleModules.push_back(name);
        }
    }
    for (const auto& name : staleModules) {
        computeModuleStates.erase(name);
    }

    return Result::SUCCESS;
}

Result Scheduler::arrangeDependencyOrder(ExecutionOrder& executionOrder) {
    JST_DEBUG("[SCHEDULER] Calculating module degrees.");
    std::unordered_set<std::string> queue;
    std::unordered_map<std::string, U64> degrees;
    for (const auto& [name, state] : computeModuleStates) {
        degrees[name] = state.activeInputs.size();
        if (state.activeInputs.size() == 0) {
            queue.emplace(name);
        }
    }
    JST_TRACE("Block degrees: {}", degrees);
    JST_TRACE("Initial sorting queue: {}", queue);

    JST_DEBUG("[SCHEDULER] Creating module cache.");
    std::unordered_map<std::string, std::unordered_set<std::string>> moduleEdgesCache;
    std::unordered_map<U64, std::vector<std::string>> moduleInputCache;
    std::unordered_map<U64, std::string> moduleOutputCache;
    
    for (const auto& [name, state] : computeModuleStates) {
        for (const auto& [_, inputMeta] : state.activeInputs) {
            moduleInputCache[inputMeta->phash].push_back(name);
        }
        for (const auto& [_, outputMeta] : state.activeOutputs) {
            moduleOutputCache[outputMeta->phash] = name;
        }
    }

    for (const auto& [name, state] : computeModuleStates) {
        auto& edges = moduleEdgesCache[name];

        for (const auto& [_, inputMeta] : state.activeInputs) {
            edges.insert(moduleOutputCache[inputMeta->phash]);
        }
        for (const auto& [_, outputMeta] : state.activeOutputs) {
            const auto& matches = moduleInputCache[outputMeta->phash];
            edges.insert(matches.begin(), matches.end());
        }
    }
    
    JST_DEBUG("[SCHEDULER] Calculating primitive execution order.");
    Device lastDevice = Device::None; 
    while (!queue.empty()) {
        std::string nextName;
        for (const auto& name : queue) {
            const auto& module = computeModuleStates[name].block->module;
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

        for (const auto& [nextOutputName, nextOutputMeta] : computeModuleStates[nextName].activeOutputs) {
            for (const auto& inputModuleName : moduleInputCache[nextOutputMeta->phash]) {
                if (--degrees[inputModuleName] == 0) {
                    queue.emplace(inputModuleName);
                }
            }
        }
    }
    JST_TRACE("Primitive execution order: {}", executionOrder);
    if (executionOrder.size() != computeModuleStates.size()) {
        JST_FATAL("[SCHEDULER] Dependency cycle detected. Expected ({}) and actual "
                  "({}) execution order size mismatch.", computeModuleStates.size(), executionOrder.size());
        return Result::ERROR;
    }
    JST_TRACE("Primitive execution order: {}", executionOrder);

    JST_DEBUG("[SCHEDULER] Spliting graph into sub-graphs.");
    U64 clusterCount = 0;
    std::unordered_set<std::string> visited;
    for (const auto& [name, state] : computeModuleStates) {
        if (!visited.contains(name)) {
            std::stack<std::string> stack;
            stack.push(name);

            while (!stack.empty()) {
                std::string current = stack.top();
                stack.pop();

                for (const auto& neighbor : moduleEdgesCache[current]) {
                    if (!visited.contains(neighbor)) {
                        stack.push(neighbor);
                    }
                }

                if (!visited.contains(current)) {
                    visited.insert(current);
                    computeModuleStates[current].clusterId = clusterCount;
                }
            }

            clusterCount += 1;
        }
    }

    JST_DEBUG("[SCHEDULER] Calculating graph execution order.");
    lastDevice = Device::None;
    U64 lastCluster = 0;
    for (const auto& name : executionOrder) {
        const auto& module = computeModuleStates[name];
        const auto& currentCluster = module.clusterId;
        const auto& currentDevice = module.block->module->device();

        if ((currentDevice & lastDevice) != lastDevice || 
            currentCluster != lastCluster) {
            deviceExecutionOrder.push_back({currentDevice, {}});
        }

        lastCluster = currentCluster;
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

Result Scheduler::checkSequenceValidity(ExecutionOrder& executionOrder) {
    JST_DEBUG("[SCHEDULER] Gathering modules with inplace operations.");
    std::unordered_map<U64, std::vector<std::string>> inplaceVectorsMap;
    for (const auto& name : executionOrder) {
        auto& state = computeModuleStates[name];
            
        std::unordered_set<U64> inputs; 
        for (const auto& [_, inputMeta] : state.activeInputs) {
            inputs.emplace(inputMeta->hash);
        }

        std::unordered_set<U64> outputs;
        for (const auto& [_, outputMeta] : state.activeOutputs) {
            outputs.emplace(outputMeta->hash);
        }

        std::vector<U64> inplaceVectors;
        std::ranges::set_intersection(inputs, outputs, std::back_inserter(inplaceVectors));

        for (const auto& inplaceVector : inplaceVectors) {
            inplaceVectorsMap[inplaceVector].push_back(name);
        }
    }
    JST_TRACE("In-place module map: {}", inplaceVectorsMap)

    JST_DEBUG("[SCHEDULER] Gathering positional memory layout.");
    std::map<std::pair<U64, U64>, std::vector<std::string>> pMap;
    for (const auto& name : executionOrder) {
        for (const auto& [_, inputMeta] : computeModuleStates[name].activeInputs) {
            pMap[{inputMeta->hash, inputMeta->phash}].push_back(name);
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

Result Scheduler::createExecutionGraphs() {
    JST_DEBUG("[SCHEDULER] Instantiating compute graphs and adding wired Vectors.");
    for (const auto& [device, blocksNames] : deviceExecutionOrder) {
        auto graph = NewGraph(device);

        for (const auto& blockName : blocksNames) {
            auto& state = computeModuleStates[blockName];

            for (const auto& [_, inputMeta] : state.activeInputs) {
                graph->setWiredInput(inputMeta->phash);
            }

            for (const auto& [_, outputMeta] : state.activeOutputs) {
                graph->setWiredOutput(outputMeta->phash);
            }

            graph->setModule(state.block->compute);
        }

        graphs.push_back(std::move(graph));
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

void Scheduler::drawDebugMessage() const {
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::Text("Pipeline:");
    ImGui::TableSetColumnIndex(1);
    ImGui::TextFormatted("{} graph(s)", graphs.size());

    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::Text("Present:");
    ImGui::TableSetColumnIndex(1);
    ImGui::TextFormatted("{} block(s)", presentModuleStates.size());

    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::Text("Compute:");
    ImGui::TableSetColumnIndex(1);
    ImGui::SetNextItemWidth(-1);
    ImGui::TextFormatted("{} block(s)", computeModuleStates.size());

    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::TextFormatted("Graph List:");
    ImGui::TableSetColumnIndex(1);
    ImGui::TextUnformatted("");

    U64 count = 0;
    for (const auto& [device, blocks] : deviceExecutionOrder) {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("");
        ImGui::TableSetColumnIndex(1);
        ImGui::TextFormatted("[{}] {}: {} blocks", count, GetDevicePrettyName(device), blocks.size());
    }
}

}  // namespace Jetstream
