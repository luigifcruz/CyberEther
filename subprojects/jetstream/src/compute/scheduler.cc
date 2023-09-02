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
// TODO: Redo PHash logic with locale.

Result Scheduler::addModule(const Locale& locale, const std::shared_ptr<BlockState>& block) {
    JST_DEBUG("[SCHEDULER] Adding new module '{}' to the pipeline.", locale);

    if (!block->module) {
        JST_DEBUG("[SCHEDULER] Ignoring non-module block.");
        return Result::SUCCESS;
    }

    // Print new module metadata.
    JST_INFO("——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————");
    JST_INFO("[{}] [{}] [Device::{}] [C: {}, P: {}] [Internal: {}]", block->module->prettyName(),
                                                                     locale,
                                                                     block->module->device(),
                                                                     (block->compute) ? "YES" : "NO",
                                                                     (block->present) ? "YES" : "NO",
                                                                     (!block->interface) ? "YES" : "NO");
    JST_INFO("——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————");

    {
        JST_INFO("Configuration:");
        block->module->summary();
    }

    {
        JST_INFO("Block I/O:");

        U64 inputCount = 0;
        const auto& inputs = block->record.inputMap;
        JST_INFO("  Inputs:");
        if (inputs.empty()) {
            JST_INFO("    None")
        }
        for (const auto& [name, meta] : inputs) {
            JST_INFO("    {}: [{:>4s}] {} | [Device::{}] | Pointer: 0x{:016X} | Hash: 0x{:016X} | [{}]", inputCount++,
                                                                                                         meta.dataType,
                                                                                                         meta.shape,
                                                                                                         meta.device,
                                                                                                         reinterpret_cast<uintptr_t>(meta.data),
                                                                                                         meta.hash,
                                                                                                         name);
        }

        U64 outputCount = 0;
        const auto& outputs = block->record.outputMap;
        JST_INFO("  Outputs:");
        if (outputs.empty()) {
            JST_INFO("    None")
        }
        for (const auto& [name, meta] : outputs) {
            JST_INFO("    {}: [{:>4s}] {} | [Device::{}] | Pointer: 0x{:016X} | Hash: 0x{:016X} | [{}]", outputCount++,
                                                                                                         meta.dataType,
                                                                                                         meta.shape,
                                                                                                         meta.device,
                                                                                                         reinterpret_cast<uintptr_t>(meta.data),
                                                                                                         meta.hash,
                                                                                                         name);
        }
    }
    JST_INFO("——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————");

    // Stop execution.
    lock();

    // Destroy compute logic from modules.
    for (const auto& graph : graphs) {
        graph->destroy();
    }

    // Add module to present and/or compute.
    if (block->present) {
        presentModuleStates[locale.str()].block = block;
    }
    if (block->compute) {
        computeModuleStates[locale.str()].block = block;
    }

    // Process modules into graph.
    JST_CHECK(removeInactive());
    JST_CHECK(arrangeDependencyOrder());
    JST_CHECK(checkSequenceValidity());
    JST_CHECK(createExecutionGraphs());

    // Initialize graphs.
    for (const auto& graph : graphs) {
        JST_CHECK_THROW(graph->create());
    }

    // Restart execution.
    unlock();

    return Result::SUCCESS;
}

Result Scheduler::removeModule(const Locale& locale) {
    JST_DEBUG("[SCHEDULER] Removing module '{}' from the pipeline.", locale);

    // Stop execution.
    lock();

    // Destroy compute logic from modules.
    for (const auto& graph : graphs) {
        graph->destroy();
    }

    // Remove module from present and/or compute.
    if (presentModuleStates.contains(locale.str())) {
        presentModuleStates.erase(locale.str());
    }
    if (computeModuleStates.contains(locale.str())) {
        computeModuleStates.erase(locale.str());
    }

    // Process modules into graph.
    JST_CHECK(removeInactive());
    JST_CHECK(arrangeDependencyOrder());
    JST_CHECK(checkSequenceValidity());
    JST_CHECK(createExecutionGraphs());

    // Initialize graphs.
    for (const auto& graph : graphs) {
        JST_CHECK_THROW(graph->create());
    }

    // Restart execution.
    unlock();

    return Result::SUCCESS;
}

Result Scheduler::destroy() {
    JST_DEBUG("[SCHEDULER] Destroying compute graph.");

    // Stop execution.
    lock();

    // Destroy compute logic from modules.
    for (const auto& graph : graphs) {
        graph->destroy();
    }

    // Blanks internal memory.
    computeModuleStates.clear();
    presentModuleStates.clear();
    validComputeModuleStates.clear();
    validPresentModuleStates.clear();
    executionOrder.clear();
    deviceExecutionOrder.clear();
    graphs.clear();

    // Restart execution.
    unlock();

    return Result::SUCCESS;
}

Result Scheduler::compute() {
    if (computeHalt.test()) {
        computeHalt.wait(true);
        return Result::SUCCESS;
    }

    // The state cannot change while we are waiting for
    // a module to finish computing. This is a workaround
    // blocking the state from changing while we are waiting.
    // TODO: Replace with something else can can cancel the wait.
    {
        computeWait.test_and_set();

        wait:
        for (const auto& graph : graphs) {
            const auto& res = graph->computeReady();
            if (res == Result::TIMEOUT) {
                // Yes, I used a goto. Sue me.
                goto wait;
            }
            JST_CHECK(res);
        }

        computeWait.clear();
        computeWait.notify_all();
    }

    presentSync.wait(true);
    computeSync.test_and_set();

    for (const auto& graph : graphs) {
        JST_CHECK(graph->compute());
    }

    computeSync.clear();
    computeSync.notify_all();

    return Result::SUCCESS;
}

Result Scheduler::present() {
    if (presentHalt.test()) {
        return Result::SUCCESS;
    }

    presentSync.test_and_set();
    computeSync.wait(true);

    for (const auto& [_, state] : validPresentModuleStates) {
        JST_CHECK(state.block->present->present());
    }

    presentSync.clear();
    presentSync.notify_all();

    return Result::SUCCESS;
}

void Scheduler::lock() {
    presentHalt.test_and_set();
    computeHalt.test_and_set();
    computeWait.wait(true);
    presentSync.wait(true);
    computeSync.wait(true);
}

void Scheduler::unlock() {
    presentHalt.clear();
    presentHalt.notify_all();
    computeHalt.clear();
    computeHalt.notify_all();
}

Result Scheduler::removeInactive() {
    validComputeModuleStates.clear();
    validPresentModuleStates.clear();

    JST_DEBUG("[SCHEDULER] Removing inactive I/O.");
    std::unordered_map<U64, U64> valid;
    for (const auto& [name, state] : computeModuleStates) {
        for (const auto& [_, meta] : state.block->record.inputMap) {
            if (meta.hash) {
                valid[meta.hash]++;
            }
        }
        for (const auto& [_, meta] : state.block->record.outputMap) {
            if (meta.hash) {
                valid[meta.hash]++;
            }
        }
    }

    JST_DEBUG("[SCHEDULER] Generating I/O map for each module.");
    for (auto& [name, state] : computeModuleStates) {
        for (const auto& [inputName, meta] : state.block->record.inputMap) {
            if (valid[meta.hash] > 1) {
                state.activeInputs[inputName] = &meta;
            } else {
                JST_TRACE("Nulling '{}' input from '{}' module ({:#016x}).", inputName, name, meta.hash);
            }
        }
        for (const auto& [outputName, meta] : state.block->record.outputMap) {
            if (valid[meta.hash] > 1) {
                state.activeOutputs[outputName] = &meta;
            } else {
                JST_TRACE("Nulling '{}' output from '{}' module ({:#016x}).", outputName, name, meta.hash);
            }
        }
    }

    JST_DEBUG("[SCHEDULER] Removing stale modules.");
    std::unordered_set<std::string> staleModules;
    for (const auto& [name, state] : computeModuleStates) {
        if (state.activeInputs.empty() && state.activeOutputs.empty()) {
            JST_TRACE("Removing stale module '{}'.", name);
            staleModules.insert(name);
        }
    }
    for (const auto& [name, state] : computeModuleStates) {
        if (!staleModules.contains(name)) {
            validComputeModuleStates[name] = state;
        }
    }
    for (const auto& [name, state] : presentModuleStates) {
        if (!staleModules.contains(name)) {
            validPresentModuleStates[name] = state;
        }
    }

    return Result::SUCCESS;
}

Result Scheduler::arrangeDependencyOrder() {
    executionOrder.clear();
    deviceExecutionOrder.clear();

    JST_DEBUG("[SCHEDULER] Calculating module degrees.");
    std::unordered_set<std::string> queue;
    std::unordered_map<std::string, U64> degrees;
    for (const auto& [name, state] : validComputeModuleStates) {
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

    for (const auto& [name, state] : validComputeModuleStates) {
        for (const auto& [_, inputMeta] : state.activeInputs) {
            moduleInputCache[inputMeta->locale.hash()].push_back(name);
        }
        for (const auto& [_, outputMeta] : state.activeOutputs) {
            moduleOutputCache[outputMeta->locale.hash()] = name;
        }
    }

    for (const auto& [name, state] : validComputeModuleStates) {
        auto& edges = moduleEdgesCache[name];

        for (const auto& [_, inputMeta] : state.activeInputs) {
            edges.insert(moduleOutputCache[inputMeta->locale.hash()]);
        }
        for (const auto& [_, outputMeta] : state.activeOutputs) {
            const auto& matches = moduleInputCache[outputMeta->locale.hash()];
            edges.insert(matches.begin(), matches.end());
        }
    }

    JST_DEBUG("[SCHEDULER] Calculating primitive execution order.");
    Device lastDevice = Device::None;
    while (!queue.empty()) {
        std::string nextName;
        for (const auto& name : queue) {
            const auto& module = validComputeModuleStates[name].block->module;
            if (lastDevice == Device::None) {
                lastDevice = module->device();
            }
            if (module->device() == lastDevice) {
                lastDevice = module->device();
                nextName = name;
                break;
            }
        }
        if (nextName.empty()) {
            lastDevice = Device::None;
            continue;
        }
        JST_ASSERT(!nextName.empty());
        queue.erase(queue.find(nextName));
        executionOrder.push_back(nextName);

        for (const auto& [nextOutputName, nextOutputMeta] : validComputeModuleStates[nextName].activeOutputs) {
            for (const auto& inputModuleName : moduleInputCache[nextOutputMeta->locale.hash()]) {
                if (--degrees[inputModuleName] == 0) {
                    queue.emplace(inputModuleName);
                }
            }
        }
    }
    if (executionOrder.size() != validComputeModuleStates.size()) {
        JST_ERROR("[SCHEDULER] Dependency cycle detected. Expected ({}) and actual "
                  "({}) execution order size mismatch.", validComputeModuleStates.size(), executionOrder.size());
        return Result::ERROR;
    }
    JST_TRACE("Primitive execution order: {}", executionOrder);

    JST_DEBUG("[SCHEDULER] Spliting graph into sub-graphs.");
    U64 clusterCount = 0;
    std::unordered_set<std::string> visited;
    for (const auto& [name, state] : validComputeModuleStates) {
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
                    validComputeModuleStates[current].clusterId = clusterCount;
                }
            }

            clusterCount += 1;
        }
    }

    JST_DEBUG("[SCHEDULER] Calculating graph execution order.");
    lastDevice = Device::None;
    U64 lastCluster = 0;
    for (const auto& name : executionOrder) {
        const auto& module = validComputeModuleStates[name];
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

Result Scheduler::checkSequenceValidity() {
    JST_DEBUG("[SCHEDULER] Gathering modules with inplace operations.");
    std::unordered_map<U64, std::vector<std::string>> inplaceVectorsMap;
    for (const auto& name : executionOrder) {
        auto& state = validComputeModuleStates[name];

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
        for (const auto& [_, inputMeta] : validComputeModuleStates[name].activeInputs) {
            pMap[{inputMeta->hash, inputMeta->locale.hash()}].push_back(name);
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
                JST_ERROR("[SCHEDULER] Vector is being shared by at least two modules after a branch "
                          "and at least one of them is an in-place module.");
                JST_ERROR("    Hash: 0x{:016x} | Pos: {} | Modules: {}", hash,
                                                                         phash - hash,
                                                                         blocks);
                return Result::ERROR;
            }
        }
    }

    return Result::SUCCESS;
}

Result Scheduler::createExecutionGraphs() {
    graphs.clear();

    JST_DEBUG("[SCHEDULER] Instantiating compute graphs and adding wired Vectors.");
    for (const auto& [device, blocksNames] : deviceExecutionOrder) {
        auto graph = NewGraph(device);

        for (const auto& blockName : blocksNames) {
            auto& state = validComputeModuleStates[blockName];

            for (const auto& [_, inputMeta] : state.activeInputs) {
                graph->setWiredInput(inputMeta->locale.hash());
            }

            for (const auto& [_, outputMeta] : state.activeOutputs) {
                graph->setWiredOutput(outputMeta->locale.hash());
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
    ImGui::TextFormatted("{} block(s)", validPresentModuleStates.size());

    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::Text("Compute:");
    ImGui::TableSetColumnIndex(1);
    ImGui::SetNextItemWidth(-1);
    ImGui::TextFormatted("{} block(s)", validComputeModuleStates.size());

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
