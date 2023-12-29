#include <regex>

#include "jetstream/benchmark.hh"
#include "jetstream/compositor.hh"
#include "jetstream/instance.hh"
#include "jetstream/store.hh"
#include "jetstream/platform.hh"
#include "jetstream/tools/stb_image.hh"

#include "resources/resources.hh"

namespace Jetstream {

Compositor::Compositor(Instance& instance)
     : instance(instance),
       running(true),
       graphSpatiallyOrganized(false),
       rightClickMenuEnabled(false),
       sourceEditorEnabled(false),
       moduleStoreEnabled(true),
       infoPanelEnabled(true),
       flowgraphEnabled(true),
       debugDemoEnabled(false),
       debugLatencyEnabled(false),
       debugEnableTrace(false),
       globalModalContentId(0),
       nodeContextMenuNodeId(0),
       globalModalToggle(false) {
    stacks["Graph"] = {true, 0};
    JST_CHECK_THROW(refreshState());
}

Result Compositor::addBlock(const Locale& locale,
                            const std::shared_ptr<Block>& block, 
                            const Parser::RecordMap& inputMap, 
                            const Parser::RecordMap& outputMap, 
                            const Parser::RecordMap& stateMap,
                            const Block::Fingerprint& fingerprint) {
    JST_DEBUG("[COMPOSITOR] Adding block '{}'.", locale);

    // Make sure compositor is running.
    running = true;
    globalModalToggle = false;

    // Check if the locale is already created.
    if (nodeStates.contains(locale)) {
        JST_ERROR("[COMPOSITOR] Entry for node '{}' already exists.", locale);
        return Result::ERROR;
    }

    // Prevent drawMainInterface from running.
    lock();

    // Save block in node state.

    auto& nodeState = nodeStates[locale];

    nodeState.block = block;
    nodeState.inputMap = inputMap;
    nodeState.outputMap = outputMap;
    nodeState.stateMap = stateMap;
    nodeState.fingerprint = fingerprint;
    nodeState.title = fmt::format("{} ({})", block->name(), locale);

    JST_CHECK(refreshState());

    // Resume drawMainInterface.
    unlock();

    return Result::SUCCESS;
}

Result Compositor::removeBlock(const Locale& locale) {
    JST_DEBUG("[COMPOSITOR] Removing block '{}'.", locale);

    // Return early if the scheduler is not running.
    if (!running) {
        return Result::SUCCESS;
    }

    // Check if the locale is already created.
    if (!nodeStates.contains(locale)) {
        JST_ERROR("[COMPOSITOR] Entry for node '{}' doesn't exist.", locale);
        return Result::ERROR;
    }

    // Prevent drawMainInterface from running.
    lock();

    // Save block in node state.
    nodeStates.erase(locale);

    JST_CHECK(refreshState());

    // Resume drawMainInterface.
    unlock();

    return Result::SUCCESS;
}

Result Compositor::refreshState() {
    JST_DEBUG("[COMPOSITOR] Refreshing interface state.");

    // Create interface state, input, output and pin cache.
    pinLocaleMap.clear();
    nodeLocaleMap.clear();
    inputLocalePinMap.clear();
    outputLocalePinMap.clear();
    outputInputCache.clear();

    U64 id = 1;
    for (auto& [locale, state] : nodeStates) {
        // Generate id for node.
        state.id = id;
        nodeLocaleMap[id++] = locale;

        // Cleanup and create pin map and convert locale to interface locale.

        state.inputs.clear();
        state.outputs.clear();

        for (const auto& [inputPinId, inputRecord] : state.inputMap) {
            // Generate clean locale.
            Locale inputLocale = {locale.blockId, "", inputPinId};

            // Save the input pin locale.
            pinLocaleMap[id] = inputLocale;
            inputLocalePinMap[inputLocale] = id;

            // Generate clean locale.
            Locale outputLocale = inputRecord.locale.pin();

            // Save the incoming input locale.
            state.inputs[id++] = outputLocale;

            // Save the output to input locale map cache.
            outputInputCache[outputLocale].push_back(inputLocale);
        }

        for (const auto& [outputPinId, outputRecord] : state.outputMap) {
            // Generate clean locale.
            Locale outputLocale = {locale.blockId, "", outputPinId};

            // Save the output pin locale.
            pinLocaleMap[id] = outputLocale;
            outputLocalePinMap[outputLocale] = id;

            // Save the outgoing output locale.
            state.outputs[id++] = outputLocale;
        }
    }

    // Create link and edges.
    linkLocaleMap.clear();

    U64 linkId = 0;
    for (auto& [locale, state] : nodeStates) {
        // Cleanup buffers.
        state.edges.clear();

        for (const auto& [_, inputLocale] : state.inputs) {
            if (nodeStates.contains(inputLocale.block())) {
                state.edges.insert(nodeStates.at(inputLocale.block()).id);
            }
        }

        for (const auto& [_, outputLocale] : state.outputs) {
            for (const auto& inputLocale : outputInputCache[outputLocale]) {
                state.edges.insert(nodeStates.at(inputLocale.block()).id);

                linkLocaleMap[linkId++] = {inputLocale, outputLocale};
            }
        }
    }

    return Result::SUCCESS;
}

Result Compositor::checkAutoLayoutState() {
    JST_DEBUG("[COMPOSITOR] Checking auto layout state.");

    bool graphHasPos = false;
    for (const auto& [_, state] : nodeStates) {
        if (state.block->getState().nodePos != Size2D<F32>{0.0f, 0.0f}) {
            graphHasPos = true;
        }
    }

    if (!graphHasPos) {
        JST_CHECK(updateAutoLayoutState());
    }

    return Result::SUCCESS;
}

Result Compositor::updateAutoLayoutState() {
    JST_DEBUG("[COMPOSITOR] Updating auto layout state.");

    graphSpatiallyOrganized = false;

    // Separate graph in sub-graphs if applicable.
    U64 clusterCount = 0;
    std::unordered_set<NodeId> visited;

    for (const auto& [nodeId, _] : nodeLocaleMap) {
        if (visited.contains(nodeId)) {
            continue;
        }

        std::stack<U64> stack;
        stack.push(nodeId);

        while (!stack.empty()) {
            U64 current = stack.top();
            stack.pop();

            if (visited.contains(current)) {
                continue;
            }

            visited.insert(current);
            auto& state = nodeStates.at(nodeLocaleMap.at(current));
            state.clusterLevel = clusterCount;
            for (const auto& neighbor : state.edges) {
                if (!visited.contains(neighbor)) {
                    stack.push(neighbor);
                }
            }
        }
        clusterCount += 1;
    }

    // Create automatic graph layout.
    nodeTopology.clear();

    U64 columnId = 0;
    std::unordered_set<NodeId> S;

    for (const auto& [nodeId, nodeLocale] : nodeLocaleMap) {
        if (nodeStates.at(nodeLocale).inputs.size() == 0) {
            S.insert(nodeId);
        }
    }

    while (!S.empty()) {
        std::unordered_set<U64> nodesToInsert;
        std::unordered_set<U64> nodesToExclude;
        std::unordered_map<NodeId, std::unordered_set<NodeId>> nodeMatches;

        // Build the matches map.
        for (const auto& sourceNodeId : S) {
            const auto& outputList = nodeStates.at(nodeLocaleMap.at(sourceNodeId)).outputs;

            if (outputList.empty()) {
                nodesToInsert.insert(sourceNodeId);
                continue;
            }

            for (const auto& [_, outputLocale] : outputList) {
                const auto& inputList = outputInputCache.at(outputLocale);

                if (inputList.empty()) {
                    nodesToInsert.insert(sourceNodeId);
                    continue;
                }

                for (const auto& inputLocale : inputList) {
                    nodeMatches[nodeStates.at(inputLocale.block()).id].insert(sourceNodeId);
                }
            }
        }

        U64 previousSetSize = S.size();

        // Determine which nodes to insert and which to exclude.
        for (const auto& [targetNodeId, sourceNodes] : nodeMatches) {
            U64 inputCount = 0;
            for (const auto& [_, locale] : nodeStates.at(nodeLocaleMap.at(targetNodeId)).inputs) {
                if (!locale.empty()) {
                    inputCount += 1;
                }
            }
            if (inputCount >= sourceNodes.size()) {
                S.insert(targetNodeId);
                nodesToInsert.insert(sourceNodes.begin(), sourceNodes.end());
            } else {
                nodesToExclude.insert(sourceNodes.begin(), sourceNodes.end());
            }
        }

        // Exclude nodes from the nodesToInsert set.
        for (const auto& node : nodesToExclude) {
            nodesToInsert.erase(node);
        }

        // If no new nodes were added to S, insert all nodes from S into nodesToInsert.
        if (previousSetSize == S.size()) {
            nodesToInsert.insert(S.begin(), S.end());
        }

        for (const auto& nodeId : nodesToInsert) {
            const U64& clusterId = nodeStates.at(nodeLocaleMap[nodeId]).clusterLevel;
            if (nodeTopology.size() <= clusterId) {
                nodeTopology.resize(clusterId + 1);
            }
            if (nodeTopology.at(clusterId).size() <= columnId) {
                nodeTopology[clusterId].resize(columnId + 1);
            }
            nodeTopology[clusterId][columnId].push_back(nodeId);
            S.erase(nodeId);
        }

        columnId += 1;
    }

    return Result::SUCCESS;
}

Result Compositor::destroy() {
    JST_DEBUG("[COMPOSITOR] Destroying compositor.");

    // Stop execution.
    running = false;

    // Reseting variables.

    rightClickMenuEnabled = false;
    graphSpatiallyOrganized = false;
    sourceEditorEnabled = false;
    moduleStoreEnabled = true;
    debugDemoEnabled = false;
    debugLatencyEnabled = false;
    flowgraphEnabled = true;
    infoPanelEnabled = true;
    globalModalContentId = 0;
    nodeContextMenuNodeId = 0;

    // Reseting locks.

    interfaceHalt.clear();
    interfaceHalt.notify_all();

    // Clearing buffers.

    linkLocaleMap.clear();
    inputLocalePinMap.clear();
    outputLocalePinMap.clear();
    pinLocaleMap.clear();
    nodeLocaleMap.clear();
    stacks.clear();
    nodeTopology.clear();
    outputInputCache.clear();
    nodeStates.clear();

    // Add static.

    stacks["Graph"] = {true, 0};

    // Unload assets.
    if (assetsLoaded) {
        JST_CHECK(unloadAssets());
    }

    return Result::SUCCESS;
}

Result Compositor::loadImageAsset(const uint8_t* binary_data, 
                                  const U64& binary_len, 
                                  std::shared_ptr<Render::Texture>& texture) {
    int image_width, image_height, image_channels;

    uint8_t* raw_data = stbi_load_from_memory(
        binary_data, 
        binary_len, 
        &image_width, 
        &image_height, 
        &image_channels, 
        4
    );
    if (raw_data == nullptr) {
        JST_FATAL("[COMPOSITOR] Could not load image asset.");
        return Result::ERROR;
    }

    Render::Texture::Config config;
    config.size = {
        static_cast<U64>(image_width), 
        static_cast<U64>(image_height)
    };
    config.buffer = static_cast<uint8_t*>(raw_data);
    JST_CHECK(instance.window().build(texture, config));
    JST_CHECK(texture->create());
    stbi_image_free(raw_data);

    return Result::SUCCESS;
}

Result Compositor::loadAssets() {
    JST_DEBUG("[COMPOSITOR] Loading assets.");

    JST_CHECK(loadImageAsset(Resources::compositor_banner_primary_bin, 
                             Resources::compositor_banner_primary_len,
                             primaryBannerTexture));

    JST_CHECK(loadImageAsset(Resources::compositor_banner_secondary_bin, 
                             Resources::compositor_banner_secondary_len,
                             secondaryBannerTexture));

    assetsLoaded = true;
    return Result::SUCCESS;
}

Result Compositor::unloadAssets() {
    JST_CHECK(primaryBannerTexture->destroy());
    JST_CHECK(secondaryBannerTexture->destroy());

    assetsLoaded = false;
    return Result::SUCCESS;
}

Result Compositor::draw() {
    if (!assetsLoaded) {
        JST_CHECK(loadAssets());
    }

    // Prevent state from refreshing while drawing these methods.
    interfaceHalt.wait(true);
    interfaceHalt.test_and_set();

    JST_CHECK(drawStatic());
    JST_CHECK(drawGraph());

    interfaceHalt.clear();
    interfaceHalt.notify_one();

    return Result::SUCCESS;
}

Result Compositor::processInteractions() {
    if (createBlockMailbox) {
        JST_DISPATCH_ASYNC([&, metadata = *createBlockMailbox](){
            ImGui::InsertNotification({ ImGuiToastType_Info, 2500, "Adding module..." });

            // Create new node fingerprint.
            const auto& [module, device] = metadata;
            const auto moduleEntry = Store::BlockMetadataList().at(module);
            if (moduleEntry.options.at(device).empty()) {
                ImGui::InsertNotification({ ImGuiToastType_Error, 5000, "No compatible data types for this module." });
                createBlockMailbox.reset();
            }
            const auto& [inputDataType, outputDataType] = moduleEntry.options.at(device).at(0);

            Block::Fingerprint fingerprint = {};
            fingerprint.id = module;
            fingerprint.device = GetDeviceName(device);
            fingerprint.inputDataType = inputDataType;
            fingerprint.outputDataType = outputDataType;

            // Create node where the mouse dropped the module.
            Parser::RecordMap configMap, inputMap, stateMap;
            const auto [x, y] = ImNodes::ScreenSpaceToGridSpace(ImGui::GetMousePos());
            stateMap["nodePos"] = {Size2D<F32>{x, y}};

            // Create module.
            JST_CHECK_NOTIFY(Store::BlockConstructorList().at(fingerprint)(instance, "", configMap, inputMap, stateMap));

            // Update source.
            updateFlowgraphBlobMailbox = true;
        });
        
        createBlockMailbox.reset();
    }
    
    if (deleteBlockMailbox) {
        JST_DISPATCH_ASYNC([&](){
            JST_CHECK_NOTIFY(instance.removeBlock(*deleteBlockMailbox));
            updateFlowgraphBlobMailbox = true;
        });
        deleteBlockMailbox.reset();
    }

    if (renameBlockMailbox) {
        JST_DISPATCH_ASYNC([&, metadata = *renameBlockMailbox](){
            const auto& [locale, id] = metadata;
            ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Renaming block..." });
            JST_CHECK_NOTIFY(instance.renameBlock(locale, id));
        });
        renameBlockMailbox.reset();
    }

    if (reloadBlockMailbox) {
        JST_DISPATCH_ASYNC([&, locale = *reloadBlockMailbox](){
            ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Reloading block..." });
            JST_CHECK_NOTIFY(instance.reloadBlock(locale));
        });
        reloadBlockMailbox.reset();
    }

    if (linkMailbox) {
        JST_DISPATCH_ASYNC([&, metadata = *linkMailbox](){
            const auto& [inputLocale, outputLocale] = metadata;
            JST_CHECK_NOTIFY(instance.linkBlocks(inputLocale, outputLocale));
            updateFlowgraphBlobMailbox = true;
        });
        linkMailbox.reset();
    }

    if (unlinkMailbox) {
        JST_DISPATCH_ASYNC([&, metadata = *unlinkMailbox](){
            const auto& [inputLocale, outputLocale] = metadata;
            JST_CHECK_NOTIFY(instance.unlinkBlocks(inputLocale, outputLocale));
            updateFlowgraphBlobMailbox = true;
        });
        unlinkMailbox.reset();
    }

    if (changeBlockBackendMailbox) {
        JST_DISPATCH_ASYNC([&, metadata = *changeBlockBackendMailbox](){
            const auto& [locale, device] = metadata;
            JST_CHECK_NOTIFY(instance.changeBlockBackend(locale, device));
            updateFlowgraphBlobMailbox = true;
        });
        changeBlockBackendMailbox.reset();
    }

    if (changeBlockDataTypeMailbox) {
        JST_DISPATCH_ASYNC([&, metadata = *changeBlockDataTypeMailbox](){
            const auto& [locale, type] = metadata;
            JST_CHECK_NOTIFY(instance.changeBlockDataType(locale, type));
            updateFlowgraphBlobMailbox = true;
        });
        changeBlockDataTypeMailbox.reset();
    }
    
    if (toggleBlockMailbox) {
        JST_DISPATCH_ASYNC([&](){
            // TODO: Implement.
            ImGui::InsertNotification({ ImGuiToastType_Warning, 5000, "Toggling a node is not implemented yet." });
            updateFlowgraphBlobMailbox = true;
        });
        toggleBlockMailbox.reset();
    }

    if (resetFlowgraphMailbox) {
        JST_DISPATCH_ASYNC([&](){
            JST_CHECK_NOTIFY(instance.reset());
            updateFlowgraphBlobMailbox = true;
        });
        resetFlowgraphMailbox.reset();
    }

    if (closeFlowgraphMailbox) {
        JST_DISPATCH_ASYNC([&](){
            JST_CHECK_NOTIFY([&]{
                JST_CHECK(instance.reset());
                JST_CHECK(instance.flowgraph().destroy());
                return Result::SUCCESS;
            }());
            updateFlowgraphBlobMailbox = true;
        });
        closeFlowgraphMailbox.reset();
    }

    if (openFlowgraphUrlMailbox) {
        JST_DISPATCH_ASYNC([&](){
            // TODO: Implement.
            ImGui::InsertNotification({ ImGuiToastType_Warning, 5000, "Remote flowgraph is not implemented yet." });
            updateFlowgraphBlobMailbox = true;
        });
        openFlowgraphUrlMailbox.reset();
    }

    if (openFlowgraphPathMailbox) {
        JST_DISPATCH_ASYNC([&, filepath = *openFlowgraphPathMailbox](){
            ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Loading flowgraph..." });
            JST_CHECK_NOTIFY([&]{
                JST_CHECK(instance.flowgraph().create(filepath));
                JST_CHECK(checkAutoLayoutState());
                return Result::SUCCESS;
            }());
            updateFlowgraphBlobMailbox = true;
        });        
        openFlowgraphPathMailbox.reset();
    }

    if (openFlowgraphBlobMailbox) {
        JST_DISPATCH_ASYNC([&, blob = *openFlowgraphBlobMailbox](){
            ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Loading flowgraph..." });
            JST_CHECK_NOTIFY([&]{
                JST_CHECK(instance.flowgraph().create(blob));
                JST_CHECK(checkAutoLayoutState());
                return Result::SUCCESS;
            }());
            updateFlowgraphBlobMailbox = true;
        });
        openFlowgraphBlobMailbox.reset();
    }

    if (saveFlowgraphMailbox) {
        JST_DISPATCH_ASYNC([&](){
            ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Saving flowgraph..." });
            JST_CHECK_NOTIFY(instance.flowgraph().exportToFile());
        });
        saveFlowgraphMailbox.reset();
    }

    if (newFlowgraphMailbox) {
        JST_DISPATCH_ASYNC([&](){
            JST_CHECK_NOTIFY(instance.flowgraph().create());
            updateFlowgraphBlobMailbox = true;
        });
        newFlowgraphMailbox.reset();
    }

    if (updateFlowgraphBlobMailbox) {
        if (sourceEditorEnabled) {
            JST_DISPATCH_ASYNC([&](){
                JST_CHECK(instance.flowgraph().exportToBlob());
                return Result::SUCCESS;
            });
            updateFlowgraphBlobMailbox.reset();
        }
    }

    return Result::SUCCESS;
}

Result Compositor::drawStatic() {
    // Load local variables.
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    const auto& io = ImGui::GetIO();
    const auto& scalingFactor = instance.window().scalingFactor();
    const bool flowgraphLoaded = instance.flowgraph().imported();
    const F32 variableWidth = 100.0f * scalingFactor;
    I32 interactionTrigger = 0;

    //
    // Grab Shortcuts.
    //

    const auto flag = ImGuiInputFlags_RouteAlways;

    if (ImGui::Shortcut(ImGuiKey_Escape, 0, flag)) {
        JST_TRACE("[COMPOSITOR] Escape shortcut pressed.");
        interactionTrigger = 99;
    }

    if (ImGui::Shortcut(ImGuiMod_Shortcut | ImGuiKey_S, 0, flag)) {
        JST_TRACE("[COMPOSITOR] Save document shortcut pressed.");
        interactionTrigger = 1;
    }

    if (ImGui::Shortcut(ImGuiMod_Shortcut | ImGuiKey_N, 0, flag)) {
        JST_TRACE("[COMPOSITOR] New document shortcut pressed.");
        interactionTrigger = 2;
    }

    if (ImGui::Shortcut(ImGuiMod_Shortcut | ImGuiKey_O, 0, flag)) {
        JST_TRACE("[COMPOSITOR] Open document shortcut pressed.");
        interactionTrigger = 3;
    }

    if (ImGui::Shortcut(ImGuiMod_Shortcut | ImGuiKey_I, 0, flag)) {
        JST_TRACE("[COMPOSITOR] Info document shortcut pressed.");
        interactionTrigger = 4;
    }

    if (ImGui::Shortcut(ImGuiMod_Shortcut | ImGuiKey_W, 0, flag)) {
        JST_TRACE("[COMPOSITOR] Close document shortcut pressed.");
        interactionTrigger = 5;
    }

    //
    // Menu Bar.
    //

    F32 currentHeight = 0.0f;

    if (ImGui::BeginMainMenuBar()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.65f, 0.75f, 0.62f, 1.0f));
        if (ImGui::BeginMenu("CyberEther")) {
            ImGui::PopStyleColor();

            if (ImGui::MenuItem("About CyberEther")) {
                globalModalToggle = true;
                globalModalContentId = 0;
            }
            if (ImGui::MenuItem("View License")) {
                globalModalToggle = true;
                globalModalContentId = 7;
            }
            if (ImGui::MenuItem("Third-Party OSS")) {
                globalModalToggle = true;
                globalModalContentId = 8;
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Block Backend Matrix")) {
                globalModalToggle = true;
                globalModalContentId = 1;
            }
#ifndef __EMSCRIPTEN__
            ImGui::Separator();
            if (ImGui::MenuItem("Quit CyberEther")) {
                std::exit(0);
            }
#endif
            ImGui::EndMenu();
        } else {
            ImGui::PopStyleColor();
        }

        if (ImGui::BeginMenu("Flowgraph")) {
            if (ImGui::MenuItem("New", "CTRL+N", false, !flowgraphLoaded)) {
                interactionTrigger = 2;
            }
            if (ImGui::MenuItem("Open", "CTRL+O", false, !flowgraphLoaded)) {
                interactionTrigger = 3;
            }
            if (ImGui::MenuItem("Save", "CTRL+S", false, flowgraphLoaded)) {
                interactionTrigger = 1;
            }
            if (ImGui::MenuItem("Info", "CTRL+I", false, flowgraphLoaded)) {
                interactionTrigger = 4;
            }
            if (ImGui::MenuItem("Close", "CTRL+W", false, flowgraphLoaded)) {
                interactionTrigger = 5;
            }
            if (ImGui::MenuItem("Rename", nullptr, false, flowgraphLoaded)) {
                interactionTrigger = 8;
            }
            if (ImGui::MenuItem("Reset", nullptr, false, flowgraphLoaded)) {
                interactionTrigger = 6;
            }
            if (ImGui::MenuItem("Source", nullptr, false, flowgraphLoaded)) {
                interactionTrigger = 7;
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("View")) {
            if (ImGui::MenuItem("Show Info Panel", nullptr, &infoPanelEnabled)) { }
            if (ImGui::MenuItem("Show Flowgraph Source", nullptr, &sourceEditorEnabled, flowgraphLoaded)) { }
            if (ImGui::MenuItem("Show Flowgraph", nullptr, &flowgraphEnabled, flowgraphLoaded)) { }
            if (ImGui::MenuItem("Show Block Store", nullptr, &moduleStoreEnabled, flowgraphLoaded && flowgraphEnabled)) { }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Developer")) {
            if (ImGui::MenuItem("Show Demo Window", nullptr, &debugDemoEnabled)) { }
            if (ImGui::MenuItem("Show Latency Window", nullptr, &debugLatencyEnabled)) { }
            if (ImGui::MenuItem("Enable Trace", nullptr, &debugEnableTrace)) {
                if (debugEnableTrace) {
                    JST_LOG_SET_DEBUG_LEVEL(4);
                } else {
                    JST_LOG_SET_DEBUG_LEVEL(JST_LOG_DEBUG_DEFAULT_LEVEL);
                }
            }
            if (ImGui::MenuItem("Open Benchmarking Tool", nullptr, nullptr)) {
                globalModalToggle = true;
                globalModalContentId = 9;
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Help")) {
            if (ImGui::MenuItem("Getting started")) {
                globalModalToggle = true;
                globalModalContentId = 2;
            }
            if (ImGui::MenuItem("Luigi's Twitter")) {
                JST_CHECK_NOTIFY(Platform::OpenUrl("https://twitter.com/luigifcruz"));
            }
            if (ImGui::MenuItem("Documentation")) {
                JST_CHECK_NOTIFY(Platform::OpenUrl("https://github.com/luigifcruz/CyberEther"));
            }
            if (ImGui::MenuItem("Open repository")) {
                JST_CHECK_NOTIFY(Platform::OpenUrl("https://github.com/luigifcruz/CyberEther"));
            }
            if (ImGui::MenuItem("Report an issue")) {
                JST_CHECK_NOTIFY(Platform::OpenUrl("https://github.com/luigifcruz/CyberEther/issues"));
            }
            ImGui::Separator();
            if (ImGui::MenuItem("View license")) {
                globalModalToggle = true;
                globalModalContentId = 7;
            }
            if (ImGui::MenuItem("Third-Party OSS")) {
                globalModalToggle = true;
                globalModalContentId = 8;
            }
            ImGui::EndMenu();
        }

        currentHeight += ImGui::GetWindowSize().y;
        ImGui::EndMainMenuBar();
    }

    //
    // Tool Bar.
    //

    ImGui::SetNextWindowPos(ImVec2(viewport->Pos.x, currentHeight));
    ImGui::SetNextWindowSize(ImVec2(viewport->Size.x, 0));

    {
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::Begin("##ToolBar", nullptr, ImGuiWindowFlags_NoDecoration |
                                           ImGuiWindowFlags_NoNav |
                                           ImGuiWindowFlags_NoDocking |
                                           ImGuiWindowFlags_NoSavedSettings);
        ImGui::PopStyleVar();
        ImGui::PopStyleVar();

        {
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(scalingFactor * 6.0f, scalingFactor * 6.0f));

            if (!flowgraphLoaded) {
                if (ImGui::Button(ICON_FA_FILE " New")) {
                    interactionTrigger = 2;
                }
                ImGui::SameLine();

                if (ImGui::Button(ICON_FA_FOLDER_OPEN " Open")) {
                    interactionTrigger = 3;
                }
                ImGui::SameLine();
            } else {
                if (ImGui::Button(ICON_FA_FLOPPY_DISK " Save")) {
                    interactionTrigger = 1;
                }
                ImGui::SameLine();

                if (ImGui::Button(ICON_FA_CIRCLE_XMARK " Close")) {
                    interactionTrigger = 5;
                }
                ImGui::SameLine();

                if (ImGui::Button(ICON_FA_ERASER " Reset")) {
                    interactionTrigger = 6;
                }
                ImGui::SameLine();

                if (ImGui::Button(ICON_FA_FILE_CODE " Source")) {
                    interactionTrigger = 7;
                }
                ImGui::SameLine();

                if (ImGui::Button(ICON_FA_CIRCLE_INFO " Info")) {
                    interactionTrigger = 4;
                }
                ImGui::SameLine();

                ImGui::Dummy(ImVec2(5.0f, 0.0f));
                ImGui::SameLine();

                if (ImGui::Button(ICON_FA_HAND_SPARKLES " Auto Layout")) {
                    JST_CHECK_NOTIFY(updateAutoLayoutState());
                }
                ImGui::SameLine();

                if (ImGui::Button(ICON_FA_LAYER_GROUP " New Stack")) {
                    stacks[fmt::format("Stack #{}", stacks.size())] = {true, 0};
                }
                ImGui::SameLine();
            }

#ifdef __EMSCRIPTEN__
            ImGui::Dummy(ImVec2(5.0f, 0.0f));
            ImGui::SameLine();

            if (ImGui::Button(ICON_FA_PLUG " Connect WebUSB Device")) {
                EM_ASM({
                    openUsbDevice();
                });
            }
            ImGui::SameLine();
#endif

            if (flowgraphLoaded) {
                ImGui::Dummy(ImVec2(5.0f, 0.0f));
                ImGui::SameLine();

                const auto& title = instance.flowgraph().title();
                if (title.empty()) {
                    ImGui::TextFormatted("Title: N/A", title);
                } else {
                    ImGui::TextFormatted("Title: {}", title);
                }

                ImGui::SameLine();
                ImGui::TextDisabled(ICON_FA_CIRCLE_QUESTION);
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
                    ImGui::BeginTooltip();
                    ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);

                    std::string textSummary, textAuthor, textLicense, textDescription;

                    if (!instance.flowgraph().summary().empty()) {
                        textSummary = fmt::format("Summary: {}\n", instance.flowgraph().summary());
                    }

                    if (!instance.flowgraph().author().empty()) {
                        textAuthor = fmt::format("Author:  {}\n", instance.flowgraph().author());
                    }

                    if (!instance.flowgraph().license().empty()) {
                        textLicense = fmt::format("License: {}\n", instance.flowgraph().license());
                    }

                    if (!instance.flowgraph().description().empty()) {
                        textDescription = fmt::format("Description:\n{}", instance.flowgraph().description());
                    }

                    ImGui::TextWrapped("%s%s%s%s", textSummary.c_str(),
                                                   textAuthor.c_str(),
                                                   textLicense.c_str(),
                                                   textDescription.c_str());
                    ImGui::PopTextWrapPos();
                    ImGui::EndTooltip();
                }
            }

            ImGui::PopStyleVar();
        }

        currentHeight += ImGui::GetWindowSize().y;
        ImGui::End();
    }

    //
    // Submit interactions.
    //

    // Save
    if (interactionTrigger == 1 && flowgraphLoaded) {
        if (instance.flowgraph().filename().empty()) {
            globalModalToggle = true;
            globalModalContentId = 4;
        } else {
            saveFlowgraphMailbox = true;
        }
    }

    // New
    if (interactionTrigger == 2 && !flowgraphLoaded) {
        newFlowgraphMailbox = true;
    }

    // Open
    if (interactionTrigger == 3 && !flowgraphLoaded) {
        globalModalToggle = true;
        globalModalContentId = 3;
    }

    // Info
    if (interactionTrigger == 4 && flowgraphLoaded) {
        globalModalToggle = true;
        globalModalContentId = 4;
    }

    // Close
    if (interactionTrigger == 5 && flowgraphLoaded) {
        if (instance.flowgraph().filename().empty() &&
            !instance.flowgraph().empty()) {
            globalModalToggle = true;
            globalModalContentId = 5;
        } else {
            closeFlowgraphMailbox = true;
        }
    }

    // Reset
    if (interactionTrigger == 6 && flowgraphLoaded) {
        resetFlowgraphMailbox = true;
    }

    // Source
    if (interactionTrigger == 7 && flowgraphLoaded) {
        sourceEditorEnabled = !sourceEditorEnabled;
    }

    // Rename
    if (interactionTrigger == 8 && flowgraphLoaded) {
        globalModalToggle = true;
        globalModalContentId = 4;
    }

    //
    // Docking Arena.
    //

    const ImGuiWindowFlags windowFlags = ImGuiWindowFlags_NoDecoration |
                                         ImGuiWindowFlags_NoNav |
                                         ImGuiWindowFlags_NoDocking |
                                         ImGuiWindowFlags_NoBackground |
                                         ImGuiWindowFlags_NoBringToFrontOnFocus |
                                         ImGuiWindowFlags_NoSavedSettings;

    ImGui::SetNextWindowPos(ImVec2(viewport->Pos.x, currentHeight));
    ImGui::SetNextWindowSize(ImVec2(viewport->Size.x, viewport->Size.y - currentHeight));

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::Begin("##ArenaWindow", nullptr, windowFlags);
    ImGui::PopStyleVar();

    mainNodeId = ImGui::GetID("##Arena");
    ImGui::DockSpace(mainNodeId, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_PassthruCentralNode | 
                                                     ImGuiDockNodeFlags_AutoHideTabBar);

    ImGui::End();

    //
    // Draw notifications.
    //

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, scalingFactor * 5.0f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(43.0f  / 255.0f,
                                                    43.0f  / 255.0f,
                                                    43.0f  / 255.0f,
                                                    100.0f / 255.0f));
    ImGui::RenderNotifications();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();

    //
    // Draw Source Editor.
    //

    // TODO: Implement editing inside the source editor.

    [&](){
        if (!sourceEditorEnabled) {
            return;
        }

        ImGui::SetNextWindowSize(ImVec2(500.0f * scalingFactor, 300.0f * scalingFactor), ImGuiCond_FirstUseEver);
        if (!ImGui::Begin("Source File", &sourceEditorEnabled)) {
            ImGui::End();
            return;
        }

        auto& file = instance.flowgraph().blob();

        if (file.empty()) {
            ImGui::Text("Empty source file.");
        } else {
            ImGui::InputTextMultiline("##SourceFileData", 
                                      const_cast<char*>(file.data()), 
                                      file.size(), 
                                      ImGui::GetContentRegionAvail(), 
                                      ImGuiInputTextFlags_ReadOnly |
                                      ImGuiInputTextFlags_NoHorizontalScroll);
        }

        ImGui::End();
    }();

    //
    // Draw stacks.
    //

    std::vector<std::string> stacksToRemove;
    for (auto& [stack, state] : stacks) {
        auto& [enabled, id] = state;

        if (!enabled) {
            stacksToRemove.push_back(stack);
            continue;
        }

        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
        ImGui::SetNextWindowDockID(mainNodeId, ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(500.0f * scalingFactor, 300.0f * scalingFactor), ImGuiCond_FirstUseEver);
        ImGui::Begin(stack.c_str(), &enabled);
        ImGui::PopStyleVar();
    
        bool isDockNew = false;

        if (!id) {
            isDockNew = true;
            id = ImGui::GetID(fmt::format("##Stack{}", stack).c_str());
        }
        
        ImGui::DockSpace(id, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_PassthruCentralNode);

        if (isDockNew && stack == "Graph") {
            ImGuiID dock_id_left, dock_id_right;

            ImGui::DockBuilderRemoveNode(id);
            ImGui::DockBuilderAddNode(id);
            ImGui::DockBuilderSetNodePos(id, ImVec2(viewport->Pos.x, currentHeight));
            ImGui::DockBuilderSetNodeSize(id, ImVec2(viewport->Size.x, viewport->Size.y - currentHeight));

            ImGui::DockBuilderSplitNode(id, ImGuiDir_Left, 0.8f, &dock_id_left, &dock_id_right);
            ImGui::DockBuilderDockWindow("Flowgraph", dock_id_left);
            ImGui::DockBuilderDockWindow("Store", dock_id_right);

            ImGui::DockBuilderFinish(id);
        }

        ImGui::End();
    }
    for (const auto& stack : stacksToRemove) {
        stacks.erase(stack);
    }

    //
    // Info HUD.
    //

    [&](){
        if (!infoPanelEnabled) {
            return;
        }

        const ImGuiWindowFlags windowFlags = ImGuiWindowFlags_NoDecoration |
                                             ImGuiWindowFlags_NoDocking |
                                             ImGuiWindowFlags_AlwaysAutoResize |
                                             ImGuiWindowFlags_NoSavedSettings |
                                             ImGuiWindowFlags_NoFocusOnAppearing |
                                             ImGuiWindowFlags_NoNav |
                                             ImGuiWindowFlags_NoMove |
                                             ImGuiWindowFlags_Tooltip;

        const F32 windowPad = 6.0f * scalingFactor;
        ImVec2 workPos = viewport->WorkPos;
        ImVec2 workSize = viewport->WorkSize;
        ImVec2 windowPos, windowPosPivot;
        windowPos.x = workPos.x + windowPad;
        windowPos.y = workPos.y + workSize.y - windowPad;
        windowPosPivot.x = 0.0f;
        windowPosPivot.y = 1.0f;
        ImGui::SetNextWindowPos(windowPos, ImGuiCond_Always, windowPosPivot);
        ImGui::SetNextWindowViewport(viewport->ID);

        ImGui::SetNextWindowBgAlpha(0.35f);
        ImGui::Begin("Info", nullptr, windowFlags);

        if (ImGui::TreeNodeEx("Graphics", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::BeginTable("##InfoTableGraphics", 2, ImGuiTableFlags_None);
            ImGui::TableSetupColumn("Variable", ImGuiTableColumnFlags_WidthFixed, variableWidth);
            ImGui::TableSetupColumn("Info", ImGuiTableColumnFlags_WidthStretch);

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Text("FPS:");
            ImGui::TableSetColumnIndex(1);
            ImGui::TextFormatted("{:.1f} Hz", ImGui::GetIO().Framerate);

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Text("Dropped Frames:");
            ImGui::TableSetColumnIndex(1);
            ImGui::TextFormatted("{}", instance.window().stats().droppedFrames);

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Text("Render Backend:");
            ImGui::TableSetColumnIndex(1);
            ImGui::SetNextItemWidth(-1);
            ImGui::TextFormatted("{}", GetDevicePrettyName(instance.window().device()));

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Text("Viewport:");
            ImGui::TableSetColumnIndex(1);
            ImGui::SetNextItemWidth(-1);
            ImGui::TextFormatted("{}", instance.viewport().name());

            instance.window().drawDebugMessage();

            ImGui::EndTable();
            ImGui::TreePop();
        }

        if (ImGui::TreeNodeEx("Compute", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::BeginTable("##InfoTableGraphics", 2, ImGuiTableFlags_None);
            ImGui::TableSetupColumn("Variable", ImGuiTableColumnFlags_WidthFixed, variableWidth);
            ImGui::TableSetupColumn("Info", ImGuiTableColumnFlags_WidthStretch);

            instance.scheduler().drawDebugMessage();

            ImGui::EndTable();
            ImGui::TreePop();
        }

        ImGui::Dummy(ImVec2(variableWidth * 2.3f, 0.0f));

        ImGui::End();
    }();

    //
    // Welcome HUD.
    //

    [&](){
        if (instance.flowgraph().imported() || ImGui::IsPopupOpen("##help_modal")) {
            return;
        }

        const ImGuiWindowFlags windowFlags = ImGuiWindowFlags_NoDecoration |
                                             ImGuiWindowFlags_NoDocking |
                                             ImGuiWindowFlags_AlwaysAutoResize |
                                             ImGuiWindowFlags_NoSavedSettings |
                                             ImGuiWindowFlags_NoFocusOnAppearing |
                                             ImGuiWindowFlags_NoNav |
                                             ImGuiWindowFlags_NoMove |
                                             ImGuiWindowFlags_Tooltip;

        ImVec2 center = ImGui::GetMainViewport()->GetCenter();
        ImGui::SetNextWindowPos(center, ImGuiCond_Always, ImVec2(0.5f, 0.5f));
        ImGui::SetNextWindowViewport(viewport->ID);

        ImGui::SetNextWindowBgAlpha(0.35f);
        ImGui::Begin("Welcome", nullptr, windowFlags);

        ImGui::TextUnformatted(ICON_FA_USER_ASTRONAUT " Welcome to CyberEther!");
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 0.6f));
        ImGui::TextFormatted("Version: {} (Alpha)", JETSTREAM_VERSION_STR);
        ImGui::PopStyleColor();
        
        ImGui::Separator();
        ImGui::Spacing();

        const char* largestText = "To get started, create a new flowgraph or open an existing one using";
        const auto largestTextSize = ImGui::CalcTextSize(largestText).x;

        if (assetsLoaded) {
            static bool usePrimaryTexture = true;
            auto texture = usePrimaryTexture ? primaryBannerTexture : secondaryBannerTexture;
            const auto& [w, h] = texture->size();
            const auto ratio = static_cast<F32>(w) / static_cast<F32>(h);
            ImGui::Image(texture->raw(), ImVec2(largestTextSize, largestTextSize / ratio));

            if (ImGui::IsItemHovered() && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
                usePrimaryTexture = false;
            } else {
                usePrimaryTexture = true;
            }

            ImGui::Spacing();
        }

        ImGui::Text("CyberEther is a tool designed for graphical visualization of radio");
        ImGui::Text("signals and general computing focusing in heterogeneous systems.");

        ImGui::Spacing();
        ImGui::Spacing();

        ImGui::TextUnformatted(largestText);
        ImGui::Text("the toolbar or the buttons below.");

        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(scalingFactor * 6.0f, scalingFactor * 6.0f));

        if (ImGui::Button(ICON_FA_FILE " New Flowgraph")) {
            newFlowgraphMailbox = true;
        }
        ImGui::SameLine();

        if (ImGui::Button(ICON_FA_FOLDER_OPEN " Open Flowgraph")) {
            globalModalToggle = true;
            globalModalContentId = 3;
        }
        ImGui::SameLine();

        if (ImGui::Button(ICON_FA_VIAL " Open Examples")) {
            globalModalToggle = true;
            globalModalContentId = 3;
        }
        ImGui::SameLine();

        ImGui::PopStyleVar();

        ImGui::Spacing();
        ImGui::Spacing();

        ImGui::Text("To learn more about CyberEther, check the Help menu.");

        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(scalingFactor * 6.0f, scalingFactor * 6.0f));

        if (ImGui::Button(ICON_FA_CIRCLE_QUESTION " Getting Started")) {
            globalModalToggle = true;
            globalModalContentId = 2;
        }
        ImGui::SameLine();

        if (ImGui::Button(ICON_FA_CIRCLE_INFO " About CyberEther")) {
            globalModalToggle = true;
            globalModalContentId = 0;
        }
        ImGui::SameLine();

        if (ImGui::Button(ICON_FA_CUBE " Block Backend Matrix")) {
            globalModalToggle = true;
            globalModalContentId = 1;
        }
        ImGui::SameLine();

        ImGui::PopStyleVar();

        ImGui::End();
    }();

    //
    // Global Modal
    //

    if (globalModalToggle) {
        ImGui::OpenPopup("##help_modal");
        globalModalToggle = false;
    }

    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

    if (ImGui::BeginPopupModal("##help_modal", nullptr, ImGuiWindowFlags_AlwaysAutoResize |
                                                        ImGuiWindowFlags_NoTitleBar |
                                                        ImGuiWindowFlags_NoResize |
                                                        ImGuiWindowFlags_NoMove|
                                                        ImGuiWindowFlags_NoScrollbar)) {
        if (globalModalContentId == 0) {
            ImGui::TextUnformatted(ICON_FA_CIRCLE_INFO " About CyberEther");
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::Text("CyberEther is a state-of-the-art tool designed for graphical visualization");
            ImGui::Text("of radio signals and computing focusing in heterogeneous systems.");

            ImGui::Spacing();

            ImGui::BulletText(ICON_FA_GAUGE_HIGH   " Graphical support for any device with Vulkan, Metal, or WebGPU.");
            ImGui::BulletText(ICON_FA_BATTERY_FULL " Portable GPU-acceleration for compute: NVIDIA (CUDA), Apple (Metal), etc.");
            ImGui::BulletText(ICON_FA_SHUFFLE      " Runtime flowgraph pipeline with heterogeneously-accelerated modular blocks.");

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::TextFormatted("Version: {}-{}", JETSTREAM_VERSION_STR, JETSTREAM_BUILD_TYPE);
            ImGui::Text("License: MIT License");
            ImGui::Text("Copyright (c) 2021-2023 Luigi F. Cruz");

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            if (ImGui::Button("Close", ImVec2(-1, 0))) {
                ImGui::CloseCurrentPopup();
            }
        } else if (globalModalContentId == 1) {
            ImGui::TextUnformatted(ICON_FA_CUBE " Block Backend Matrix");
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::Text("A CyberEther flowgraph is composed of modules, each module has a backend and a data type:");
            ImGui::BulletText("Backend is the hardware or software that will be used to process the data.");
            ImGui::BulletText("Data type is the format of the data that will be processed.");
            ImGui::Text("The following matrix shows the current installation available backends and data types.");

            ImGui::Spacing();

            const F32 screenHeight = io.DisplaySize.y;
            const ImVec2 tableHeight = ImVec2(0.0f, 0.60f * screenHeight);
            const ImGuiTableFlags tableFlags = ImGuiTableFlags_PadOuterX | ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollY;

            if (ImGui::BeginTable("table1", 5, tableFlags, tableHeight)) {
                static std::vector<std::tuple<const char*, float, U32, bool, Device>> columns = {
                    {"Block Name",  0.25f,   DefaultColor, false, Device::None},
                    {"CPU",         0.1875f, CpuColor,     true,  Device::CPU},
#if defined(JST_OS_MAC) || defined(JST_OS_IOS)
                    {"Metal",       0.1875f, MetalColor,   true,  Device::Metal},
#else
                    {"CUDA",        0.1875f, CudaColor,  true,  Device::CUDA},
#endif
                    {"Vulkan",      0.1875f, VulkanColor,  true,  Device::Vulkan},
                    {"WebGPU",      0.1875f, WebGPUColor,  true,  Device::WebGPU}
                };

                ImGui::TableSetupScrollFreeze(0, 1);
                for (U64 col = 0; col < columns.size(); ++col) {
                    const auto& [name, width, _1, _2, _3] = columns[col];
                    ImGui::TableSetupColumn(name, ImGuiTableColumnFlags_WidthStretch, width);
                }

                ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
                for (U64 col = 0; col < columns.size(); ++col) {
                    const auto& [name, _1, color, cube, _2] = columns[col]; 
                    ImGui::TableSetColumnIndex(col);
                    if (cube) {
                        ImGui::PushStyleColor(ImGuiCol_Text, color);
                        ImGui::Text(ICON_FA_CUBE);
                        ImGui::PopStyleColor();
                        ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
                    }
                    ImGui::TableHeader(name);
                }

                for (const auto& [_, store] : Store::BlockMetadataList()) {
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::TextUnformatted(store.title.c_str());

                    for (U64 col = 1; col < 5; ++col) {
                        ImGui::TableSetColumnIndex(col);

                        const auto& device = std::get<4>(columns[col]);
                        if (!store.options.contains(device)) {
                            ImGui::TableSetBgColor(ImGuiTableBgTarget_CellBg, IM_COL32(255, 0, 0, 90));
                            continue;
                        }

                        ImGui::TableSetBgColor(ImGuiTableBgTarget_CellBg, IM_COL32(0, 255, 0, 30));
                        for (const auto& [inputDataType, outputDataType] : store.options.at(device)) {
                            const auto label = (outputDataType.empty()) ? fmt::format("{}", inputDataType) : 
                                                                          fmt::format("{} -> {}", inputDataType, outputDataType);
                            ImGui::TextUnformatted(label.c_str());
                            ImGui::SameLine();
                        }
                    }
                }

                ImGui::EndTable();
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            if (ImGui::Button("Close", ImVec2(-1, 0))) {
                ImGui::CloseCurrentPopup();
            }
        } else if (globalModalContentId == 2) {
            ImGui::TextUnformatted(ICON_FA_CIRCLE_QUESTION " Getting started");
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::Text("To get started:");
            ImGui::BulletText("There is no start or stop button, the graph will run automatically.");
            ImGui::BulletText("Drag and drop a module from the Block Store to the Flowgraph.");
            ImGui::BulletText("Configure the module settings as needed.");
            ImGui::BulletText("To connect modules, drag from the output pin of one module to the input pin of another.");
            ImGui::BulletText("To disconnect modules, click on the input pin.");
            ImGui::BulletText("To remove a module, right click on it and select 'Remove Block'.");
            ImGui::Text("Ensure your device compatibility for optimal performance.");
            ImGui::Text("Need more help? Check the 'Help' section or visit our official documentation.");

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            if (ImGui::Button("Close", ImVec2(-1, 0))) {
                ImGui::CloseCurrentPopup();
            }
        } else if (globalModalContentId == 3) {
            ImGui::TextUnformatted(ICON_FA_STORE " Flowgraph Store");
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::Text("This is the Flowgraph Store, a place where you can find and load flowgraphs.");
            ImGui::Text("You can also create your own flowgraphs and share them with the community.");
            ImGui::Text("To load a flowgraph, click in one of the bubbles below:");

            ImGui::Spacing();

            const F32 screenHeight = io.DisplaySize.y;
            const F32 maxTableHeight = 0.40f * screenHeight;

            const F32 lineHeight = ImGui::GetTextLineHeightWithSpacing();
            const F32 textPadding = lineHeight / 3.0f;
            const F32 rowHeight = lineHeight + (textPadding * 5.25f);
            const F32 totalTableHeight = rowHeight * std::ceil(Store::FlowgraphMetadataList().size() / 2.0f);

            const F32 tableHeight = (totalTableHeight < maxTableHeight) ? totalTableHeight : maxTableHeight;

            const ImGuiTableFlags tableFlags = ImGuiTableFlags_PadOuterX | 
                                               ImGuiTableFlags_NoBordersInBody | 
                                               ImGuiTableFlags_NoBordersInBodyUntilResize | 
                                               ImGuiTableFlags_ScrollY;

            if (ImGui::BeginTable("flowgraph_table", 2, tableFlags, ImVec2(0, tableHeight))) {
                for (U64 i = 0; i < 2; ++i) {
                    ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch, 0.5f);
                }

                U64 cellCount = 0;
                for (const auto& [id, flowgraph] : Store::FlowgraphMetadataList()) {
                    if ((cellCount % 2) == 0) {
                        ImGui::TableNextRow();
                    }
                    ImGui::TableSetColumnIndex(cellCount % 2);

                    const ImVec2 cellMin = ImGui::GetCursorScreenPos();
                    const ImVec2 cellSize = ImVec2(ImGui::GetColumnWidth(), lineHeight * 2 + textPadding);
                    const ImVec2 cellMax = ImVec2(cellMin.x + cellSize.x, cellMin.y + cellSize.y);

                    ImDrawList* drawList = ImGui::GetWindowDrawList();
                    drawList->AddRectFilled(cellMin, cellMax, IM_COL32(13, 13, 13, 138), 5.0f);

                    if (ImGui::InvisibleButton(("cell_button_" + id).c_str(), cellSize)) {
                        openFlowgraphBlobMailbox = flowgraph.data;
                        ImGui::CloseCurrentPopup();
                    }

                    ImGui::SetCursorScreenPos(ImVec2(cellMin.x + textPadding, cellMin.y + textPadding));
                    ImGui::Text("%s", flowgraph.title.c_str());
                    ImGui::SameLine();
                    ImGui::TextDisabled(ICON_FA_CIRCLE_QUESTION);
                    if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
                        ImGui::BeginTooltip();
                        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);

                        ImGui::TextWrapped("%s", flowgraph.description.c_str());

                        ImGui::PopTextWrapPos();
                        ImGui::EndTooltip();
                    }

                    ImGui::SetCursorScreenPos(ImVec2(cellMin.x + textPadding, cellMin.y + textPadding + lineHeight));
                    ImGui::Text("%s", flowgraph.summary.c_str());

                    cellCount += 1;
                }

                ImGui::EndTable();
            }

            ImGui::Spacing();
            ImGui::Separator();

            ImGui::Text(ICON_FA_FOLDER_OPEN " Or paste the path or URL of a flowgraph file here:");
            static std::string globalModalPath;
            if (ImGui::BeginTable("flowgraph_table_path", 2, ImGuiTableFlags_NoBordersInBody | 
                                                             ImGuiTableFlags_NoBordersInBodyUntilResize)) {
                ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch, 80.0f);
                ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch, 20.0f);

                ImGui::TableNextRow();

                ImGui::TableSetColumnIndex(0);
                ImGui::SetNextItemWidth(-1);
                ImGui::InputText("##FlowgraphPath", &globalModalPath);

                ImGui::TableSetColumnIndex(1);
                if (ImGui::Button("Browse File", ImVec2(-1, 0))) {
                    JST_CHECK_NOTIFY(Platform::PickFile(globalModalPath));
                }

                ImGui::EndTable();
            }

            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(scalingFactor * 6.0f, scalingFactor * 6.0f));
            if (ImGui::Button(ICON_FA_PLAY " Load")) {
                if (globalModalPath.size() == 0) {
                    ImGui::InsertNotification({ ImGuiToastType_Error, 5000, "Please enter a valid path or URL." });
                } else if (std::regex_match(globalModalPath, std::regex("^(https?)://.*$"))) {
                    openFlowgraphUrlMailbox = globalModalPath;
                    ImGui::CloseCurrentPopup();
                } else if (std::filesystem::exists(globalModalPath)) {
                    openFlowgraphPathMailbox = globalModalPath;
                    ImGui::CloseCurrentPopup();
                } else {
                    ImGui::InsertNotification({ ImGuiToastType_Error, 5000, "The specified path doesn't exist." });
                }
            }
            ImGui::PopStyleVar();

            ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 0.6f));
            ImGui::SetCursorPosY(ImGui::GetCursorPosY());
            ImGui::Text("Make sure you trust the flowgraph source!");
            ImGui::PopStyleColor();

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            if (ImGui::Button("Close", ImVec2(-1, 0))) {
                ImGui::CloseCurrentPopup();
            }
        } else if (globalModalContentId == 4) {
            ImGui::TextUnformatted(ICON_FA_CIRCLE_INFO " Flowgraph Information");
            ImGui::Separator();
            ImGui::Spacing();

            const ImGuiTableFlags tableFlags = ImGuiTableFlags_PadOuterX;
            if (ImGui::BeginTable("##flowgraph-info-table", 2, tableFlags)) {
                ImGui::TableSetupColumn("##flowgraph-info-table-labels", ImGuiTableColumnFlags_WidthStretch, 0.20f);
                ImGui::TableSetupColumn("##flowgraph-info-table-values", ImGuiTableColumnFlags_WidthStretch, 0.80f);

                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::Text("Title:");
                ImGui::TableSetColumnIndex(1);
                ImGui::SetNextItemWidth(-1);
                auto title = instance.flowgraph().title();
                if (ImGui::InputText("##flowgraph-info-title", &title)) {
                    JST_CHECK_THROW(instance.flowgraph().setTitle(title));
                }

                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::Text("Summary:");
                ImGui::TableSetColumnIndex(1);
                ImGui::SetNextItemWidth(-1);
                auto summary = instance.flowgraph().summary();
                if (ImGui::InputText("##flowgraph-info-summary", &summary)) {
                    JST_CHECK_THROW(instance.flowgraph().setSummary(summary));
                }

                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::Text("Author:");
                ImGui::TableSetColumnIndex(1);
                ImGui::SetNextItemWidth(-1);
                auto author = instance.flowgraph().author();
                if (ImGui::InputText("##flowgraph-info-author", &author)) {
                    JST_CHECK_THROW(instance.flowgraph().setAuthor(author));
                }

                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::Text("License:");
                ImGui::TableSetColumnIndex(1);
                ImGui::SetNextItemWidth(-1);
                auto license = instance.flowgraph().license();
                if (ImGui::InputText("##flowgraph-info-license", &license)) {
                    JST_CHECK_THROW(instance.flowgraph().setLicense(license));
                }

                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::Text("Description:");
                ImGui::TableSetColumnIndex(1);
                ImGui::SetNextItemWidth(-1);
                auto description = instance.flowgraph().description();
                // TODO: Implement automatic line wrapping.
                if (ImGui::InputTextMultiline("##flowgraph-info-description", &description)) {
                    JST_CHECK_THROW(instance.flowgraph().setDescription(description));
                }

                ImGui::EndTable();
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(scalingFactor * 5.0f, scalingFactor * 5.0f));

            if (ImGui::BeginTable("##flowgraph-info-path", 2, ImGuiTableFlags_NoBordersInBody | 
                                                              ImGuiTableFlags_NoBordersInBodyUntilResize)) {
                ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch, 80.0f);
                ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch, 20.0f);

                ImGui::TableNextRow();

                ImGui::TableSetColumnIndex(0);
                ImGui::SetNextItemWidth(-1);
                auto filename = instance.flowgraph().filename();
                if (ImGui::InputText("##flowgraph-info-filename", &filename)) {
                    JST_CHECK_THROW(instance.flowgraph().setFilename(filename));
                }

                ImGui::TableSetColumnIndex(1);
                if (ImGui::Button("Browse File", ImVec2(-1, 0))) {
                    JST_CHECK_NOTIFY([&]{
                        std::string filename;
                        JST_CHECK(Platform::SaveFile(filename));
                        JST_CHECK_THROW(instance.flowgraph().setFilename(filename));
                        return Result::SUCCESS;
                    }());
                }

                ImGui::EndTable();
            }

            if (ImGui::Button(ICON_FA_FLOPPY_DISK " Save Flowgraph", ImVec2(-1, 0))) {
                saveFlowgraphMailbox = true;
                ImGui::CloseCurrentPopup();
            }

            ImGui::PopStyleVar();

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            if (ImGui::Button("Close", ImVec2(-1, 0))) {
                ImGui::CloseCurrentPopup();
            }
        } else if (globalModalContentId == 5) {
            ImGui::TextUnformatted(ICON_FA_TRIANGLE_EXCLAMATION " Close Flowgraph");
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::Text("You are about to close a flowgraph without saving it.");
            ImGui::Text("Are you sure you want to continue?");

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            if (ImGui::Button("Save", ImVec2(-1, 0))) {
                globalModalContentId = 4;
            }
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.0f, 0.0f, 1.0f));
            if (ImGui::Button("Close Anyway", ImVec2(-1, 0))) {
                closeFlowgraphMailbox = true;
                ImGui::CloseCurrentPopup();
            }
            ImGui::PopStyleColor();
            if (ImGui::Button("Cancel", ImVec2(-1, 0))) {
                ImGui::CloseCurrentPopup();
            }
        } else if (globalModalContentId == 6) {
            ImGui::TextUnformatted(ICON_FA_PENCIL " Rename Block");
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::SetNextItemWidth(-1);
            ImGui::InputText("##rename-block-new-id", &renameBlockNewId);

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.0f, 1.0f, 1.0f));
            if (ImGui::Button("Rename Block", ImVec2(-1, 0))) {
                renameBlockMailbox = {renameBlockLocale, renameBlockNewId};
                ImGui::CloseCurrentPopup();
            }
            ImGui::PopStyleColor();
            if (ImGui::Button("Close", ImVec2(-1, 0))) {
                ImGui::CloseCurrentPopup();
            }
        } else if (globalModalContentId == 7) {
            ImGui::TextUnformatted(ICON_FA_KEY " CyberEther License");
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::Text("MIT License");

            ImGui::Spacing();

            ImGui::Text("Copyright (c) 2021-2023 Luigi F. Cruz");

            ImGui::Spacing();

            ImGui::Text("Permission is hereby granted, free of charge, to any person obtaining a copy");
            ImGui::Text("of this software and associated documentation files (the \"Software\"), to deal");
            ImGui::Text("in the Software without restriction, including without limitation the rights");
            ImGui::Text("to use, copy, modify, merge, publish, distribute, sublicense, and/or sell");
            ImGui::Text("copies of the Software, and to permit persons to whom the Software is");
            ImGui::Text("furnished to do so, subject to the following conditions:");

            ImGui::Spacing();

            ImGui::Text("The above copyright notice and this permission notice shall be");
            ImGui::Text("included in all copies or substantial portions of the Software.");

            ImGui::Spacing();

            ImGui::Text("THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR");
            ImGui::Text("IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,");
            ImGui::Text("FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE");
            ImGui::Text("AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER");
            ImGui::Text("LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,");
            ImGui::Text("OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE");
            ImGui::Text("SOFTWARE.");

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            if (ImGui::Button("View Third-Party Licenses")) {
                Platform::OpenUrl("https://github.com/luigifcruz/CyberEther/blob/main/ACKNOWLEDGMENTS.md");
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            if (ImGui::Button("Close", ImVec2(-1, 0))) {
                ImGui::CloseCurrentPopup();
            }
        } else if (globalModalContentId == 8) {
            ImGui::TextUnformatted(ICON_FA_BOX_OPEN " Third-Party OSS");
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::Text("CyberEther utilizes the following open-source third-party software,");
            ImGui::Text("and we extend our gratitude to the creators of these libraries for");
            ImGui::Text("their valuable contributions to the open-source community.");

            ImGui::BulletText("Miniaudio - MIT License");
            ImGui::BulletText("Dear ImGui - MIT License");
            ImGui::BulletText("ImNodes - MIT License");
            ImGui::BulletText("PocketFFT - BSD-3-Clause License");
            ImGui::BulletText("RapidYAML - MIT License");
            ImGui::BulletText("vkFFT - MIT License");
            ImGui::BulletText("stb - MIT License");
            ImGui::BulletText("fmtlib - MIT License");
            ImGui::BulletText("SoapySDR - Boost Software License");
            ImGui::BulletText("GLFW - zlib/libpng License");
            ImGui::BulletText("imgui-notify - MIT License");
            ImGui::BulletText("spirv-cross - MIT License");
            ImGui::BulletText("glslang - BSD-3-Clause License");
            ImGui::BulletText("naga - Apache License 2.0");
            ImGui::BulletText("gstreamer - LGPL-2.1 License");
            ImGui::BulletText("libusb - LGPL-2.1 License");
            ImGui::BulletText("nanobench - MIT License");
            // [NEW DEPENDENCY HOOK]

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            if (ImGui::Button("View Third-Party Licenses")) {
                Platform::OpenUrl("https://github.com/luigifcruz/CyberEther/blob/main/ACKNOWLEDGMENTS.md");
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            if (ImGui::Button("Close", ImVec2(-1, 0))) {
                ImGui::CloseCurrentPopup();
            }
        } else if (globalModalContentId == 9) {
            ImGui::TextUnformatted(ICON_FA_GAUGE_HIGH " Benchmarking Tool");
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::Text("This is the Benchmarking Tool, a place where you can run benchmarks");
            ImGui::Text("to compare the performance between different devices and backends.");

            if (ImGui::Button("Run Benchmark")) {
                std::thread([&]{
                    ImGui::InsertNotification({ ImGuiToastType_Info, 5000, "Running benchmark..." });
                    Benchmark::Run("quiet");
                }).detach();
            }
            ImGui::SameLine();
            if (ImGui::Button("Reset Benchmark")) {
                if (Benchmark::TotalCount() == Benchmark::CurrentCount()) {
                    Benchmark::ResetResults();
                }
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            const auto& results = Benchmark::GetResults();

            const ImVec2 tableHeight = ImVec2(0.50f * io.DisplaySize.x, 0.50f * io.DisplaySize.y);
            const ImGuiTableFlags mainTableFlags = ImGuiTableFlags_PadOuterX | 
                                                   ImGuiTableFlags_Borders | 
                                                   ImGuiTableFlags_ScrollY | 
                                                   ImGuiTableFlags_Reorderable | 
                                                   ImGuiTableFlags_Hideable;

            if (ImGui::BeginTable("benchmark-table", 2, mainTableFlags, tableHeight)) {
                ImGui::TableSetupColumn("Module", ImGuiTableColumnFlags_WidthStretch, 0.175f);
                ImGui::TableSetupColumn("Results", ImGuiTableColumnFlags_WidthStretch, 0.825f);
                ImGui::TableHeadersRow();

                for (const auto& [name, entries] : results) {
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted(name.c_str());

                    const ImGuiTableFlags nestedTableFlags = ImGuiTableFlags_Borders |
                                                             ImGuiTableFlags_Reorderable |
                                                             ImGuiTableFlags_Hideable;

                    ImGui::TableNextColumn();
                    if (ImGui::BeginTable(("benchmark-subtable-" + name).c_str(), 4, nestedTableFlags)) {
                        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch, 0.60f);
                        ImGui::TableSetupColumn("ms/op", ImGuiTableColumnFlags_WidthStretch, 0.10f);
                        ImGui::TableSetupColumn("op/s", ImGuiTableColumnFlags_WidthStretch, 0.20f);
                        ImGui::TableSetupColumn("err%", ImGuiTableColumnFlags_WidthStretch, 0.10f);
                        ImGui::TableHeadersRow();

                        for (const auto& entry : entries) {
                            ImGui::TableNextRow();
                            ImGui::TableNextColumn();
                            ImGui::Text("%s", entry.name.c_str());
                            ImGui::TableNextColumn();
                            ImGui::Text("%.2f", entry.ms_per_op);
                            ImGui::TableNextColumn();
                            ImGui::Text("%.2f", entry.ops_per_sec);
                            ImGui::TableNextColumn();
                            ImGui::Text("%.2f", entry.error);
                        }

                        ImGui::EndTable();
                    }
                }

                ImGui::EndTable();
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            std::string progressTaint;
            if (Benchmark::TotalCount() == Benchmark::CurrentCount()) {
                progressTaint = "COMPLETE";
            } else {
                progressTaint = fmt::format("{}/{}", Benchmark::CurrentCount(), Benchmark::TotalCount());
            }
            ImGui::TextFormatted("Benchmark Progress [{}]", progressTaint);
            F32 progress =  Benchmark::TotalCount() > 0 ? static_cast<F32>(Benchmark::CurrentCount()) /  Benchmark::TotalCount() : 0.0f;
            ImGui::ProgressBar(progress, ImVec2(-1, 0), "");

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            if (ImGui::Button("Close", ImVec2(-1, 0))) {
                ImGui::CloseCurrentPopup();
            }
        }

        if (interactionTrigger == 99) {
            ImGui::CloseCurrentPopup();
        }

        ImGui::SetItemDefaultFocus();
        ImGui::Dummy(ImVec2(500.0f * scalingFactor, 0.0f));
        ImGui::EndPopup();
    }

    return Result::SUCCESS;
}

Result Compositor::drawGraph() {
    // Load local variables.
    const auto& scalingFactor = instance.window().scalingFactor();
    const auto& nodeStyle = ImNodes::GetStyle();
    const auto& guiStyle = ImGui::GetStyle();
    const auto& io = ImGui::GetIO();

    const F32 windowMinWidth = 300.0f * scalingFactor;
    const F32 variableWidth = 100.0f * scalingFactor;

    //
    // View Render
    //

    for (const auto& [_, state] : nodeStates) {
        if (!state.block->getState().viewEnabled ||
            !state.block->shouldDrawView() ||
            !state.block->complete()) {
            continue;
        }

        ImGui::SetNextWindowSizeConstraints(ImVec2(64.0f, 64.0f), 
                                            ImVec2(io.DisplaySize.x, io.DisplaySize.y));
        if (!ImGui::Begin(fmt::format("View - {}", state.title).c_str(),
                          &state.block->state.viewEnabled)) {
            ImGui::End();
            continue;
        }
        state.block->drawView();
        ImGui::End();
    }

    //
    // Control Render
    //

    for (const auto& [_, state] : nodeStates) {
        if (!state.block->getState().controlEnabled ||
            !state.block->shouldDrawControl()) {
            continue;
        }

        ImGui::SetNextWindowSizeConstraints(ImVec2(64.0f, 64.0f), 
                                            ImVec2(io.DisplaySize.x, io.DisplaySize.y));
        if (!ImGui::Begin(fmt::format("Control - {}", state.title).c_str(),
                          &state.block->state.controlEnabled)) {
            ImGui::End();
            continue;
        }

        ImGui::BeginTable("##ControlTable", 2, ImGuiTableFlags_None);
        ImGui::TableSetupColumn("Variable", ImGuiTableColumnFlags_WidthFixed, variableWidth);
        ImGui::TableSetupColumn("Control", ImGuiTableColumnFlags_WidthFixed, ImGui::GetWindowWidth() - variableWidth -
                                                                             (guiStyle.CellPadding.x * 2.0f));
        state.block->drawControl();
        ImGui::EndTable();

        if (state.block->shouldDrawInfo()) {
            if (ImGui::TreeNode("Info")) {
                ImGui::BeginTable("##InfoTableAttached", 2, ImGuiTableFlags_None);
                ImGui::TableSetupColumn("Variable", ImGuiTableColumnFlags_WidthFixed, variableWidth);
                ImGui::TableSetupColumn("Info", ImGuiTableColumnFlags_WidthFixed, ImGui::GetWindowWidth() - variableWidth -
                                                                                  (guiStyle.CellPadding.x * 2.0f));
                state.block->drawInfo();
                ImGui::EndTable();
                ImGui::TreePop();
            }
        }

        ImGui::Dummy(ImVec2(windowMinWidth, 0.0f));

        ImGui::End();
    }

    //
    // Flowgraph Render
    //

    [&](){
        if (!flowgraphEnabled || !instance.flowgraph().imported()) {
            return;
        }

        ImGui::SetNextWindowSize(ImVec2(500.0f * scalingFactor, 300.0f * scalingFactor), ImGuiCond_FirstUseEver);
        
        if (!ImGui::Begin("Flowgraph")) {
            ImGui::End();
            return;
        }

        // Set node position according to the internal state.
        for (const auto& [locale, state] : nodeStates) {
            const auto& [x, y] = state.block->getState().nodePos;
            ImNodes::SetNodeGridSpacePos(state.id, ImVec2(x, y));
        }

        ImNodes::BeginNodeEditor();
        ImNodes::MiniMap(0.075f * scalingFactor, ImNodesMiniMapLocation_TopRight);

        for (const auto& [locale, state] : nodeStates) {
            const auto& block = state.block;
            const auto& moduleEntry = Store::BlockMetadataList().at(block->id());

            F32& nodeWidth = block->state.nodeWidth;
            const F32 titleWidth = ImGui::CalcTextSize(state.title.c_str()).x +
                                   ImGui::CalcTextSize(" " ICON_FA_CIRCLE_QUESTION).x +
                                   ((!block->complete()) ? 
                                        ImGui::CalcTextSize(" " ICON_FA_SKULL).x : 0) +
                                   ((!block->warning().empty() && block->complete()) ? 
                                        ImGui::CalcTextSize(" " ICON_FA_TRIANGLE_EXCLAMATION).x : 0);
            const F32 controlWidth = block->shouldDrawControl() ? windowMinWidth: 0.0f;
            const F32 previewWidth = block->shouldDrawPreview() ? windowMinWidth : 0.0f;
            nodeWidth = std::max({titleWidth, nodeWidth, controlWidth, previewWidth});

            // Push node-specific style.
            if (block->complete()) {
                switch (block->device()) {
                    case Device::CPU:
                        ImNodes::PushColorStyle(ImNodesCol_TitleBar,         CpuColor);
                        ImNodes::PushColorStyle(ImNodesCol_TitleBarHovered,  CpuColor);
                        ImNodes::PushColorStyle(ImNodesCol_TitleBarSelected, CpuColorSelected);
                        ImNodes::PushColorStyle(ImNodesCol_Pin,              CpuColor);
                        ImNodes::PushColorStyle(ImNodesCol_PinHovered,       CpuColorSelected);
                        break;
                    case Device::CUDA:
                        ImNodes::PushColorStyle(ImNodesCol_TitleBar,         CudaColor);
                        ImNodes::PushColorStyle(ImNodesCol_TitleBarHovered,  CudaColor);
                        ImNodes::PushColorStyle(ImNodesCol_TitleBarSelected, CudaColorSelected);
                        ImNodes::PushColorStyle(ImNodesCol_Pin,              CudaColor);
                        ImNodes::PushColorStyle(ImNodesCol_PinHovered,       CudaColorSelected);
                        break;
                    case Device::Metal:
                        ImNodes::PushColorStyle(ImNodesCol_TitleBar,         MetalColor);
                        ImNodes::PushColorStyle(ImNodesCol_TitleBarHovered,  MetalColor);
                        ImNodes::PushColorStyle(ImNodesCol_TitleBarSelected, MetalColorSelected);
                        ImNodes::PushColorStyle(ImNodesCol_Pin,              MetalColor);
                        ImNodes::PushColorStyle(ImNodesCol_PinHovered,       MetalColorSelected);
                        break;
                    case Device::Vulkan:
                        ImNodes::PushColorStyle(ImNodesCol_TitleBar,         VulkanColor);
                        ImNodes::PushColorStyle(ImNodesCol_TitleBarHovered,  VulkanColor);
                        ImNodes::PushColorStyle(ImNodesCol_TitleBarSelected, VulkanColorSelected);
                        ImNodes::PushColorStyle(ImNodesCol_Pin,              VulkanColor);
                        ImNodes::PushColorStyle(ImNodesCol_PinHovered,       VulkanColorSelected);
                        break;
                    case Device::WebGPU:
                        ImNodes::PushColorStyle(ImNodesCol_TitleBar,         WebGPUColor);
                        ImNodes::PushColorStyle(ImNodesCol_TitleBarHovered,  WebGPUColor);
                        ImNodes::PushColorStyle(ImNodesCol_TitleBarSelected, WebGPUColorSelected);
                        ImNodes::PushColorStyle(ImNodesCol_Pin,              WebGPUColor);
                        ImNodes::PushColorStyle(ImNodesCol_PinHovered,       WebGPUColorSelected);
                        break;
                    case Device::None:
                        break;
                }
            } else {
                ImNodes::PushColorStyle(ImNodesCol_TitleBar,         DisabledColor);
                ImNodes::PushColorStyle(ImNodesCol_TitleBarHovered,  DisabledColor);
                ImNodes::PushColorStyle(ImNodesCol_TitleBarSelected, DisabledColorSelected);
                ImNodes::PushColorStyle(ImNodesCol_Pin,              DisabledColor);
                ImNodes::PushColorStyle(ImNodesCol_PinHovered,       DisabledColorSelected);
            }

            ImNodes::BeginNode(state.id);

            // Draw node title.
            ImNodes::BeginNodeTitleBar();

            ImGui::TextUnformatted(state.title.c_str());

            ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 0.4f));
            ImGui::Text(ICON_FA_CIRCLE_QUESTION);
            ImGui::PopStyleColor();
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(scalingFactor * 8.0f, scalingFactor * 8.0f));
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
                ImGui::BeginTooltip();
                ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
                ImGui::TextWrapped(ICON_FA_BOOK " Description");
                ImGui::Separator();
                ImGui::TextWrapped("%s", moduleEntry.description.c_str());
                ImGui::PopTextWrapPos();
                ImGui::EndTooltip();
            }
            ImGui::PopStyleVar();

            if (!block->complete()) {
                ImGui::SameLine();
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 0.4f));
                ImGui::Text(ICON_FA_SKULL);
                ImGui::PopStyleColor();
                ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(scalingFactor * 8.0f, scalingFactor * 8.0f));
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
                    ImGui::BeginTooltip();
                    ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
                    ImGui::TextWrapped(ICON_FA_SKULL " Error Message");
                    ImGui::Separator();
                    ImGui::TextWrapped("%s", block->error().c_str());
                    ImGui::PopTextWrapPos();
                    ImGui::EndTooltip();
                }
                ImGui::PopStyleVar();
            } else if (!block->warning().empty()) {
                ImGui::SameLine();
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 0.4f));
                ImGui::Text(ICON_FA_TRIANGLE_EXCLAMATION);
                ImGui::PopStyleColor();
                ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(scalingFactor * 8.0f, scalingFactor * 8.0f));
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
                    ImGui::BeginTooltip();
                    ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
                    ImGui::TextWrapped(ICON_FA_TRIANGLE_EXCLAMATION " Warning Message");
                    ImGui::Separator();
                    ImGui::TextWrapped("%s", block->warning().c_str());
                    ImGui::PopTextWrapPos();
                    ImGui::EndTooltip();
                }
                ImGui::PopStyleVar();
            }

            ImNodes::EndNodeTitleBar();

            // Suspend ImNodes default styling.
            ImGui::PopStyleVar();
            ImGui::PopStyleVar();

            // Draw node info.
            if (block->shouldDrawInfo()) {
                ImGui::BeginTable("##NodeInfoTable", 2, ImGuiTableFlags_None);
                ImGui::TableSetupColumn("Variable", ImGuiTableColumnFlags_WidthFixed, variableWidth);
                ImGui::TableSetupColumn("Info", ImGuiTableColumnFlags_WidthFixed, nodeWidth -  variableWidth -
                                                                                  (guiStyle.CellPadding.x * 2.0f));
                block->drawInfo();
                ImGui::EndTable();
            }

            // Draw node control.
            if (block->shouldDrawControl()) {
                ImGui::BeginTable("##NodeControlTable", 2, ImGuiTableFlags_None);
                ImGui::TableSetupColumn("Variable", ImGuiTableColumnFlags_WidthFixed, variableWidth);
                ImGui::TableSetupColumn("Control", ImGuiTableColumnFlags_WidthFixed, nodeWidth -  variableWidth -
                                                                                     (guiStyle.CellPadding.x * 2.0f));
                block->drawControl();
                ImGui::EndTable();
            }

            // Restore ImNodes default styling.
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(1.0f, 1.0f));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));

            // Draw node input and output pins.
            if (!state.inputs.empty() || !state.outputs.empty()) {
                ImGui::Spacing();

                ImNodes::PushAttributeFlag(ImNodesAttributeFlags_EnableLinkDetachWithDragClick);
                for (const auto& [inputPinId, _] : state.inputs) {
                    const auto& pinName = pinLocaleMap.at(inputPinId).pinId;

                    ImNodes::BeginInputAttribute(inputPinId);
                    ImGui::TextUnformatted(pinName.c_str());
                    ImNodes::EndInputAttribute();
                }
                ImNodes::PopAttributeFlag();

                for (const auto& [outputPinId, _] : state.outputs) {
                    const auto& pinName = pinLocaleMap.at(outputPinId).pinId;
                    const F32 textWidth = ImGui::CalcTextSize(pinName.c_str()).x;

                    ImNodes::BeginOutputAttribute(outputPinId);
                    ImGui::Indent(nodeWidth - textWidth);
                    ImGui::TextUnformatted(pinName.c_str());
                    ImNodes::EndOutputAttribute();
                }
            }

            // Draw node preview.
            if (block->complete() &&
                block->shouldDrawPreview() &&
                block->getState().previewEnabled) {
                ImGui::Spacing();
                block->drawPreview(nodeWidth);
            }

            // Ensure minimum width set by the internal state.
            ImGui::Dummy(ImVec2(nodeWidth, 0.0f));

            // Draw interfacing options.
            if (block->shouldDrawView()    ||
                block->shouldDrawPreview() ||
                block->shouldDrawControl() ||
                block->shouldDrawInfo()) {
                ImGui::BeginTable("##NodeInterfacingOptionsTable", 3, ImGuiTableFlags_None);
                const F32 buttonSize = 25.0f * scalingFactor;
                ImGui::TableSetupColumn("Switches", ImGuiTableColumnFlags_WidthFixed, nodeWidth - (buttonSize * 2.0f) -
                                                                                      (guiStyle.CellPadding.x * 4.0f));
                ImGui::TableSetupColumn("Minus", ImGuiTableColumnFlags_WidthFixed, buttonSize);
                ImGui::TableSetupColumn("Plus", ImGuiTableColumnFlags_WidthFixed, buttonSize);
                ImGui::TableNextRow();

                // Switches
                ImGui::TableSetColumnIndex(0);

                if (block->shouldDrawView()) {
                    ImGui::Checkbox("Window", &block->state.viewEnabled);

                    if (block->shouldDrawControl() ||
                        block->shouldDrawInfo()    ||
                        block->shouldDrawPreview()) {
                        ImGui::SameLine();
                    }
                }

                if (block->shouldDrawControl() ||
                    block->shouldDrawInfo()) {
                    ImGui::Checkbox("Control", &block->state.controlEnabled);

                    if (block->shouldDrawPreview()) {
                        ImGui::SameLine();
                    }
                }

                if (block->shouldDrawPreview()) {
                    ImGui::Checkbox("Preview", &block->state.previewEnabled);
                }

                ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(scalingFactor * 2.0f, scalingFactor * 1.0f));

                // Minus Button
                ImGui::TableSetColumnIndex(1);
                ImGui::SetNextItemWidth(-1);
                if (ImGui::Button(" - ")) {
                    nodeWidth -= 25.0f * scalingFactor;
                }

                // Plus Button
                ImGui::TableSetColumnIndex(2);
                ImGui::SetNextItemWidth(-1);
                if (ImGui::Button(" + ")) {
                    nodeWidth += 25.0f * scalingFactor;
                }

                ImGui::PopStyleVar();

                ImGui::EndTable();
            }

            ImNodes::EndNode();

            // Pop node-specific style.
            ImNodes::PopColorStyle(); // TitleBar
            ImNodes::PopColorStyle(); // TitleBarHovered
            ImNodes::PopColorStyle(); // TitleBarSelected
            ImNodes::PopColorStyle(); // Pin
            ImNodes::PopColorStyle(); // PinHovered
        }

        // Draw node links.
        for (const auto& [linkId, locales] : linkLocaleMap) {
            const auto& [inputLocale, outputLocale] = locales;
            const auto& inputPinId = inputLocalePinMap.at(inputLocale);
            const auto& outputPinId = outputLocalePinMap.at(outputLocale);
            const auto& outputBlock = nodeStates.at(outputLocale.block()).block;

            if (outputBlock->complete()) {
                switch (outputBlock->device()) {
                    case Device::CPU:
                        ImNodes::PushColorStyle(ImNodesCol_Link,         CpuColor);
                        ImNodes::PushColorStyle(ImNodesCol_LinkHovered,  CpuColor);
                        ImNodes::PushColorStyle(ImNodesCol_LinkSelected, CpuColorSelected);
                        break;
                    case Device::CUDA:
                        ImNodes::PushColorStyle(ImNodesCol_Link,         CudaColor);
                        ImNodes::PushColorStyle(ImNodesCol_LinkHovered,  CudaColor);
                        ImNodes::PushColorStyle(ImNodesCol_LinkSelected, CudaColorSelected);
                        break;
                    case Device::Metal:
                        ImNodes::PushColorStyle(ImNodesCol_Link,         MetalColor);
                        ImNodes::PushColorStyle(ImNodesCol_LinkHovered,  MetalColor);
                        ImNodes::PushColorStyle(ImNodesCol_LinkSelected, MetalColorSelected);
                        break;
                    case Device::Vulkan:
                        ImNodes::PushColorStyle(ImNodesCol_Link,         VulkanColor);
                        ImNodes::PushColorStyle(ImNodesCol_LinkHovered,  VulkanColor);
                        ImNodes::PushColorStyle(ImNodesCol_LinkSelected, VulkanColorSelected);
                        break;
                    case Device::WebGPU:
                        ImNodes::PushColorStyle(ImNodesCol_Link,         WebGPUColor);
                        ImNodes::PushColorStyle(ImNodesCol_LinkHovered,  WebGPUColor);
                        ImNodes::PushColorStyle(ImNodesCol_LinkSelected, WebGPUColorSelected);
                        break;
                    case Device::None:
                        break;
                }
            } else {
                ImNodes::PushColorStyle(ImNodesCol_Link,         DisabledColor);
                ImNodes::PushColorStyle(ImNodesCol_LinkHovered,  DisabledColor);
                ImNodes::PushColorStyle(ImNodesCol_LinkSelected, DisabledColorSelected);
            }

            ImNodes::Link(linkId, inputPinId, outputPinId);

            ImNodes::PopColorStyle(); // Link
            ImNodes::PopColorStyle(); // LinkHovered
            ImNodes::PopColorStyle(); // LinkSelected
        }

        ImNodes::EndNodeEditor();

        // Create drop zone for new module creation.
        if (ImGui::BeginDragDropTarget()) {
            if (ImGui::AcceptDragDropPayload("MODULE_DRAG")) {
                createBlockMailbox = createBlockStagingMailbox;
            }
            ImGui::EndDragDropTarget();
        }

        // Spatially organize graph.
        if (!graphSpatiallyOrganized) {
            JST_DEBUG("[COMPOSITOR] Running graph auto-route.");

            F32 previousClustersHeight = 0.0f;

            for (const auto& cluster : nodeTopology) {
                F32 largestColumnHeight = 0.0f;
                F32 previousColumnsWidth = 0.0f;

                for (const auto& column : cluster) {
                    F32 largestNodeWidth = 0.0f;
                    F32 previousNodesHeight = 0.0f;

                    for (const auto& nodeId : column) {
                        const auto& dims = ImNodes::GetNodeDimensions(nodeId);
                        auto& block = nodeStates.at(nodeLocaleMap.at(nodeId)).block;
                        auto& [x, y] = block->state.nodePos;

                        // Add previous columns horizontal offset.
                        x = previousColumnsWidth;
                        // Add previous clusters and rows vertical offset.
                        y = previousNodesHeight + previousClustersHeight;

                        previousNodesHeight += dims.y + (25.0f * scalingFactor);
                        largestNodeWidth = std::max({
                            dims.x,
                            largestNodeWidth,
                        });

                        // Add left padding to nodes in the same column.
                        x += (largestNodeWidth - dims.x);

                        ImNodes::SetNodeGridSpacePos(nodeId, ImVec2(x, y));
                    }

                    largestColumnHeight = std::max({
                        previousNodesHeight,
                        largestColumnHeight,
                    });

                    previousColumnsWidth += largestNodeWidth + (37.5f * scalingFactor);
                }

                previousClustersHeight += largestColumnHeight + (12.5f * scalingFactor);
            }

            ImNodes::EditorContextResetPanning({0.0f, 0.0f});
            graphSpatiallyOrganized = true;
        }

        // Update internal state node position.
        for (const auto& [locale, state] : nodeStates) {
            const auto& [x, y] = ImNodes::GetNodeGridSpacePos(state.id);
            state.block->state.nodePos = {x, y};
        }

        // Render underlying buffer information about the link.
        I32 linkId;
        if (ImNodes::IsLinkHovered(&linkId)) {
            const auto& [inputLocale, outputLocale] = linkLocaleMap.at(linkId);
            const auto& rec = nodeStates.at(outputLocale.block()).outputMap.at(outputLocale.pinId);

            ImGui::BeginTooltip();
            ImGui::TextWrapped(ICON_FA_MEMORY " Tensor Metadata");
            ImGui::Separator();

            const auto firstLine  = fmt::format("[{} -> {}]", outputLocale, inputLocale);
            const auto secondLine = fmt::format("[{}] {} [Device::{}]", rec.dataType, rec.shape, rec.device);
            const auto thirdLine  = fmt::format("[PTR: 0x{:016X}] [HASH: 0x{:016X}]", reinterpret_cast<uintptr_t>(rec.data), rec.hash);

            ImGui::TextUnformatted(firstLine.c_str());
            ImGui::TextUnformatted(secondLine.c_str());
            ImGui::TextUnformatted(thirdLine.c_str());

            if (!rec.attributes.empty()) {
                std::string attributes;
                U64 i = 0;
                for (const auto& [key, value] : rec.attributes) {
                    attributes += fmt::format("{}{}: {}{}", i == 0 ? "" : "             ", 
                                                             key, 
                                                             value, 
                                                             i == rec.attributes.size() - 1 ? "" : ", \n");
                    i++;
                }
                ImGui::TextFormatted("[ATTRIBUTES: {}]", attributes);
            }

            ImGui::EndTooltip();
        }

        // Resize node by dragging interface logic.
        // TODO: I think there might be a bug here when initializing the flowgraph.
        I32 nodeId;
        if (ImNodes::IsNodeHovered(&nodeId)) {
            const auto nodeDims = ImNodes::GetNodeDimensions(nodeId);
            const auto nodeOrigin = ImNodes::GetNodeScreenSpacePos(nodeId);

            F32& nodeWidth = nodeStates.at(nodeLocaleMap.at(nodeId)).block->state.nodeWidth;

            bool isNearRightEdge =
                std::abs((nodeOrigin.x + nodeDims.x) - ImGui::GetMousePos().x) < 10.0f &&
                ImGui::GetMousePos().y >= nodeOrigin.y &&
                ImGui::GetMousePos().y <= (nodeOrigin.y + nodeDims.y);

            if (isNearRightEdge && ImGui::IsMouseDown(0) && !nodeDragId) {
                ImNodes::SetNodeDraggable(nodeId, false);
                nodeDragId = nodeId;
            }

            if (nodeDragId) {
                nodeWidth = (ImGui::GetMousePos().x - nodeOrigin.x) - (nodeStyle.NodePadding.x * 2.0f);
            }

            if (isNearRightEdge || nodeDragId) {
                ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
            }
        }

        if (ImGui::IsMouseReleased(0)) {
            if (nodeDragId) {
                ImNodes::SetNodeDraggable(nodeDragId, true);
                nodeDragId = 0;
            }
        }

        ImGui::End();
    }();

    // Update the internal state when a link is deleted.
    I32 linkId;
    if (ImNodes::IsLinkDestroyed(&linkId)) {
        unlinkMailbox = linkLocaleMap.at(linkId);
    }

    // Update the internal state when a link is created.
    I32 startId, endId;
    if (ImNodes::IsLinkCreated(&startId, &endId)) {
        linkMailbox = {pinLocaleMap.at(endId), pinLocaleMap.at(startId)};
    }

    // Draw right-click menu for node actions.
    if (ImNodes::IsNodeHovered(&nodeContextMenuNodeId) &&
        (ImGui::IsMouseClicked(ImGuiMouseButton_Right))) {
        ImGui::CloseCurrentPopup();
        ImGui::OpenPopup("##node_context_menu");
    }

    // Draw node context menu.
    if (ImGui::BeginPopup("##node_context_menu")) {
        const auto& locale = nodeLocaleMap.at(nodeContextMenuNodeId);
        const auto& state = nodeStates.at(locale.block());
        const auto moduleEntry = Store::BlockMetadataList().at(state.fingerprint.id);

        ImGui::Text("Node ID: %d", nodeContextMenuNodeId);
        ImGui::Separator();

        // Delete node.
        if (ImGui::MenuItem("Delete Node")) {
            deleteBlockMailbox = locale;
        }

        // Rename node.
        if (ImGui::MenuItem("Rename Node")) {
            globalModalToggle = true;
            renameBlockLocale = locale;
            renameBlockNewId = locale.blockId;
            globalModalContentId = 6;
        }

        // Enable/disable node toggle.
        if (ImGui::MenuItem("Enable Node", nullptr, state.block->complete())) {
            toggleBlockMailbox = {locale, !state.block->complete()};
        }

        // Reload node.
        if (ImGui::MenuItem("Reload Node")) {
            reloadBlockMailbox = locale;
        }

        // Device backend options.
        if (ImGui::BeginMenu("Backend Device")) {
            for (const auto& [device, _] : moduleEntry.options) {
                const auto enabled = (state.block->device() == device);
                if (ImGui::MenuItem(GetDevicePrettyName(device), nullptr, enabled)) {
                    changeBlockBackendMailbox = {locale, device};
                }
            }
            ImGui::EndMenu();
        }

        // Data type options.
        if (ImGui::BeginMenu("Data Type")) {
            for (const auto& types : moduleEntry.options.at(state.block->device())) {
                const auto& [inputDataType, outputDataType] = types;
                const auto enabled = state.fingerprint.inputDataType == inputDataType &&
                                     state.fingerprint.outputDataType == outputDataType;
                const auto label = (outputDataType.empty()) ? fmt::format("{}", inputDataType) : 
                                                              fmt::format("{} -> {}", inputDataType, outputDataType);
                if (ImGui::MenuItem(label.c_str(), NULL, enabled)) {
                    changeBlockDataTypeMailbox = {locale, types};
                }
            }
            ImGui::EndMenu();
        }

        ImGui::EndPopup();
    } else if ((ImGui::IsMouseClicked(0) ||
                ImGui::IsMouseClicked(1) ||
                ImGui::IsMouseClicked(2))) {
        ImGui::CloseCurrentPopup();
        nodeContextMenuNodeId = 0;
    }

    //
    // Block Store
    //

    [&](){
        if (!moduleStoreEnabled || !flowgraphEnabled || !instance.flowgraph().imported()) {
            return;
        }

        ImGui::SetNextWindowSize(ImVec2(250.0f * scalingFactor, 300.0f * scalingFactor), ImGuiCond_FirstUseEver);
        if (!ImGui::Begin("Store")) {
            ImGui::End();
            return;
        }

        static char filterText[256] = "";

        ImGui::Text("Search Blocks");
        ImGui::SameLine();
        ImGui::TextDisabled(ICON_FA_CIRCLE_QUESTION);
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
            ImGui::BeginTooltip();
            ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
            ImGui::TextWrapped("Use the text box below to filter the list of available modules.\n"
                               "The cube icon below the module represents the backends available.\n"
                               "Drag and drop a cube into the flowgraph to create a new module.\n");
            ImGui::PopTextWrapPos();
            ImGui::EndTooltip();
        }
        ImGui::PushItemWidth(-1);
        ImGui::InputText("##Filter", filterText, IM_ARRAYSIZE(filterText));
        ImGui::Spacing();

        ImGui::BeginChild("Block List", ImVec2(0, 0), true);

        for (const auto& [id, module] : Store::BlockMetadataList(filterText)) {
            ImGui::TextUnformatted(module.title.c_str());
            ImGui::SameLine();
            ImGui::TextDisabled(ICON_FA_CIRCLE_QUESTION);
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
                ImGui::BeginTooltip();
                ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
                ImGui::TextWrapped("%s", module.description.c_str());
                ImGui::PopTextWrapPos();
                ImGui::EndTooltip();
            }
            ImGui::TextWrapped("%s", module.summary.c_str());

            for (const auto& [device, _] : module.options) {
                switch (device) {
                    case Device::CPU:
                        ImGui::PushStyleColor(ImGuiCol_Text, CpuColor);
                        break;
                    case Device::CUDA:
                        ImGui::PushStyleColor(ImGuiCol_Text, CudaColor);
                        break;
                    case Device::Metal:
                        ImGui::PushStyleColor(ImGuiCol_Text, MetalColor);
                        break;
                    case Device::Vulkan:
                        ImGui::PushStyleColor(ImGuiCol_Text, VulkanColor);
                        break;
                    case Device::WebGPU:
                        ImGui::PushStyleColor(ImGuiCol_Text, WebGPUColor);
                        break;
                    default:
                        continue;
                }
                ImGui::Text(ICON_FA_CUBE " ");
                if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID)) {
                    createBlockStagingMailbox = {id, device};
                    ImGui::SetDragDropPayload("MODULE_DRAG", nullptr, 0);
                    ImGui::Text(ICON_FA_CUBE);
                    ImGui::SameLine();
                    ImGui::PopStyleColor();
                    ImGui::Text(" %s (%s)", module.title.c_str(), GetDevicePrettyName(device));
                    ImGui::EndDragDropSource();
                } else {
                    ImGui::PopStyleColor();
                }

                if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
                    ImGui::BeginTooltip();
                    ImGui::Text("%s (%s)", module.title.c_str(), GetDevicePrettyName(device));
                    ImGui::EndTooltip();
                }
                ImGui::SameLine();
            }

            ImGui::Spacing();
            ImGui::Separator();
        }

        const std::string text = "\n"
                                 "       " ICON_FA_HAND_SPOCK 
                                 "\n\n"
                                 "-- END OF LIST --\n\n";
        auto windowWidth = ImGui::GetWindowSize().x;
        auto textWidth   = ImGui::CalcTextSize(text.c_str()).x;

        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 0.4f));
        ImGui::SetCursorPosX((windowWidth - textWidth) * 0.5f);
        ImGui::TextUnformatted(text.c_str());
        ImGui::PopStyleColor();

        ImGui::EndChild();

        ImGui::End();
    }();

    //
    // Debug Latency Render
    //

    [&](){
        if (!debugLatencyEnabled) {
            return;
        }

        const auto& mainWindowWidth = io.DisplaySize.x;
        const auto& mainWindowHeight = io.DisplaySize.y;

        const auto timerWindowWidth  = 200.0f * scalingFactor;
        const auto timerWindowHeight = 85.0f * scalingFactor;

        static F32 x = 0.0f;
        static F32 xd = 1.0f;

        x += xd;

        if (x > (mainWindowWidth - timerWindowWidth)) {
            xd = -xd;
        }
        if (x < 0.0f) {
            xd = -xd;
        }
    
        ImGui::SetNextWindowSize(ImVec2(timerWindowWidth, timerWindowHeight));
        ImGui::SetNextWindowPos(ImVec2(x, (mainWindowHeight / 2.0f) - (timerWindowHeight / 2.0f)));

        if (!ImGui::Begin("Timer", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse)) {
            ImGui::End();
            return;
        }

        U64 ms = duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        ImGui::TextFormatted("Time: {} ms", ms);

        const F32 blockWidth = timerWindowWidth / 6.0f;
        const F32 blockHeight = 35.0f;
        ImVec2 windowPos = ImGui::GetWindowPos();
        ImDrawList* drawList = ImGui::GetWindowDrawList();

        ImVec2 blockPos = ImVec2(windowPos.x, windowPos.y + timerWindowHeight - blockHeight);
        drawList->AddRectFilled(blockPos, ImVec2(blockPos.x + blockWidth, blockPos.y + blockHeight), IM_COL32(255, 0, 0, 255));      // Red
        blockPos.x += blockWidth;
        drawList->AddRectFilled(blockPos, ImVec2(blockPos.x + blockWidth, blockPos.y + blockHeight), IM_COL32(0, 255, 0, 255));      // Green
        blockPos.x += blockWidth;
        drawList->AddRectFilled(blockPos, ImVec2(blockPos.x + blockWidth, blockPos.y + blockHeight), IM_COL32(0, 0, 255, 255));      // Blue
        blockPos.x += blockWidth;
        drawList->AddRectFilled(blockPos, ImVec2(blockPos.x + blockWidth, blockPos.y + blockHeight), IM_COL32(255, 255, 0, 255));    // Yellow
        blockPos.x += blockWidth;
        drawList->AddRectFilled(blockPos, ImVec2(blockPos.x + blockWidth, blockPos.y + blockHeight), IM_COL32(255, 255, 255, 255));  // White
        blockPos.x += blockWidth;
        drawList->AddRectFilled(blockPos, ImVec2(blockPos.x + blockWidth, blockPos.y + blockHeight), IM_COL32(0, 0, 0, 255));        // Black

        ImGui::End();
    }();

    //
    // Debug Demo Render
    //

    if (debugDemoEnabled) {
        ImGui::ShowDemoWindow();
    }

    return Result::SUCCESS;
}

void Compositor::lock() {
    interfaceHalt.wait(true);
    interfaceHalt.test_and_set();
}

void Compositor::unlock() {
    interfaceHalt.clear();
    interfaceHalt.notify_one();
}

}  // namespace Jetstream
