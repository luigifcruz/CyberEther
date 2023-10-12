#include "jetstream/compositor.hh"
#include "jetstream/instance.hh"
#include "jetstream/store.hh"

namespace Jetstream {

Result Compositor::addModule(const Locale& locale, const std::shared_ptr<BlockState> &block) {
    JST_DEBUG("[COMPOSITOR] Adding module '{}'.", locale);

    // Check if the locale is not for a internal module.
    if (locale.internal()) {
        JST_ERROR("[COMPOSITOR] Can't add internal module '{}' to compositor.", locale);
        return Result::ERROR;
    }

    // Check if the locale is already created.
    if (nodeStates.contains(locale)) {
        JST_ERROR("[COMPOSITOR] Entry for node '{}' already exists.", locale);
        return Result::ERROR;
    }

    // Check if module has interface.
    if (!block->interface) {
        JST_ERROR("[COMPOSITOR] Can't add interfaceless module '{}' to compositor.", locale);
        return Result::ERROR;
    }

    // Prevent drawMainInterface from running.
    lock();

    // Save module in node state.
    nodeStates[locale].block = block;

    JST_CHECK(refreshState());

    // Resume drawMainInterface.
    unlock();

    return Result::SUCCESS;
}

Result Compositor::removeModule(const Locale& locale) {
    JST_DEBUG("[COMPOSITOR] Removing module '{}'.", locale);

    // Check if the locale is not for a internal module.
    if (locale.internal()) {
        JST_ERROR("[COMPOSITOR] Can't remove internal module '{}' from compositor.", locale);
        return Result::ERROR;
    }

    // Check if the locale is already created.
    if (!nodeStates.contains(locale)) {
        JST_ERROR("[COMPOSITOR] Entry for node '{}' doesn't exist.", locale);
        return Result::ERROR;
    }

    // Prevent drawMainInterface from running.
    lock();

    // Save module in node state.
    nodeStates.erase(locale);

    JST_CHECK(refreshState());

    if (nodeStates.empty()) {
        JST_CHECK(instance.destroy());
    }

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
        // Create shortcuts.
        auto& interface = state.block->interface;

        // Generate id for node.
        state.id = id;
        nodeLocaleMap[id++] = locale;

        // Check if the graph is already organized.
        if (interface->getConfig().nodePos != Size2D<F32>{0.0f, 0.0f}) {
            graphSpatiallyOrganized = true;
        }

        // Cleanup and create pin map and convert locale to interface locale.

        state.inputs.clear();
        state.outputs.clear();

        for (const auto& [inputPinId, inputRecord] : state.block->record.inputMap) {
            // Generate clean locale.
            const Locale inputLocale = {locale.id, "", inputPinId};

            // Save the input pin locale.
            pinLocaleMap[id] = inputLocale;
            inputLocalePinMap[inputLocale] = id;

            // Generate clean locale.
            const Locale outputLocale = inputRecord.locale.idPin();

            // Save the incoming input locale.
            state.inputs[id++] = outputLocale;

            // Save the output to input locale map cache.
            outputInputCache[outputLocale].push_back(inputLocale);
        }

        for (const auto& [outputPinId, outputRecord] : state.block->record.outputMap) {
            // Generate clean locale.
            const Locale outputLocale = {locale.id, "", outputPinId};

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
            if (nodeStates.contains(inputLocale.idOnly())) {
                state.edges.insert(nodeStates.at(inputLocale.idOnly()).id);
            }
        }

        for (const auto& [_, outputLocale] : state.outputs) {
            for (const auto& inputLocale : outputInputCache[outputLocale]) {
                state.edges.insert(nodeStates.at(inputLocale.idOnly()).id);

                linkLocaleMap[linkId++] = {inputLocale, outputLocale};
            }
        }
    }

    if (!graphSpatiallyOrganized) {
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
                    nodeMatches[nodeStates.at(inputLocale.idOnly()).id].insert(sourceNodeId);
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
            if (inputCount == sourceNodes.size()) {
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

    // Reseting variables.

    rightClickMenuEnabled = false;
    graphSpatiallyOrganized = false;
    sourceEditorEnabled = false;
    moduleStoreEnabled = true;
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

    return Result::SUCCESS;
}

Result Compositor::draw() {
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
    if (createModuleMailbox) {
        const auto& [module, device] = *createModuleMailbox;
        const auto moduleEntry = Store::ModuleList().at(module);
        if (moduleEntry.options.at(device).empty()) {
            ImGui::InsertNotification({ ImGuiToastType_Error, 5000, "No compatible data types for this module." });
            createModuleMailbox.reset();
        }
        const auto& [dataType, inputDataType, outputDataType] = moduleEntry.options.at(device).at(0);

        Parser::ModuleRecord record = {};

        record.fingerprint.module = module;
        record.fingerprint.device = GetDeviceName(device);
        record.fingerprint.dataType = dataType;
        record.fingerprint.inputDataType = inputDataType;
        record.fingerprint.outputDataType = outputDataType;

        // Create node where the mouse dropped the module.
        const auto [x, y] = ImNodes::ScreenSpaceToGridSpace(ImGui::GetMousePos());
        record.interfaceMap["nodePos"] = {Size2D<F32>{x, y}};

        JST_CHECK_NOTIFY(Store::Modules().at(record.fingerprint)(instance, record));
        createModuleMailbox.reset();
    }
    
    if (deleteModuleMailbox) {
        JST_CHECK_NOTIFY(instance.removeModule(deleteModuleMailbox->id));
        deleteModuleMailbox.reset();
    }

    if (renameModuleMailbox) {
        // TODO: Implement.
        renameModuleMailbox.reset();
    }

    if (reloadModuleMailbox) {
        JST_DISPATCH_ASYNC([&, locale = *reloadModuleMailbox](){
            ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Reloading module..." });
            JST_CHECK_NOTIFY(instance.reloadModule(locale));
        });
        reloadModuleMailbox.reset();
    }

    if (linkMailbox) {
        const auto& [inputLocale, outputLocale] = *linkMailbox;
        JST_CHECK_NOTIFY(instance.linkModules(inputLocale, outputLocale));
        linkMailbox.reset();
    }

    if (unlinkMailbox) {
        const auto& [inputLocale, outputLocale] = *unlinkMailbox;
        JST_CHECK_NOTIFY(instance.unlinkModules(inputLocale, outputLocale));
        unlinkMailbox.reset();
    }

    if (changeModuleBackendMailbox) {
        const auto& [locale, device] = *changeModuleBackendMailbox;
        JST_CHECK_NOTIFY(instance.changeModuleBackend(locale, device));
        changeModuleBackendMailbox.reset();
    }

    if (changeModuleDataTypeMailbox) {
        const auto& [locale, type] = *changeModuleDataTypeMailbox;
        JST_CHECK_NOTIFY(instance.changeModuleDataType(locale, type));
        changeModuleDataTypeMailbox.reset();
    }
    
    if (toggleModuleMailbox) {
        // TODO: Implement.
        ImGui::InsertNotification({ ImGuiToastType_Warning, 5000, "Toggling a node is not implemented yet." });
        toggleModuleMailbox.reset();
    }

    if (resetFlowgraphMailbox) {
        JST_CHECK_NOTIFY(instance.resetFlowgraph());
        resetFlowgraphMailbox.reset();
    }

    if (closeFlowgraphMailbox) {
        JST_CHECK_NOTIFY(instance.closeFlowgraph());
        closeFlowgraphMailbox.reset();
    }

    if (openFlowgraphUrlMailbox) {
        //const auto& filePath = *openFlowgraphUrlMailbox;
        // TODO: Implement.
        ImGui::InsertNotification({ ImGuiToastType_Warning, 5000, "Remote flowgraph is not implemented yet." });
        openFlowgraphUrlMailbox.reset();
    }

    if (openFlowgraphPathMailbox) {
        const auto& filePath = *openFlowgraphPathMailbox;
        JST_CHECK_NOTIFY(instance.openFlowgraphFile(filePath));
        openFlowgraphPathMailbox.reset();
    }

    if (openFlowgraphBlobMailbox) {
        JST_DISPATCH_ASYNC([&, blob = *openFlowgraphBlobMailbox](){
            ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Loading flowgraph..." });

            Result res = Result::SUCCESS;
            res |= instance.openFlowgraphBlob(blob);
            res |= updateAutoLayoutState();
            JST_CHECK_NOTIFY(res);
        });
        openFlowgraphBlobMailbox.reset();
    }

    if (saveFlowgraphMailbox) {
        // TODO: Add real save path.
        //JST_CHECK_NOTIFY(instance.saveFlowgraph(""));
        ImGui::InsertNotification({ ImGuiToastType_Warning, 5000, "Save Graph is not implemented yet." });
        saveFlowgraphMailbox.reset();
    }

    if (newFlowgraphMailbox) {
        JST_CHECK_NOTIFY(instance.newFlowgraph());
        newFlowgraphMailbox.reset();
    }

    return Result::SUCCESS;
}

#ifdef __EMSCRIPTEN__
EM_JS(void, openWebUsbPicker, (), {
    openUsbDevice();
});
#endif

Result Compositor::drawStatic() {
    // Load local variables.
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    const auto& io = ImGui::GetIO();
    const auto& scalingFactor = instance.window().scalingFactor();

    const F32 variableWidth = 100.0f * scalingFactor;

    //
    // Menu Bar.
    //

    F32 currentHeight = 0.0f;
    bool globalModalToggle = false;
    bool flowgraphLoaded = instance.parser().haveFlowgraph();

    if (ImGui::BeginMainMenuBar()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.65f, 0.75f, 0.62f, 1.0f));
        if (ImGui::BeginMenu("CyberEther")) {
            ImGui::PopStyleColor();

            if (ImGui::MenuItem("About CyberEther")) {
                globalModalToggle = true;
                globalModalContentId = 0;
            }
            if (ImGui::MenuItem("Module Backend Matrix")) {
                globalModalToggle = true;
                globalModalContentId = 1;
            }
#ifndef __EMSCRIPTEN__
            ImGui::Separator();
            if (ImGui::MenuItem("Quit CyberEther")) {
                // TODO: Implement a cleaner way to do this.
                exit(0);
            }
#endif
            ImGui::EndMenu();
        } else {
            ImGui::PopStyleColor();
        }

        if (ImGui::BeginMenu("Flowgraph")) {
            if (ImGui::MenuItem("New", nullptr, false, !flowgraphLoaded)) {
                newFlowgraphMailbox = true;
            }
            if (ImGui::MenuItem("Open", nullptr, false, !flowgraphLoaded)) {
                globalModalToggle = true;
                globalModalContentId = 3;
            }
            if (ImGui::MenuItem("Save", nullptr, false, flowgraphLoaded)) {
                // TODO: Add path dialog.
                saveFlowgraphMailbox = true;
            }
            if (ImGui::MenuItem("Close", nullptr, false, flowgraphLoaded)) {
                closeFlowgraphMailbox = true;
            }
            if (ImGui::MenuItem("Reset", nullptr, false, flowgraphLoaded)) {
                resetFlowgraphMailbox = true;
            }
            if (ImGui::MenuItem("Show Source", nullptr, false, flowgraphLoaded)) {
                sourceEditorEnabled = !sourceEditorEnabled;
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("View")) {
            if (ImGui::MenuItem("Show Info Panel", nullptr, &infoPanelEnabled)) { }
            if (ImGui::MenuItem("Show Flowgraph Source", nullptr, &sourceEditorEnabled, flowgraphLoaded)) { }
            if (ImGui::MenuItem("Show Flowgraph", nullptr, &flowgraphEnabled, flowgraphLoaded)) { }
            if (ImGui::MenuItem("Show Module Store", nullptr, &moduleStoreEnabled, flowgraphLoaded && flowgraphEnabled)) { }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Help")) {
            if (ImGui::MenuItem("Getting started")) {
                globalModalToggle = true;
                globalModalContentId = 2;
            }
            if (ImGui::MenuItem("Documentation")) {
                // TODO: Add URL handler.
            }
            if (ImGui::MenuItem("Open repository")) {
                // TODO: Add URL handler.
            }
            if (ImGui::MenuItem("Report an issue")) {
                // TODO: Add URL handler.
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
                    newFlowgraphMailbox = true;
                }
                ImGui::SameLine();

                if (ImGui::Button(ICON_FA_FOLDER_OPEN " Open")) {
                    globalModalToggle = true;
                    globalModalContentId = 3;
                }
                ImGui::SameLine();
            } else {
                if (ImGui::Button(ICON_FA_FLOPPY_DISK " Save")) {
                    // TODO: Add path dialog.
                    saveFlowgraphMailbox = true;
                }
                ImGui::SameLine();

                if (ImGui::Button(ICON_FA_CIRCLE_XMARK " Close")) {
                    closeFlowgraphMailbox = true;
                }
                ImGui::SameLine();

                if (ImGui::Button(ICON_FA_ERASER" Reset")) {
                    resetFlowgraphMailbox = true;
                }
                ImGui::SameLine();

                if (ImGui::Button(ICON_FA_FILE_CODE " Source")) {
                    sourceEditorEnabled = !sourceEditorEnabled;
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

            ImGui::Dummy(ImVec2(5.0f, 0.0f));
            ImGui::SameLine();

#ifdef __EMSCRIPTEN__
            if (ImGui::Button(ICON_FA_PLUG " Connect WebUSB Device")) {
                openWebUsbPicker();
            }
            ImGui::SameLine();
#endif

            ImGui::PopStyleVar();
        }

        currentHeight += ImGui::GetWindowSize().y;
        ImGui::End();
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

    [&](){
        if (!sourceEditorEnabled) {
            return;
        }

        ImGui::SetNextWindowSize(ImVec2(500.0f * scalingFactor, 300.0f * scalingFactor), ImGuiCond_FirstUseEver);
        if (!ImGui::Begin("Source File", &sourceEditorEnabled)) {
            ImGui::End();
            return;
        }

        auto& file = instance.parser().getFlowgraphBlob();

        ImGui::InputTextMultiline("##SourceFileData", 
                                  const_cast<char*>(file.data()), 
                                  file.size(), 
                                  ImGui::GetContentRegionAvail(), 
                                  ImGuiInputTextFlags_ReadOnly |
                                  ImGuiInputTextFlags_NoHorizontalScroll);

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
            ImGui::TextFormatted("{}", instance.viewport().prettyName());

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
        if (instance.parser().haveFlowgraph() || ImGui::IsPopupOpen("##help_modal")) {
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
        ImGui::TextFormatted("Version: {}", JETSTREAM_VERSION_STR);
        ImGui::PopStyleColor();
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::Text("CyberEther is a tool designed for graphical visualization of radio");
        ImGui::Text("signals and general computing focusing in heterogeneous systems.");

        ImGui::Spacing();
        ImGui::Spacing();

        ImGui::Text("To get started, create a new flowgraph or open an existing one using");
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

        if (ImGui::Button(ICON_FA_CUBE " Module Backend Matrix")) {
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

            ImGui::TextFormatted("Version: {}-{}", JETSTREAM_VERSION_STR, JETSTREAM_BUILD_TYPE);
            ImGui::Text("Developed by Luigi Cruz");

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::Text("CyberEther is a state-of-the-art tool designed for graphical visualization");
            ImGui::Text("of radio signals and computing focusing in heterogeneous systems.");

            ImGui::Spacing();

            ImGui::BulletText(ICON_FA_GAUGE_HIGH   " Broad compatibility including NVIDIA, Apple Silicon, Raspberry Pi, and AMD.");
            ImGui::BulletText(ICON_FA_BATTERY_FULL " Broad graphical backend support with Vulkan, Metal, and WebGPU.");
            ImGui::BulletText(ICON_FA_SHUFFLE      " Dynamic hardware acceleration settings for tailored performance.");
        } else if (globalModalContentId == 1) {
            ImGui::TextUnformatted(ICON_FA_CUBE " Module Backend Matrix");
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
                    {"Module Name", 0.25f,   DefaultColor, false, Device::None},
                    {"CPU",         0.1875f, CpuColor,     true,  Device::CPU},
                    {"Metal",       0.1875f, MetalColor,   true,  Device::Metal},
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

                for (const auto& [_, store] : Store::ModuleList()) {
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
                        for (const auto& [dataType, inputDataType, outputDataType] : store.options.at(device)) {
                            const auto label = (!dataType.empty()) ? fmt::format("{}", dataType) : 
                                                                     fmt::format("{} -> {}", inputDataType, outputDataType);
                            ImGui::TextUnformatted(label.c_str());
                            ImGui::SameLine();
                        }
                    }
                }

                ImGui::EndTable();
            }
        } else if (globalModalContentId == 2) {
            ImGui::TextUnformatted(ICON_FA_CIRCLE_QUESTION " Getting started");
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::Text("To get started:");
            ImGui::BulletText("There is no start or stop button, the graph will run automatically.");
            ImGui::BulletText("Drag and drop a module from the Module Store to the Flowgraph.");
            ImGui::BulletText("Configure the module settings as needed.");
            ImGui::BulletText("To connect modules, drag from the output pin of one module to the input pin of another.");
            ImGui::BulletText("To disconnect modules, click on the input pin.");
            ImGui::BulletText("To remove a module, right click on it and select 'Remove Module'.");
            ImGui::Text("Ensure your device compatibility for optimal performance.");
            ImGui::Text("Need more help? Check the 'Help' section or visit our official documentation.");
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
            const F32 totalTableHeight = rowHeight * std::ceil(Store::FlowgraphList().size() / 2.0f);

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
                for (const auto& [id, flowgraph] : Store::FlowgraphList()) {
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

                    ImGui::SetCursorScreenPos(ImVec2(cellMin.x + textPadding, cellMin.y + textPadding + lineHeight));
                    ImGui::Text("%s", flowgraph.description.c_str());

                    cellCount += 1;
                }

                ImGui::EndTable();
            }

            ImGui::Spacing();
            ImGui::Separator();

            ImGui::Text(ICON_FA_FOLDER_OPEN " Or paste the path or URL of a flowgraph file here:");
            static char globalModalPath[1024] = "";
            if (ImGui::BeginTable("flowgraph_table_path", 2, ImGuiTableFlags_NoBordersInBody | 
                                                             ImGuiTableFlags_NoBordersInBodyUntilResize)) {
                ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch, 80.0f);
                ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch, 20.0f);

                ImGui::TableNextRow();

                ImGui::TableSetColumnIndex(0);
                ImGui::SetNextItemWidth(-1);
                ImGui::InputText("##FlowgraphPath", globalModalPath, IM_ARRAYSIZE(globalModalPath));

                ImGui::TableSetColumnIndex(1);
                float buttonWidth = ImGui::GetColumnWidth() - ImGui::GetStyle().ItemSpacing.x;
                if (ImGui::Button("Browse File", ImVec2(buttonWidth, 0.0f))) {
                    // TODO: Implement file picker. 
                    ImGui::InsertNotification({ ImGuiToastType_Warning, 5000, "File browser is not implemented yet." });
                }

                ImGui::EndTable();
            }

            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(scalingFactor * 6.0f, scalingFactor * 6.0f));
            if (ImGui::Button(ICON_FA_PLAY " Load")) {
                if (strlen(globalModalPath) == 0) {
                    ImGui::InsertNotification({ ImGuiToastType_Error, 5000, "Please enter a valid path or URL." });
                } else if (strstr(globalModalPath, "http://") || strstr(globalModalPath, "https://")) {
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
            ImGui::SetCursorPosY(ImGui::GetCursorPosY() + (3.0f * scalingFactor));
            ImGui::Text("Make sure you trust the flowgraph source!");
            ImGui::PopStyleColor();
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        if (ImGui::Button("Close", ImVec2(-1, 0))) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::SetItemDefaultFocus();
        ImGui::Dummy(ImVec2(350.0f * scalingFactor, 0.0f));
        ImGui::EndPopup();
    }

    return Result::SUCCESS;
}

Result Compositor::drawGraph() {
    // Load local variables.
    const auto& scalingFactor = instance.window().scalingFactor();
    const auto& nodeStyle = ImNodes::GetStyle();
    const auto& guiStyle = ImGui::GetStyle();

    const F32 windowMinWidth = 300.0f * scalingFactor;
    const F32 variableWidth = 100.0f * scalingFactor;

    //
    // View Render
    //

    for (const auto& [_, state] : nodeStates) {
        const auto& interface = state.block->interface;

        if (!interface->getConfig().viewEnabled ||
            !interface->shouldDrawView() ||
            !interface->complete()) {
            continue;
        }

        if (!ImGui::Begin(fmt::format("View - {}", interface->title()).c_str(),
                          &interface->getConfig().viewEnabled)) {
            ImGui::End();
            continue;
        }
        interface->drawView();
        ImGui::End();
    }

    //
    // Control Render
    //

    for (const auto& [_, state] : nodeStates) {
        const auto& interface = state.block->interface;

        if (!interface->getConfig().controlEnabled ||
            !interface->shouldDrawControl()) {
            continue;
        }

        if (!ImGui::Begin(fmt::format("Control - {}", interface->title()).c_str(),
                          &interface->getConfig().controlEnabled)) {
            ImGui::End();
            continue;
        }

        ImGui::BeginTable("##ControlTable", 2, ImGuiTableFlags_None);
        ImGui::TableSetupColumn("Variable", ImGuiTableColumnFlags_WidthFixed, variableWidth);
        ImGui::TableSetupColumn("Control", ImGuiTableColumnFlags_WidthFixed, ImGui::GetWindowWidth() - variableWidth -
                                                                             (guiStyle.CellPadding.x * 2.0f));
        interface->drawControl();
        ImGui::EndTable();

        if (interface->shouldDrawInfo()) {
            if (ImGui::TreeNode("Info")) {
                ImGui::BeginTable("##InfoTableAttached", 2, ImGuiTableFlags_None);
                ImGui::TableSetupColumn("Variable", ImGuiTableColumnFlags_WidthFixed, variableWidth);
                ImGui::TableSetupColumn("Info", ImGuiTableColumnFlags_WidthFixed, ImGui::GetWindowWidth() - variableWidth -
                                                                                  (guiStyle.CellPadding.x * 2.0f));
                interface->drawInfo();
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
        if (!flowgraphEnabled || !instance.parser().haveFlowgraph()) {
            return;
        }

        ImGui::SetNextWindowSize(ImVec2(500.0f * scalingFactor, 300.0f * scalingFactor), ImGuiCond_FirstUseEver);
        
        if (!ImGui::Begin("Flowgraph")) {
            ImGui::End();
            return;
        }

        // Set node position according to the internal state.
        for (const auto& [locale, state] : nodeStates) {
            const auto& [x, y] = state.block->interface->getConfig().nodePos;
            ImNodes::SetNodeGridSpacePos(state.id, ImVec2(x, y));
        }

        ImNodes::BeginNodeEditor();
        ImNodes::MiniMap(0.075f * scalingFactor, ImNodesMiniMapLocation_TopRight);

        for (const auto& [locale, state] : nodeStates) {
            const auto& block = state.block;
            const auto& interface = block->interface;

            F32& nodeWidth = interface->getConfig().nodeWidth;
            const F32 titleWidth = ImGui::CalcTextSize(interface->title().c_str()).x +
                                   ImGui::CalcTextSize(" " ICON_FA_CIRCLE_QUESTION).x +
                                   ((!interface->complete()) ? ImGui::CalcTextSize(" " ICON_FA_TRIANGLE_EXCLAMATION).x : 0);
            const F32 controlWidth = interface->shouldDrawControl() ? windowMinWidth: 0.0f;
            const F32 previewWidth = interface->shouldDrawPreview() ? windowMinWidth : 0.0f;
            nodeWidth = std::max({titleWidth, nodeWidth, controlWidth, previewWidth});

            // Push node-specific style.
            if (interface->complete()) {
                switch (interface->device()) {
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

            ImGui::TextUnformatted(interface->title().c_str());

            ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 0.4f));
            ImGui::Text(ICON_FA_CIRCLE_QUESTION);
            ImGui::PopStyleColor();
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(scalingFactor * 8.0f, scalingFactor * 8.0f));
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
                ImGui::BeginTooltip();
                ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
                ImGui::TextWrapped("%s", Store::ModuleList().at(block->record.fingerprint.module).detailed.c_str());
                ImGui::PopTextWrapPos();
                ImGui::EndTooltip();
            }
            ImGui::PopStyleVar();

            if (!interface->complete()) {
                ImGui::SameLine();
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 0.4f));
                ImGui::Text(ICON_FA_TRIANGLE_EXCLAMATION);
                ImGui::PopStyleColor();
                ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(scalingFactor * 8.0f, scalingFactor * 8.0f));
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
                    ImGui::BeginTooltip();
                    ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
                    ImGui::TextWrapped("%s", interface->error().c_str());
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
            if (interface->shouldDrawInfo()) {
                ImGui::BeginTable("##NodeInfoTable", 2, ImGuiTableFlags_None);
                ImGui::TableSetupColumn("Variable", ImGuiTableColumnFlags_WidthFixed, variableWidth);
                ImGui::TableSetupColumn("Info", ImGuiTableColumnFlags_WidthFixed, nodeWidth -  variableWidth -
                                                                                  (guiStyle.CellPadding.x * 2.0f));
                interface->drawInfo();
                ImGui::EndTable();
            }

            // Draw node control.
            if (interface->shouldDrawControl()) {
                ImGui::BeginTable("##NodeControlTable", 2, ImGuiTableFlags_None);
                ImGui::TableSetupColumn("Variable", ImGuiTableColumnFlags_WidthFixed, variableWidth);
                ImGui::TableSetupColumn("Control", ImGuiTableColumnFlags_WidthFixed, nodeWidth -  variableWidth -
                                                                                     (guiStyle.CellPadding.x * 2.0f));
                interface->drawControl();
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
            if (interface->complete() &&
                interface->shouldDrawPreview() &&
                interface->getConfig().previewEnabled) {
                ImGui::Spacing();
                interface->drawPreview(nodeWidth);
            }

            // Ensure minimum width set by the internal state.
            ImGui::Dummy(ImVec2(nodeWidth, 0.0f));

            // Draw interfacing options.
            if (interface->shouldDrawView()    ||
                interface->shouldDrawPreview() ||
                interface->shouldDrawControl() ||
                interface->shouldDrawInfo()) {
                ImGui::BeginTable("##NodeInterfacingOptionsTable", 3, ImGuiTableFlags_None);
                const F32 buttonSize = 25.0f * scalingFactor;
                ImGui::TableSetupColumn("Switches", ImGuiTableColumnFlags_WidthFixed, nodeWidth - (buttonSize * 2.0f) -
                                                                                      (guiStyle.CellPadding.x * 4.0f));
                ImGui::TableSetupColumn("Minus", ImGuiTableColumnFlags_WidthFixed, buttonSize);
                ImGui::TableSetupColumn("Plus", ImGuiTableColumnFlags_WidthFixed, buttonSize);
                ImGui::TableNextRow();

                // Switches
                ImGui::TableSetColumnIndex(0);

                if (interface->shouldDrawView()) {
                    ImGui::Checkbox("Window", &interface->getConfig().viewEnabled);

                    if (interface->shouldDrawControl() ||
                        interface->shouldDrawInfo()    ||
                        interface->shouldDrawPreview()) {
                        ImGui::SameLine();
                    }
                }

                if (interface->shouldDrawControl() ||
                    interface->shouldDrawInfo()) {
                    ImGui::Checkbox("Control", &interface->getConfig().controlEnabled);

                    if (interface->shouldDrawPreview()) {
                        ImGui::SameLine();
                    }
                }

                if (interface->shouldDrawPreview()) {
                    ImGui::Checkbox("Preview", &interface->getConfig().previewEnabled);
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
            const auto& outputInterface = nodeStates.at(outputLocale.idOnly()).block->interface;

            if (outputInterface->complete()) {
                switch (outputInterface->device()) {
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
                createModuleMailbox = createModuleStagingMailbox;
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
                        auto& [x, y] = block->interface->getConfig().nodePos;

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

            graphSpatiallyOrganized = true;
        }

        // Update internal state node position.
        for (const auto& [locale, state] : nodeStates) {
            const auto& [x, y] = ImNodes::GetNodeGridSpacePos(state.id);
            state.block->interface->getConfig().nodePos = {x, y};
        }

        // Render underlying buffer information about the link.
        I32 linkId;
        if (ImNodes::IsLinkHovered(&linkId)) {
            const auto& [inputLocale, outputLocale] = linkLocaleMap.at(linkId);
            const auto& rec = nodeStates.at(outputLocale.idOnly()).block->record.outputMap.at(outputLocale.pinId);

            const auto firstLine  = fmt::format("Vector [{} -> {}]", outputLocale, inputLocale);
            const auto secondLine = fmt::format("[{}] {} [Device::{}]", rec.dataType, rec.shape, rec.device);
            const auto thirdLine  = fmt::format("[PTR: 0x{:016X}] [HASH: 0x{:016X}]", reinterpret_cast<uintptr_t>(rec.data), rec.hash);

            ImGui::BeginTooltip();
            ImGui::TextUnformatted(firstLine.c_str());
            ImGui::TextUnformatted(secondLine.c_str());
            ImGui::TextUnformatted(thirdLine.c_str());
            ImGui::EndTooltip();
        }

        // Resize node by dragging interface logic.
        I32 nodeId;
        if (ImNodes::IsNodeHovered(&nodeId)) {
            const auto nodeDims = ImNodes::GetNodeDimensions(nodeId);
            const auto nodeOrigin = ImNodes::GetNodeScreenSpacePos(nodeId);

            F32& nodeWidth = nodeStates.at(nodeLocaleMap.at(nodeId)).block->interface->getConfig().nodeWidth;

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
        const auto& block = nodeStates.at(locale.idOnly()).block;
        const auto& interface = block->interface;
        const auto& fingerprint = block->record.fingerprint;
        const auto moduleEntry = Store::ModuleList().at(fingerprint.module);

        ImGui::Text("Node ID: %d", nodeContextMenuNodeId);
        ImGui::Separator();

        // Delete node.
        if (ImGui::MenuItem("Delete Node")) {
            deleteModuleMailbox = locale;
        }

        // Rename node.
        if (ImGui::MenuItem("Rename Node")) {
            // TODO: Implement.
            ImGui::InsertNotification({ ImGuiToastType_Warning, 5000, "Renaming a node is not implemented yet." });
        }

        // Enable/disable node toggle.
        if (ImGui::MenuItem("Enable Node", nullptr, interface->complete())) {
            toggleModuleMailbox = {locale, !interface->complete()};
        }

        // Reload node.
        if (ImGui::MenuItem("Reload Node")) {
            reloadModuleMailbox = locale;
        }

        // Device backend options.
        if (ImGui::BeginMenu("Backend Device")) {
            for (const auto& [device, _] : moduleEntry.options) {
                const auto enabled = (interface->device() == device);
                if (ImGui::MenuItem(GetDevicePrettyName(device), nullptr, enabled)) {
                    changeModuleBackendMailbox = {locale, device};
                }
            }
            ImGui::EndMenu();
        }

        // Data type options.
        if (ImGui::BeginMenu("Data Type")) {
            for (const auto& types : moduleEntry.options.at(interface->device())) {
                const auto& [dataType, inputDataType, outputDataType] = types;
                const auto enabled = fingerprint.dataType == dataType &&
                                     fingerprint.inputDataType == inputDataType &&
                                     fingerprint.outputDataType == outputDataType;
                const auto label = (!dataType.empty()) ? fmt::format("{}", dataType) : 
                                                         fmt::format("{} -> {}", inputDataType, outputDataType);
                if (ImGui::MenuItem(label.c_str(), NULL, enabled)) {
                    changeModuleDataTypeMailbox = {locale, types};
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
    // Module Store
    //

    [&](){
        if (!moduleStoreEnabled || !flowgraphEnabled || !instance.parser().haveFlowgraph()) {
            return;
        }

        ImGui::SetNextWindowSize(ImVec2(250.0f * scalingFactor, 300.0f * scalingFactor), ImGuiCond_FirstUseEver);
        if (!ImGui::Begin("Store")) {
            ImGui::End();
            return;
        }

        static char filterText[256] = "";
        static bool showModules = false;

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
        ImGui::Checkbox("Show Modules", &showModules);
        ImGui::Spacing();

        ImGui::BeginChild("Block List", ImVec2(0, 0), true);

        for (const auto& [id, module] : Store::ModuleList(filterText, showModules)) {
            ImGui::TextUnformatted(module.title.c_str());
            ImGui::SameLine();
            ImGui::TextDisabled(ICON_FA_CIRCLE_QUESTION);
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
                ImGui::BeginTooltip();
                ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
                ImGui::TextWrapped("%s", module.detailed.c_str());
                ImGui::PopTextWrapPos();
                ImGui::EndTooltip();
            }
            ImGui::TextWrapped("%s", module.small.c_str());

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
                    createModuleStagingMailbox = {id, device};
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

        ImGui::EndChild();

        ImGui::End();
    }();

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
