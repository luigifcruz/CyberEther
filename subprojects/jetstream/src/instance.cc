#include "jetstream/instance.hh"

namespace Jetstream {

Result Instance::create() {
    JST_DEBUG("[INSTANCE] Creating instance.");

    if (commited) {
        JST_FATAL("[INSTANCE] The instance was already commited.");
        return Result::ERROR;
    }

    // Create output and input cache and populate interface states.
    std::unordered_map<U64, std::vector<std::pair<U64, U64>>> inputCache;
    std::unordered_map<U64, std::pair<U64, U64>> outputCache;

    U64 id = 1;
    for (auto& [blockName, blockState] : blockStates) {
        if (!blockState.interface) {
            continue;
        }

        const U64 nodeId = id++;
        auto& state = interfaceStates[nodeId];

        state.name = blockName;
        state.block = &blockState;
        state.title = fmt::format("{} ({})", blockState.interface->prettyName(), blockName);

        if (blockState.interface->config.nodePos != Size2D<F32>{0.0f, 0.0f}) {
            graphSpatiallyOrganized = true;
        }
        
        for (auto& [inputName, inputMeta] : blockState.record.data.inputMap) {
            inputCache[inputMeta.vector.phash].push_back({nodeId, id});
            state.inputs[id++] = {inputName, &inputMeta.vector};
        }

        for (auto& [outputName, outputMeta] : blockState.record.data.outputMap) {
            outputCache[outputMeta.vector.phash] = {nodeId, id};
            state.outputs[id++] = {outputName, &outputMeta.vector};
        }
    }

    // Create node edge cache.
    std::unordered_map<U64, std::unordered_set<U64>> edgesCache;

    U64 linkId = 0;
    for (auto& [nodeId, nodeState] : interfaceStates) {
        auto& edges = edgesCache[nodeId];

        for (const auto& [_1, input] : nodeState.inputs) {
            const auto& [outputNodeId, _2] = outputCache[input.second->phash];
            edges.insert(outputNodeId);
        }

        for (const auto& [outputId, output] : nodeState.outputs) {
            for (const auto& [inputNodeId, inputId] : inputCache[output.second->phash]) {
                edges.insert(inputNodeId);

                nodeConnections[linkId++] = {
                    {inputId, &interfaceStates[inputNodeId]},
                    {outputId, &nodeState}
                };
            }
        }
    }
    
    // Separate graph in sub-graphs if applicable.
    U64 clusterCount = 0;
    std::unordered_set<U64> visited;

    for (auto& [nodeId, node] : interfaceStates) {
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
            interfaceStates[current].clusterId = clusterCount;

            for (const auto& neighbor : edgesCache[current]) {
                if (!visited.contains(neighbor)) {
                    stack.push(neighbor);
                }
            }
        }
        clusterCount += 1;
    }

    // Create automatic graph layout.
    U64 columnId = 0;
    std::unordered_set<U64> S;

    for (const auto& [id, state] : interfaceStates) {
        if (state.inputs.size() == 0) {
            S.insert(id);
        }
    }

    while (!S.empty()) {
        std::unordered_map<U64, std::unordered_set<U64>> nodeMatches;

        // Build the matches map.
        for (const auto& sourceNodeId : S) {
            for (const auto& [_, output] : interfaceStates[sourceNodeId].outputs) {
                for (const auto& [targetNodeId, _] : inputCache[output.second->phash]) {
                    nodeMatches[targetNodeId].insert(sourceNodeId);
                }
            }
        }

        U64 previousSetSize = S.size();
        std::unordered_set<U64> nodesToInsert;
        std::unordered_set<U64> nodesToExclude;

        // Determine which nodes to insert and which to exclude.
        for (const auto& [targetNodeId, sourceNodes] : nodeMatches) {
            if (interfaceStates[targetNodeId].inputs.size() == sourceNodes.size()) {
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

        for (const auto& node : nodesToInsert) {
            const U64& clusterId = interfaceStates[node].clusterId;
            if (topological.size() <= clusterId) {
                topological.resize(clusterId + 1);
            }
            if (topological[clusterId].size() <= columnId) {
                topological[clusterId].resize(columnId + 1);
            }
            topological[clusterId][columnId].push_back(node);
            S.erase(node);
        }

        columnId += 1;
    }

    // Create scheduler.
    _scheduler = std::make_shared<Scheduler>(_window, blockStates, blockStateOrder);

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
    JST_DEBUG("[INSTANCE] Destroying instance.");

    // Check if instance is commited.
    if (!commited) {
        JST_FATAL("Can't create instance that wasn't created.");
        return Result::ERROR;
    }

    // Destroy instance window.
    if (_window && _viewport) {
        JST_CHECK(_window->destroy());
        JST_CHECK(_viewport->destroy());
    }

    // Destroy scheduler.
    _scheduler.reset();

    // Unlock instance.
    commited = false;

    return Result::SUCCESS;
}

Result Instance::compute() {
    return _scheduler->compute();
}

Result Instance::begin() {
    if (_window) {
        JST_CHECK(_window->begin());
    }

    ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());

    return Result::SUCCESS;
}

Result Instance::present() {
    return _scheduler->present();
}

Result Instance::end() {
    const auto& style = ImNodes::GetStyle();

    static const U32 CpuColor            = IM_COL32(224, 146,   0, 255);
    static const U32 CpuColorSelected    = IM_COL32(184, 119,   0, 255);
    static const U32 CudaColor           = IM_COL32(118, 201,   3, 255);
    static const U32 CudaColorSelected   = IM_COL32( 95, 161,   2, 255);
    static const U32 MetalColor          = IM_COL32( 98,  60, 234, 255);
    static const U32 MetalColorSelected  = IM_COL32( 76,  33, 232, 255);
    static const U32 VulkanColor         = IM_COL32(238,  27,  52, 255);
    static const U32 VulkanColorSelected = IM_COL32(209,  16,  38, 255);
    static const U32 WebGPUColor         = IM_COL32( 59, 165, 147, 255);
    static const U32 WebGPUColorSelected = IM_COL32( 49, 135, 121, 255);

    //
    // View Render
    //

    for (const auto& [_, state] : interfaceStates) {
        if (!state.block->interface->config.viewEnabled ||
            !state.block->interface->shouldDrawView()) {
            continue;
        }

        ImGui::Begin(state.title.c_str(), &state.block->interface->config.viewEnabled);
        state.block->interface->drawView();
        ImGui::End();
    }

    //
    // Control Render
    //

    ImGui::Begin("Control");
    for (auto& [_, state] : interfaceStates) {
        if (!state.block->interface->shouldDrawControl()) {
            continue;
        }

        if (ImGui::CollapsingHeader(state.title.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
            state.block->interface->drawControl();
        }
    }
    ImGui::End();

    //
    // Info Render
    //

    ImGui::Begin("Info");

    if (ImGui::CollapsingHeader("Compute", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::TextFormatted("Graphs: {}", _scheduler->getNumberOfGraphs());
        ImGui::TextFormatted("Present Blocks: {}", _scheduler->getNumberOfPresentBlocks());
        ImGui::TextFormatted("Compute Blocks: {}", _scheduler->getNumberOfComputeBlocks());
        // TODO: Add printout of compute devices here.
    }

    if (ImGui::CollapsingHeader("Graphics", ImGuiTreeNodeFlags_DefaultOpen)) {
        _window->drawDebugMessage();
        ImGui::TextFormatted("Dropped Frames: {}", _window->stats().droppedFrames);
        ImGui::TextFormatted("Render Device: {}", GetDevicePrettyName(_window->device()));
        ImGui::TextFormatted("Viewport Platform: {}", _viewport->prettyName());
    }

    for (auto& [_, state] : interfaceStates) {
        if (!state.block->interface->shouldDrawInfo()) {
            continue;
        }

        if (ImGui::CollapsingHeader(state.title.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
            state.block->interface->drawInfo();
        }
    }

    ImGui::End();

    //
    // Flowgraph Render
    //

    ImGui::Begin("Flowgraph");
    
    ImNodes::BeginNodeEditor();
    ImNodes::MiniMap(0.2f, ImNodesMiniMapLocation_TopRight);

    for (auto& [id, state] : interfaceStates) {
        F32& nodeWidth = state.block->interface->config.nodeWidth;
        const F32 titleWidth = ImGui::CalcTextSize(state.title.c_str()).x;
        const F32 controlWidth = state.block->interface->shouldDrawControl() ? 500.0f : 0.0f;
        const F32 previewWidth = state.block->interface->shouldDrawPreview() ? 500.0f : 0.0f;
        nodeWidth = std::max({titleWidth, nodeWidth, controlWidth, previewWidth});

        // Push node-specific style.
        switch (state.block->interface->device()) {
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

        ImNodes::BeginNode(id);

        // Draw node title.
        ImNodes::BeginNodeTitleBar();
        ImGui::TextUnformatted(state.title.c_str());
        ImNodes::EndNodeTitleBar();

        // Draw node control.
        if (state.block->interface->shouldDrawControl()) {
            ImGui::Spacing();
            ImGui::PushItemWidth(nodeWidth - 200.0f);
            state.block->interface->drawControl();
            ImGui::PopItemWidth();
        }

        // Draw node input and output pins.
        if (!state.inputs.empty() || !state.outputs.empty()) {
            ImGui::Spacing();

            for (const auto& [inputId, input] : state.inputs) {
                const auto& [inputName, _] = input;
                ImNodes::BeginInputAttribute(inputId);
                ImGui::TextUnformatted(inputName.c_str());
                ImNodes::EndInputAttribute();
            }

            for (const auto& [outputId, output] : state.outputs) {
                const auto& [outputName, _] = output;
                ImNodes::BeginOutputAttribute(outputId);
                const F32 textWidth = ImGui::CalcTextSize(outputName.c_str()).x;
                ImGui::Indent(nodeWidth - textWidth);
                ImGui::TextUnformatted(outputName.c_str());
                ImNodes::EndInputAttribute();
            }
        }

        // Draw node preview.
        if (state.block->interface->shouldDrawPreview() &&
            state.block->interface->config.previewEnabled) {
            ImGui::Spacing();
            state.block->interface->drawPreview(nodeWidth);
        }

        // Ensure minimum width set by the internal state.
        ImGui::Dummy(ImVec2(nodeWidth, 0.0f));

        // Draw interfacing options.
        if (state.block->interface->shouldDrawView() || state.block->interface->shouldDrawPreview()) {
            ImGui::Spacing();

            if (state.block->interface->shouldDrawView()) {
                ImGui::Checkbox("Window", &state.block->interface->config.viewEnabled);

                if (state.block->interface->shouldDrawPreview()) {
                    ImGui::SameLine();
                }
            }

            if (state.block->interface->shouldDrawPreview()) {
                ImGui::Checkbox("Preview", &state.block->interface->config.previewEnabled);
                ImGui::SameLine();

                const auto& nodeOrigin = ImNodes::GetNodeScreenSpacePos(id);
                ImGui::SetCursorPosX(nodeOrigin.x + nodeWidth - 100.0f);

                if (ImGui::Button(" - ", ImVec2(40.0f, 0.0f))) {
                    nodeWidth -= 50.0f;
                }
                ImGui::SameLine();
                if (ImGui::Button(" + ", ImVec2(40.0f, 0.0f))) {
                    nodeWidth += 50.0f;
                }
            }
        }

        ImNodes::EndNode();

        // Pop node-specific style.
        ImNodes::PopColorStyle(); // TitleBar
        ImNodes::PopColorStyle(); // TitleBarHovered
        ImNodes::PopColorStyle(); // TitleBarSelected
        ImNodes::PopColorStyle(); // Pin
        ImNodes::PopColorStyle(); // PinHovered

        // Set node position according to the internal state.
        const auto& [x, y] = state.block->interface->config.nodePos;
        ImNodes::SetNodeGridSpacePos(id, ImVec2(x, y));
    }

    // Spatially organize graph.
    if (!graphSpatiallyOrganized) {
        F32 previousClustersHeight = 0.0f;
        
        for (const auto& cluster : topological) {
            F32 largestColumnHeight = 0.0f;
            F32 previousColumnsWidth = 0.0f;
            
            for (const auto& column : cluster) {
                F32 largestNodeWidth = 0.0f;
                F32 previousNodesHeight = 0.0f;

                for (const auto& node : column) {
                    auto& [x, y] = interfaceStates[node].block->interface->config.nodePos;
                    x = previousColumnsWidth;
                    y = previousNodesHeight + previousClustersHeight;
                    ImNodes::SetNodeGridSpacePos(node, ImVec2(x, y));

                    const auto& dims = ImNodes::GetNodeDimensions(node);
                    previousNodesHeight += dims.y + 50.0f;
                    largestNodeWidth = std::max({
                        dims.x,
                        largestNodeWidth,
                    });
                }

                // Extra pass to add left padding to nodes in the same column.
                for (const auto& node : column) {
                    const auto& dims = ImNodes::GetNodeDimensions(node);
                    auto& [x, y] = interfaceStates[node].block->interface->config.nodePos;
                    x += (largestNodeWidth - dims.x);
                    ImNodes::SetNodeGridSpacePos(node, ImVec2(x, y));
                }

                largestColumnHeight = std::max({
                    previousNodesHeight,
                    largestColumnHeight,
                });

                previousColumnsWidth += largestNodeWidth + 150.0f;
            }

            previousClustersHeight += largestColumnHeight + 25.0f;
        }

        graphSpatiallyOrganized = true;
    }

    // Draw node links.
    for (const auto& [id, connection] : nodeConnections) {
        const auto& [input, output] = connection;
        const auto& [inputId, _] = input;
        const auto& [outputId, outputState] = output;

        switch (outputState->block->interface->device()) {
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

        ImNodes::Link(id, inputId, outputId);

        ImNodes::PopColorStyle(); // Link
        ImNodes::PopColorStyle(); // LinkHovered
        ImNodes::PopColorStyle(); // LinkSelected
    }

    // Update internal state node position via drag callback. 
    I32 numSelectedNodes;
    if (ImNodes::IsNodesDragStopped(&numSelectedNodes)) {
        std::vector<I32> selectedNodes(numSelectedNodes);
        ImNodes::GetSelectedNodes(selectedNodes.data());

        for (const auto& id : selectedNodes) {
            const auto& [x, y] = ImNodes::GetNodeGridSpacePos(id);
            interfaceStates[id].block->interface->config.nodePos = {x, y};
        }
    }

    ImNodes::EndNodeEditor();

    // Render underlying buffer information about the link.
    I32 linkId;
    if (ImNodes::IsLinkHovered(&linkId)) {
        const auto& [_, output] = nodeConnections[linkId];
        const auto& [outputId, outputState] = output;
        const auto& [outputName, vec] = outputState->outputs[outputId];

        const auto firstLine = fmt::format("Vector ({})", outputName);
        const auto secondLine = fmt::format("[{}] {} [Device::{}] [{:02}]", vec->type, vec->shape, vec->device, vec->phash - vec->hash);
        const auto thirdLine = fmt::format("[PTR: 0x{:016X}] [HASH: 0x{:016X}]", reinterpret_cast<uintptr_t>(vec->ptr), vec->hash);

        ImGui::BeginTooltip();
        ImGui::TextUnformatted(firstLine.c_str());
        ImGui::TextUnformatted(secondLine.c_str());
        ImGui::TextUnformatted(thirdLine.c_str());
        ImGui::EndTooltip();
    }

    // Update the internal state when a link is deleted.
    if (ImNodes::IsLinkDestroyed(&linkId)) {
        // TODO: Add link deletion.
    }

    // Update the internal state when a link is created.
    I32 startId, endId;
    if (ImNodes::IsLinkCreated(&startId, &endId)) {
        // TODO: Add link creation.
    }

    // Resize node by dragging interface logic.
    I32 nodeId;
    if (ImNodes::IsNodeHovered(&nodeId)) {
        const auto nodeDims = ImNodes::GetNodeDimensions(nodeId);
        const auto nodeOrigin = ImNodes::GetNodeScreenSpacePos(nodeId);

        F32& nodeWidth = interfaceStates[nodeId].block->interface->config.nodeWidth;

        bool isNearRightEdge = 
            std::abs((nodeOrigin.x + nodeDims.x) - ImGui::GetMousePos().x) < 10.0f &&
            ImGui::GetMousePos().y >= nodeOrigin.y &&
            ImGui::GetMousePos().y <= (nodeOrigin.y + nodeDims.y);

        if (isNearRightEdge && ImGui::IsMouseDown(0) && !nodeDragId) {
            ImNodes::SetNodeDraggable(nodeId, false);
            nodeDragId = nodeId;
        }

        if (nodeDragId) {
            nodeWidth = (ImGui::GetMousePos().x - nodeOrigin.x) - (style.NodePadding.x * 2.0f);
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

    if (_window) {
        JST_CHECK(_window->end());
    }

    return Result::SUCCESS;
}

}  // namespace Jetstream
