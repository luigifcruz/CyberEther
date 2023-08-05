#include "jetstream/instance.hh"

namespace Jetstream {

Result Instance::create() {
    JST_DEBUG("Creating instance.");

    // Check if instance isn't commited.
    if (commited) {
        JST_FATAL("The instance was already commited.");
        return Result::ERROR;
    }

    std::vector<std::pair<U64, U64>> nodeInputMap;
    std::unordered_map<U64, std::tuple<U64, U64>> nodeOutputMap;

    U64 id = 1;
    for (const auto& [name, block] : blockStates) {
        if (!block.interface) {
            continue;
        }

        const U64 nodeId = id++;
        auto& state = nodeStates[nodeId];

        state.name = name;
        state.title = fmt::format("{} ({})", block.interface->prettyName(), name);

        for (const auto& [inputName, inputMeta] : block.record.data.inputMap) {
            nodeInputMap.push_back({inputMeta.vector.phash, id});
            state.inputs[id++] = inputName;
        }

        for (const auto& [outputName, outputMeta] : block.record.data.outputMap) {
            nodeOutputMap[outputMeta.vector.phash] = {nodeId, id};
            state.outputs[id++] = outputName;
        }
    }

    for (const auto& [inputHash, inputId] : nodeInputMap) {
        const auto& [nodeId, outputId] = nodeOutputMap[inputHash];
        nodeConnections.push_back({inputId, outputId, nodeId});
    }
    
    // TODO: Implement graph layout algorithm.

    // Create scheduler.
    _scheduler = std::make_shared<Scheduler>(_window, blockStates, blockStateMap);

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
    JST_DEBUG("Destroying instance.");

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

    for (auto& [_, nodeState] : nodeStates) {
        auto& blockState = blockStates[nodeState.name];

        if (!blockState.interface->config.viewEnabled ||
            !blockState.interface->shouldDrawView()) {
            continue;
        }

        ImGui::Begin(nodeState.title.c_str(), &blockState.interface->config.viewEnabled);
        blockState.interface->drawView();
        ImGui::End();
    }

    //
    // Control Render
    //

    ImGui::Begin("Control");
    for (auto& [_, nodeState] : nodeStates) {
        const auto& blockState = blockStates[nodeState.name];

        if (!blockState.interface->shouldDrawControl()) {
            continue;
        }

        if (ImGui::CollapsingHeader(nodeState.title.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
            blockState.interface->drawControl();
        }
    }
    ImGui::End();

    //
    // Info Render
    //

    ImGui::Begin("Info");

    if (ImGui::CollapsingHeader("Compute", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("Graphs: %lu", _scheduler->getNumberOfGraphs());
        ImGui::Text("Present Blocks: %lu", _scheduler->getNumberOfPresentBlocks());
        ImGui::Text("Compute Blocks: %lu", _scheduler->getNumberOfComputeBlocks());
        const auto computeDevices = fmt::format("{}", _scheduler->getComputeDevices());
        ImGui::Text("Compute Devices: %s", computeDevices.c_str());
    }

    if (ImGui::CollapsingHeader("Graphics", ImGuiTreeNodeFlags_DefaultOpen)) {
        _window->drawDebugMessage();
        ImGui::Text("Dropped Frames: %lu", _window->stats().droppedFrames);
        ImGui::Text("Render Device: %s", GetDevicePrettyName(_window->device()));
        ImGui::Text("Viewport Platform: %s", _viewport->prettyName().c_str());
    }

    for (auto& [_, nodeState] : nodeStates) {
        const auto& blockState = blockStates[nodeState.name];

        if (!blockState.interface->shouldDrawInfo()) {
            continue;
        }

        if (ImGui::CollapsingHeader(nodeState.title.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
            blockState.interface->drawInfo();
        }
    }

    ImGui::End();

    //
    // Flowgraph Render
    //

    ImGui::Begin("Flowgraph");
    
    ImNodes::BeginNodeEditor();
    ImNodes::MiniMap(0.2f, ImNodesMiniMapLocation_TopRight);

    for (auto& [id, nodeState] : nodeStates) {
        auto& blockState = blockStates[nodeState.name];

        F32& nodeWidth = blockState.interface->config.nodeWidth;
        const F32 titleWidth = ImGui::CalcTextSize(nodeState.title.c_str()).x;
        const F32 controlWidth = blockState.interface->shouldDrawControl() ? 500.0f : 0.0f;
        const F32 previewWidth = blockState.interface->shouldDrawPreview() ? 500.0f : 0.0f;
        nodeWidth = std::max({titleWidth, nodeWidth, controlWidth, previewWidth});

        // Push node-specific style.
        switch (blockState.interface->device()) {
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
        ImGui::TextUnformatted(nodeState.title.c_str());
        ImNodes::EndNodeTitleBar();

        // Draw node control.
        if (blockState.interface->shouldDrawControl()) {
            ImGui::Spacing();
            ImGui::PushItemWidth(nodeWidth - 200.0f);
            blockState.interface->drawControl();
            ImGui::PopItemWidth();
        }

        // Draw node input and output pins.
        if (!nodeState.inputs.empty() || !nodeState.outputs.empty()) {
            ImGui::Spacing();

            for (const auto& [inputId, inputName] : nodeState.inputs) {
                ImNodes::BeginInputAttribute(inputId);
                ImGui::TextUnformatted(inputName.c_str());
                ImNodes::EndInputAttribute();
            }

            for (const auto& [outputId, outputName] : nodeState.outputs) {
                ImNodes::BeginOutputAttribute(outputId);
                const F32 textWidth = ImGui::CalcTextSize(outputName.c_str()).x;
                ImGui::Indent(nodeWidth - textWidth);
                ImGui::TextUnformatted(outputName.c_str());
                ImNodes::EndInputAttribute();
            }
        }

        // Draw node preview.
        if (blockState.interface->shouldDrawPreview() &&
            blockState.interface->config.previewEnabled) {
            ImGui::Spacing();
            blockState.interface->drawPreview(nodeWidth);
        }

        // Ensure minimum width set by the internal state.
        ImGui::Dummy(ImVec2(nodeWidth, 0.0f));

        // Draw interfacing options.
        if (blockState.interface->shouldDrawView() || blockState.interface->shouldDrawPreview()) {
            ImGui::Spacing();

            if (blockState.interface->shouldDrawView()) {
                ImGui::Checkbox("Window", &blockState.interface->config.viewEnabled);

                if (blockState.interface->shouldDrawPreview()) {
                    ImGui::SameLine();
                }
            }

            if (blockState.interface->shouldDrawPreview()) {
                ImGui::Checkbox("Preview", &blockState.interface->config.previewEnabled);
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
        const auto& [x, y] = blockState.interface->config.nodePos;
        ImNodes::SetNodeGridSpacePos(id, ImVec2(x, y));
    }

    // Draw node links.
    U64 nodeConnectionId = 0;
    for (const auto& [a, b, nodeId] : nodeConnections) {
        switch (blockStates[nodeStates[nodeId].name].interface->device()) {
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

        ImNodes::Link(nodeConnectionId++, a, b);

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
            blockStates[nodeStates[id].name].interface->config.nodePos = {x, y};
        }
    }

    ImNodes::EndNodeEditor();

    // Render underlying buffer information about the link.
    I32 linkId;
    if (ImNodes::IsLinkHovered(&linkId)) {
        const auto& [_, outputId, nodeId] = nodeConnections[linkId];
        auto& nodeState = nodeStates[nodeId];
        const auto& outputName = nodeState.outputs[outputId];
        const auto& vec = blockStates[nodeState.name].record.data.outputMap[outputName].vector;

        const auto firstLine = fmt::format("Vector ({})", outputName);
        const auto secondLine = fmt::format("[{}] {} [Device::{}] [{:02}]", vec.type, vec.shape, vec.device, vec.phash - vec.hash);
        const auto thirdLine = fmt::format("[PTR: 0x{:016X}] [HASH: 0x{:016X}]", reinterpret_cast<uintptr_t>(vec.ptr), vec.hash);

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

        auto& blockState = blockStates[nodeStates[nodeId].name];
        F32& nodeWidth = blockState.interface->config.nodeWidth;

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
