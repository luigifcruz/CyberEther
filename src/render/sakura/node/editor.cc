#include <jetstream/render/sakura/node/editor.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct NodeEditor::Impl {
    Config config;
};

NodeEditor::NodeEditor() {
    this->impl = std::make_unique<Impl>();
}

NodeEditor::~NodeEditor() = default;
NodeEditor::NodeEditor(NodeEditor&&) noexcept = default;
NodeEditor& NodeEditor::operator=(NodeEditor&&) noexcept = default;

bool NodeEditor::update(Config config) {
    this->impl->config = std::move(config);
    return true;
}

void NodeEditor::render(const Context& ctx, Child child) const {
    const auto& config = this->impl->config;

    ImNodesContext* nodeContext = Private::NativeNodeContext(ctx.nodeContext(config.contextId));
    if (!nodeContext) {
        return;
    }

    const auto previousContext = ImNodes::GetCurrentContext();
    ImNodes::SetCurrentContext(nodeContext);
    Private::ClearNodeEditorRegistries();
    ImGui::PushID(config.id.c_str());
    ImGui::PushFont(ImGui::GetFont(), ImGui::GetStyle().FontSizeBase * config.fontScale);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, Scale(ctx, config.childRounding));
    const ImVec2 editorMin = ImGui::GetCursorScreenPos();
    const ImVec2 editorSize = ImGui::GetContentRegionAvail();
    const ImVec2 editorCenter(editorMin.x + editorSize.x * 0.5f,
                              editorMin.y + editorSize.y * 0.5f);
    ImNodes::BeginNodeEditor();
    ImGui::PopStyleVar();
    if (child) {
        child(ctx);
    }
    ImNodes::EndNodeEditor();

    int hoveredNodeId = 0;
    const bool nodeHovered = ImNodes::IsNodeHovered(&hoveredNodeId);
    int hoveredPinId = 0;
    const bool pinHovered = ImNodes::IsPinHovered(&hoveredPinId);
    int hoveredLinkObjectId = 0;
    const bool linkHovered = !nodeHovered && !pinHovered && ImNodes::IsLinkHovered(&hoveredLinkObjectId);
    const bool editorHovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows);
    const ImVec2 mousePos = ImGui::GetMousePos();
    const Extent2D<F32> screenPosition = Private::ToExtent2D(mousePos);
    const Extent2D<F32> mouseGridPosition = Unscale(ctx, Private::ToExtent2D(ImNodes::ScreenSpaceToGridSpace(mousePos)));

    if (config.onMouseGridPositionChange) {
        config.onMouseGridPositionChange(mouseGridPosition);
    }
    if (config.onViewportGridCenterChange) {
        config.onViewportGridCenterChange(Unscale(ctx, Private::ToExtent2D(ImNodes::ScreenSpaceToGridSpace(editorCenter))));
    }

    if (editorHovered && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
        if (!nodeHovered && config.onEditorContextMenu) {
            config.onEditorContextMenu(mouseGridPosition, screenPosition);
        }
    }

    if (editorHovered && !nodeHovered && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
        if (config.onEditorDoubleClick) {
            config.onEditorDoubleClick(mouseGridPosition);
        }
    }

    int startPinId = 0;
    int endPinId = 0;
    if (config.onLinkCreated && ImNodes::IsLinkCreated(&startPinId, &endPinId)) {
        const auto startIt = Private::NodeEditorPinRegistry().find(startPinId);
        const auto endIt = Private::NodeEditorPinRegistry().find(endPinId);
        if (startIt != Private::NodeEditorPinRegistry().end() && endIt != Private::NodeEditorPinRegistry().end()) {
            config.onLinkCreated(startIt->second, endIt->second);
        }
    }

    int imNodesLinkId = 0;
    if (config.onLinkDestroyed && ImNodes::IsLinkDestroyed(&imNodesLinkId)) {
        const auto it = Private::NodeEditorLinkRegistry().find(imNodesLinkId);
        if (it != Private::NodeEditorLinkRegistry().end()) {
            config.onLinkDestroyed(it->second);
        }
    }

    for (const auto& [linkObjectId, onHover] : Private::NodeEditorLinkHoverRegistry()) {
        if (onHover) {
            onHover(linkHovered && hoveredLinkObjectId == linkObjectId);
        }
    }
    if (linkHovered) {
        const auto tooltipIt = Private::NodeEditorLinkTooltipRegistry().find(hoveredLinkObjectId);
        if (tooltipIt != Private::NodeEditorLinkTooltipRegistry().end() && tooltipIt->second) {
            tooltipIt->second(ctx);
        }
    }

    const int numSelectedNodes = ImNodes::NumSelectedNodes();
    std::vector<int> selectedNodeIds(numSelectedNodes);
    if (numSelectedNodes > 0) {
        ImNodes::GetSelectedNodes(selectedNodeIds.data());
    }

    std::vector<std::string> selected;
    selected.reserve(selectedNodeIds.size());
    for (const auto nodeId : selectedNodeIds) {
        const auto it = Private::NodeEditorNodeRegistry().find(nodeId);
        if (it != Private::NodeEditorNodeRegistry().end()) {
            selected.push_back(it->second);
        }
    }

    if (config.onSelectionChange) {
        config.onSelectionChange(selected);
    }

    const ImGuiIO& io = ImGui::GetIO();
    const bool commandPressed = (io.KeyMods & ImGuiMod_Super) != 0;
    const bool controlPressed = (io.KeyMods & ImGuiMod_Ctrl) != 0;
    if ((commandPressed || controlPressed) && ImGui::IsKeyPressed(ImGuiKey_C, false)) {
        if (config.onCopyShortcut) {
            config.onCopyShortcut(selected);
        }
    }
    if ((commandPressed || controlPressed) && ImGui::IsKeyPressed(ImGuiKey_V, false)) {
        if (config.pasteEnabled && config.onPasteShortcut) {
            config.onPasteShortcut(mouseGridPosition);
        }
    }

    ImGui::PopFont();
    ImGui::PopID();
    ImNodes::SetCurrentContext(previousContext);
}

}  // namespace Jetstream::Sakura
