#include <jetstream/render/sakura/components/node/editor.hh>

#include "base.hh"
#include "../../state.hh"

namespace Jetstream::Sakura {

struct NodeEditor::Impl {
    Impl() {
        nodeContext = ImNodes::CreateContext();
    }

    ~Impl() {
        if (!nodeContext) {
            return;
        }
        if (ImNodes::GetCurrentContext() == nodeContext) {
            ImNodes::SetCurrentContext(nullptr);
        }
        ImNodes::DestroyContext(nodeContext);
    }

    void applyImNodesStyle(const Context& ctx) const {
        const auto color = [&](const std::string& key) {
            return ImGui::ColorConvertFloat4ToU32(Private::ImColor(ctx, key));
        };

        auto& colors = ImNodes::GetStyle().Colors;
        colors[ImNodesCol_NodeBackground]         = color("node_background");
        colors[ImNodesCol_NodeBackgroundHovered]  = color("node_background");
        colors[ImNodesCol_NodeBackgroundSelected] = color("node_background");
        colors[ImNodesCol_NodeOutline]            = color("node_outline");
        colors[ImNodesCol_TitleBar]               = color("node_title_bar");
        colors[ImNodesCol_TitleBarHovered]        = color("node_title_bar");
        colors[ImNodesCol_TitleBarSelected]       = color("node_title_bar");
        colors[ImNodesCol_Pin]                    = color("node_pin");
        colors[ImNodesCol_PinHovered]             = color("node_pin");
        colors[ImNodesCol_Link]                   = color("node_link");
        colors[ImNodesCol_LinkHovered]            = color("node_link");
        colors[ImNodesCol_LinkSelected]           = color("node_link");
        colors[ImNodesCol_GridLine]               = color("grid_line");
        colors[ImNodesCol_GridBackground]         = color("grid_background");

        const F32 scale = ScalingFactor(ctx);
        auto& style = ImNodes::GetStyle();
        style.NodePadding               = ImVec2(8.0f * scale, 8.0f * scale);
        style.PinCircleRadius           = 4.0f  * scale;
        style.GridSpacing               = 23.0f * scale;
        style.NodeBorderThickness       = 2.0f  * scale;
        style.NodeCornerRounding        = 12.0f * scale;
        style.LinkThickness             = 1.5f  * scale;
        style.PinLineThickness          = 1.0f  * scale;
        style.LinkLineSegmentsPerLength = 0.2f  / scale;
        style.MiniMapOffset             = ImVec2(8.0f * scale, 8.0f * scale);
    }

    Config config;
    ImNodesContext* nodeContext = nullptr;
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
    if (!this->impl->nodeContext) {
        return;
    }

    const auto previousContext = ImNodes::GetCurrentContext();
    ImNodes::SetCurrentContext(this->impl->nodeContext);
    this->impl->applyImNodesStyle(ctx);
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
    const bool uiInputActive = io.WantTextInput || ImGui::IsAnyItemActive() ||
                               Private::IsKeyboardInputCaptured();
    if (!uiInputActive && (commandPressed || controlPressed) && ImGui::IsKeyPressed(ImGuiKey_C, false)) {
        if (config.onCopyShortcut) {
            config.onCopyShortcut(selected);
        }
    }
    if (!uiInputActive && (commandPressed || controlPressed) && ImGui::IsKeyPressed(ImGuiKey_V, false)) {
        if (config.pasteEnabled && config.onPasteShortcut) {
            config.onPasteShortcut(mouseGridPosition);
        }
    }

    ImGui::PopFont();
    ImGui::PopID();
    ImNodes::SetCurrentContext(previousContext);
}

}  // namespace Jetstream::Sakura
