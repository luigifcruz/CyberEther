#include <jetstream/render/sakura/node/runtime_overlay.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct NodeRuntimeOverlay::Impl {
    Config config;
};

NodeRuntimeOverlay::NodeRuntimeOverlay() {
    this->impl = std::make_unique<Impl>();
}

NodeRuntimeOverlay::~NodeRuntimeOverlay() = default;
NodeRuntimeOverlay::NodeRuntimeOverlay(NodeRuntimeOverlay&&) noexcept = default;
NodeRuntimeOverlay& NodeRuntimeOverlay::operator=(NodeRuntimeOverlay&&) noexcept = default;

bool NodeRuntimeOverlay::update(Config config) {
    this->impl->config = std::move(config);
    return true;
}

void NodeRuntimeOverlay::render(const Context& ctx) const {
    const auto& config = this->impl->config;

    if (config.lines.empty()) {
        return;
    }

    Geometry geometry{
        .nodePos = config.nodePos,
        .nodeSize = config.nodeSize,
    };
    if (config.onResolveGeometry) {
        const auto resolved = config.onResolveGeometry();
        if (!resolved.has_value()) {
            return;
        }
        geometry = resolved.value();
    }

    const ImU32 primaryColor = ImGui::ColorConvertFloat4ToU32(Private::ImColor(ctx, "text_primary", ImVec4(1.0f, 1.0f, 1.0f, 1.0f)));
    const ImU32 secondaryColor = ImGui::ColorConvertFloat4ToU32(Private::ImColor(ctx, "text_secondary"));
    const Extent2D<F32> textPos = {geometry.nodePos.x,
                                   geometry.nodePos.y + geometry.nodeSize.y + Scale(ctx, config.offset)};
    ImDrawList* drawList = ImGui::GetWindowDrawList();

    for (U64 i = 0; i < config.lines.size(); ++i) {
        const ImVec2 pos(textPos.x, textPos.y + static_cast<F32>(i) * ImGui::GetTextLineHeight());
        const ImU32 lineColor = (i == 0 || i + 1 == config.lines.size()) ? primaryColor : secondaryColor;
        drawList->AddText(pos, lineColor, config.lines[i].c_str());
    }
}

}  // namespace Jetstream::Sakura
