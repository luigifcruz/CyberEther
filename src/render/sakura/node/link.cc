#include <jetstream/render/sakura/node/link.hh>

#include <jetstream/render/sakura/tooltip.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct NodeLink::Impl {
    Config config;
    Tooltip linkTooltip;
    bool hovered = false;
};

NodeLink::NodeLink() {
    this->impl = std::make_unique<Impl>();
}

NodeLink::~NodeLink() = default;
NodeLink::NodeLink(NodeLink&&) noexcept = default;
NodeLink& NodeLink::operator=(NodeLink&&) noexcept = default;

bool NodeLink::update(Config config) {
    auto& impl = *this->impl;
    impl.config = std::move(config);
    impl.linkTooltip.update({
        .id = impl.config.id + ":tooltip",
        .wrapWidth = 420.0f,
        .delayed = false,
        .visible = true,
    });
    return true;
}

void NodeLink::render(const Context& ctx, Child tooltip) const {
    auto& impl = *this->impl;
    const auto& config = impl.config;

    Private::RegisterNodeEditorLink(config.id);
    Private::RegisterNodeEditorLinkHover(config.id, [this](const bool nextHovered) {
        this->impl->hovered = nextHovered;
        if (this->impl->config.onHover) {
            this->impl->config.onHover(nextHovered);
        }
    });
    if (!config.unresolved && impl.hovered && tooltip) {
        Private::RegisterNodeEditorLinkTooltip(config.id, [this, tooltip = std::move(tooltip)](const Context& ctx) {
            this->impl->linkTooltip.render(ctx, tooltip);
        });
    }

    if (config.unresolved) {
        const ImU32 color = ImGui::ColorConvertFloat4ToU32(Private::ImColor(ctx, "node_outline_pending"));
        ImNodes::PushColorStyle(ImNodesCol_Link, color);
        ImNodes::PushColorStyle(ImNodesCol_LinkHovered, color);
        ImNodes::PushColorStyle(ImNodesCol_LinkSelected, color);
    }

    const NodeEditorPinRef startPin{
        .nodeId = config.start.nodeId,
        .pinId = config.start.pinId,
        .isInput = config.start.isInput,
    };
    const NodeEditorPinRef endPin{
        .nodeId = config.end.nodeId,
        .pinId = config.end.pinId,
        .isInput = config.end.isInput,
    };
    ImNodes::Link(Private::NodeEditorObjectId(config.id),
                  Private::NodeEditorPinObjectId(startPin),
                  Private::NodeEditorPinObjectId(endPin));

    if (config.unresolved) {
        ImNodes::PopColorStyle();
        ImNodes::PopColorStyle();
        ImNodes::PopColorStyle();
    }
}

}  // namespace Jetstream::Sakura
