#include <jetstream/render/sakura/node/pin.hh>

#include <jetstream/render/sakura/text.hh>
#include <jetstream/render/sakura/tooltip.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct NodePin::Impl {
    Config config;
    Text label;
    Text help;
    Tooltip helpTooltip;
};

NodePin::NodePin() {
    this->impl = std::make_unique<Impl>();
}

NodePin::~NodePin() = default;
NodePin::NodePin(NodePin&&) noexcept = default;
NodePin& NodePin::operator=(NodePin&&) noexcept = default;

bool NodePin::update(Config config) {
    auto& impl = *this->impl;
    impl.config = std::move(config);
    const std::string id = "NodePin" + impl.config.id;
    impl.label.update({
        .id = id + "Label",
        .str = impl.config.label,
        .align = impl.config.direction == Direction::Output ? Text::Align::Right : Text::Align::Left,
    });
    impl.help.update({
        .id = id + "Help",
        .str = impl.config.help,
        .wrapped = true,
    });
    impl.helpTooltip.update({
        .id = id + "HelpTooltip",
        .wrapWidth = 560.0f,
    });
    return true;
}

void NodePin::render(const Context& ctx) const {
    const auto& impl = *this->impl;
    const auto& config = impl.config;

    const auto renderLabel = [&]() {
        impl.label.render(ctx);
        if (!config.help.empty()) {
            impl.helpTooltip.render(ctx, [this](const Context& ctx) {
                this->impl->help.render(ctx);
            });
        }
    };

    const NodeEditorPinRef pin{
        .nodeId = Private::NodeEditorNodeStack().back(),
        .pinId = config.id,
        .isInput = config.direction == Direction::Input,
    };
    Private::RegisterNodeEditorPin(pin);

    const int imNodesId = Private::NodeEditorPinObjectId(pin);
    if (config.direction == Direction::Input) {
        if (config.enableDetach) {
            ImNodes::PushAttributeFlag(ImNodesAttributeFlags_EnableLinkDetachWithDragClick);
        }

        ImNodes::BeginInputAttribute(imNodesId, ImNodesPinShape_CircleFilled);
        renderLabel();
        ImNodes::EndInputAttribute();

        if (config.enableDetach) {
            ImNodes::PopAttributeFlag();
        }
        return;
    }

    ImNodes::BeginOutputAttribute(imNodesId, ImNodesPinShape_CircleFilled);
    renderLabel();
    ImNodes::EndOutputAttribute();
}

}  // namespace Jetstream::Sakura
