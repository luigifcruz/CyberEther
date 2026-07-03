#include <jetstream/render/sakura/components/node/pin.hh>

#include <jetstream/render/sakura/components/table.hh>
#include <jetstream/render/sakura/components/text.hh>
#include <jetstream/render/sakura/components/tooltip.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct NodePin::Impl {
    Config config;
    Text label;
    Text help;
    Tooltip helpTooltip;
    bool tensorTooltip = false;
    Text tensorSection;
    Table tensorMetadataTable;
    Text tensorAttributesSection;
    Table tensorAttributeMetadataTable;
};

NodePin::NodePin() {
    this->impl = std::make_unique<Impl>();
}

NodePin::~NodePin() = default;
NodePin::NodePin(NodePin&&) noexcept = default;
NodePin& NodePin::operator=(NodePin&&) noexcept = default;

bool NodePin::update(Config config) {
    impl->config = std::move(config);
    const std::string id = "NodePin" + impl->config.id;
    impl->label.update({
        .id = id + "Label",
        .str = impl->config.label,
        .align = impl->config.direction == Direction::Output ? Text::Align::Right : Text::Align::Left,
    });
    impl->help.update({
        .id = id + "Help",
        .str = impl->config.help,
        .wrapped = true,
    });
    impl->helpTooltip.update({
        .id = id + "HelpTooltip",
        .wrapWidth = 560.0f,
    });
    impl->tensorTooltip = false;
    if (impl->config.dataShape.size() > 0) {
        impl->tensorTooltip = true;
        impl->tensorSection.update({
            .id = impl->config.id + ":tooltip:tensor-section",
            .str = "Layout",
            .font = Sakura::Text::Font::Bold,
        });
        impl->tensorMetadataTable.update({
            .id = impl->config.id + ":tooltip:tensor-table",
            .columns = {"", ""},
            .rows = {
                {"Device", jst::fmt::format("{}", impl->config.dataDevice)},
                {"Type", jst::fmt::format("{}", impl->config.dataType)},
                {"Shape", jst::fmt::format("{}", impl->config.dataShape)},
                {"Strides", jst::fmt::format("{} ({})", impl->config.dataStride,
                                                        impl->config.dataContiguous ? "Contiguous" :
                                                                                      "Non-contiguous")},
                {"Offset", jst::fmt::format("{} bytes", impl->config.dataOffsetBytes)},
            },
            .fixedColumnWidths = {72.0f},
            .showHeaders = false,
            .wrapped = true,
        });
        if (!impl->config.dataAttributes.empty()) {
            impl->tensorAttributesSection.update({
                .id = impl->config.id + ":tooltip:tensor-attributes-section",
                .str = "Attributes",
                .font = Sakura::Text::Font::Bold,
            });
            impl->tensorAttributeMetadataTable.update({
                .id = impl->config.id + ":tooltip:tensor-attribute-table",
                .columns = {"", ""},
                .rows = impl->config.dataAttributes,
                .fixedColumnWidths = {120.0f},
                .showHeaders = false,
                .wrapped = true,
            });
        }
    }
    return true;
}

void NodePin::render(const Context& ctx) const {
    const auto& config = impl->config;

    const auto renderLabel = [&]() {
        impl->label.render(ctx);
        if (!config.help.empty() || impl->tensorTooltip) {
            impl->helpTooltip.render(ctx, [this](const Context& ctx) {
                if (!this->impl->config.help.empty()) {
                    this->impl->help.render(ctx);
                }
                if (this->impl->tensorTooltip) {
                    this->impl->tensorSection.render(ctx);
                    this->impl->tensorMetadataTable.render(ctx);
                    if (!this->impl->config.dataAttributes.empty()) {
                        this->impl->tensorAttributesSection.render(ctx);
                        this->impl->tensorAttributeMetadataTable.render(ctx);
                    }
                }
            });
        }
    };

    const NodeEditor::PinRef pin{
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
