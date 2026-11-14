#include "link.hh"

namespace Jetstream {

std::string FlowgraphLinkId(const std::string& consumerName, const std::string& inputSlot) {
    return jst::fmt::format("{}:{}link", consumerName, inputSlot);
}

void FlowgraphLink::update(Config config) {
    this->config = std::move(config);
    const bool asked = linkTooltip.requested();
    const bool resolved = this->config.connection.resolved;
    bool changed = false;
    if (resolved && linkTooltip.prepared()) {
        const auto& connection = this->config.connection;
        FlowgraphTooltipRows link = {
            {"Producer", jst::fmt::format("{}.{}", connection.producerName, connection.producerPort)},
            {"Consumer", jst::fmt::format("{}.{}", connection.consumerName, connection.inputSlot)},
            {"Tensor ID", jst::fmt::format("{}", connection.tensor.id())},
        };
        auto layout = FlowgraphTensorLayoutRows(connection.tensor);
        auto attributes = FlowgraphTensorAttributeRows(connection.tensor);
        changed = link != linkRows || layout != layoutRows || attributes != attributeRows;
        if (changed) {
            ++tooltipRevision;
            linkRows = std::move(link);
            layoutRows = std::move(layout);
            attributeRows = std::move(attributes);
        }
    }
    linkSpec = {
        .id = this->config.id,
        .start = {
            .nodeId = this->config.start.nodeId,
            .pinId = this->config.start.pinId,
            .isInput = this->config.start.isInput,
        },
        .end = {
            .nodeId = this->config.end.nodeId,
            .pinId = this->config.end.pinId,
            .isInput = this->config.end.isInput,
        },
        .unresolved = !this->config.connection.resolved,
        .tooltip = {
            .content = [this]() -> Sakura::Component* {
                if (!this->config.connection.resolved) {
                    return nullptr;
                }
                return linkTooltip.show();
            },
            .revision = tooltipRevision,
        },
    };
    if (resolved && (asked || changed)) {
        prepareTooltip();
    }
}

Sakura::NodeEditor::Link FlowgraphLink::takeLink() {
    return std::move(linkSpec);
}

const FlowgraphLink::Connection& FlowgraphLink::connection() const {
    return config.connection;
}

void FlowgraphLink::prepareTooltip() {
    using Text = Sakura::Text;

    const bool hasAttributes = !attributeRows.empty();

    tooltipTitle.update({
        .str = ICON_LC_MEMORY_STICK " Tensor Metadata",
        .wrapped = true,
    });
    tooltipHeaderDivider.update({.spacing = 0.0f});
    tooltipHelp.update({
        .str = ICON_LC_INFO " Click on the end of the link to detach it.",
        .tone = Text::Tone::Secondary,
        .wrapped = true,
    });
    tooltipHelpDivider.update({.spacing = 0.0f});
    linkSection.update({
        .str = "Link",
        .font = Text::Font::Bold,
    });
    layoutSection.update({
        .str = "Layout",
        .font = Text::Font::Bold,
    });
    attributesSection.update({
        .str = "Attributes",
        .font = Text::Font::Bold,
    });
    linkMetadataTable.update({
        .columns = {"", ""},
        .rows = linkRows,
        .fixedColumnWidths = {72.0f},
        .showHeaders = false,
        .wrapped = true,
    });
    layoutMetadataTable.update({
        .columns = {"", ""},
        .rows = layoutRows,
        .fixedColumnWidths = {72.0f},
        .showHeaders = false,
        .wrapped = true,
    });
    attributeMetadataTable.update({
        .columns = {"", ""},
        .rows = attributeRows,
        .fixedColumnWidths = {120.0f},
        .showHeaders = false,
        .wrapped = true,
    });

    std::vector<Sakura::Stack::Item> items = {
        {.child = &tooltipTitle},
        {.child = &tooltipHeaderDivider},
        {.child = &tooltipHelp},
        {.child = &tooltipHelpDivider},
        {.child = &linkSection},
        {.child = &linkMetadataTable},
        {.child = &layoutSection},
        {.child = &layoutMetadataTable},
    };
    if (hasAttributes) {
        items.push_back({.child = &attributesSection});
        items.push_back({.child = &attributeMetadataTable});
    }
    tooltipContent.update({.items = std::move(items)});
    linkTooltip.update({
        .id = this->config.id + ":tooltip",
        .content = &tooltipContent,
        .wrapWidth = 420.0f,
    });
}

}  // namespace Jetstream
