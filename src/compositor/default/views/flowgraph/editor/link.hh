#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_LINK_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_LINK_HH

#include "node.hh"

#include "jetstream/render/tools/imgui_icons_ext.hh"

#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

inline std::string FlowgraphLinkId(const std::string& consumerName, const std::string& inputSlot) {
    return consumerName + ":" + inputSlot + "link";
}

struct FlowgraphLink {
    struct Connection {
        std::string consumerName;
        std::string inputSlot;
        std::string producerName;
        std::string producerPort;
        bool resolved = false;
        Tensor tensor;
    };

    struct Endpoint {
        std::string nodeId;
        std::string pinId;
        bool isInput = false;
    };

    struct Config {
        std::string id;
        Endpoint start;
        Endpoint end;
        Connection connection;
    };

    void update(Config config) {
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

    Sakura::NodeEditor::Link takeLink() {
        return std::move(linkSpec);
    }

    const Connection& connection() const {
        return config.connection;
    }

 private:
    void prepareTooltip() {
        using Text = Sakura::Text;

        const bool hasAttributes = !attributeRows.empty();

        tooltipTitle.update({
            .str = ICON_FA_MEMORY " Tensor Metadata",
            .wrapped = true,
        });
        tooltipHeaderDivider.update({.spacing = 0.0f});
        tooltipHelp.update({
            .str = ICON_FA_CIRCLE_INFO " Click on the end of the link to detach it.",
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

    Config config;
    Sakura::NodeEditor::Link linkSpec;
    FlowgraphTooltipRows linkRows;
    FlowgraphTooltipRows layoutRows;
    FlowgraphTooltipRows attributeRows;
    U64 tooltipRevision = 0;
    Tooltip linkTooltip;
    Sakura::Text tooltipTitle;
    Sakura::Divider tooltipHeaderDivider;
    Sakura::Text tooltipHelp;
    Sakura::Divider tooltipHelpDivider;
    Sakura::Text linkSection;
    Sakura::Table linkMetadataTable;
    Sakura::Text layoutSection;
    Sakura::Table layoutMetadataTable;
    Sakura::Text attributesSection;
    Sakura::Table attributeMetadataTable;
    Sakura::Stack tooltipContent;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_LINK_HH
