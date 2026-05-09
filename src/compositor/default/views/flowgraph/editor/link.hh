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

struct FlowgraphLink : public Sakura::Component {
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

        nodeLink.update({
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
            .onHover = [this](const bool hovered) {
                tooltipActive = hovered;
            },
        });

        if (!tooltipActive || !this->config.connection.resolved) {
            hasAttributes = false;
            return;
        }

        const auto& tensor = this->config.connection.tensor;

        std::vector<std::vector<std::string>> attributeRows;
        for (const auto& key : tensor.attributeKeys()) {
            std::string encoded;
            if (Parser::TypedToString(tensor.attribute(key), encoded) != Result::SUCCESS) {
                encoded = "?";
            }
            attributeRows.push_back({key, encoded});
        }
        hasAttributes = !attributeRows.empty();

        tooltipTitle.update({
            .id = this->config.id + ":tooltip:title",
            .str = ICON_FA_MEMORY " Tensor Metadata",
            .font = Sakura::Text::Font::Bold,
        });
        tooltipHeaderDivider.update({.id = this->config.id + ":tooltip:header-divider"});
        tooltipHelp.update({
            .id = this->config.id + ":tooltip:help",
            .str = ICON_FA_CIRCLE_INFO " Click on the end of the link to detach it.",
            .tone = Sakura::Text::Tone::Secondary,
        });
        tooltipHelpDivider.update({.id = this->config.id + ":tooltip:help-divider"});
        linkSection.update({
            .id = this->config.id + ":tooltip:link-section",
            .str = "Link",
            .font = Sakura::Text::Font::Bold,
        });
        layoutSection.update({
            .id = this->config.id + ":tooltip:layout-section",
            .str = "Layout",
            .font = Sakura::Text::Font::Bold,
        });
        attributesSection.update({
            .id = this->config.id + ":tooltip:attributes-section",
            .str = "Attributes",
            .font = Sakura::Text::Font::Bold,
        });
        linkMetadataTable.update({
            .id = this->config.id + ":tooltip:link-table",
            .columns = {"", ""},
            .rows = {
                {"Producer", jst::fmt::format("{}.{}",
                                               this->config.connection.producerName,
                                               this->config.connection.producerPort)},
                {"Consumer", jst::fmt::format("{}.{}",
                                               this->config.connection.consumerName,
                                               this->config.connection.inputSlot)},
                {"Tensor ID", jst::fmt::format("{}", tensor.id())},
            },
            .fixedColumnWidths = {72.0f},
            .showHeaders = false,
            .wrapped = true,
        });
        layoutMetadataTable.update({
            .id = this->config.id + ":tooltip:layout-table",
            .columns = {"", ""},
            .rows = {
                {"Device", jst::fmt::format("{}", tensor.device())},
                {"Type", jst::fmt::format("{}", tensor.dtype())},
                {"Shape", jst::fmt::format("{}", tensor.shape())},
                {"Strides", jst::fmt::format("{} ({})", tensor.stride(),
                                                        tensor.contiguous() ? "Contiguous" :
                                                                              "Non-contiguous")},
                {"Offset", jst::fmt::format("{} bytes", tensor.offsetBytes())},
            },
            .fixedColumnWidths = {72.0f},
            .showHeaders = false,
            .wrapped = true,
        });
        attributeMetadataTable.update({
            .id = this->config.id + ":tooltip:attribute-table",
            .columns = {"", ""},
            .rows = std::move(attributeRows),
            .fixedColumnWidths = {120.0f},
            .showHeaders = false,
            .wrapped = true,
        });
    }

    void render(const Sakura::Context& ctx) const {
        nodeLink.render(ctx, [this](const Sakura::Context& ctx) {
            tooltipTitle.render(ctx);
            tooltipHeaderDivider.render(ctx);
            tooltipHelp.render(ctx);
            tooltipHelpDivider.render(ctx);

            linkSection.render(ctx);
            linkMetadataTable.render(ctx);
            layoutSection.render(ctx);
            layoutMetadataTable.render(ctx);
            if (hasAttributes) {
                attributesSection.render(ctx);
                attributeMetadataTable.render(ctx);
            }
        });
    }

 private:
    Config config;
    Sakura::NodeLink nodeLink;
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
    bool hasAttributes = false;
    bool tooltipActive = false;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_LINK_HH
