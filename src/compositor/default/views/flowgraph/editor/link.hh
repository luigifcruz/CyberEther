#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_LINK_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_LINK_HH

#include "node.hh"

#include "jetstream/render/tools/icons_lucide.hh"

#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

std::string FlowgraphLinkId(const std::string& consumerName, const std::string& inputSlot);

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

    void update(Config config);
    Sakura::NodeEditor::Link takeLink();
    const Connection& connection() const;

 private:
    void prepareTooltip();

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
