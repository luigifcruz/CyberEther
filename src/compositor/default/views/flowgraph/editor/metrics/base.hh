#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_METRICS_BASE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_METRICS_BASE_HH

#include "error.hh"
#include "label.hh"
#include "progressbar.hh"
#include "table.hh"
#include "types.hh"

#include "../field_tooltip.hh"

#include "jetstream/parser.hh"

#include <string>
#include <utility>

namespace Jetstream {

struct FlowgraphMetricInstance {
    void update(FlowgraphMetricConfig config) {
        kind = Parser::Get<std::string>(config.format, "type");

        if (kind == "progressbar") {
            progress.update(std::move(config));
        } else if (kind == "label") {
            label.update(std::move(config));
        } else if (kind == "table") {
            table.update(std::move(config));
        } else {
            unknownHelp.update(config.id, config.help);
            unknownFrame.update({
                .id = config.id,
                .label = config.label,
                .tooltip = unknownHelp.overlay(),
                .content = &unknownText.component(),
            });
            unknownText.update({
                .id = config.id + "Unsupported",
                .str = "Unsupported metric: " + kind,
            });
        }
    }

    Sakura::Component& component() {
        if (kind == "progressbar") {
            return progress.component();
        } else if (kind == "label") {
            return label.component();
        } else if (kind == "table") {
            return table.component();
        }
        return unknownFrame;
    }

 private:
    std::string kind;
    FlowgraphMetricProgressBar progress;
    FlowgraphMetricLabel label;
    FlowgraphMetricTable table;
    FlowgraphFieldTooltip unknownHelp;
    Sakura::NodeField unknownFrame;
    FlowgraphMetricError unknownText;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_METRICS_BASE_HH
