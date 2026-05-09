#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_METRICS_BASE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_METRICS_BASE_HH

#include "label.hh"
#include "progressbar.hh"
#include "table.hh"
#include "types.hh"

#include "jetstream/parser.hh"

#include <string>
#include <utility>

namespace Jetstream {

struct FlowgraphMetricInstance : public Sakura::Component {
    void update(FlowgraphMetricConfig config) {
        const auto parts = Parser::SplitString(config.format, ":");
        kind = parts.empty() ? "" : parts[0];

        if (kind == "progressbar") {
            progress.update(std::move(config));
        } else if (kind == "label") {
            label.update(std::move(config));
        } else if (kind == "table") {
            table.update(std::move(config));
        } else {
            unknownFrame.update({
                .id = config.id,
                .label = config.label,
                .help = config.help,
            });
            unknownText.update({
                .id = config.id + "Unsupported",
                .str = "Unsupported metric: " + kind,
                .tone = Sakura::Text::Tone::Warning,
            });
        }
    }

    void render(const Sakura::Context& ctx) const {
        if (kind == "progressbar") {
            progress.render(ctx);
        } else if (kind == "label") {
            label.render(ctx);
        } else if (kind == "table") {
            table.render(ctx);
        } else {
            unknownFrame.render(ctx, [this](const Sakura::Context& ctx) {
                unknownText.render(ctx);
            });
        }
    }

 private:
    std::string kind;
    FlowgraphMetricProgressBar progress;
    FlowgraphMetricLabel label;
    FlowgraphMetricTable table;
    Sakura::NodeField unknownFrame;
    Sakura::NodeLabel unknownText;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_METRICS_BASE_HH
