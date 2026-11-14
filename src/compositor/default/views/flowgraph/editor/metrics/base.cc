#include "base.hh"

namespace Jetstream {

void FlowgraphMetricInstance::update(FlowgraphMetricConfig config) {
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

Sakura::Component& FlowgraphMetricInstance::component() {
    if (kind == "progressbar") {
        return progress.component();
    } else if (kind == "label") {
        return label.component();
    } else if (kind == "table") {
        return table.component();
    }
    return unknownFrame;
}

}  // namespace Jetstream
