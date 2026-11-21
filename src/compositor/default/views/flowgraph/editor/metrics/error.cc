#include "error.hh"

namespace Jetstream {

void FlowgraphMetricError::update(Config config) {
    text.update({
        .id = config.id,
        .str = std::move(config.str),
        .colorKey = "text_secondary",
        .align = Sakura::NodeLabel::Align::Center,
        .boxed = true,
    });
}

Sakura::Component& FlowgraphMetricError::component() {
    return text;
}

}  // namespace Jetstream
