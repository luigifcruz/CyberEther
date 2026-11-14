#include "tooltip.hh"

namespace Jetstream {

bool Tooltip::requested() {
    const bool asked = pending;
    pending = false;
    ready = ready || asked;
    return asked;
}

void Tooltip::update(Config config) {
    panel.update({
        .id = config.id + ":panel",
        .content = config.content,
        .size = {0.0f, 0.0f},
        .maxWidth = config.wrapWidth - 2.0f * WindowPadding,
        .transparent = true,
    });
}

bool Tooltip::prepared() const {
    return ready;
}

Sakura::Component* Tooltip::show() {
    pending = true;
    return ready ? &panel : nullptr;
}

}  // namespace Jetstream
