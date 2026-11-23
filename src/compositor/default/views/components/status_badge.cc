#include "status_badge.hh"

namespace Jetstream {

void StatusBadge::update(Config config) {
    switch (config.tone) {
        case Tone::Neutral:
            Badge::update({
                .label = std::move(config.label),
                .backgroundColorKey = "pill_bg",
                .borderColorKey = "tile_border",
                .textColorKey = "text_secondary",
                .icon = std::move(config.icon),
                .onIcon = std::move(config.onIcon),
            });
            return;
        case Tone::Success:
            return banner(std::move(config), "success");
        case Tone::Warning:
            return banner(std::move(config), "warning");
        case Tone::Error:
            return banner(std::move(config), "error");
    }
}

void StatusBadge::banner(Config config, const char* key) {
    Badge::update({
        .label = std::move(config.label),
        .backgroundColorKey = jst::fmt::format("banner_{}_bg", key),
        .borderColorKey = jst::fmt::format("banner_{}_border", key),
        .textColorKey = jst::fmt::format("banner_{}_text", key),
        .icon = std::move(config.icon),
        .onIcon = std::move(config.onIcon),
    });
}

}  // namespace Jetstream
