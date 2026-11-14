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
            });
            return;
        case Tone::Success:
            return banner(std::move(config.label), "success");
        case Tone::Warning:
            return banner(std::move(config.label), "warning");
        case Tone::Error:
            return banner(std::move(config.label), "error");
    }
}

void StatusBadge::banner(std::string label, const char* key) {
    Badge::update({
        .label = std::move(label),
        .backgroundColorKey = jst::fmt::format("banner_{}_bg", key),
        .borderColorKey = jst::fmt::format("banner_{}_border", key),
        .textColorKey = jst::fmt::format("banner_{}_text", key),
    });
}

}  // namespace Jetstream
