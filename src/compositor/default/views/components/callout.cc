#include "callout.hh"

namespace Jetstream {

void Callout::update(Config config) {
    this->config = std::move(config);
    const std::string key = toneKey();
    icon.update({
        .str = this->config.icon.empty() ? toneIcon() : this->config.icon,
        .colorKey = jst::fmt::format("banner_{}_text", key),
    });
    text.update({
        .str = this->config.str,
        .font = this->config.font,
        .colorKey = jst::fmt::format("banner_{}_text", key),
        .wrapped = true,
    });
    row.update({
        .axis = Sakura::Stack::Axis::Horizontal,
        .spacing = Sakura::Spacing::Small,
        .items = {
            {.child = &icon},
            {.child = &text, .grow = 1.0f, .shrink = 1.0f},
        },
    });
    Div::update({
        .content = &row,
        .padding = Sakura::Spacing::Medium,
        .rounding = this->config.rounding,
        .border = true,
        .colorKey = jst::fmt::format("banner_{}_bg", key),
        .borderColorKey = jst::fmt::format("banner_{}_border", key),
    });
}

std::string Callout::toneKey() const {
    switch (config.tone) {
        case Tone::Success:
            return "success";
        case Tone::Warning:
            return "warning";
        case Tone::Error:
            return "error";
        case Tone::Info:
            break;
    }
    return "info";
}

std::string Callout::toneIcon() const {
    switch (config.tone) {
        case Tone::Success:
            return ICON_LC_CIRCLE_CHECK;
        case Tone::Warning:
            return ICON_LC_TRIANGLE_ALERT;
        case Tone::Error:
            return ICON_LC_CIRCLE_X;
        case Tone::Info:
            break;
    }
    return ICON_LC_INFO;
}

}  // namespace Jetstream
