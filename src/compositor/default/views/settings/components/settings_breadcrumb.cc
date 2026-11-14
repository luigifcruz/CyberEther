#include "settings_breadcrumb.hh"

namespace Jetstream {

void SettingsBreadcrumb::update(Config config) {
    using Text = Sakura::Text;

    rootButton.update({
        .str = std::move(config.root),
        .variant = Sakura::Button::Variant::Text,
        .textColorKey = "text_secondary",
        .onClick = std::move(config.onRoot),
        .font = Text::Font::Sans,
    });
    chevron.update({
        .str = ICON_LC_CHEVRON_RIGHT,
        .tone = Text::Tone::Secondary,
        .scale = 0.7f,
        .verticalOffset = 1.0f,
    });
    current.update({
        .str = std::move(config.current),
        .font = Text::Font::Sans,
    });
    trail.update({
        .axis = Axis::Horizontal,
        .spacing = Sakura::Spacing::Small,
        .items = {
            {.child = &rootButton},
            {.child = &chevron},
            {.child = &current},
        },
    });
    spacing.update({.lines = 3});
    Stack::update({
        .spacing = Sakura::Spacing::XSmall,
        .items = {
            {.child = &trail},
            {.child = &spacing},
        },
    });
}

}  // namespace Jetstream
