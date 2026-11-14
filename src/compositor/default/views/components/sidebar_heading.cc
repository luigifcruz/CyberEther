#include "sidebar_heading.hh"

namespace Jetstream {

void SidebarHeading::update(Config config) {
    using Text = Sakura::Text;

    inset.update({});
    text.update({
        .str = std::move(config.label),
        .font = config.title ? Text::Font::DisplaySemiBold : Text::Font::Display,
        .tone = config.title ? Text::Tone::Primary : Text::Tone::Secondary,
        .scale = config.title ? 1.8f : Sakura::Typography::Scale::Title,
    });
    Stack::update({
        .axis = Axis::Horizontal,
        .spacing = 0.0f,
        .items = {
            {.child = &inset, .size = Inset},
            {.child = &text},
        },
    });
}

}  // namespace Jetstream
