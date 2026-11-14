#include "about.hh"

namespace Jetstream {

void AboutView::update(Config) {
    using Text = Sakura::Text;

    topSpacing.update({.lines = 2});
    middleSpacing.update({.lines = 2});
    title.update({
        .str = "CyberEther",
        .font = Text::Font::Display,
        .align = Text::Align::Center,
        .scale = 3.0f,
    });
    slogan.update({
        .str = "The final frontier!",
        .font = Text::Font::Sans,
        .tone = Text::Tone::Secondary,
        .align = Text::Align::Center,
        .scale = Sakura::Typography::Scale::Large,
    });
    details.update({
        .str = jst::fmt::format("MIT Licensed\nCopyright (c) 2021-2026 Luigi F. Cruz\nv{}-{}",
                                JETSTREAM_VERSION_STR,
                                JETSTREAM_BUILD_TYPE),
        .font = Text::Font::Sans,
        .align = Text::Align::Center,
    });
    stack.update({
        .items = {
            {.child = &topSpacing},
            {.child = &title},
            {.child = &slogan},
            {.child = &middleSpacing},
            {.child = &details},
        },
    });
    panel.update({
        .id = "AboutModal",
        .content = &stack,
        .backgroundColorKey = "popup_bg",
    });
}

Sakura::Component& AboutView::component() {
    return panel;
}

}  // namespace Jetstream
