#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_ABOUT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_ABOUT_HH

#include "jetstream/render/sakura/base.hh"

#include "jetstream/config.hh"

#include <string>

namespace Jetstream {

struct AboutView {
    struct Config {};

    void update(Config config) {
        using Text = Sakura::Text;

        this->config = std::move(config);
        topSpacing.update({.lines = 2});
        middleSpacing.update({.lines = 2});
        title.update({
            .str = "CyberEther",
            .font = Text::Font::H1,
            .align = Text::Align::Center,
            .scale = 3.0f,
        });
        slogan.update({
            .str = "The final frontier!",
            .tone = Text::Tone::Secondary,
            .align = Text::Align::Center,
            .scale = 1.15f,
        });
        details.update({
            .str = jst::fmt::format("MIT Licensed\nCopyright (c) 2021-2026 Luigi F. Cruz\nv{}-{}",
                                    JETSTREAM_VERSION_STR,
                                    JETSTREAM_BUILD_TYPE),
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

    Sakura::Component& component() {
        return panel;
    }

 private:
    Config config;
    Sakura::Spacer topSpacing;
    Sakura::Spacer middleSpacing;
    Sakura::Text title;
    Sakura::Text slogan;
    Sakura::Text details;
    Sakura::Stack stack;
    Sakura::Panel panel;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_ABOUT_HH
