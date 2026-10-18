#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_HUD_INFO_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_HUD_INFO_HH

#include "jetstream/render/sakura/base.hh"

#include <string>
#include <utility>

namespace Jetstream {

struct InfoHudView {
    struct Config {
        F32 frameRate = 0.0f;
        std::string viewportName;
        std::string renderInfo;
    };

    void update(Config config) {
        this->config = std::move(config);
        using Text = Sakura::Text;
        using Stack = Sakura::Stack;

        frameRateText.update({
            .str = jst::fmt::format("{:.0f} Hz", this->config.frameRate),
            .tone = this->config.frameRate > 50.0f ? Text::Tone::Success : Text::Tone::Primary,
        });
        viewportText.update({.str = this->config.viewportName});
        renderText.update({.str = this->config.renderInfo});
        row.update({
            .axis = Stack::Axis::Horizontal,
            .spacing = 12.0f,
            .items = {{.child = &frameRateText}, {.child = &viewportText}},
        });
        stack.update({.items = {{.child = &row}, {.child = &renderText}}});
        panel.update({
            .id = "InfoHud",
            .content = &stack,
            .size = {0.0f, 0.0f},
            .transparent = true,
            .backgroundColorKey = "info_hud_background",
        });
    }

    Sakura::HudLayer::Hud hud() {
        return {
            .content = &panel,
            .anchor = Sakura::Anchor::BottomLeft,
            .margin = 16.0f,
            .backgroundColorKey = "info_hud_background",
        };
    }

 private:
    Config config;
    Sakura::Text frameRateText;
    Sakura::Text viewportText;
    Sakura::Text renderText;
    Sakura::Stack row;
    Sakura::Stack stack;
    Sakura::Panel panel;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_HUD_INFO_HH
