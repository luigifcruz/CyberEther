#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_HUD_INFO_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_HUD_INFO_HH

#include "jetstream/render/sakura/sakura.hh"

#include <string>
#include <utility>

namespace Jetstream {

struct InfoHudView : public Sakura::Component {
    struct Config {
        bool visible = false;
        F32 frameRate = 0.0f;
        std::string viewportName;
        std::string renderInfo;
    };

    void update(Config config) {
        this->config = std::move(config);
        hud.update({
            .id = "Info HUD",
            .anchor = Sakura::Hud::Anchor::BottomLeft,
            .padding = 16.0f,
            .backgroundAlpha = 0.35f,
        });
        row.update({
            .id = "InfoHudRow",
            .spacing = 4.0f,
        });
        frameRateText.update({
            .id = "InfoHudFrameRate",
            .str = jst::fmt::format("{:.0f} Hz", this->config.frameRate),
            .tone = this->config.frameRate > 50.0f ? Sakura::Text::Tone::Success : Sakura::Text::Tone::Primary,
        });
        viewportText.update({
            .id = "InfoHudViewport",
            .str = this->config.viewportName,
        });
        renderText.update({
            .id = "InfoHudRenderInfo",
            .str = this->config.renderInfo,
        });
    }

    void render(const Sakura::Context& ctx) const {
        if (!config.visible) {
            return;
        }

        hud.render(ctx, [this](const Sakura::Context& ctx) {
            row.render(ctx, {
                [this](const Sakura::Context& ctx) { frameRateText.render(ctx); },
                [this](const Sakura::Context& ctx) { viewportText.render(ctx); },
            });
            renderText.render(ctx);
        });
    }

 private:
    Config config;
    Sakura::Hud hud;
    Sakura::HStack row;
    Sakura::Text frameRateText;
    Sakura::Text viewportText;
    Sakura::Text renderText;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_HUD_INFO_HH
