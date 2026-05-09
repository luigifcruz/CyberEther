#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_HUD_REMOTE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_HUD_REMOTE_HH

#include "jetstream/render/sakura/sakura.hh"
#include "jetstream/render/tools/imgui_icons_ext.hh"

#include <functional>
#include <string>
#include <utility>

namespace Jetstream {

struct RemoteHudView : public Sakura::Component {
    struct Config {
        bool visible = false;
        U64 clientCount = 0;
        std::function<void()> onOpen;
    };

    void update(Config config) {
        this->config = std::move(config);
        hud.update({
            .id = "Remote HUD",
            .anchor = Sakura::Hud::Anchor::TopRight,
            .padding = 12.0f,
            .clickable = true,
            .windowPadding = Extent2D<F32>{10.0f, 10.0f},
            .borderSize = 1.0f,
            .backgroundColorKey = "remote_hud_background",
            .borderColorKey = "remote_hud_border",
            .onClick = [this]() {
                if (this->config.onOpen) {
                    this->config.onOpen();
                }
            },
        });
        row.update({
            .id = "RemoteHudRow",
            .spacing = 4.0f,
        });
        icon.update({
            .id = "RemoteHudIcon",
            .str = ICON_FA_TOWER_BROADCAST,
            .tone = Sakura::Text::Tone::Success,
        });
        status.update({
            .id = "RemoteHudStatus",
            .str = jst::fmt::format("Remote Sharing ({})", this->config.clientCount),
        });
    }

    void render(const Sakura::Context& ctx) const {
        if (!config.visible) {
            return;
        }

        hud.render(ctx, [this](const Sakura::Context& ctx) {
            row.render(ctx, {
                [this](const Sakura::Context& ctx) { icon.render(ctx); },
                [this](const Sakura::Context& ctx) { status.render(ctx); },
            });
        });
    }

 private:
    Config config;
    Sakura::Hud hud;
    Sakura::HStack row;
    Sakura::Text icon;
    Sakura::Text status;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_HUD_REMOTE_HH
