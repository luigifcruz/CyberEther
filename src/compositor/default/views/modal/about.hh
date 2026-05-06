#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_ABOUT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_ABOUT_HH

#include "jetstream/render/sakura/sakura.hh"

#include "jetstream/config.hh"

#include <string>

namespace Jetstream {

struct AboutView : public Sakura::Component {
    struct Config {};

    void update(Config config) {
        this->config = std::move(config);
        spacing.update({
            .id = "AboutModalSpacing",
            .lines = 2,
        });
        title.update({
            .id = "AboutModalTitle",
            .str = "CyberEther",
            .font = Sakura::Text::Font::H1,
            .align = Sakura::Text::Align::Center,
            .scale = 3.0f,
        });
        slogan.update({
            .id = "AboutModalSlogan",
            .str = "The final frontier!",
            .tone = Sakura::Text::Tone::Secondary,
            .align = Sakura::Text::Align::Center,
            .scale = 1.15f,
        });
        details.update({
            .id = "AboutModalDetails",
            .str = jst::fmt::format("MIT Licensed\nCopyright (c) 2021-2026 Luigi F. Cruz\nv{}-{}",
                                    JETSTREAM_VERSION_STR,
                                    JETSTREAM_BUILD_TYPE),
            .align = Sakura::Text::Align::Center,
        });
    }

    void render(const Sakura::Context& ctx) const {
        spacing.render(ctx);
        title.render(ctx);
        slogan.render(ctx);
        spacing.render(ctx);
        details.render(ctx);
    }

 private:
    Config config;
    Sakura::Spacing spacing;
    Sakura::Text title;
    Sakura::Text slogan;
    Sakura::Text details;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_ABOUT_HH
