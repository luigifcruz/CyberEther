#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_COMPONENTS_ABOUT_UPDATE_CARD_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_COMPONENTS_ABOUT_UPDATE_CARD_HH

#include "jetstream/render/sakura/sakura.hh"
#include "jetstream/render/tools/imgui_icons_ext.hh"

#include <functional>
#include <string>

namespace Jetstream {

struct AboutUpdateCard : public Sakura::Component {
    struct Config {
        std::string version;
        std::string buildInfo;
        bool updateAvailable = false;
        std::string updateVersion;
        std::string accentKey = "accent_color";
        std::function<void()> onOpenReleases;
        std::function<void()> onDownloadUpdate;
        std::function<void()> onDismissUpdate;
    };

    void update(Config config) {
        this->config = std::move(config);

        div.update({
            .id = "AboutUpdateDiv",
            .padding = 16.0f,
            .rounding = 8.0f,
            .border = true,
            .scrollbar = false,
            .mouseScroll = false,
        });

        versionText.update({
            .id = "AboutVersion",
            .str = this->config.version,
            .font = Sakura::Text::Font::H1,
            .colorKey = this->config.accentKey,
            .scale = 1.5f,
        });

        buildText.update({
            .id = "AboutBuildInfo",
            .str = this->config.buildInfo,
            .tone = Sakura::Text::Tone::Disabled,
        });

        updateAvailableText.update({
            .id = "AboutUpdateAvailable",
            .str = ICON_FA_CIRCLE_CHECK " Update Available",
            .tone = Sakura::Text::Tone::Success,
        });

        updateVersionText.update({
            .id = "AboutUpdateVersion",
            .str = "Version " + this->config.updateVersion + " is ready to download.",
        });

        downloadButton.update({
            .id = "AboutDownloadUpdate",
            .str = ICON_FA_DOWNLOAD " Download Update",
            .size = {-1.0f, 40.0f},
            .variant = Sakura::Button::Variant::Action,
            .onClick = [this]() {
                if (this->config.onDownloadUpdate) {
                    this->config.onDownloadUpdate();
                }
            },
        });

        dismissButton.update({
            .id = "AboutDismissUpdate",
            .str = "Dismiss",
            .size = {-1.0f, 0.0f},
            .onClick = [this]() {
                if (this->config.onDismissUpdate) {
                    this->config.onDismissUpdate();
                }
            },
        });

        releasesButton.update({
            .id = "AboutOpenReleases",
            .str = ICON_FA_ARROW_UP_RIGHT_FROM_SQUARE " Open Releases",
            .size = {-1.0f, 40.0f},
            .onClick = [this]() {
                if (this->config.onOpenReleases) {
                    this->config.onOpenReleases();
                }
            },
        });

        spacing.update({
            .id = "AboutUpdateSpacing",
        });
    }

    void render(const Sakura::Context& ctx) const {
        div.render(ctx, [&](const Sakura::Context& ctx) {
            versionText.render(ctx);
            buildText.render(ctx);
            spacing.render(ctx);

            if (config.updateAvailable) {
                updateAvailableText.render(ctx);
                spacing.render(ctx);
                updateVersionText.render(ctx);
                spacing.render(ctx);
                downloadButton.render(ctx);
                spacing.render(ctx);
                dismissButton.render(ctx);
            } else {
                releasesButton.render(ctx);
            }
        });
    }

 private:
    Config config;
    Sakura::Div div;
    Sakura::Text versionText;
    Sakura::Text buildText;
    Sakura::Text updateAvailableText;
    Sakura::Text updateVersionText;
    Sakura::Button downloadButton;
    Sakura::Button dismissButton;
    Sakura::Button releasesButton;
    Sakura::Spacing spacing;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_COMPONENTS_ABOUT_UPDATE_CARD_HH
