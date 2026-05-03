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
        bool checkingForUpdate = false;
        std::string updateVersion;
        std::string accentKey = "accent_color";
        std::function<void()> onCheckForUpdates;
        std::function<void()> onDownloadUpdate;
        std::function<void()> onDismissUpdate;
    };

    void update(Config config) {
        this->config = std::move(config);

        div.update({
            .id = "AboutUpdateDiv",
            .padding = 16.0f,
            .rounding = 8.0f,
            .border = false,
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

        checkingText.update({
            .id = "AboutCheckingForUpdate",
            .str = "Checking...",
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

        checkButton.update({
            .id = "AboutCheckForUpdates",
            .str = ICON_FA_ROTATE " Check for Updates",
            .size = {-1.0f, 40.0f},
            .onClick = [this]() {
                if (this->config.onCheckForUpdates) {
                    this->config.onCheckForUpdates();
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
            spacing.render(ctx);
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
                checkButton.render(ctx);
                if (config.checkingForUpdate) {
                    spacing.render(ctx);
                    checkingText.render(ctx);
                }
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
    Sakura::Text checkingText;
    Sakura::Button downloadButton;
    Sakura::Button dismissButton;
    Sakura::Button checkButton;
    Sakura::Spacing spacing;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_COMPONENTS_ABOUT_UPDATE_CARD_HH
