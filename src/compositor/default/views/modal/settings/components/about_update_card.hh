#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_COMPONENTS_ABOUT_UPDATE_CARD_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_COMPONENTS_ABOUT_UPDATE_CARD_HH

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/sakura/components/retained/markdown_view.hh"
#include "jetstream/render/tools/imgui_icons_ext.hh"

#include <functional>
#include <string>

namespace Jetstream {

struct AboutUpdateCard {
    struct Config {
        std::string version;
        bool supported = false;
        bool upToDate = false;
        bool failed = false;
        bool checking = false;
        bool updateAvailable = false;
        bool downloading = false;
        bool ready = false;
        bool applying = false;
        F32 progress = 0.0f;
        std::string updateVersion;
        std::string releaseNotes;
        std::string message;
        std::string accentKey = "accent_color";
        std::function<void()> onCheckForUpdates;
        std::function<void()> onDownloadUpdate;
        std::function<void()> onApplyUpdate;
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

        headerDivider.update({
            .id = "AboutUpdateHeaderDivider",
        });

        upToDateText.update({
            .id = "AboutUpToDate",
            .str = ICON_FA_CIRCLE_CHECK " Up to Date",
            .tone = Sakura::Text::Tone::Success,
            .align = Sakura::Text::Align::Right,
            .verticalOffset = 6.0f,
        });

        updateAvailableText.update({
            .id = "AboutUpdateAvailable",
            .str = ICON_FA_CIRCLE_EXCLAMATION " Update Available",
            .tone = Sakura::Text::Tone::Warning,
            .align = Sakura::Text::Align::Right,
            .verticalOffset = 6.0f,
        });

        updateVersionText.update({
            .id = "AboutUpdateVersion",
            .str = (this->config.ready || this->config.applying)
                ? "A new version of CyberEther (" + this->config.updateVersion + ") is ready to install."
                : "A new version of CyberEther (" + this->config.updateVersion + ") is available to download.",
        });

        changelog.update({
            .id = "AboutUpdateChangelog",
            .value = this->config.releaseNotes,
            .fontSize = 14.5f,
            .backgroundColorKey = "card",
        });

        statusText.update({
            .id = "AboutUpdateStatus",
            .str = this->config.message,
            .tone = Sakura::Text::Tone::Secondary,
            .wrapped = true,
        });

        downloadingText.update({
            .id = "AboutDownloading",
            .str = "Downloading update...",
            .tone = Sakura::Text::Tone::Secondary,
        });

        progressBar.update({
            .id = "AboutUpdateProgress",
            .value = this->config.progress,
            .overlay = jst::fmt::format("{:.0f}%", this->config.progress * 100.0f),
            .colorKey = "text_disabled",
            .backgroundColorKey = "border",
        });

        checkButton.update({
            .id = "AboutCheckUpdate",
            .str = this->config.checking ? ICON_FA_HOURGLASS " Checking for updates..."
                                         : ICON_FA_ROTATE " Check for Updates",
            .size = {-1.0f, 40.0f},
            .variant = Sakura::Button::Variant::Action,
            .disabled = !this->config.supported || this->config.checking,
            .onClick = [this]() {
                if (this->config.onCheckForUpdates) {
                    this->config.onCheckForUpdates();
                }
            },
        });

        downloadButton.update({
            .id = "AboutDownloadUpdate",
            .str = ICON_FA_DOWNLOAD " Download Update " + this->config.updateVersion,
            .size = {-1.0f, 40.0f},
            .variant = Sakura::Button::Variant::Action,
            .onClick = [this]() {
                if (this->config.onDownloadUpdate) {
                    this->config.onDownloadUpdate();
                }
            },
        });

        applyButton.update({
            .id = "AboutApplyUpdate",
            .str = ICON_FA_ROTATE " Restart to Update",
            .size = {-1.0f, 40.0f},
            .variant = Sakura::Button::Variant::Action,
            .disabled = this->config.applying,
            .colorKey = "warning_btn",
            .hoveredColorKey = "warning_btn_hovered",
            .activeColorKey = "warning_btn_active",
            .borderColorKey = "warning_btn_outline",
            .textColorKey = "warning_btn_text",
            .onClick = [this]() {
                if (this->config.onApplyUpdate) {
                    this->config.onApplyUpdate();
                }
            },
        });

        headerRow.update({
            .id = "AboutUpdateHeaderRow",
            .spacing = 8.0f,
        });

        spacing.update({
            .id = "AboutUpdateSpacing",
        });

        changelogDivider.update({
            .id = "AboutChangelogDivider",
        });
    }

    void render(const Sakura::Context& ctx) const {
        const bool hasUpdate = config.updateAvailable || config.downloading ||
                               config.ready || config.applying;
        const bool showUpToDate = config.supported && config.upToDate && !config.failed;
        const bool showStatus = !config.message.empty() &&
                                (config.failed ||
                                 (!config.checking && !hasUpdate && !config.upToDate));

        div.render(ctx, [&](const Sakura::Context& ctx) {
            Sakura::HStack::Children headerChildren;
            headerChildren.push_back([this](const Sakura::Context& ctx) {
                versionText.render(ctx);
            });
            if (showUpToDate) {
                headerChildren.push_back([this](const Sakura::Context& ctx) {
                    upToDateText.render(ctx);
                });
            } else if (config.updateAvailable) {
                headerChildren.push_back([this](const Sakura::Context& ctx) {
                    updateAvailableText.render(ctx);
                });
            }
            headerRow.render(ctx, std::move(headerChildren));
            headerDivider.render(ctx);

            if (hasUpdate && !config.releaseNotes.empty()) {
                changelog.render(ctx);
                changelogDivider.render(ctx);
            }

            if (showStatus) {
                statusText.render(ctx);
                spacing.render(ctx);
            }

            if (hasUpdate) {
                if (config.downloading) {
                    downloadingText.render(ctx);
                    spacing.render(ctx);
                    progressBar.render(ctx);
                } else if (config.ready || config.applying) {
                    applyButton.render(ctx);
                } else {
                    downloadButton.render(ctx);
                }
            } else {
                checkButton.render(ctx);
            }
        });
    }

 private:
    Config config;
    Sakura::Div div;
    Sakura::Text versionText;
    Sakura::Divider headerDivider;
    Sakura::HStack headerRow;
    Sakura::Text upToDateText;
    Sakura::Text updateAvailableText;
    Sakura::Text updateVersionText;
    Sakura::Retained::MarkdownView changelog;
    Sakura::Divider changelogDivider;
    Sakura::Text statusText;
    Sakura::Text downloadingText;
    Sakura::ProgressBar progressBar;
    Sakura::Button checkButton;
    Sakura::Button downloadButton;
    Sakura::Button applyButton;
    Sakura::Spacing spacing;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_COMPONENTS_ABOUT_UPDATE_CARD_HH
