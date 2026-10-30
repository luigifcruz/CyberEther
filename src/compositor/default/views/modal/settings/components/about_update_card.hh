#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_COMPONENTS_ABOUT_UPDATE_CARD_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_COMPONENTS_ABOUT_UPDATE_CARD_HH

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_fa.hh"

#include <functional>
#include <string>

namespace Jetstream {

struct AboutUpdateCard : public Sakura::Div {
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
        using Text = Sakura::Text;
        using Stack = Sakura::Stack;
        using Button = Sakura::Button;

        this->config = std::move(config);
        versionText.update({
            .str = this->config.version,
            .font = Text::Font::H1,
            .colorKey = this->config.accentKey,
            .scale = 1.5f,
        });
        headerDivider.update({});
        upToDateText.update({
            .str = ICON_FA_CIRCLE_CHECK " Up to Date",
            .tone = Text::Tone::Success,
            .align = Text::Align::Right,
            .verticalOffset = 6.0f,
        });
        updateAvailableText.update({
            .str = ICON_FA_CIRCLE_EXCLAMATION " Update Available",
            .tone = Text::Tone::Warning,
            .align = Text::Align::Right,
            .verticalOffset = 6.0f,
        });
        changelog.update({.value = this->config.releaseNotes});
        statusText.update({
            .str = this->config.message,
            .tone = Text::Tone::Secondary,
            .wrapped = true,
        });
        downloadingText.update({
            .str = "Downloading update...",
            .tone = Text::Tone::Secondary,
        });
        progressBar.update({
            .value = this->config.progress,
            .overlay = jst::fmt::format("{:.0f}%", this->config.progress * 100.0f),
            .colorKey = "text_disabled",
            .backgroundColorKey = "border",
        });
        checkButton.update({
            .str = this->config.checking ? ICON_FA_HOURGLASS " Checking for updates..."
                                         : ICON_FA_ROTATE " Check for Updates",
            .size = {Sakura::Fill, 40.0f},
            .variant = Button::Variant::Action,
            .disabled = !this->config.supported || this->config.checking,
            .onClick = [this]() {
                if (this->config.onCheckForUpdates) {
                    this->config.onCheckForUpdates();
                }
            },
        });
        downloadButton.update({
            .str = ICON_FA_DOWNLOAD " Download Update " + this->config.updateVersion,
            .size = {Sakura::Fill, 40.0f},
            .variant = Button::Variant::Action,
            .onClick = [this]() {
                if (this->config.onDownloadUpdate) {
                    this->config.onDownloadUpdate();
                }
            },
        });
        applyButton.update({
            .str = ICON_FA_ROTATE " Restart to Update",
            .size = {Sakura::Fill, 40.0f},
            .variant = Button::Variant::Action,
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
        statusSpacing.update({});
        downloadSpacing.update({});
        headerFill.update({});
        changelogDivider.update({});

        const bool hasUpdate = this->config.updateAvailable || this->config.downloading ||
                               this->config.ready || this->config.applying;
        const bool showUpToDate = this->config.supported && this->config.upToDate && !this->config.failed;
        const bool showStatus = !this->config.message.empty() &&
                                (this->config.failed ||
                                 (!this->config.checking && !hasUpdate && !this->config.upToDate));

        std::vector<Stack::Item> headerItems = {{.child = &versionText}, {.child = &headerFill, .grow = 1.0f}};
        if (showUpToDate) {
            headerItems.push_back({.child = &upToDateText});
        } else if (this->config.updateAvailable) {
            headerItems.push_back({.child = &updateAvailableText});
        }
        headerRow.update({
            .axis = Stack::Axis::Horizontal,
            .spacing = 16.0f,
            .centerCross = false,
            .items = std::move(headerItems),
        });

        std::vector<Stack::Item> items = {{.child = &headerRow}, {.child = &headerDivider}};
        if (hasUpdate && !this->config.releaseNotes.empty()) {
            items.push_back({.child = &changelog});
            items.push_back({.child = &changelogDivider});
        }
        if (showStatus) {
            items.push_back({.child = &statusText});
            items.push_back({.child = &statusSpacing});
        }
        if (hasUpdate) {
            if (this->config.downloading) {
                items.push_back({.child = &downloadingText});
                items.push_back({.child = &downloadSpacing});
                items.push_back({.child = &progressBar});
            } else if (this->config.ready || this->config.applying) {
                items.push_back({.child = &applyButton});
            } else {
                items.push_back({.child = &downloadButton});
            }
        } else {
            items.push_back({.child = &checkButton});
        }
        content.update({.items = std::move(items)});
        Div::update({
            .content = &content,
            .padding = 16.0f,
            .rounding = 8.0f,
            .border = true,
        });
    }

 private:
    Config config;
    Sakura::Text versionText;
    Sakura::Spacer headerFill;
    Sakura::Text upToDateText;
    Sakura::Text updateAvailableText;
    Sakura::Stack headerRow;
    Sakura::Divider headerDivider;
    Sakura::Markdown changelog;
    Sakura::Divider changelogDivider;
    Sakura::Text statusText;
    Sakura::Spacer statusSpacing;
    Sakura::Text downloadingText;
    Sakura::Spacer downloadSpacing;
    Sakura::ProgressBar progressBar;
    Sakura::Button checkButton;
    Sakura::Button downloadButton;
    Sakura::Button applyButton;
    Sakura::Stack content;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_COMPONENTS_ABOUT_UPDATE_CARD_HH
