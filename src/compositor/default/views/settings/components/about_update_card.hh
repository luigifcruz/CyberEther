#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_COMPONENTS_ABOUT_UPDATE_CARD_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_COMPONENTS_ABOUT_UPDATE_CARD_HH

#include "settings_group.hh"

#include "jetstream/render/sakura/base.hh"

#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct AboutUpdateCard : public SettingsGroup {
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
        const bool hasUpdate = this->config.updateAvailable || this->config.downloading ||
                               this->config.ready || this->config.applying;
        const bool showUpToDate = this->config.supported && this->config.upToDate && !this->config.failed;
        const bool showStatus = this->config.supported && !this->config.message.empty() &&
                                (this->config.failed ||
                                 (!this->config.checking && !hasUpdate && !this->config.upToDate));

        versionBadge.update(tone("Up to Date", "banner_success"));
        channelCombo.update({
            .id = "settings:update-channel",
            .options = {"Stable"},
            .value = "Stable",
            .disabled = true,
            .borderColorKey = "border",
        });
        unsupportedBadge.update({
            .label = "Unsupported",
            .backgroundColorKey = "header_hovered",
            .borderColorKey = "border",
            .textColorKey = "text_secondary",
        });

        changelog.update({.value = this->config.releaseNotes});
        progressBar.update({
            .value = this->config.progress,
            .overlay = jst::fmt::format("{:.0f}%", this->config.progress * 100.0f),
            .colorKey = "text_disabled",
            .backgroundColorKey = "border",
        });

        checkButton.update(compact(this->config.checking ? "Checking..." : "Check for Updates",
                                   !this->config.supported || this->config.checking,
                                   Sakura::Button::Variant::Default,
                                   [this]() {
                                       if (this->config.onCheckForUpdates) {
                                           this->config.onCheckForUpdates();
                                       }
                                   }));
        if (this->config.ready || this->config.applying) {
            auto restart = compact("Restart to Update", this->config.applying, Sakura::Button::Variant::Action,
                                   [this]() {
                                       if (this->config.onApplyUpdate) {
                                           this->config.onApplyUpdate();
                                       }
                                   });
            restart.colorKey = "warning_btn";
            restart.hoveredColorKey = "warning_btn_hovered";
            restart.activeColorKey = "warning_btn_active";
            restart.borderColorKey = "warning_btn_outline";
            restart.textColorKey = "warning_btn_text";
            actionButton.update(std::move(restart));
        } else {
            actionButton.update(compact("Download", this->config.downloading, Sakura::Button::Variant::Action,
                                        [this]() {
                                            if (this->config.onDownloadUpdate) {
                                                this->config.onDownloadUpdate();
                                            }
                                        }));
        }

        std::vector<Sakura::Stack::Item> versionItems;
        if (showUpToDate) {
            versionItems.push_back({.child = &versionBadge});
        }
        versionItems.push_back({.child = &checkButton});
        versionControls.update({
            .axis = Sakura::Stack::Axis::Horizontal,
            .spacing = 6.0f,
            .items = std::move(versionItems),
        });

        std::vector<Sakura::SettingCard::Row> rows;
        if (!hasUpdate) {
            rows.push_back({
                .label = "Installed Version",
                .description = this->config.version,
                .control = &versionControls,
                .wide = true,
            });
        }
        rows.push_back({
            .label = "Update Channel",
            .description = "Release feed this installation follows.",
            .control = &channelCombo,
        });
        if (!this->config.supported) {
            rows.push_back({
                .label = "Automatic Updates",
                .description = "This installation is updated through its package manager or build.",
                .control = &unsupportedBadge,
            });
        }
        if (showStatus) {
            rows.push_back({.label = "Status", .description = this->config.message});
        }
        if (this->config.downloading) {
            rows.push_back({
                .label = "Downloading",
                .description = this->config.updateVersion,
                .control = &progressBar,
            });
        }

        hero.update({
            .title = jst::fmt::format("CyberEther {} is available!", this->config.updateVersion),
            .description = this->config.ready || this->config.applying
                               ? jst::fmt::format("Downloaded and ready to install. You are running {}.",
                                                  this->config.version)
                               : jst::fmt::format("You are running {}.", this->config.version),
            .action = hasUpdate && !this->config.downloading ? &actionButton : nullptr,
            .content = this->config.releaseNotes.empty() ? nullptr : &changelog,
        });

        SettingsGroup::update({
            .title = "Updates",
            .description = "Keep CyberEther up to date with the latest release.",
            .warning = this->config.supported ? "" : this->config.message,
            .rows = std::move(rows),
            .toolbar = hasUpdate ? &hero : nullptr,
        });
    }

 private:
    static Sakura::Badge::Config tone(std::string label, const std::string& key) {
        return {
            .label = std::move(label),
            .backgroundColorKey = key + "_bg",
            .borderColorKey = key + "_border",
            .textColorKey = key + "_text",
        };
    }

    static Sakura::Button::Config compact(std::string label,
                                          bool disabled,
                                          Sakura::Button::Variant variant,
                                          std::function<void()> onClick) {
        return {
            .str = std::move(label),
            .variant = variant,
            .disabled = disabled,
            .padding = {14.0f, 5.0f},
            .onClick = std::move(onClick),
            .font = Sakura::Text::Font::SansBold,
        };
    }

    Config config;
    Sakura::Badge versionBadge;
    Sakura::Badge unsupportedBadge;
    Sakura::Combo channelCombo;
    Sakura::Markdown changelog;
    Sakura::FeatureCard hero;
    Sakura::ProgressBar progressBar;
    Sakura::Button checkButton;
    Sakura::Button actionButton;
    Sakura::Stack versionControls;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_COMPONENTS_ABOUT_UPDATE_CARD_HH
