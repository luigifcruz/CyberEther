#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_COMPONENTS_ABOUT_UPDATE_CARD_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_COMPONENTS_ABOUT_UPDATE_CARD_HH

#include "../../components/compact_button.hh"
#include "../../components/section_group.hh"
#include "../../components/status_badge.hh"

#include "jetstream/render/sakura/base.hh"

#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct AboutUpdateCard : public SectionGroup {
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

        versionBadge.update({.label = "Up to Date", .tone = StatusBadge::Tone::Success});
        channelCombo.update({
            .id = "settings:update-channel",
            .options = {"Stable"},
            .value = "Stable",
            .disabled = true,
            .borderColorKey = "border",
        });
        unsupportedBadge.update({.label = "Unsupported"});

        changelog.update({.value = this->config.releaseNotes});
        progressBar.update({
            .value = this->config.progress,
            .overlay = jst::fmt::format("{:.0f}%", this->config.progress * 100.0f),
            .colorKey = "text_disabled",
            .backgroundColorKey = "border",
        });

        checkButton.update({
            .str = this->config.checking ? "Checking..." : "Check for Updates",
            .tone = CompactButton::Tone::Strong,
            .disabled = !this->config.supported || this->config.checking,
            .onClick = [this]() {
                if (this->config.onCheckForUpdates) {
                    this->config.onCheckForUpdates();
                }
            },
        });
        if (this->config.ready || this->config.applying) {
            actionButton.update({
                .str = "Restart to Update",
                .tone = CompactButton::Tone::Warning,
                .disabled = this->config.applying,
                .onClick = [this]() {
                    if (this->config.onApplyUpdate) {
                        this->config.onApplyUpdate();
                    }
                },
            });
        } else {
            actionButton.update({
                .str = "Download",
                .tone = CompactButton::Tone::Action,
                .disabled = this->config.downloading,
                .onClick = [this]() {
                    if (this->config.onDownloadUpdate) {
                        this->config.onDownloadUpdate();
                    }
                },
            });
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

        SectionGroup::update({
            .title = "Updates",
            .description = "Keep CyberEther up to date with the latest release.",
            .warning = this->config.supported ? "" : this->config.message,
            .rows = std::move(rows),
            .toolbar = hasUpdate ? &hero : nullptr,
        });
    }

 private:
    Config config;
    StatusBadge versionBadge;
    StatusBadge unsupportedBadge;
    Sakura::Combo channelCombo;
    Sakura::Markdown changelog;
    Sakura::FeatureCard hero;
    Sakura::ProgressBar progressBar;
    CompactButton checkButton;
    CompactButton actionButton;
    Sakura::Stack versionControls;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_COMPONENTS_ABOUT_UPDATE_CARD_HH
