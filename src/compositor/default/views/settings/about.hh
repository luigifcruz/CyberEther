#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_ABOUT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_ABOUT_HH

#include "components/about_info_table.hh"
#include "components/about_update_card.hh"
#include "../components/compact_button.hh"
#include "../components/section_group.hh"
#include "components/settings_header.hh"
#include "jetstream/render/sakura/base.hh"

#include "jetstream/config.hh"

#include <memory>
#include <string>
#include <vector>

namespace Jetstream {

struct AboutSettingsView {
    struct Config {
        bool updateSupported = false;
        bool updateUpToDate = false;
        bool updateFailed = false;
        bool updateChecking = false;
        bool updateAvailable = false;
        bool updateDownloading = false;
        bool updateReady = false;
        bool updateApplying = false;
        F32 updateProgress = 0.0f;
        std::string updateVersion;
        std::string updateReleaseNotes;
        std::string updateMessage;
        std::vector<AboutInfoTable::Config> infoTables;
        Sakura::Callback<void()> onCheckForUpdates;
        Sakura::Callback<void()> onDownloadUpdate;
        Sakura::Callback<void()> onApplyUpdate;
        Sakura::Callback<void()> onSendFeedback;
    };

    void update(Config config) {
        this->config = std::move(config);
        header.update({
            .title = "About",
            .description = "Installation details and update management.",
        });
        updateCard.update({
            .version = jst::fmt::format("CyberEther v{}", JETSTREAM_VERSION_STR),
            .supported = this->config.updateSupported,
            .upToDate = this->config.updateUpToDate,
            .failed = this->config.updateFailed,
            .checking = this->config.updateChecking,
            .updateAvailable = this->config.updateAvailable,
            .downloading = this->config.updateDownloading,
            .ready = this->config.updateReady,
            .applying = this->config.updateApplying,
            .progress = this->config.updateProgress,
            .updateVersion = this->config.updateVersion,
            .releaseNotes = this->config.updateReleaseNotes,
            .message = this->config.updateMessage,
            .onCheckForUpdates = this->config.onCheckForUpdates,
            .onDownloadUpdate = this->config.onDownloadUpdate,
            .onApplyUpdate = this->config.onApplyUpdate,
        });
        feedbackButton.update({
            .str = "Send",
            .onClick = [this]() {
                if (this->config.onSendFeedback) {
                    this->config.onSendFeedback();
                }
            },
        });
        feedbackGroup.update({
            .title = "Feedback",
            .description = "Help us improve CyberEther.",
            .rows = {{
                .label = "Send Feedback",
                .description = "Report bugs or suggest improvements. Shortcut: Ctrl+G.",
                .control = &feedbackButton,
            }},
        });
        while (infoTables.size() < this->config.infoTables.size()) {
            infoTables.push_back(std::make_unique<AboutInfoTable>());
        }
        for (U64 i = 0; i < this->config.infoTables.size(); ++i) {
            infoTables[i]->update(this->config.infoTables[i]);
        }

        std::vector<Sakura::Stack::Item> items = {
            {.child = &header},
            {.child = &updateCard},
            {.child = &feedbackGroup},
        };
        for (U64 i = 0; i < this->config.infoTables.size(); ++i) {
            items.push_back({.child = infoTables[i].get()});
        }
        stack.update({
            .spacing = Sakura::Spacing::XSmall,
            .items = std::move(items),
        });
    }

    Sakura::Component& content() {
        return stack;
    }

 private:
    Config config;
    SettingsHeader header;
    AboutUpdateCard updateCard;
    CompactButton feedbackButton;
    SectionGroup feedbackGroup;
    std::vector<std::unique_ptr<AboutInfoTable>> infoTables;
    Sakura::Stack stack;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_ABOUT_HH
