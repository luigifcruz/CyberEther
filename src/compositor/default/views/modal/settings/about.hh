#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_ABOUT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_ABOUT_HH

#include "components/about_info_table.hh"
#include "components/about_update_card.hh"
#include "jetstream/render/sakura/base.hh"

#include "jetstream/config.hh"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace Jetstream {

struct AboutSettingsPanel {
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
        std::string accentKey = "accent_color";
        std::vector<AboutInfoTable::Config> infoTables;
        std::function<void()> onCheckForUpdates;
        std::function<void()> onDownloadUpdate;
        std::function<void()> onApplyUpdate;
        std::function<void()> onSendFeedback;
    };

    void update(Config config) {
        using Text = Sakura::Text;

        this->config = std::move(config);
        title.update({
            .str = "About",
            .font = Text::Font::Bold,
            .scale = 1.2f,
        });
        description.update({
            .str = "Installation details and update management.",
            .tone = Text::Tone::Secondary,
            .wrapped = true,
        });
        divider.update({});
        spacing.update({.lines = 2});
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
            .accentKey = this->config.accentKey,
            .onCheckForUpdates = this->config.onCheckForUpdates,
            .onDownloadUpdate = this->config.onDownloadUpdate,
            .onApplyUpdate = this->config.onApplyUpdate,
        });
        while (infoTables.size() < this->config.infoTables.size()) {
            infoTables.push_back(std::make_unique<AboutInfoTable>());
        }
        for (U64 i = 0; i < this->config.infoTables.size(); ++i) {
            infoTables[i]->update(this->config.infoTables[i]);
        }
        feedbackDividerTop.update({});
        feedbackDivider.update({});
        feedbackButton.update({
            .str = ICON_FA_COMMENT_DOTS " Send Feedback (CTRL+G)",
            .size = {Sakura::Fill, 38.0f},
            .onClick = [this]() {
                if (this->config.onSendFeedback) {
                    this->config.onSendFeedback();
                }
            },
        });
        feedbackField.update({
            .label = "Feedback",
            .description = "Help us improve. Report bugs or suggest improvements.",
            .control = &feedbackButton,
            .divider = false,
        });

        std::vector<Sakura::Stack::Item> items = {
            {.child = &title},
            {.child = &description},
            {.child = &divider},
            {.child = &updateCard},
            {.child = &spacing},
            {.child = &feedbackDividerTop},
            {.child = &feedbackField},
            {.child = &feedbackDivider},
        };
        for (U64 i = 0; i < this->config.infoTables.size(); ++i) {
            items.push_back({.child = infoTables[i].get()});
        }
        stack.update({.items = std::move(items)});
    }

    Sakura::Component& content() {
        return stack;
    }

 private:
    Config config;
    Sakura::Text title;
    Sakura::Text description;
    Sakura::Divider divider;
    Sakura::Spacer spacing;
    AboutUpdateCard updateCard;
    std::vector<std::unique_ptr<AboutInfoTable>> infoTables;
    Sakura::Divider feedbackDividerTop;
    Sakura::Divider feedbackDivider;
    Sakura::Button feedbackButton;
    Sakura::SettingField feedbackField;
    Sakura::Stack stack;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_ABOUT_HH
