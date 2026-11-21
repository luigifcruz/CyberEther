#include "about.hh"

namespace Jetstream {

void AboutSettingsView::update(Config config) {
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
        .str = "Send Feedback",
        .icon = ICON_LC_SEND,
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
            .description = "Report bugs or suggest improvements.",
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

Sakura::Component& AboutSettingsView::content() {
    return stack;
}

}  // namespace Jetstream
