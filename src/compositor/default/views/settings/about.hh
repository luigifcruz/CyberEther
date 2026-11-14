#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_ABOUT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_ABOUT_HH

#include "components/about_info_table.hh"
#include "components/about_update_card.hh"
#include "../components/compact_button.hh"
#include "../components/section_group.hh"
#include "components/settings_header.hh"
#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"

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

    void update(Config config);
    Sakura::Component& content();

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
