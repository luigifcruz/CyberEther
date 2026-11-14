#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_COMPONENTS_ABOUT_UPDATE_CARD_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_COMPONENTS_ABOUT_UPDATE_CARD_HH

#include "../../components/compact_button.hh"
#include "../../components/section_group.hh"
#include "../../components/status_badge.hh"

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"

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
        Sakura::Callback<void()> onCheckForUpdates;
        Sakura::Callback<void()> onDownloadUpdate;
        Sakura::Callback<void()> onApplyUpdate;
    };

    void update(Config config);

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
