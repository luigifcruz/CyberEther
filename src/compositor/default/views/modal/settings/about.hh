#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_ABOUT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_ABOUT_HH

#include "components/about_info_table.hh"
#include "components/about_update_card.hh"
#include "jetstream/render/sakura/base.hh"

#include "jetstream/config.hh"

#include <functional>
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
    };

    void update(Config config) {
        this->config = std::move(config);

        title.update({
            .id = "AboutTitle",
            .str = "About",
            .font = Sakura::Text::Font::Bold,
            .scale = 1.2f,
        });

        description.update({
            .id = "AboutDescription",
            .str = "Installation details and update management.",
            .tone = Sakura::Text::Tone::Secondary,
            .wrapped = true,
        });

        divider.update({
            .id = "AboutHeaderDivider",
        });

        spacing.update({
            .id = "AboutSpacing",
            .lines = 2,
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
            .accentKey = this->config.accentKey,
            .onCheckForUpdates = this->config.onCheckForUpdates,
            .onDownloadUpdate = this->config.onDownloadUpdate,
            .onApplyUpdate = this->config.onApplyUpdate,
        });

        infoTables.resize(this->config.infoTables.size());
        for (U64 i = 0; i < infoTables.size(); ++i) {
            infoTables[i].update(this->config.infoTables[i]);
        }
    }

    void render(const Sakura::Context& ctx) const {
        title.render(ctx);
        description.render(ctx);
        divider.render(ctx);
        updateCard.render(ctx);
        spacing.render(ctx);

        for (const auto& infoTable : infoTables) {
            infoTable.render(ctx);
        }
    }

 private:
    Config config;
    Sakura::Text title;
    Sakura::Text description;
    Sakura::Divider divider;
    Sakura::Spacing spacing;
    AboutUpdateCard updateCard;
    std::vector<AboutInfoTable> infoTables;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_ABOUT_HH
