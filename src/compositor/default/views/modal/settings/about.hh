#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_ABOUT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_ABOUT_HH

#include "components/about_info_table.hh"
#include "components/about_update_card.hh"
#include "jetstream/render/sakura/sakura.hh"

#include "jetstream/config.hh"

#include <functional>
#include <string>
#include <vector>

namespace Jetstream {

struct AboutSettingsPanel : public Sakura::Component {
    struct Config {
        bool updateAvailable = false;
        bool checkingForUpdate = false;
        std::string updateVersion;
        std::string accentKey = "accent_color";
        std::vector<AboutInfoTable::Config> infoTables;
        std::function<void()> onCheckForUpdates;
        std::function<void()> onDownloadUpdate;
        std::function<void()> onDismissUpdate;
    };

    void update(Config config) {
        this->config = std::move(config);

        title.update({
            .id = "AboutTitle",
            .str = "About",
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
            .buildInfo = jst::fmt::format("Built on {} at {}", __DATE__, __TIME__),
            .updateAvailable = this->config.updateAvailable,
            .checkingForUpdate = this->config.checkingForUpdate,
            .updateVersion = this->config.updateVersion,
            .accentKey = this->config.accentKey,
            .onCheckForUpdates = this->config.onCheckForUpdates,
            .onDownloadUpdate = this->config.onDownloadUpdate,
            .onDismissUpdate = this->config.onDismissUpdate,
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
