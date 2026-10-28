#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_BASE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_BASE_HH

#include "../../model/ui.hh"
#include "../components/sidebar_navigation.hh"
#include "../components/sidebar_tiles.hh"
#include "about.hh"
#include "developer.hh"
#include "general.hh"
#include "intelligence.hh"
#include "legal.hh"
#include "registry.hh"
#include "remote.hh"
#include "runtime.hh"

#include "jetstream/render/tools/icons_lucide.hh"

#include <array>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct SettingsView {
    static constexpr const char* PaneId = "settings";
    static constexpr F32 PageWidth = 720.0f;

    struct Config {
        SettingsSection section = SettingsSection::General;
        GeneralSettingsView::Config general;
        RemoteSettingsView::Config remote;
        RuntimeSettingsView::Config runtime;
        RegistrySettingsView::Config registry;
        IntelligenceSettingsView::Config intelligence;
        DeveloperSettingsView::Config developer;
        AboutSettingsView::Config about;
        LegalSettingsView::Config legal;
        Sakura::Callback<void(SettingsSection)> onSectionChange;
        Sakura::Callback<void()> onClose;
        Sakura::Callback<void()> onFocus;
    };

    void update(Config config) {
        this->config = std::move(config);
        navItem(SettingsSection::General, ICON_LC_MONITOR, "General");
        navItem(SettingsSection::Remote, ICON_LC_RADIO_TOWER, "Remote");
        navItem(SettingsSection::Runtime, ICON_LC_CPU, "Runtime");
        navItem(SettingsSection::Registry, ICON_LC_DATABASE, "Registry");
        navItem(SettingsSection::Intelligence, ICON_LC_WAND_SPARKLES, "Intelligence");
        navItem(SettingsSection::Developer, ICON_LC_FLASK_CONICAL, "Developer");
        navItem(SettingsSection::About, ICON_LC_INFO, "About");
        navItem(SettingsSection::Legal, ICON_LC_SCALE, "Legal");
        Sakura::Component& content = updateSection();

        const auto button = [this](SettingsSection section) -> Sakura::Component* {
            return &navigationButtons[static_cast<U64>(section)];
        };
        std::vector<Sakura::Component*> preferences;
        for (const auto section : {SettingsSection::General,
                                   SettingsSection::Remote,
                                   SettingsSection::Runtime,
                                   SettingsSection::Registry,
                                   SettingsSection::Intelligence}) {
#ifdef JST_OS_BROWSER
            if (section == SettingsSection::Intelligence) {
                continue;
            }
#endif
            preferences.push_back(button(section));
        }
        navigation.update({
            .title = "Settings",
            .sections = {
                {.heading = "Preferences", .items = std::move(preferences)},
                {.heading = "Advanced", .items = {button(SettingsSection::Developer)}},
                {.heading = "Application", .items = {button(SettingsSection::About), button(SettingsSection::Legal)}},
            },
        });
        page.update({
            .id = "settings-page",
            .content = &content,
            .maxWidth = PageWidth,
            .transparent = true,
        });
        pageFrame.update({
            .content = &page,
            .padding = Sakura::Spacing::XXLarge,
            .border = false,
            .colorKey = "transparent",
        });
        pageArea.update({
            .content = &pageFrame,
            .alwaysScrollbar = false,
            .border = false,
            .overlay = true,
            .scrollbarInset = {0.0f, Sakura::Dock::TileRounding},
        });
        tile.update({
            .sidebar = &navigation,
            .page = &pageArea,
        });
    }

    Sakura::Component& component() {
        return tile;
    }

    Sakura::Dock::Pane pane() {
        return {
            .id = PaneId,
            .title = "Settings",
            .group = "main",
            .icon = ICON_LC_SETTINGS,
            .content = &tile,
            .onClose = config.onClose,
            .onFocus = config.onFocus,
        };
    }

 private:
    void navItem(SettingsSection section, std::string icon, std::string label) {
        navigationButtons[static_cast<U64>(section)].update({
            .icon = std::move(icon),
            .label = std::move(label),
            .selected = config.section == section,
            .onSelect = [this, section]() {
                if (config.onSectionChange) {
                    config.onSectionChange(section);
                }
            },
        });
    }

    template<typename View, typename SectionConfig>
    static Sakura::Component& Show(View& view, const SectionConfig& sectionConfig) {
        view.update(sectionConfig);
        return view.content();
    }

    Sakura::Component& updateSection() {
        switch (config.section) {
            case SettingsSection::General:
                return Show(general, config.general);
            case SettingsSection::Remote:
                return Show(remote, config.remote);
            case SettingsSection::Runtime:
                return Show(runtime, config.runtime);
            case SettingsSection::Registry:
                return Show(registry, config.registry);
            case SettingsSection::Intelligence:
                return Show(intelligence, config.intelligence);
            case SettingsSection::Developer:
                return Show(developer, config.developer);
            case SettingsSection::About:
                return Show(about, config.about);
            case SettingsSection::Legal:
                return Show(legal, config.legal);
        }
        return Show(general, config.general);
    }

    Config config;
    std::array<Sakura::SidebarItem, 8> navigationButtons;
    SidebarNavigation navigation;
    GeneralSettingsView general;
    RemoteSettingsView remote;
    RuntimeSettingsView runtime;
    RegistrySettingsView registry;
    IntelligenceSettingsView intelligence;
    DeveloperSettingsView developer;
    AboutSettingsView about;
    LegalSettingsView legal;
    Sakura::Panel page;
    Sakura::Div pageFrame;
    Sakura::ScrollArea pageArea;
    SidebarTiles tile;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_BASE_HH
