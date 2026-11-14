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
#include "maps.hh"
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
        MapsSettingsView::Config maps;
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

    void update(Config config);
    Sakura::Component& component();
    Sakura::Dock::Pane pane();

 private:
    void navItem(SettingsSection section, std::string icon, std::string label);

    template<typename View, typename SectionConfig>
    static Sakura::Component& Show(View& view, const SectionConfig& sectionConfig) {
        view.update(sectionConfig);
        return view.content();
    }

    Sakura::Component& updateSection();

    Config config;
    std::array<Sakura::SidebarItem, 9> navigationButtons;
    SidebarNavigation navigation;
    GeneralSettingsView general;
    RemoteSettingsView remote;
    MapsSettingsView maps;
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
