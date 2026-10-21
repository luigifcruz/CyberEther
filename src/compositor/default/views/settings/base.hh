#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_BASE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_BASE_HH

#include "../../model/ui.hh"
#include "about.hh"
#include "developer.hh"
#include "general.hh"
#include "intelligence.hh"
#include "legal.hh"
#include "registry.hh"
#include "remote.hh"
#include "runtime.hh"

#include "jetstream/render/tools/icons_fa.hh"

#include <array>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct SettingsSidebarHeading : public Sakura::Stack {
    static constexpr F32 Inset = 9.0f;

    void update(std::string label, bool title = false) {
        using Text = Sakura::Text;

        inset.update({});
        text.update({
            .str = std::move(label),
            .font = title ? Text::Font::DisplaySemiBold : Text::Font::Display,
            .tone = title ? Text::Tone::Primary : Text::Tone::Secondary,
            .scale = title ? 1.8f : 18.0f / 15.0f,
        });
        Stack::update({
            .axis = Axis::Horizontal,
            .spacing = 0.0f,
            .items = {
                {.child = &inset, .size = Inset},
                {.child = &text},
            },
        });
    }

 private:
    Sakura::Spacer inset;
    Sakura::Text text;
};

struct SettingsView {
    static constexpr const char* PaneId = "settings";
    static constexpr const char* TileId = "settings:page";
    static constexpr F32 SidebarWidth = 240.0f;
    static constexpr F32 PageWidth = 720.0f;

    struct Config {
        SettingsSection section = SettingsSection::General;
        GeneralSettingsPanel::Config general;
        RemoteSettingsPanel::Config remote;
        RuntimeSettingsPanel::Config runtime;
        RegistrySettingsPanel::Config registry;
        IntelligenceSettingsPanel::Config intelligence;
        DeveloperSettingsPanel::Config developer;
        AboutSettingsPanel::Config about;
        LegalSettingsPanel::Config legal;
        std::function<void(SettingsSection)> onSectionChange;
        std::function<void()> onClose;
        std::function<void()> onFocus;
    };

    void update(Config config) {
        this->config = std::move(config);
        title.update("Settings", true);
        titleSpacing.update({.lines = 3});
        preferencesHeading.update("Preferences");
        advancedSpacing.update({.lines = 3});
        advancedHeading.update("Advanced");
        applicationSpacing.update({.lines = 3});
        applicationHeading.update("Application");
        headingSpacing.update({});
        navItem(SettingsSection::General, ICON_FA_DESKTOP, "General");
        navItem(SettingsSection::Remote, ICON_FA_TOWER_BROADCAST, "Remote");
        navItem(SettingsSection::Runtime, ICON_FA_MICROCHIP, "Runtime");
        navItem(SettingsSection::Registry, ICON_FA_DATABASE, "Registry");
        navItem(SettingsSection::Intelligence, ICON_FA_WAND_MAGIC_SPARKLES, "Intelligence");
        navItem(SettingsSection::Developer, ICON_FA_FLASK, "Developer");
        navItem(SettingsSection::About, ICON_FA_CIRCLE_INFO, "About");
        navItem(SettingsSection::Legal, ICON_FA_SCALE_BALANCED, "Legal");
        general.update(this->config.general);
        remote.update(this->config.remote);
        runtime.update(this->config.runtime);
        registry.update(this->config.registry);
        intelligence.update(this->config.intelligence);
        developer.update(this->config.developer);
        about.update(this->config.about);
        legal.update(this->config.legal);

        std::vector<Sakura::Stack::Item> navigationItems = {
            {.child = &title},
            {.child = &titleSpacing},
            {.child = &preferencesHeading},
            {.child = &headingSpacing},
        };
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
            navigationItems.push_back({.child = &navigationButtons[static_cast<U64>(section)]});
        }
        navigationItems.push_back({.child = &advancedSpacing});
        navigationItems.push_back({.child = &advancedHeading});
        navigationItems.push_back({.child = &advancedHeadingSpacing});
        navigationItems.push_back({.child = &navigationButtons[static_cast<U64>(SettingsSection::Developer)]});
        navigationItems.push_back({.child = &applicationSpacing});
        navigationItems.push_back({.child = &applicationHeading});
        navigationItems.push_back({.child = &applicationHeadingSpacing});
        navigationItems.push_back({.child = &navigationButtons[static_cast<U64>(SettingsSection::About)]});
        navigationItems.push_back({.child = &navigationButtons[static_cast<U64>(SettingsSection::Legal)]});
        advancedHeadingSpacing.update({});
        applicationHeadingSpacing.update({});
        navigation.update({
            .spacing = 1.0f,
            .items = std::move(navigationItems),
        });
        navigationFrame.update({
            .content = &navigation,
            .padding = 9.0f,
            .border = false,
            .colorKey = "transparent",
        });
        navigationArea.update({
            .content = &navigationFrame,
            .alwaysScrollbar = false,
            .border = false,
        });
        page.update({
            .id = "settings-page",
            .content = &section(),
            .maxWidth = PageWidth,
            .transparent = true,
        });
        pageFrame.update({
            .content = &page,
            .padding = 32.0f,
            .border = false,
            .colorKey = "transparent",
        });
        pageArea.update({
            .content = &pageFrame,
            .alwaysScrollbar = false,
            .border = false,
        });
        layout.update({
            .left = &navigationArea,
            .right = &pageArea,
            .leftWidth = SidebarWidth,
            .height = Sakura::Fill,
            .gutter = 0.0f,
        });
        tile.update({
            .panes = {{
                .id = TileId,
                .title = "Settings",
                .group = TileId,
                .content = &layout,
                .closable = false,
                .maximizable = false,
            }},
            .columns = true,
            .flexPane = TileId,
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

    Sakura::Component& section() {
        switch (config.section) {
            case SettingsSection::General:
                return general.content();
            case SettingsSection::Remote:
                return remote.content();
            case SettingsSection::Runtime:
                return runtime.content();
            case SettingsSection::Registry:
                return registry.content();
            case SettingsSection::Intelligence:
                return intelligence.content();
            case SettingsSection::Developer:
                return developer.content();
            case SettingsSection::About:
                return about.content();
            case SettingsSection::Legal:
                return legal.content();
        }
        return general.content();
    }

    Config config;
    SettingsSidebarHeading title;
    Sakura::Spacer titleSpacing;
    SettingsSidebarHeading preferencesHeading;
    Sakura::Spacer headingSpacing;
    Sakura::Spacer advancedSpacing;
    SettingsSidebarHeading advancedHeading;
    Sakura::Spacer advancedHeadingSpacing;
    Sakura::Spacer applicationSpacing;
    SettingsSidebarHeading applicationHeading;
    Sakura::Spacer applicationHeadingSpacing;
    std::array<Sakura::SidebarItem, 8> navigationButtons;
    Sakura::Stack navigation;
    Sakura::Div navigationFrame;
    Sakura::ScrollArea navigationArea;
    GeneralSettingsPanel general;
    RemoteSettingsPanel remote;
    RuntimeSettingsPanel runtime;
    RegistrySettingsPanel registry;
    IntelligenceSettingsPanel intelligence;
    DeveloperSettingsPanel developer;
    AboutSettingsPanel about;
    LegalSettingsPanel legal;
    Sakura::Panel page;
    Sakura::Div pageFrame;
    Sakura::ScrollArea pageArea;
    Sakura::SplitView layout;
    Sakura::Dock tile;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_BASE_HH
