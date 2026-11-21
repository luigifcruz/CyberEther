#include "base.hh"

namespace Jetstream {

void SettingsView::update(Config config) {
    this->config = std::move(config);
    navItem(SettingsSection::General, ICON_LC_MONITOR, "General");
    navItem(SettingsSection::Remote, ICON_LC_SCREEN_SHARE, "Remote");
    navItem(SettingsSection::Maps, ICON_LC_MAP, "Maps");
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
                               SettingsSection::Maps,
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

Sakura::Component& SettingsView::component() {
    return tile;
}

Sakura::Dock::Pane SettingsView::pane() {
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

void SettingsView::navItem(SettingsSection section, std::string icon, std::string label) {
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

Sakura::Component& SettingsView::updateSection() {
    switch (config.section) {
        case SettingsSection::General:
            return Show(general, config.general);
        case SettingsSection::Remote:
            return Show(remote, config.remote);
        case SettingsSection::Maps:
            return Show(maps, config.maps);
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

}  // namespace Jetstream
