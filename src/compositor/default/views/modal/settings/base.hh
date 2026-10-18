#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_BASE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_BASE_HH

#include "../../components/modal_header.hh"
#include "../../../model/ui.hh"
#include "about.hh"
#include "developer.hh"
#include "general.hh"
#include "legal.hh"
#include "registry.hh"
#include "remote.hh"
#include "runtime.hh"

#include "jetstream/render/tools/imgui_icons_ext.hh"

#include <array>
#include <functional>

namespace Jetstream {

struct SettingsView {
    struct Config {
        SettingsSection section = SettingsSection::General;
        GeneralSettingsPanel::Config general;
        RemoteSettingsPanel::Config remote;
        RuntimeSettingsPanel::Config runtime;
        RegistrySettingsPanel::Config registry;
        DeveloperSettingsPanel::Config developer;
        AboutSettingsPanel::Config about;
        LegalSettingsPanel::Config legal;
        std::function<void(SettingsSection)> onSectionChange;
    };

    void update(Config config) {
        this->config = std::move(config);
        header.update({
            .title = ICON_FA_SLIDERS " Preferences",
        });
        navigationTitle.update({.str = "Sections"});
        navigationDivider.update({.spacing = 0.0f});
        navigationSpacing.update({});
        navItem(SettingsSection::General).update({
            .label = ICON_FA_DESKTOP " General",
            .selected = this->config.section == SettingsSection::General,
            .onSelect = [this]() {
                if (this->config.onSectionChange) {
                    this->config.onSectionChange(SettingsSection::General);
                }
            },
        });
        navItem(SettingsSection::Remote).update({
            .label = ICON_FA_TOWER_BROADCAST " Remote",
            .selected = this->config.section == SettingsSection::Remote,
            .onSelect = [this]() {
                if (this->config.onSectionChange) {
                    this->config.onSectionChange(SettingsSection::Remote);
                }
            },
        });
        navItem(SettingsSection::Runtime).update({
            .label = ICON_FA_MICROCHIP " Runtime",
            .selected = this->config.section == SettingsSection::Runtime,
            .onSelect = [this]() {
                if (this->config.onSectionChange) {
                    this->config.onSectionChange(SettingsSection::Runtime);
                }
            },
        });
        navItem(SettingsSection::Registry).update({
            .label = ICON_FA_DATABASE " Registry",
            .selected = this->config.section == SettingsSection::Registry,
            .onSelect = [this]() {
                if (this->config.onSectionChange) {
                    this->config.onSectionChange(SettingsSection::Registry);
                }
            },
        });
        navItem(SettingsSection::Developer).update({
            .label = ICON_FA_FLASK " Developer",
            .selected = this->config.section == SettingsSection::Developer,
            .onSelect = [this]() {
                if (this->config.onSectionChange) {
                    this->config.onSectionChange(SettingsSection::Developer);
                }
            },
        });
        navItem(SettingsSection::About).update({
            .label = ICON_FA_CIRCLE_INFO " About",
            .selected = this->config.section == SettingsSection::About,
            .onSelect = [this]() {
                if (this->config.onSectionChange) {
                    this->config.onSectionChange(SettingsSection::About);
                }
            },
        });
        navItem(SettingsSection::Legal).update({
            .label = ICON_FA_SCALE_BALANCED " Legal",
            .selected = this->config.section == SettingsSection::Legal,
            .onSelect = [this]() {
                if (this->config.onSectionChange) {
                    this->config.onSectionChange(SettingsSection::Legal);
                }
            },
        });
        general.update(this->config.general);
        remote.update(this->config.remote);
        runtime.update(this->config.runtime);
        registry.update(this->config.registry);
        developer.update(this->config.developer);
        about.update(this->config.about);
        legal.update(this->config.legal);

        std::vector<Sakura::Stack::Item> navigationItems = {
            {.child = &navigationTitle},
            {.child = &navigationDivider},
            {.child = &navigationSpacing},
        };
        for (auto& item : navigationButtons) {
            navigationItems.push_back({.child = &item});
        }
        navigation.update({.items = std::move(navigationItems)});
        navigationArea.update({
            .content = &navigation,
            .alwaysScrollbar = false,
        });
        editorArea.update({
            .content = &section(),
            .alwaysScrollbar = false,
        });
        layout.update({
            .left = &navigationArea,
            .right = &editorArea,
            .leftWidth = 220.0f,
            .height = Sakura::Fill,
        });
        stack.update({
            .items = {
                {.child = &header},
                {.child = &layout, .grow = 1.0f},
            },
        });
        panel.update({
            .id = "AppSettings",
            .content = &stack,
            .size = {Sakura::Fill, Sakura::Fill},
            .backgroundColorKey = "popup_bg",
        });
    }

    Sakura::Component& component() {
        return panel;
    }

 private:
    Sakura::NavigationItem& navItem(SettingsSection section) {
        return navigationButtons[static_cast<U64>(section)];
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
    ModalHeader header;
    Sakura::Text navigationTitle;
    Sakura::Divider navigationDivider;
    Sakura::Spacer navigationSpacing;
    std::array<Sakura::NavigationItem, 7> navigationButtons;
    Sakura::Stack navigation;
    Sakura::ScrollArea navigationArea;
    GeneralSettingsPanel general;
    RemoteSettingsPanel remote;
    RuntimeSettingsPanel runtime;
    RegistrySettingsPanel registry;
    DeveloperSettingsPanel developer;
    AboutSettingsPanel about;
    LegalSettingsPanel legal;
    Sakura::ScrollArea editorArea;
    Sakura::SplitView layout;
    Sakura::Stack stack;
    Sakura::Panel panel;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_BASE_HH
