#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_BASE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_BASE_HH

#include "../../components/modal_header.hh"
#include "../../../model/state.hh"
#include "about.hh"
#include "developer.hh"
#include "general.hh"
#include "legal.hh"
#include "remote.hh"

#include "jetstream/render/tools/imgui_icons_ext.hh"

#include <functional>

namespace Jetstream {

struct SettingsView : public Sakura::Component {
    using Section = DefaultCompositorState::SettingsState::Section;

    struct Config {
        Section section = Section::General;
        std::function<void(Section)> onSectionChange;
        GeneralSettingsPanel::Config general;
        RemoteSettingsPanel::Config remote;
        DeveloperSettingsPanel::Config developer;
        AboutSettingsPanel::Config about;
        LegalSettingsPanel::Config legal;
    };

    void update(Config config) {
        this->config = std::move(config);

        header.update({
            .id = "AppSettingsHeader",
            .title = ICON_FA_SLIDERS " Preferences",
            .description = "",
        });

        layout.update({
            .id = "AppSettingsLayout",
            .leftWidth = 220.0f,
            .fillHeight = true,
            .reservedHeight = 80.0f,
        });

        navigationDiv.update({
            .id = "AppSettingsNavigationDiv",
            .scrollbar = false,
            .mouseScroll = false,
        });

        editorDiv.update({
            .id = "AppSettingsEditorDiv",
        });

        navigation.update({
            .id = "AppSettingsNavigation",
            .title = "Sections",
            .items = {
                {
                    .label = ICON_FA_DESKTOP " General",
                    .selected = this->config.section == Section::General,
                    .onSelect = [this]() {
                        if (this->config.onSectionChange) {
                            this->config.onSectionChange(Section::General);
                        }
                    },
                },
                {
                    .label = ICON_FA_TOWER_BROADCAST " Remote",
                    .selected = this->config.section == Section::Remote,
                    .onSelect = [this]() {
                        if (this->config.onSectionChange) {
                            this->config.onSectionChange(Section::Remote);
                        }
                    },
                },
                {
                    .label = ICON_FA_FLASK " Developer",
                    .selected = this->config.section == Section::Developer,
                    .onSelect = [this]() {
                        if (this->config.onSectionChange) {
                            this->config.onSectionChange(Section::Developer);
                        }
                    },
                },
                {
                    .label = ICON_FA_CIRCLE_INFO " About",
                    .selected = this->config.section == Section::About,
                    .onSelect = [this]() {
                        if (this->config.onSectionChange) {
                            this->config.onSectionChange(Section::About);
                        }
                    },
                },
                {
                    .label = ICON_FA_SCALE_BALANCED " Legal",
                    .selected = this->config.section == Section::Legal,
                    .onSelect = [this]() {
                        if (this->config.onSectionChange) {
                            this->config.onSectionChange(Section::Legal);
                        }
                    },
                },
            },
        });

        general.update(this->config.general);
        remote.update(this->config.remote);
        developer.update(this->config.developer);
        about.update(this->config.about);
        legal.update(this->config.legal);
    }

    void render(const Sakura::Context& ctx) const {
        header.render(ctx);
        layout.render(ctx, {
            [this](const Sakura::Context& ctx) {
                navigationDiv.render(ctx, [this](const Sakura::Context& ctx) {
                    navigation.render(ctx);
                });
            },
            [this](const Sakura::Context& ctx) {
                editorDiv.render(ctx, [this](const Sakura::Context& ctx) {
                    switch (config.section) {
                        case Section::General:
                            general.render(ctx);
                            break;
                        case Section::Remote:
                            remote.render(ctx);
                            break;
                        case Section::Developer:
                            developer.render(ctx);
                            break;
                        case Section::About:
                            about.render(ctx);
                            break;
                        case Section::Legal:
                            legal.render(ctx);
                            break;
                    }
                });
            },
        });
    }

 private:
    Config config;
    ModalHeader header;
    Sakura::SplitView layout;
    Sakura::Div navigationDiv;
    Sakura::Div editorDiv;
    Sakura::NavigationList navigation;
    GeneralSettingsPanel general;
    RemoteSettingsPanel remote;
    DeveloperSettingsPanel developer;
    AboutSettingsPanel about;
    LegalSettingsPanel legal;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_BASE_HH
