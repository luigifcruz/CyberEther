#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_REGISTRY_ADD_PLUGIN_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_REGISTRY_ADD_PLUGIN_HH

#include "components/settings_breadcrumb.hh"
#include "../components/section_group.hh"
#include "components/settings_header.hh"

#include "jetstream/render/sakura/base.hh"

#include <functional>
#include <string>
#include <utility>

namespace Jetstream {

struct RegistryAddPluginView {
    struct Config {
        std::function<void(const std::string&, std::function<void(std::string)>)> onBrowse;
        std::function<void(const std::string&)> onRegister;
        std::function<void()> onBack;
    };

    void reset() {
        pendingPath.clear();
        trusted = false;
    }

    void update(Config config) {
        this->config = std::move(config);
        breadcrumb.update({
            .root = "Registry",
            .current = "Add Plugin",
            .onRoot = this->config.onBack,
        });
        const bool ready = trusted && !pendingPath.empty();
        registerButton.update({
            .str = "Register Plugin",
            .size = {Sakura::Fill, 40.0f},
            .variant = Sakura::Button::Variant::Action,
            .disabled = !ready,
            .rounding = Sakura::Rounding::Card,
            .onClick = [this]() {
                if (trusted && !pendingPath.empty() && this->config.onRegister) {
                    this->config.onRegister(pendingPath);
                }
            },
            .font = Sakura::Text::Font::SansBold,
        });
        header.update({
            .title = "Add Plugin",
            .description = "Load a plugin bundle that registers additional blocks.",
        });
        pathInput.update({
            .value = pendingPath,
            .hint = "/path/to/plugin.cep",
            .submit = Sakura::TextInput::Submit::OnEdit,
            .onChange = [this](const std::string& value) {
                setPendingPath(value);
            },
            .borderColorKey = "border",
        });
        browseButton.update({
            .str = "Browse",
            .onClick = [this]() {
                if (this->config.onBrowse) {
                    this->config.onBrowse(pendingPath, [this](std::string path) {
                        setPendingPath(std::move(path));
                    });
                }
            },
            .font = Sakura::Text::Font::Sans,
        });
        pathControls.update({
            .axis = Sakura::Stack::Axis::Horizontal,
            .items = {
                {.child = &pathInput, .size = 1.0f, .grow = 1.0f},
                {.child = &browseButton},
            },
        });
        pathField.update({
            .content = &pathControls,
            .border = false,
            .colorKey = "transparent",
            .hoveredColorKey = "transparent",
        });
        trustToggle.update({
            .value = trusted,
            .onChange = [this](bool value) {
                trusted = value;
            },
        });
        sourceGroup.update({
            .warning = "Only register plugins you built yourself or received from a source you trust.",
            .rows = {
                {
                    .label = "Plugin Path",
                    .description = "Path to a .cep plugin bundle that registers additional blocks.",
                    .control = &pathField,
                },
                {
                    .label = "Trust Source",
                    .description = "Confirm that you trust the source of this plugin.",
                    .control = &trustToggle,
                },
            },
            .footer = &registerButton,
        });
        stack.update({
            .spacing = Sakura::Spacing::XSmall,
            .items = {
                {.child = &breadcrumb},
                {.child = &header},
                {.child = &sourceGroup},
            },
        });
    }

    Sakura::Component& content() {
        return stack;
    }

 private:
    void setPendingPath(std::string path) {
        if (path != pendingPath) {
            trusted = false;
        }
        pendingPath = std::move(path);
    }

    Config config;
    std::string pendingPath;
    bool trusted = false;
    SettingsBreadcrumb breadcrumb;
    SettingsHeader header;
    Sakura::Button registerButton;
    Sakura::TextInput pathInput;
    Sakura::Button browseButton;
    Sakura::Stack pathControls;
    Sakura::Div pathField;
    Sakura::Toggle trustToggle;
    SectionGroup sourceGroup;
    Sakura::Stack stack;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_REGISTRY_ADD_PLUGIN_HH
