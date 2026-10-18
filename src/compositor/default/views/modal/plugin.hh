#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_PLUGIN_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_PLUGIN_HH

#include "../components/modal_header.hh"

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/imgui_icons_ext.hh"

#include <functional>
#include <string>
#include <utility>

namespace Jetstream {

struct PluginView {
    struct Config {
        std::function<void(const std::string&, std::function<void(std::string)>)> onBrowse;
        std::function<void(const std::string&)> onRegister;
        std::function<void()> onCancel;
    };

    void update(Config config) {
        this->config = std::move(config);

        using Button = Sakura::Button;

        header.update({
            .title = ICON_FA_FOLDER_OPEN " Register Plugin",
            .description = "Load a plugin that registers additional blocks.",
        });

        pathInput.update({
            .value = path,
            .submit = Sakura::TextInput::Submit::OnEdit,
            .onChange = [this](const std::string& value) {
                if (path != value) {
                    sourceTrusted = false;
                }
                path = value;
            },
        });

        browseButton.update({
            .str = "Browse File",
            .size = {Sakura::Fill, 0.0f},
            .onClick = [this]() {
                if (this->config.onBrowse) {
                    this->config.onBrowse(path, [this](std::string nextPath) {
                        if (path != nextPath) {
                            sourceTrusted = false;
                        }
                        path = std::move(nextPath);
                    });
                }
            },
        });

        trustCheckbox.update({
            .label = "I trust the source of this plugin.",
            .value = sourceTrusted,
            .onChange = [this](bool value) {
                sourceTrusted = value;
            },
        });

        registerButton.update({
            .str = "Register Plugin",
            .size = {Sakura::Fill, 40.0f},
            .variant = Button::Variant::Action,
            .disabled = !sourceTrusted,
            .onClick = [this]() {
                if (!sourceTrusted) {
                    return;
                }
                if (this->config.onRegister) {
                    this->config.onRegister(path);
                }
            },
        });

        pathControls.update({.items = {{.child = &pathInput}, {.child = &browseButton}}});
        pathField.update({
            .label = "Plugin Path",
            .description = "Path to a .cep plugin bundle that registers additional blocks.",
            .control = &pathControls,
        });
        trustField.update({
            .label = "Trust Source",
            .description = "Plugins run native code inside CyberEther as soon as they load. Only register plugins you built yourself or received from a source you trust.",
            .control = &trustCheckbox,
            .divider = false,
        });
        divider.update({});
        stack.update({
            .items = {
                {.child = &header},
                {.child = &pathField},
                {.child = &trustField},
                {.child = &divider},
                {.child = &registerButton},
            },
        });
        panel.update({
            .id = "RegistryPlugin",
            .content = &stack,
            .backgroundColorKey = "popup_bg",
        });
    }

    Sakura::Component& component() {
        return panel;
    }

 private:
    Config config;
    std::string path;
    bool sourceTrusted = false;
    ModalHeader header;
    Sakura::TextInput pathInput;
    Sakura::Button browseButton;
    Sakura::Stack pathControls;
    Sakura::SettingField pathField;
    Sakura::Checkbox trustCheckbox;
    Sakura::SettingField trustField;
    Sakura::Divider divider;
    Sakura::Button registerButton;
    Sakura::Stack stack;
    Sakura::Panel panel;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_PLUGIN_HH
