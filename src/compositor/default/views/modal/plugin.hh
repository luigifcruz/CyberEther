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

        header.update({
            .id = "RegistryPluginHeader",
            .title = ICON_FA_FOLDER_OPEN " Register Plugin",
            .description = "Load a plugin that registers additional blocks.",
        });

        pathField.update({
            .id = "RegistryPluginPathField",
            .label = "Plugin Path",
            .description = "Path to a .cep plugin bundle that registers additional blocks.",
        });

        pathInput.update({
            .id = "##registry-plugin-path",
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
            .id = "RegistryPluginBrowse",
            .str = "Browse File",
            .size = {-1.0f, 0.0f},
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

        trustField.update({
            .id = "RegistryPluginTrustField",
            .label = "Trust Source",
            .description = "Plugins run native code inside CyberEther as soon as they load. Only register plugins you built yourself or received from a source you trust.",
            .divider = false,
        });

        trustCheckbox.update({
            .id = "RegistryPluginTrustCheckbox",
            .label = "I trust the source of this plugin.",
            .value = sourceTrusted,
            .onChange = [this](bool value) {
                sourceTrusted = value;
            },
        });

        registerButton.update({
            .id = "RegistryPluginRegister",
            .str = "Register Plugin",
            .size = {-1.0f, 40.0f},
            .variant = Sakura::Button::Variant::Action,
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
    }

    void render(const Sakura::Context& ctx) const {
        header.render(ctx);
        pathField.render(ctx, [this](const Sakura::Context& ctx) {
            pathInput.render(ctx);
            browseButton.render(ctx);
        });
        trustField.render(ctx, [this](const Sakura::Context& ctx) {
            trustCheckbox.render(ctx);
        });
        divider.render(ctx);
        registerButton.render(ctx);
    }

 private:
    Config config;
    std::string path;
    bool sourceTrusted = false;
    ModalHeader header;
    Sakura::SettingField pathField;
    Sakura::TextInput pathInput;
    Sakura::Button browseButton;
    Sakura::SettingField trustField;
    Sakura::Checkbox trustCheckbox;
    Sakura::Divider divider;
    Sakura::Button registerButton;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_PLUGIN_HH
