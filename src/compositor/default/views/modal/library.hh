#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_LIBRARY_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_LIBRARY_HH

#include "../components/modal_header.hh"

#include "jetstream/render/sakura/sakura.hh"
#include "jetstream/render/tools/imgui_icons_ext.hh"

#include <functional>
#include <string>
#include <utility>

namespace Jetstream {

struct LibraryView : public Sakura::Component {
    struct Config {
        std::function<void(const std::string&, std::function<void(std::string)>)> onBrowse;
        std::function<void(const std::string&)> onRegister;
        std::function<void()> onCancel;
    };

    void update(Config config) {
        this->config = std::move(config);

        header.update({
            .id = "RegistryLibraryHeader",
            .title = ICON_FA_FOLDER_OPEN " Register Dynamic Library",
            .description = "Load a dynamic library that registers additional blocks.",
        });

        pathField.update({
            .id = "RegistryLibraryPathField",
            .label = "Library Path",
            .description = "Path to a .so, .dylib, or .dll file that registers additional blocks.",
            .divider = false,
        });

        pathInput.update({
            .id = "##registry-library-path",
            .value = path,
            .submit = Sakura::TextInput::Submit::OnEdit,
            .onChange = [this](const std::string& value) {
                path = value;
            },
        });

        browseButton.update({
            .id = "RegistryLibraryBrowse",
            .str = "Browse File",
            .size = {-1.0f, 0.0f},
            .onClick = [this]() {
                if (this->config.onBrowse) {
                    this->config.onBrowse(path, [this](std::string nextPath) {
                        path = std::move(nextPath);
                    });
                }
            },
        });

        registerButton.update({
            .id = "RegistryLibraryRegister",
            .str = ICON_FA_FLOPPY_DISK " Register Library",
            .size = {-1.0f, 40.0f},
            .variant = Sakura::Button::Variant::Action,
            .onClick = [this]() {
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
        divider.render(ctx);
        registerButton.render(ctx);
    }

 private:
    Config config;
    std::string path;
    ModalHeader header;
    Sakura::SettingField pathField;
    Sakura::TextInput pathInput;
    Sakura::Button browseButton;
    Sakura::Divider divider;
    Sakura::Button registerButton;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_LIBRARY_HH
