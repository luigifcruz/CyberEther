#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_MODALS_RENAME_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_MODALS_RENAME_HH

#include "../../components/modal_header.hh"
#include "jetstream/render/sakura/sakura.hh"
#include "jetstream/render/tools/imgui_icons_ext.hh"

#include <functional>
#include <string>

namespace Jetstream {

struct RenameBlockView : public Sakura::Component {
    struct Config {
        std::string oldName;
        std::function<void(const std::string&)> onRename;
    };

    void update(Config config) {
        if (this->config.oldName != config.oldName) {
            newName.clear();
        }
        this->config = std::move(config);

        header.update({
            .id = "RenameBlockHeader",
            .title = ICON_FA_PENCIL " Rename Block",
            .description = "Enter a new block identifier.",
        });
        input.update({
            .id = "##rename-block-new-id",
            .value = newName,
            .submit = Sakura::TextInput::Submit::OnEdit,
            .onChange = [this](const std::string& value) {
                newName = value;
            },
        });
        renameButton.update({
            .id = "RenameBlockConfirm",
            .str = "Rename Block",
            .size = {-1.0f, 40.0f},
            .variant = Sakura::Button::Variant::Action,
            .onClick = [this]() {
                if (this->config.onRename) {
                    this->config.onRename(newName);
                }
            },
        });
    }

    void render(const Sakura::Context& ctx) const {
        header.render(ctx);
        input.render(ctx);
        divider.render(ctx);
        renameButton.render(ctx);
    }

 private:
    Config config;
    std::string newName;
    ModalHeader header;
    Sakura::TextInput input;
    Sakura::Divider divider;
    Sakura::Button renameButton;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_MODALS_RENAME_HH
