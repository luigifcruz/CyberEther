#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_MODALS_RENAME_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_MODALS_RENAME_HH

#include "../../components/modal_header.hh"
#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/imgui_icons_ext.hh"

#include <functional>
#include <string>

namespace Jetstream {

struct RenameBlockView {
    struct Config {
        std::string oldName;
        std::function<void(const std::string&)> onRename;
    };

    void update(Config config) {
        if (this->config.oldName != config.oldName) {
            newName = config.oldName;
        }
        this->config = std::move(config);

        header.update({
            .id = "RenameBlockHeader",
            .title = ICON_FA_TAG " Rename Block",
            .description = "Enter a new block identifier.",
        });
        input.update({
            .id = "##rename-block-new-id",
            .value = newName,
            .hint = "Block name",
            .submit = Sakura::TextInput::Submit::OnEdit,
            .focus = true,
            .selectAllOnFocus = true,
            .onChange = [this](const std::string& value) {
                newName = value;
            },
            .onSubmit = [this](const std::string& value) {
                newName = value;
                submit();
            },
        });
        renameButton.update({
            .id = "RenameBlockConfirm",
            .str = "Rename Block",
            .size = {-1.0f, 40.0f},
            .variant = Sakura::Button::Variant::Action,
            .disabled = !canSubmit(),
            .onClick = [this]() {
                submit();
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
    bool canSubmit() const {
        return newName != config.oldName;
    }

    void submit() const {
        if (canSubmit() && config.onRename) {
            config.onRename(newName);
        }
    }

    Config config;
    std::string newName;
    ModalHeader header;
    Sakura::TextInput input;
    Sakura::Divider divider;
    Sakura::Button renameButton;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_MODALS_RENAME_HH
