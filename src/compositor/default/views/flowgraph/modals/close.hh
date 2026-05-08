#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_MODALS_CLOSE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_MODALS_CLOSE_HH

#include "../../components/modal_header.hh"
#include "jetstream/render/sakura/sakura.hh"
#include "jetstream/render/tools/imgui_icons_ext.hh"

#include <functional>

namespace Jetstream {

struct FlowgraphCloseView : public Sakura::Component {
    struct Config {
        std::function<void()> onSave;
        std::function<void()> onDontSave;
    };

    void update(Config config) {
        this->config = std::move(config);
        header.update({
            .id = "FlowgraphCloseHeader",
            .title = ICON_FA_TRIANGLE_EXCLAMATION " Close Flowgraph",
            .description = "You are about to close a flowgraph without saving it. Are you sure you want to continue?",
        });
        saveButton.update({
            .id = "FlowgraphCloseSave",
            .str = ICON_FA_FLOPPY_DISK " Save",
            .size = {-1.0f, 40.0f},
            .variant = Sakura::Button::Variant::Action,
            .onClick = [this]() {
                if (this->config.onSave) {
                    this->config.onSave();
                }
            },
        });
        dontSaveButton.update({
            .id = "FlowgraphCloseDontSave",
            .str = "Don't Save",
            .size = {-1.0f, 40.0f},
            .onClick = [this]() {
                if (this->config.onDontSave) {
                    this->config.onDontSave();
                }
            },
        });
    }

    void render(const Sakura::Context& ctx) const {
        header.render(ctx);
        saveButton.render(ctx);
        dontSaveButton.render(ctx);
    }

 private:
    Config config;
    ModalHeader header;
    Sakura::Button saveButton;
    Sakura::Button dontSaveButton;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_MODALS_CLOSE_HH
