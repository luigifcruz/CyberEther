#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_MODALS_CLOSE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_MODALS_CLOSE_HH

#include "../../components/modal_header.hh"
#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/imgui_icons_ext.hh"

#include <functional>

namespace Jetstream {

struct FlowgraphCloseView {
    struct Config {
        std::function<void()> onSave;
        std::function<void()> onDontSave;
    };

    void update(Config config) {
        using Button = Sakura::Button;

        this->config = std::move(config);
        header.update({
            .title = ICON_FA_TRIANGLE_EXCLAMATION " Close Flowgraph",
            .description = "You are about to close a flowgraph without saving it. Are you sure you want to continue?",
        });
        saveButton.update({
            .str = ICON_FA_FLOPPY_DISK " Save",
            .size = {Sakura::Fill, 40.0f},
            .variant = Button::Variant::Action,
            .onClick = [this]() {
                if (this->config.onSave) {
                    this->config.onSave();
                }
            },
        });
        dontSaveButton.update({
            .str = "Don't Save",
            .size = {Sakura::Fill, 40.0f},
            .onClick = [this]() {
                if (this->config.onDontSave) {
                    this->config.onDontSave();
                }
            },
        });
        stack.update({
            .items = {
                {.child = &header},
                {.child = &saveButton},
                {.child = &dontSaveButton},
            },
        });
        panel.update({
            .id = "FlowgraphClose",
            .content = &stack,
            .backgroundColorKey = "popup_bg",
        });
    }

    Sakura::Component& component() {
        return panel;
    }

 private:
    Config config;
    ModalHeader header;
    Sakura::Button saveButton;
    Sakura::Button dontSaveButton;
    Sakura::Stack stack;
    Sakura::Panel panel;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_MODALS_CLOSE_HH
