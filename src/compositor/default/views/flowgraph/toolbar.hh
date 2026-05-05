#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_TOOLBAR_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_TOOLBAR_HH

#include "jetstream/render/sakura/sakura.hh"
#include "jetstream/render/tools/imgui_icons_ext.hh"

#include <functional>
#include <string>
#include <utility>

namespace Jetstream {

struct FlowgraphToolbar : public Sakura::Component {
    struct Config {
        std::string id;
        std::function<void()> onSave;
        std::function<void()> onClose;
        std::function<void()> onAddBlock;
    };

    void update(Config config) {
        this->config = std::move(config);
        overlay.update({
            .id = this->config.id + ":overlay",
            .size = toolbarSize,
            .anchor = Sakura::Overlay::Anchor::TopCenter,
            .offset = {0.0f, 12.0f},
            .inputs = true,
        });
        card.update({
            .id = this->config.id + ":card",
            .size = toolbarSize,
            .padding = 6.0f,
            .rounding = 12.0f,
            .border = true,
            .scrollbar = false,
            .mouseScroll = false,
        });
        layout.update({
            .id = this->config.id + ":layout",
            .spacing = 0.0f,
        });
        saveButton.update({
            .id = this->config.id + ":save",
            .str = ICON_FA_FLOPPY_DISK " Save",
            .size = {82.0f, 34.0f},
            .onClick = this->config.onSave,
        });
        closeButton.update({
            .id = this->config.id + ":close",
            .str = ICON_FA_CIRCLE_XMARK " Close",
            .size = {86.0f, 34.0f},
            .onClick = this->config.onClose,
        });
        addBlockButton.update({
            .id = this->config.id + ":add-block",
            .str = ICON_FA_CUBE " Add Block",
            .size = {110.0f, 34.0f},
            .onClick = this->config.onAddBlock,
        });
    }

    void render(const Sakura::Context& ctx) {
        overlay.render(ctx, [this](const Sakura::Context& ctx) {
            card.render(ctx, [this](const Sakura::Context& ctx) {
                layout.render(ctx, {
                    [this](const Sakura::Context& ctx) { saveButton.render(ctx); },
                    [this](const Sakura::Context& ctx) { closeButton.render(ctx); },
                    [this](const Sakura::Context& ctx) { addBlockButton.render(ctx); },
                });
            });
        });
    }

 private:
    static constexpr Extent2D<F32> toolbarSize = {306.0f, 46.0f};

    Config config;
    Sakura::Overlay overlay;
    Sakura::Div card;
    Sakura::HStack layout;
    Sakura::Button saveButton;
    Sakura::Button closeButton;
    Sakura::Button addBlockButton;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_TOOLBAR_HH
