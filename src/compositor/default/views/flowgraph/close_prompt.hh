#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CLOSE_PROMPT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CLOSE_PROMPT_HH

#include "../components/card_button.hh"

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"

#include <functional>
#include <optional>
#include <string>
#include <utility>

namespace Jetstream {

struct FlowgraphClosePromptView {
    static constexpr F32 Width = 380.0f;
    static constexpr F32 ButtonHeight = 34.0f;

    struct Config {
        std::string flowgraph;
        std::function<void()> onSave;
        std::function<void()> onDontSave;
        std::function<void()> onCancel;
    };

    void update(Config config) {
        this->config = std::move(config);
        refresh(true);
    }

    void close() {
        config = {};
        refresh(false);
    }

    Sakura::ModalLayer::Config layer(std::optional<Rect> anchor) {
        if (!open) {
            return {};
        }
        return {
            .content = &keys,
            .size = {Width, 0.0f},
            .closeButton = false,
            .chrome = false,
            .anchor = anchor,
            .onClose = [this]() {
                if (config.onCancel) {
                    config.onCancel();
                }
            },
        };
    }

 private:
    void refresh(bool open) {
        using Text = Sakura::Text;

        this->open = open;

        title.update({
            .str = "Close Flowgraph?",
            .font = Text::Font::DisplaySemiBold,
            .scale = Sakura::Typography::Scale::Title,
        });
        message.update({
            .str = "Unsaved changes will be lost.",
            .font = Text::Font::Sans,
            .tone = Text::Tone::Secondary,
            .wrapped = true,
            .scale = Sakura::Typography::Scale::Caption,
        });
        cancel.update({
            .str = ICON_LC_X "  Cancel",
            .height = ButtonHeight,
            .onClick = config.onCancel,
        });
        discard.update({
            .str = ICON_LC_TRASH_2 "  Don't Save",
            .height = ButtonHeight,
            .onClick = config.onDontSave,
        });
        save.update({
            .str = ICON_LC_SAVE "  Save",
            .tone = CardButton::Tone::Action,
            .height = ButtonHeight,
            .onClick = config.onSave,
        });
        buttons.update({
            .columns = 3,
            .cellPadding = {3.0f, 0.0f},
            .outerPadding = false,
            .items = {&cancel, &discard, &save},
        });
        gap.update({.lines = 2});
        stack.update({
            .spacing = Sakura::Spacing::XSmall,
            .centerCross = false,
            .items = {
                {.child = &title},
                {.child = &message},
                {.child = &gap},
                {.child = &buttons},
            },
        });
        frame.update({
            .content = &stack,
            .padding = Sakura::Spacing::Large,
            .rounding = Sakura::Rounding::Popup,
            .border = true,
            .chrome = true,
            .shadow = true,
            .colorKey = "popup_bg",
            .hoveredColorKey = "popup_bg",
        });
        keys.update({
            .bindings = {
                {.key = KeyCode::Enter, .onPressed = config.onSave},
                {.key = KeyCode::KeypadEnter, .onPressed = config.onSave},
            },
            .enabled = open,
            .content = &frame,
        });
    }

    Config config;
    bool open = false;
    Sakura::Text title;
    Sakura::Text message;
    Sakura::Spacer gap;
    CardButton cancel;
    CardButton discard;
    CardButton save;
    Sakura::Grid buttons;
    Sakura::Stack stack;
    Sakura::Div frame;
    Sakura::Shortcuts keys;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CLOSE_PROMPT_HH
