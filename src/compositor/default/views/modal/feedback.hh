#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_FEEDBACK_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_FEEDBACK_HH

#include "../components/modal_header.hh"
#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_fa.hh"
#include "jetstream/types.hh"

#include <functional>
#include <string>

namespace Jetstream {

struct FeedbackView {
    struct Config {
        std::string text;
        U64 minLength = 10;
        U64 maxLength = 4000;
        enum class Status { Idle, Submitting };
        Status status = Status::Idle;
        std::function<void(const std::string&)> onTextChange;
        std::function<void()> onSubmit;
    };

    void update(Config config) {
        this->config = std::move(config);

        using Text = Sakura::Text;

        header.update({
            .title = ICON_FA_COMMENT_DOTS " Send Feedback (CTRL+G)",
            .description = "Share bugs, feature ideas, or general comments. "
                           "All feedback helps. It's anonymous, with no follow-up.",
        });

        const std::string countStr = jst::fmt::format("{}/{}", this->config.text.size(), this->config.maxLength);
        const bool overLimit = this->config.text.size() > this->config.maxLength;
        const bool underMinimum = this->config.text.size() < this->config.minLength;

        countText.update({
            .str = countStr,
            .tone = overLimit ? Text::Tone::Warning
                    : underMinimum ? Text::Tone::Disabled
                                    : Text::Tone::Secondary,
            .align = Text::Align::Right,
            .scale = 0.85f,
        });

        feedbackInput.update({
            .id = "FeedbackModalEditor",
            .value = this->config.text,
            .height = 160.0f,
            .language = Sakura::CodeEditor::Language::Markdown,
            .lineNumbers = false,
            .lineWrapping = true,
            .showActiveLine = false,
            .padding = Sakura::Padding{6.0f, 6.0f, 6.0f, 6.0f},
            .backgroundColorKey = "card",
            .rounding = 8.0f,
            .onChange = [this](std::string value) {
                if (this->config.onTextChange) {
                    this->config.onTextChange(value);
                }
            },
        });

        const bool submitting = this->config.status == Config::Status::Submitting;
        const bool canSubmit = !underMinimum && !overLimit && !submitting;

        submitButton.update({
            .str = submitting ? ICON_FA_SPINNER " Submitting..."
                              : ICON_FA_PAPER_PLANE " Submit Feedback",
            .size = {Sakura::Fill, 40.0f},
            .variant = Sakura::Button::Variant::Action,
            .disabled = !canSubmit,
            .onClick = [this]() {
                if (this->config.onSubmit) {
                    this->config.onSubmit();
                }
            },
        });
        divider.update({});
        stack.update({
            .items = {
                {.child = &header},
                {.child = &feedbackInput, .size = 160.0f},
                {.child = &countText},
                {.child = &divider},
                {.child = &submitButton},
            },
        });
        panel.update({
            .id = "FeedbackModal",
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
    Sakura::CodeEditor feedbackInput;
    Sakura::Text countText;
    Sakura::Divider divider;
    Sakura::Button submitButton;
    Sakura::Stack stack;
    Sakura::Panel panel;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_FEEDBACK_HH
