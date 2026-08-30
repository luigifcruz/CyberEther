#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_FEEDBACK_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_FEEDBACK_HH

#include "../components/modal_header.hh"
#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/imgui_icons_ext.hh"
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

        header.update({
            .id = "FeedbackModalHeader",
            .title = ICON_FA_COMMENT_DOTS " Send Feedback (CTRL+G)",
            .description = "Share bugs, feature ideas, or general comments. "
                           "All feedback helps. It's anonymous, with no follow-up.",
        });

        const std::string countStr = jst::fmt::format("{}/{}", this->config.text.size(), this->config.maxLength);
        const bool overLimit = this->config.text.size() > this->config.maxLength;
        const bool underMinimum = this->config.text.size() < this->config.minLength;

        countText.update({
            .id = "FeedbackModalCount",
            .str = countStr,
            .tone = overLimit ? Sakura::Text::Tone::Warning
                    : underMinimum ? Sakura::Text::Tone::Disabled
                                    : Sakura::Text::Tone::Secondary,
            .align = Sakura::Text::Align::Right,
            .scale = 0.85f,
        });

        feedbackInput.update({
            .id = "FeedbackModalEditor",
            .value = this->config.text,
            .size = {0.0f, 160.0f},
            .language = Sakura::Retained::CodeEditor::Language::Markdown,
            .lineNumbers = false,
            .lineWrapping = true,
            .showActiveLine = false,
            .contentPadding = 6.0f,
            .backgroundColorKey = "card",
            .onChange = [this](std::string value) {
                if (this->config.onTextChange) {
                    this->config.onTextChange(value);
                }
            },
        });

        const bool submitting = this->config.status == Config::Status::Submitting;
        const bool canSubmit = !underMinimum && !overLimit && !submitting;

        submitButton.update({
            .id = "FeedbackModalSubmit",
            .str = submitting ? ICON_FA_SPINNER " Submitting..."
                              : ICON_FA_PAPER_PLANE " Submit Feedback",
            .size = {-1.0f, 40.0f},
            .variant = Sakura::Button::Variant::Action,
            .disabled = !canSubmit,
            .onClick = [this]() {
                if (this->config.onSubmit) {
                    this->config.onSubmit();
                }
            },
        });
    }

    void render(const Sakura::Context& ctx) {
        header.render(ctx);

        feedbackInput.render(ctx);
        countText.render(ctx);

        divider.render(ctx);
        submitButton.render(ctx);
    }

 private:
    Config config;
    ModalHeader header;
    Sakura::Retained::CodeEditor feedbackInput;
    Sakura::Text countText;
    Sakura::Divider divider;
    Sakura::Button submitButton;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_FEEDBACK_HH
