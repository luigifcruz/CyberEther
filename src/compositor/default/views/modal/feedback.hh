#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_FEEDBACK_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_FEEDBACK_HH

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"
#include "jetstream/types.hh"

#include <functional>
#include <string>

namespace Jetstream {

struct FeedbackView {
    static constexpr F32 EditorHeight = 180.0f;
    static constexpr F32 ButtonHeight = 34.0f;
    static constexpr F32 CountScale = Sakura::Typography::Scale::Small;

    struct Config {
        std::string text;
        U64 minLength = 10;
        U64 maxLength = 4000;
        enum class Status { Idle, Submitting };
        Status status = Status::Idle;
        std::function<void(const std::string&)> onTextChange;
        std::function<void()> onSubmit;
        std::function<void()> onClose;
    };

    void update(Config config) {
        this->config = std::move(config);

        using Text = Sakura::Text;

        icon.update({
            .str = ICON_LC_MESSAGE_CIRCLE_MORE,
            .colorKey = "text_primary",
            .scale = Sakura::Typography::Scale::Heading,
        });
        title.update({
            .str = "Send Feedback",
            .font = Text::Font::Display,
            .scale = Sakura::Typography::Scale::Headline,
        });
        titleRow.update({
            .axis = Sakura::Stack::Axis::Horizontal,
            .spacing = Sakura::Spacing::Small,
            .items = {
                {.child = &icon},
                {.child = &title},
            },
        });
        description.update({
            .str = "Share bugs, feature ideas, or general comments. "
                   "All feedback helps. It is anonymous, with no follow-up.",
            .font = Text::Font::Sans,
            .tone = Text::Tone::Secondary,
            .wrapped = true,
            .scale = Sakura::Typography::Scale::Lead,
        });

        editor.update({
            .id = "FeedbackModalEditor",
            .value = this->config.text,
            .height = EditorHeight,
            .language = Sakura::CodeEditor::Language::Markdown,
            .lineNumbers = false,
            .lineWrapping = true,
            .showActiveLine = false,
            .padding = Sakura::Padding{Sakura::Spacing::Medium, Sakura::Spacing::Medium, Sakura::Spacing::Medium,
                                       Sakura::Spacing::Medium},
            .fontName = Sakura::Typography::BodyFont,
            .backgroundColorKey = "pill_field_bg",
            .rounding = Sakura::Rounding::Card,
            .onChange = [this](std::string value) {
                if (this->config.onTextChange) {
                    this->config.onTextChange(value);
                }
            },
        });
        editorCard.update({
            .content = &editor,
            .rounding = Sakura::Rounding::Card,
            .border = true,
            .colorKey = "pill_field_bg",
        });

        const bool overLimit = this->config.text.size() > this->config.maxLength;
        const bool underMinimum = this->config.text.size() < this->config.minLength;
        const bool submitting = this->config.status == Config::Status::Submitting;
        const bool canSubmit = !underMinimum && !overLimit && !submitting;

        count.update({
            .str = jst::fmt::format("{} / {}", this->config.text.size(), this->config.maxLength),
            .tone = overLimit      ? Text::Tone::Warning
                    : underMinimum ? Text::Tone::Disabled
                                   : Text::Tone::Secondary,
            .scale = CountScale,
        });
        cancel.update({
            .label = "Cancel",
            .height = ButtonHeight,
            .onClick = [this]() {
                if (this->config.onClose) {
                    this->config.onClose();
                }
            },
        });
        send.update({
            .icon = submitting ? ICON_LC_LOADER_CIRCLE : ICON_LC_SEND,
            .label = submitting ? "Sending..." : "Send Feedback",
            .colorKey = "action_btn_text",
            .backgroundColorKey = "action_btn",
            .hoverColorKey = "action_btn_hovered",
            .raised = true,
            .disabled = !canSubmit,
            .height = ButtonHeight,
            .onClick = [this]() {
                if (this->config.onSubmit) {
                    this->config.onSubmit();
                }
            },
        });
        footerSpacer.update({});
        footer.update({
            .axis = Sakura::Stack::Axis::Horizontal,
            .spacing = Sakura::Spacing::Small,
            .items = {
                {.child = &count},
                {.child = &footerSpacer, .grow = 1.0f},
                {.child = &cancel},
                {.child = &send},
            },
        });

        header.update({
            .spacing = Sakura::Spacing::XSmall,
            .items = {
                {.child = &titleRow},
                {.child = &description},
            },
        });
        stack.update({
            .spacing = Sakura::Spacing::Large,
            .items = {
                {.child = &header},
                {.child = &editorCard, .size = EditorHeight},
                {.child = &footer},
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
    Sakura::Text icon;
    Sakura::Text title;
    Sakura::Stack titleRow;
    Sakura::Text description;
    Sakura::Stack header;
    Sakura::CodeEditor editor;
    Sakura::Div editorCard;
    Sakura::Text count;
    Sakura::Spacer footerSpacer;
    Sakura::PillButton cancel;
    Sakura::PillButton send;
    Sakura::Stack footer;
    Sakura::Stack stack;
    Sakura::Panel panel;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_FEEDBACK_HH
