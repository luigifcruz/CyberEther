#include "feedback.hh"

#include "../../support/feedback.hh"

namespace Jetstream {

void FeedbackView::update(Config config) {
    std::unique_ptr<Sakura::CodeEditor> retiredEditor;
    std::unique_ptr<Sakura::CodeEditor> retiredDebugEditor;
    if (!editor || !debugEditor || editorSession != config.session) {
        editorSession = config.session;
        retiredEditor = std::move(editor);
        retiredDebugEditor = std::move(debugEditor);
        editor = std::make_unique<Sakura::CodeEditor>();
        debugEditor = std::make_unique<Sakura::CodeEditor>();
    }
    this->config = std::move(config);

    using Text = Sakura::Text;

    const bool blockReport = this->config.blockReport || !this->config.debugInfo.empty();
    const F32 editorHeight = blockReport ? ReportEditorHeight : EditorHeight;

    icon.update({
        .str = blockReport ? ICON_LC_BUG : ICON_LC_MESSAGE_CIRCLE_MORE,
        .colorKey = "text_primary",
        .scale = Sakura::Typography::Scale::Heading,
    });
    title.update({
        .str = blockReport ? "Report Block Problem" : "Send Feedback",
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
        .str = blockReport
                   ? "Add any extra context if you'd like, and review the block details below before sending. "
                     "Reports are anonymous, with no follow-up."
                   : "Share bugs, feature ideas, or general comments. "
                     "All feedback helps. It is anonymous, with no follow-up.",
        .font = Text::Font::Sans,
        .tone = Text::Tone::Secondary,
        .wrapped = true,
        .scale = Sakura::Typography::Scale::Lead,
    });

    editor->update({
        .id = jst::fmt::format("FeedbackModalEditor:{}", this->config.session),
        .value = this->config.text,
        .height = editorHeight - 2.0f * AreaInset,
        .language = Sakura::CodeEditor::Language::Plain,
        .lineNumbers = false,
        .lineWrapping = true,
        .scrollPastEnd = false,
        .showActiveLine = false,
        .padding = Sakura::Padding{AreaPadding, AreaVerticalPadding, AreaPadding, AreaVerticalPadding},
        .fontName = Sakura::Typography::BodyFont,
        .backgroundColorKey = AreaFill,
        .onChange = [this](std::string value) {
            if (this->config.onTextChange) {
                this->config.onTextChange(value);
            }
        },
    });
    editorHint.update({
        .str = blockReport
                   ? "What went wrong or what were you trying to do? (optional)"
                   : "Share bugs, feature ideas, or general comments...",
        .font = Text::Font::Body,
        .tone = Text::Tone::Secondary,
        .wrapped = true,
    });
    std::vector<Sakura::Overlay::Item> editorItems = {{.child = editor.get()}};
    if (this->config.text.empty()) {
        editorItems.push_back({
            .child = &editorHint,
            .anchor = Sakura::Anchor::TopLeft,
            .offset = {AreaPadding, AreaVerticalPadding},
        });
    }
    editorWrite.update({.items = std::move(editorItems)});
    editorCard.update({
        .content = &editorWrite,
        .size = {Sakura::Fill, editorHeight},
        .padding = AreaInset,
        .rounding = Sakura::Rounding::Card,
        .border = true,
        .borderWidth = AreaBorder,
        .chrome = true,
        .shadow = true,
        .shadowSoftness = AreaShadowSoftness,
        .shadowDrop = AreaShadowDrop,
        .colorKey = AreaFill,
        .hoveredColorKey = AreaFill,
    });

    if (blockReport) {
        debugLabel.update({
            .str = "Debug Information (included with report)",
            .font = Text::Font::Sans,
            .tone = Text::Tone::Secondary,
            .scale = CountScale,
        });
        debugEditor->update({
            .id = jst::fmt::format("FeedbackModalDebugEditor:{}", this->config.session),
            .value = this->config.debugInfo,
            .height = DebugEditorHeight - 2.0f * AreaInset,
            .language = Sakura::CodeEditor::Language::Plain,
            .lineNumbers = false,
            .lineWrapping = true,
            .scrollPastEnd = false,
            .showActiveLine = false,
            .padding = Sakura::Padding{AreaPadding, AreaVerticalPadding, AreaPadding, AreaVerticalPadding},
            .fontName = Sakura::Typography::MonoFont,
            .backgroundColorKey = AreaFill,
            .onChange = [this](std::string value) {
                if (this->config.onDebugInfoChange) {
                    this->config.onDebugInfoChange(value);
                }
            },
        });
        debugCard.update({
            .content = debugEditor.get(),
            .size = {Sakura::Fill, DebugEditorHeight},
            .padding = AreaInset,
            .rounding = Sakura::Rounding::Card,
            .border = true,
            .borderWidth = AreaBorder,
            .chrome = true,
            .shadow = true,
            .shadowSoftness = AreaShadowSoftness,
            .shadowDrop = AreaShadowDrop,
            .colorKey = AreaFill,
            .hoveredColorKey = AreaFill,
        });
        debugSection.update({
            .spacing = Sakura::Spacing::XSmall,
            .items = {
                {.child = &debugLabel},
                {.child = &debugCard},
            },
        });
    }

    const U64 totalLength = Feedback::Compose(this->config.text, this->config.debugInfo).size();
    const bool overLimit = totalLength > this->config.maxLength;
    const bool underMinimum = totalLength < this->config.minLength;
    const bool submitting = this->config.status == Config::Status::Submitting;
    const bool canSubmit = !underMinimum && !overLimit && !submitting;

    count.update({
        .str = jst::fmt::format("{} / {}", totalLength, this->config.maxLength),
        .tone = overLimit      ? Text::Tone::Warning
                : underMinimum ? Text::Tone::Disabled
                               : Text::Tone::Secondary,
        .scale = CountScale,
    });
    cancel.update({
        .label = "Cancel",
        .hoverColorKey = "chrome_button_hover",
        .height = ButtonHeight,
        .onClick = [this]() {
            if (this->config.onClose) {
                this->config.onClose();
            }
        },
    });
    send.update({
        .icon = submitting ? ICON_LC_LOADER_CIRCLE : ICON_LC_SEND,
        .label = submitting ? "Sending..." : (blockReport ? "Send Report" : "Send Feedback"),
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
    std::vector<Sakura::Stack::Item> items = {
        {.child = &header},
        {.child = &editorCard},
    };
    if (blockReport) {
        items.push_back({.child = &debugSection});
    }
    items.push_back({.child = &footer});
    stack.update({
        .spacing = Sakura::Spacing::Large,
        .items = std::move(items),
    });
    panel.update({
        .id = "FeedbackModal",
        .content = &stack,
        .backgroundColorKey = "popup_bg",
    });
}

Sakura::Component& FeedbackView::component() {
    return panel;
}

}  // namespace Jetstream
