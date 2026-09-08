#include <jetstream/render/sakura/components/retained/markdown_editor.hh>

#include <jetstream/render/sakura/components/retained/box.hh>
#include <jetstream/render/sakura/components/retained/button.hh>
#include <jetstream/render/sakura/components/retained/canvas.hh>
#include <jetstream/render/sakura/components/retained/text_editor.hh>
#include <jetstream/render/sakura/components/retained/text_markdown.hh>

#include "../../helpers.hh"

#include <algorithm>
#include <string>
#include <utility>

namespace Jetstream::Sakura::Retained {

namespace {

constexpr F32 kReferenceFontSize = 15.0f;
constexpr F32 kPaddingFontRatio = 6.0f / kReferenceFontSize;
constexpr F32 kLineHeightFontRatio = 18.0f / kReferenceFontSize;
constexpr F32 kSurfaceInset = 2.0f;
constexpr F32 kButtonTopGap = 10.0f;
constexpr F32 kButtonPadY = 6.0f;
constexpr F32 kContentFitSlackPixels = 1.0f;

}  // namespace

struct MarkdownEditorBody : public Component {
    Box background;
    TextMarkdown preview;
    TextEditor editor;
    Button button;

    std::string id;
    std::string value;
    bool editing = false;
    F32 fontSize = kReferenceFontSize;
    std::string backgroundColorKey = "editor_background";
    std::function<void(std::string)> onChange;
    std::function<void()> onEdit;
    std::function<void()> onDone;

    F32 maxAutoHeightWindowRatio = 0.0f;

    MarkdownEditorBody() {
        setClipsChildren(true);
        add(background);
        add(preview);
        add(editor);
        add(button);
        editor.setVisible(false);
    }

    void apply(const MarkdownEditor::Config& config) {
        if (editing != config.editing) {
            editing = config.editing;
            preview.setVisible(!editing);
            editor.setVisible(editing);
        }
        id = config.id;
        value = config.value;
        fontSize = config.fontSize;
        backgroundColorKey = config.backgroundColorKey;
        onChange = config.onChange;
        onEdit = config.onEdit;
        onDone = config.onDone;
        maxAutoHeightWindowRatio = config.maxAutoHeightWindowRatio;
    }

    Padding contentPaddingPixels(F32 pixelRatio) const {
        const F32 horizontal = (kSurfaceInset + fontSize * kPaddingFontRatio) * pixelRatio;
        return {horizontal, kSurfaceInset * pixelRatio, horizontal, 0.0f};
    }

    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override {
        const F32 pixelRatio = ctx.pixelRatio;
        applyContent(pixelRatio);
        const F32 activeContentPx = editing
            ? measureChild(editor, ctx, available).y
            : measureChild(preview, ctx, available).y;

        const F32 pad = kSurfaceInset;
        const F32 lineHeight = fontSize * kLineHeightFontRatio;
        const F32 buttonH = fontSize + 2.0f * kButtonPadY;
        const F32 contentLogical = activeContentPx / std::max(1e-3f, pixelRatio);
        const F32 contentFitSlack = kContentFitSlackPixels / std::max(1e-3f, pixelRatio);
        const F32 total = contentLogical + contentFitSlack + kButtonTopGap + buttonH + pad;

        const F32 viewportHeight = ImGui::GetMainViewport() ? ImGui::GetMainViewport()->WorkSize.y : 0.0f;
        const F32 maxHeight = Unscale(ctx, viewportHeight * std::clamp(maxAutoHeightWindowRatio, 0.0f, 1.0f));
        const F32 minHeight = pad * 3.0f + lineHeight + buttonH;
        const F32 desiredLogical = std::max(minHeight, maxHeight > 0.0f ? std::min(total, maxHeight) : total);

        return {available.x, desiredLogical * pixelRatio};
    }

    void applyContent(F32 pixelRatio) {
        const F32 fontSizePixels = fontSize * pixelRatio;
        const auto padding = contentPaddingPixels(pixelRatio);
        if (editing) {
            editor.update({
                .id = id + ":editor",
                .value = value,
                .fontSize = fontSizePixels,
                .lineNumbers = false,
                .wrap = TextGrid::Wrap::Word,
                .padding = padding,
                .language = TextEditor::Language::Markdown,
                .textColorKey = "editor_text",
                .lineNumberColorKey = "editor_line_number",
                .gutterSeparatorColorKey = "editor_gutter_separator",
                .selectionColorKey = "editor_selection",
                .selectionMatchColorKey = "editor_selection_match",
                .activeLineColorKey = "editor_active_line",
                .cursorColorKey = "editor_cursor",
                .scrollbarTrackColorKey = "editor_scrollbar_track",
                .scrollbarThumbColorKey = "editor_scrollbar_thumb",
                .onChange = onChange,
            });
        } else {
            preview.update({
                .id = id + ":preview",
                .value = value,
                .fontSize = fontSizePixels,
                .scrollbar = true,
                .padding = padding,
                .textColorKey = "editor_text",
                .lineNumberColorKey = "editor_line_number",
                .gutterSeparatorColorKey = "editor_gutter_separator",
                .selectionColorKey = "editor_selection",
                .selectionMatchColorKey = "editor_selection_match",
                .activeLineColorKey = "editor_active_line",
                .cursorColorKey = "editor_cursor",
                .scrollbarTrackColorKey = "editor_scrollbar_track",
                .scrollbarThumbColorKey = "editor_scrollbar_thumb",
            });
        }
    }

    void layout(const Context& ctx) override {
        const Rect bounds = frame();
        const F32 pixelRatio = ctx.pixelRatio;
        const F32 fontSizePixels = fontSize * pixelRatio;
        const F32 pad = kSurfaceInset * pixelRatio;
        const F32 buttonH = fontSizePixels + 2.0f * kButtonPadY * pixelRatio;
        const bool visible = !bounds.empty();

        const ColorRGBA<F32> bg = ctx.color(backgroundColorKey);
        background.update({
            .id = id + ":bg",
            .instances = {{.rect = bounds, .visible = visible, .backgroundColor = bg}},
        });

        const Rect buttonRect = {bounds.x + pad, bounds.bottom() - pad - buttonH,
                                      std::max(0.0f, bounds.width - 2.0f * pad), buttonH};
        const F32 buttonGap = kButtonTopGap * pixelRatio;
        const Rect contentRect = {bounds.x, bounds.y, bounds.width,
                                       std::max(0.0f, buttonRect.y - buttonGap - bounds.y)};

        applyContent(pixelRatio);
        if (editing) {
            layoutChild(ctx, editor, contentRect);
        } else {
            layoutChild(ctx, preview, contentRect);
        }

        button.update({
            .id = id + ":button",
            .str = editing ? "Done" : "Edit",
            .colorKey = "button",
            .hoveredColorKey = "button_hovered",
            .activeColorKey = "button_active",
            .borderColorKey = "button_outline",
            .textColorKey = "button_text",
            .fontSize = fontSizePixels,
            .cornerRadius = 8.0f * pixelRatio,
            .borderWidth = 1.0f * pixelRatio,
            .onClick = [this] {
                if (editing) {
                    if (onDone) {
                        onDone();
                    }
                } else if (onEdit) {
                    onEdit();
                }
            },
        });
        layoutChild(ctx, button, buttonRect);

        layoutChild(ctx, background, bounds);

    }
};

struct MarkdownEditor::Impl {
    Config config;

    Canvas canvas;
    MarkdownEditorBody body;

    Impl() {
        canvas.mount(body);
    }
};

MarkdownEditor::MarkdownEditor() {
    this->impl = std::make_unique<Impl>();
}

MarkdownEditor::~MarkdownEditor() = default;
MarkdownEditor::MarkdownEditor(MarkdownEditor&&) noexcept = default;
MarkdownEditor& MarkdownEditor::operator=(MarkdownEditor&&) noexcept = default;

bool MarkdownEditor::update(Config config) {
    impl->config = std::move(config);

    const F32 defaultHeight = impl->config.fontSize * (kLineHeightFontRatio * 5.0f + kPaddingFontRatio * 5.0f);

    impl->canvas.update({
        .id = impl->config.id + ":canvas",
        .size = {0.0f, impl->config.height.value_or(defaultHeight)},
        .autoHeight = !impl->config.height.has_value(),
    });

    impl->body.apply(impl->config);
    return true;
}

void MarkdownEditor::render(const Sakura::Context& ctx) const {
    impl->canvas.render(ctx);
}

}  // namespace Jetstream::Sakura::Retained
