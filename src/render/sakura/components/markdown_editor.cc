#include <jetstream/render/sakura/components/markdown_editor.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/controls.hh>
#include <jetstream/render/sakura/components/polyline.hh>
#include <jetstream/render/sakura/components/text_editor.hh>
#include <jetstream/render/sakura/components/text_markdown.hh>

#include "../context.hh"
#include "../detail/corner_mask.hh"
#include "../metrics.hh"

#include <algorithm>
#include <string>
#include <utility>

namespace Jetstream::Sakura {

namespace {

constexpr F32 kSurfaceInset = 2.0f;
constexpr F32 kButtonTopGap = 10.0f;
constexpr F32 kButtonPadY = 6.0f;
constexpr F32 kContentFitSlackPixels = 1.0f;

}  // namespace

struct MarkdownEditor::Impl {
    Box background;
    TextMarkdown preview;
    TextEditor editor;
    Button button;
    Polyline cornerMask;

    Config config;

    std::string appliedId;
    std::string appliedValue;
    F32 appliedFontSize = -1.0f;
    F32 appliedPixelRatio = -1.0f;
    bool appliedEditing = false;

    void apply(Config next) {
        if (config.editing != next.editing) {
            preview.setVisible(!next.editing);
            editor.setVisible(next.editing);
        }
        config = std::move(next);
    }

    Padding contentPaddingPixels(F32 pixelRatio) const {
        const F32 horizontal = (kSurfaceInset + config.fontSize * Typography::TextPaddingRatio) * pixelRatio;
        return {horizontal, kSurfaceInset * pixelRatio, horizontal, 0.0f};
    }

    void applyContent(F32 pixelRatio) {
        if (appliedPixelRatio == pixelRatio && appliedEditing == config.editing && appliedFontSize == config.fontSize &&
            appliedId == config.id && appliedValue == config.value) {
            return;
        }
        appliedPixelRatio = pixelRatio;
        appliedEditing = config.editing;
        appliedFontSize = config.fontSize;
        appliedId = config.id;
        appliedValue = config.value;
        const F32 fontSizePixels = config.fontSize * pixelRatio;
        const auto padding = contentPaddingPixels(pixelRatio);
        if (config.editing) {
            editor.update({
                .id = config.id + ":editor",
                .value = config.value,
                .fontSize = fontSizePixels,
                .lineHeight = Typography::BodyLineHeight,
                .fontName = TextMarkdown::BodyFont,
                .monospace = false,
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
                .onChange = [this](std::string next) {
                    if (config.onChange) {
                        config.onChange(std::move(next));
                    }
                },
            });
        } else {
            preview.update({
                .id = config.id + ":preview",
                .value = config.value,
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
};

MarkdownEditor::MarkdownEditor() {
    this->impl = std::make_unique<Impl>();
    setClipsChildren(true);
    addChild(impl->background);
    addChild(impl->preview);
    addChild(impl->editor);
    addChild(impl->button);
    addChild(impl->cornerMask);
    impl->editor.setVisible(false);
}

MarkdownEditor::~MarkdownEditor() = default;

void MarkdownEditor::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->apply(std::move(config));
}

Extent2D<F32> MarkdownEditor::measure(const Context& ctx, Extent2D<F32> available) {
    const F32 pixelRatio = ctx.pixelRatio;
    impl->applyContent(pixelRatio);
    const F32 activeContentPx = impl->config.editing
        ? measureChild(impl->editor, ctx, available).y
        : measureChild(impl->preview, ctx, available).y;

    const F32 fontSize = impl->config.fontSize;
    const F32 pad = kSurfaceInset;
    const F32 lineHeight = fontSize * Typography::BodyLineHeight;
    const F32 buttonH = fontSize + 2.0f * kButtonPadY;
    const F32 contentLogical = activeContentPx / std::max(1e-3f, pixelRatio);
    const F32 contentFitSlack = kContentFitSlackPixels / std::max(1e-3f, pixelRatio);
    const F32 total = contentLogical + contentFitSlack + kButtonTopGap + buttonH + pad;

    const F32 viewportHeight = LogicalViewportHeight(ctx);
    const F32 maxHeight = viewportHeight * std::clamp(impl->config.maxAutoHeightWindowRatio, 0.0f, 1.0f);
    const F32 minHeight = pad * 3.0f + lineHeight + buttonH;
    const F32 desiredLogical = std::max(minHeight, maxHeight > 0.0f ? std::min(total, maxHeight) : total);

    return {FillLength(available.x), FitLength(desiredLogical * pixelRatio, available.y)};
}

void MarkdownEditor::layout(const Context& ctx) {
    const Rect bounds = frame();
    const F32 pixelRatio = ctx.pixelRatio;
    const F32 fontSizePixels = impl->config.fontSize * pixelRatio;
    const F32 pad = kSurfaceInset * pixelRatio;
    const F32 buttonH = fontSizePixels + 2.0f * kButtonPadY * pixelRatio;
    const bool visible = !bounds.empty();

    const ColorRGBA<F32> bg = ctx.color(impl->config.backgroundColorKey);
    impl->background.update({
        .instances = {{.rect = bounds, .visible = visible, .backgroundColor = bg}},
    });

    const Rect buttonRect = {bounds.x + pad, bounds.bottom() - pad - buttonH,
                             std::max(0.0f, bounds.width - 2.0f * pad), buttonH};
    const F32 buttonGap = kButtonTopGap * pixelRatio;
    const Rect contentRect = {bounds.x, bounds.y, bounds.width,
                              std::max(0.0f, buttonRect.y - buttonGap - bounds.y)};

    impl->applyContent(pixelRatio);
    if (impl->config.editing) {
        layoutChild(ctx, impl->editor, contentRect);
    } else {
        layoutChild(ctx, impl->preview, contentRect);
    }

    impl->button.update({
        .str = impl->config.editing ? "Done" : "Edit",
        .textScale = impl->config.fontSize / Typography::FontSize,
        .onClick = [this] {
            if (impl->config.editing) {
                if (impl->config.onDone) {
                    impl->config.onDone();
                }
            } else if (impl->config.onEdit) {
                impl->config.onEdit();
            }
        },
    });
    layoutChild(ctx, impl->button, buttonRect);

    layoutChild(ctx, impl->background, bounds);
    UpdateCornerMask(ctx, impl->cornerMask, bounds, clip());
}

}  // namespace Jetstream::Sakura
