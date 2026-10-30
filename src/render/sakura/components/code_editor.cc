#include <jetstream/render/sakura/components/code_editor.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/polyline.hh>
#include <jetstream/render/sakura/components/status_bar.hh>
#include <jetstream/render/sakura/components/text_editor.hh>
#include <jetstream/render/sakura/components/text_view.hh>

#include "../context.hh"
#include "../detail/corner_mask.hh"
#include "../metrics.hh"
#include "tools/text.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream::Sakura {

namespace {

constexpr F32 kConsoleHeaderHeight = 12.0f;
constexpr F32 kConsoleGripWidth = 28.0f;
constexpr F32 kConsoleGripHeight = 3.0f;
constexpr F32 kConsoleDefaultHeight = 128.0f;
constexpr F32 kConsoleMinHeight = 72.0f;
constexpr F32 kConsoleMaxHeightRatio = 0.55f;

bool SameEditorConfig(const CodeEditor::Config& a, const CodeEditor::Config& b) {
    return a.id == b.id && a.value == b.value && a.fontSize == b.fontSize && a.lineNumbers == b.lineNumbers &&
           a.showActiveLine == b.showActiveLine && a.lineWrapping == b.lineWrapping && a.language == b.language &&
           a.padding == b.padding && a.rounding == b.rounding && a.backgroundColorKey == b.backgroundColorKey;
}

TextEditor::Language ToEditorLanguage(CodeEditor::Language language) {
    return language == CodeEditor::Language::Markdown ? TextEditor::Language::Markdown
                                                      : TextEditor::Language::Python;
}

}  // namespace

struct CodeEditor::Impl {
    Config config;

    Box backgroundBox;
    TextEditor textEditor;
    TextView consoleView;
    Box consoleChrome;
    Box consoleGrip;
    StatusBar statusBar;
    Polyline cornerMask;

    Rect viewRect;
    F32 fontSizePixels = Typography::FontSize;
    F32 ratio = 1.0f;
    std::vector<std::string> joinedConsoleLines;
    std::string joinedConsole;
    bool consoleCollapsed = true;
    bool consoleResizing = false;
    F32 consoleResizeHeightPixels = 0.0f;
    F32 consoleHeight = kConsoleDefaultHeight;
    bool editorStale = true;
    F32 syncedRatio = 0.0f;
    F32 barHeight = 0.0f;
    std::optional<Extent2D<F32>> reportedOverflow;

    const std::string& consoleText() {
        if (joinedConsoleLines != config.consoleOutput) {
            joinedConsoleLines = config.consoleOutput;
            joinedConsole = Jetstream::Text::JoinLines(joinedConsoleLines);
        }
        return joinedConsole;
    }

    F32 pixelRatio() const {
        return ratio;
    }
    F32 verticalPaddingLogical() const {
        return config.padding.has_value() ? config.padding->top + config.padding->bottom : 0.0f;
    }
    std::optional<Padding> editorPaddingPixels() const {
        if (!config.padding.has_value()) {
            return std::nullopt;
        }
        const F32 ratio = pixelRatio();
        return Padding{
            config.padding->left * ratio,
            config.padding->top * ratio,
            config.padding->right * ratio,
            config.padding->bottom * ratio,
        };
    }
    F32 lineHeightPixels() const { return std::max(1.0f, fontSizePixels * Typography::CodeLineHeight); }
    F32 outlineHeightPixels() const { return std::max(1.0f, std::round(pixelRatio())); }
    F32 barHeightPixels() const { return barHeight; }

    bool consoleVisible() const { return config.consoleVisible; }
    bool consoleExpanded() const { return consoleVisible() && !consoleCollapsed; }
    bool statusVisible() const { return !config.status.empty() || consoleVisible(); }

    F32 statusBarHeightPixels() const { return statusVisible() ? barHeightPixels() : 0.0f; }
    F32 statusBarTopPixels() const { return std::max(viewRect.y, viewRect.bottom() - statusBarHeightPixels()); }
    F32 editorContentBottomPixels() const {
        return statusVisible() ? statusBarTopPixels() : viewRect.bottom();
    }
    F32 consoleHeaderHeightPixels() const {
        return std::max(kConsoleHeaderHeight, kConsoleHeaderHeight * pixelRatio());
    }
    F32 consoleDividerHeightPixels() const { return consoleExpanded() ? consoleHeaderHeightPixels() : 0.0f; }

    F32 consoleMinEditorHeightPixels() const {
        return verticalPaddingLogical() * pixelRatio() + lineHeightPixels() * 3.0f;
    }
    F32 consoleMaxHeightPixels() const {
        if (!consoleVisible()) {
            return 0.0f;
        }
        const F32 contentHeight = editorContentBottomPixels() - viewRect.y;
        const F32 available = contentHeight - consoleMinEditorHeightPixels() - consoleDividerHeightPixels();
        return std::max(0.0f, std::min(available, contentHeight * kConsoleMaxHeightRatio));
    }
    F32 consoleMinHeightPixels() const { return std::min(kConsoleMinHeight * pixelRatio(), consoleMaxHeightPixels()); }
    F32 consoleHeightPixels() const {
        if (!consoleExpanded()) {
            return 0.0f;
        }
        const F32 maxHeight = consoleMaxHeightPixels();
        if (maxHeight <= 0.0f) {
            return 0.0f;
        }
        const F32 height = consoleResizing ? consoleResizeHeightPixels : consoleHeight * pixelRatio();
        return std::clamp(height, consoleMinHeightPixels(), maxHeight);
    }
    F32 consolePanelTopPixels() const { return editorContentBottomPixels() - consoleHeightPixels(); }
    F32 consoleHeaderTopPixels() const { return consolePanelTopPixels() - consoleDividerHeightPixels(); }

    Rect editorAreaRect() const {
        const F32 bottom = consoleExpanded() ? consoleHeaderTopPixels() : editorContentBottomPixels();
        return {viewRect.x, viewRect.y, viewRect.width, std::max(0.0f, bottom - viewRect.y)};
    }
    Rect consoleHeaderRect() const {
        return {viewRect.x, consoleHeaderTopPixels(), viewRect.width, consoleDividerHeightPixels()};
    }
    Rect consolePanelRect() const {
        return {viewRect.x, consolePanelTopPixels(), viewRect.width, consoleHeightPixels()};
    }
    Rect statusBarRect() const {
        return {viewRect.x, statusBarTopPixels(), viewRect.width, statusBarHeightPixels()};
    }

    std::string statusConsoleToggleText() const {
        return consoleVisible() ? "Console" : "";
    }

    void sync(const Context& ctx) {
        ratio = ctx.pixelRatio;
        fontSizePixels = config.fontSize * ratio;
        if (!editorStale && syncedRatio == ratio) {
            return;
        }
        editorStale = false;
        syncedRatio = ratio;
        textEditor.update({
            .id = config.id + ":text",
            .value = config.value,
            .fontSize = fontSizePixels,
            .lineNumbers = config.lineNumbers,
            .showActiveLine = config.showActiveLine,
            .wrap = config.language == Language::Markdown ? TextGrid::Wrap::Word
                    : config.lineWrapping                ? TextGrid::Wrap::Character
                                                         : TextGrid::Wrap::None,
            .padding = editorPaddingPixels(),
            .language = ToEditorLanguage(config.language),
            .backgroundColorKey = config.rounding > 0.0f ? std::string("transparent") : config.backgroundColorKey,
            .textColorKey = "editor_text",
            .lineNumberColorKey = "editor_line_number",
            .gutterSeparatorColorKey = "editor_gutter_separator",
            .selectionColorKey = "editor_selection",
            .selectionMatchColorKey = "editor_selection_match",
            .cursorColorKey = "editor_cursor",
            .scrollbarTrackColorKey = "editor_scrollbar_track",
            .scrollbarThumbColorKey = "editor_scrollbar_thumb",
            .onChange = [this](std::string value) {
                if (config.onChange) {
                    config.onChange(std::move(value));
                }
            },
            .onSubmit = [this](std::string value) {
                if (config.onSubmit) {
                    config.onSubmit(std::move(value));
                }
            },
        });
    }

    bool handleChromeMouse(CodeEditor& self, const MouseEvent& event) {
        switch (event.type) {
            case MouseEventType::Click: {
                if (event.button != MouseButton::Left) {
                    return false;
                }
                if (consoleExpanded() && consoleHeaderRect().contains(event.position.x, event.position.y)) {
                    consoleResizing = true;
                    consoleResizeHeightPixels = consoleHeightPixels();
                    self.invalidate(Dirty::Layout);
                    return true;
                }
                return false;
            }
            case MouseEventType::Move: {
                if (consoleResizing) {
                    self.requestCursor(SurfaceCursor::ResizeNS);
                    const F32 height = editorContentBottomPixels() - event.position.y -
                                       consoleDividerHeightPixels() * 0.5f;
                    if (height != consoleResizeHeightPixels) {
                        consoleResizeHeightPixels = height;
                        self.invalidate(Dirty::Layout);
                    }
                    return true;
                }
                return false;
            }
            case MouseEventType::Cancel: {
                if (consoleResizing) {
                    consoleResizing = false;
                    self.invalidate(Dirty::Layout);
                }
                return false;
            }
            case MouseEventType::Release: {
                if (event.button != MouseButton::Left || !consoleResizing) {
                    return false;
                }
                consoleHeight = consoleHeightPixels() / std::max(0.1f, pixelRatio());
                consoleResizing = false;
                self.invalidate(Dirty::Layout);
                return true;
            }
            default:
                return false;
        }
    }
};

CodeEditor::CodeEditor() {
    this->impl = std::make_unique<Impl>();
    setClipsChildren(true);
    addChild(impl->backgroundBox);
    addChild(impl->textEditor);
    addChild(impl->consoleView);
    addChild(impl->consoleChrome);
    addChild(impl->consoleGrip);
    addChild(impl->statusBar);
    addChild(impl->cornerMask);
}

CodeEditor::~CodeEditor() = default;

void CodeEditor::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->editorStale = impl->editorStale || !SameEditorConfig(impl->config, config);
    impl->config = std::move(config);
}

Extent2D<F32> CodeEditor::measure(const Context& ctx, Extent2D<F32> available) {
    const auto& config = impl->config;
    impl->sync(ctx);
    impl->barHeight = measureChild(impl->statusBar, ctx, {std::numeric_limits<F32>::infinity(), std::numeric_limits<F32>::infinity()}).y;
    const F32 width = FillLength(available.x);
    if (config.height != 0.0f) {
        return {width, ResolveLength(ctx, config.height, 0.0f, available.y)};
    }

    const F32 pad = impl->verticalPaddingLogical() * ctx.pixelRatio;
    const F32 editorContent = measureChild(impl->textEditor, ctx, {width, available.y}).y;
    const F32 text = std::max(0.0f, editorContent - pad);
    const F32 console = impl->consoleExpanded() ? impl->consoleHeaderHeightPixels() + impl->consoleHeightPixels() : 0.0f;
    const F32 content = pad + text + console + impl->statusBarHeightPixels();

    const F32 maxHeight = Px(ctx, LogicalViewportHeight(ctx) * std::clamp(config.maxAutoHeightWindowRatio, 0.0f, 1.0f));
    const F32 minHeight = pad + impl->lineHeightPixels();
    const F32 height = std::max(minHeight, maxHeight > 0.0f ? std::min(content, maxHeight) : content);
    return {width, FitLength(height, available.y)};
}

void CodeEditor::layout(const Context& ctx) {
    const auto& config = impl->config;
    impl->sync(ctx);
    impl->barHeight = measureChild(impl->statusBar, ctx, {std::numeric_limits<F32>::infinity(), std::numeric_limits<F32>::infinity()}).y;
    impl->viewRect = frame();

    const Rect viewRect = impl->viewRect;
    const F32 fontSizePixels = impl->fontSizePixels;
    const F32 pixelRatio = impl->pixelRatio();
    const bool visible = !viewRect.empty();
    const auto editorBackground = ctx.color(config.backgroundColorKey);
    const F32 outline = impl->outlineHeightPixels();

    impl->backgroundBox.update({
        .instances = {{.rect = viewRect, .visible = visible, .backgroundColor = editorBackground}},
        .cornerRadius = config.rounding * ctx.pixelRatio,
    });

    const bool consoleOn = visible && impl->consoleExpanded();
    const auto panel = impl->consolePanelRect();
    const auto header = impl->consoleHeaderRect();

    impl->consoleView.setVisible(consoleOn);
    if (consoleOn) {
        impl->consoleView.update({
            .id = config.id + ":console",
            .value = impl->consoleText(),
            .fontSize = fontSizePixels,
            .fontScale = Typography::ConsoleScale,
            .stickToBottom = true,
            .wrap = TextGrid::Wrap::Character,
            .backgroundColorKey = "editor_console_background",
            .textColorKey = "editor_text",
            .selectionColorKey = "editor_selection",
            .selectionMatchColorKey = "editor_selection_match",
            .scrollbarTrackColorKey = "editor_scrollbar_track",
            .scrollbarThumbColorKey = "editor_scrollbar_thumb",
        });
    }

    impl->consoleChrome.update({
        .instances = {
            {.rect = header,
             .visible = consoleOn,
             .backgroundColor = ctx.color("editor_console_background")},
            {.rect = {viewRect.x, std::floor(header.y), viewRect.width, outline},
             .visible = consoleOn,
             .backgroundColor = ctx.color("editor_console_header_outline")},
        },
    });
    const F32 gripWidth = kConsoleGripWidth * pixelRatio;
    const F32 gripHeight = std::max(2.0f, std::round(kConsoleGripHeight * pixelRatio));
    impl->consoleGrip.update({
        .instances = {{
            .rect = {std::round(header.x + (header.width - gripWidth) * 0.5f),
                     std::round(header.y + (header.height - gripHeight) * 0.5f),
                     gripWidth,
                     gripHeight},
            .visible = consoleOn,
            .backgroundColor = ctx.color("editor_console_header_outline"),
        }},
        .cornerRadius = gripHeight * 0.5f,
    });

    impl->statusBar.update({
        .id = config.id + ":status",
        .status = config.status,
        .tone = config.statusTone,
        .toggleText = impl->statusConsoleToggleText(),
        .toggleExpanded = !impl->consoleCollapsed,
        .fontSize = fontSizePixels,
        .cornerRadius = config.rounding * pixelRatio,
        .onToggle = [this]() {
            impl->consoleCollapsed = !impl->consoleCollapsed;
            invalidate(Dirty::Layout);
        },
    });

    layoutChild(ctx, impl->backgroundBox, frame());
    layoutChild(ctx, impl->textEditor, impl->editorAreaRect());
    if (impl->config.onContentOverflow) {
        const auto& metrics = impl->textEditor.metrics();
        const Rect area = impl->editorAreaRect();
        const F32 ratio = std::max(1e-3f, pixelRatio);
        const Extent2D<F32> overflow = {(metrics.contentWidth - area.width) / ratio,
                                        (metrics.contentHeight - area.height) / ratio};
        if (impl->reportedOverflow != overflow) {
            impl->reportedOverflow = overflow;
            impl->config.onContentOverflow(overflow);
        }
    }
    if (consoleOn) {
        layoutChild(ctx, impl->consoleView, panel);
    }
    layoutChild(ctx, impl->consoleChrome, frame());
    layoutChild(ctx, impl->consoleGrip, frame());
    layoutChild(ctx, impl->statusBar, impl->statusBarRect());
    UpdateCornerMask(ctx, impl->cornerMask, config.rounding > 0.0f ? Rect{} : viewRect, clip());
}

std::optional<SurfaceCursor> CodeEditor::cursorOverChildren(const Extent2D<F32>& point) const {
    if (impl->consoleExpanded() && impl->consoleHeaderRect().contains(point.x, point.y)) {
        return SurfaceCursor::ResizeNS;
    }
    return std::nullopt;
}

bool CodeEditor::mouseEvent(const MouseEvent& event) {
    if (impl->consoleResizing && impl->handleChromeMouse(*this, event)) {
        return true;
    }
    if (mouseEventChildren(event)) {
        return true;
    }
    return impl->handleChromeMouse(*this, event);
}

}  // namespace Jetstream::Sakura
