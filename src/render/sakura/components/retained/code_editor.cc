#include <jetstream/render/sakura/components/retained/code_editor.hh>

#include <jetstream/render/sakura/components/retained/box.hh>
#include <jetstream/render/sakura/components/retained/canvas.hh>
#include <jetstream/render/sakura/components/retained/status_bar.hh>
#include <jetstream/render/sakura/components/retained/text_editor.hh>
#include <jetstream/render/sakura/components/retained/text_view.hh>

#include "../../helpers.hh"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream::Sakura::Retained {

namespace {

constexpr F32 kReferenceFontSize = Typography::FontSize;
constexpr F32 kConsoleFontScale = 0.92f;
constexpr F32 kConsoleHeaderHeight = 12.0f;
constexpr F32 kConsoleGripWidth = 28.0f;
constexpr F32 kConsoleGripHeight = 3.0f;
constexpr F32 kConsoleDefaultHeight = 128.0f;
constexpr F32 kConsoleMinHeight = 72.0f;
constexpr F32 kConsoleMaxHeightRatio = 0.55f;

std::string JoinLines(const std::vector<std::string>& lines) {
    std::string value;
    for (U64 i = 0; i < lines.size(); ++i) {
        if (i > 0) {
            value += '\n';
        }
        value += lines[i];
    }
    return value;
}

TextEditor::Language ToEditorLanguage(CodeEditor::Language language) {
    return language == CodeEditor::Language::Markdown ? TextEditor::Language::Markdown
                                                      : TextEditor::Language::Python;
}

}  // namespace

struct CodeEditorRoot : public Component {
    CodeEditor::Config config;

    Box backgroundBox;
    TextEditor textEditor;
    TextView consoleView;
    Box consoleChrome;
    Box consoleGrip;
    StatusBar statusBar;

    Rect viewRect;
    F32 fontSizePixels = kReferenceFontSize;
    bool consoleCollapsed = true;
    bool consoleResizing = false;
    F32 consoleResizeHeightPixels = 0.0f;
    F32 consoleHeight = kConsoleDefaultHeight;

    CodeEditorRoot() {
        setClipsChildren(true);
        add(backgroundBox);
        add(textEditor);
        add(consoleView);
        add(consoleChrome);
        add(consoleGrip);
        add(statusBar);
    }

    F32 pixelRatio() const {
        return config.editorFontSize > 0.0f ? fontSizePixels / config.editorFontSize : 1.0f;
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
    F32 barHeightPixels() const { return StatusBar::HeightPixels(pixelRatio()); }

    bool consoleVisible() const { return config.consoleVisible; }
    bool consoleExpanded() const { return consoleVisible() && !consoleCollapsed; }
    bool statusVisible() const { return !config.status.empty() || consoleVisible(); }

    F32 statusBarHeightPixels() const { return statusVisible() ? barHeightPixels() : 0.0f; }
    F32 statusBarTopPixels() const { return std::max(0.0f, viewRect.bottom() - statusBarHeightPixels()); }
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
        const F32 contentHeight = editorContentBottomPixels();
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

    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override {
        const F32 editorContentPx = measureChild(textEditor, ctx, available).y;

        const F32 lineHeight = config.editorFontSize * Typography::CodeLineHeight;
        const F32 pad = verticalPaddingLogical();
        const F32 editorContentLogical = editorContentPx / std::max(1e-3f, pixelRatio());
        const F32 textLogical = std::max(0.0f, editorContentLogical - pad);
        const F32 consoleLogical = consoleExpanded()
            ? kConsoleHeaderHeight + consoleHeightPixels() / std::max(1e-3f, pixelRatio())
            : 0.0f;
        const F32 statusLogical = statusVisible() ? StatusBar::Height : 0.0f;
        const F32 contentHeight = pad + textLogical + consoleLogical + statusLogical;

        const F32 viewportHeight = ImGui::GetMainViewport() ? ImGui::GetMainViewport()->WorkSize.y : 0.0f;
        const F32 maxHeight = Unscale(ctx, viewportHeight * std::clamp(config.maxAutoHeightWindowRatio, 0.0f, 1.0f));
        const F32 minHeight = pad + lineHeight;
        const F32 desiredLogical = std::max(minHeight, maxHeight > 0.0f ? std::min(contentHeight, maxHeight) : contentHeight);

        return {available.x, desiredLogical * pixelRatio()};
    }

    bool handleChromeMouse(const MouseEvent& event) {
        switch (event.type) {
            case MouseEventType::Click: {
                if (event.button != MouseButton::Left) {
                    return false;
                }
                if (consoleExpanded() && consoleHeaderRect().contains(event.position.x, event.position.y)) {
                    consoleResizing = true;
                    consoleResizeHeightPixels = consoleHeightPixels();
                    return true;
                }
                return false;
            }
            case MouseEventType::Move: {
                if (consoleResizing) {
                    ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
                    consoleResizeHeightPixels = editorContentBottomPixels() - event.position.y -
                                                consoleDividerHeightPixels() * 0.5f;
                    return true;
                }
                if (consoleExpanded() && consoleHeaderRect().contains(event.position.x, event.position.y)) {
                    ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
                }
                return false;
            }
            case MouseEventType::Release: {
                if (event.button != MouseButton::Left || !consoleResizing) {
                    return false;
                }
                consoleHeight = consoleHeightPixels() / std::max(0.1f, pixelRatio());
                consoleResizing = false;
                return true;
            }
            default:
                return false;
        }
    }

    void layout(const Context& ctx) override {
        fontSizePixels = config.editorFontSize * ctx.pixelRatio;
        viewRect = frame();

        const bool visible = !viewRect.empty();
        const auto editorBackground = ctx.color(config.backgroundColorKey);
        const F32 outline = outlineHeightPixels();

        backgroundBox.update({
            .id = config.id + ":background",
            .instances = {{.rect = viewRect, .visible = visible, .backgroundColor = editorBackground}},
        });

        textEditor.update({
            .id = config.id + ":text",
            .value = config.value,
            .fontSize = fontSizePixels,
            .lineNumbers = config.lineNumbers,
            .showActiveLine = config.showActiveLine,
            .wrap = config.language == CodeEditor::Language::Markdown ? TextGrid::Wrap::Word
                    : config.lineWrapping                            ? TextGrid::Wrap::Character
                                                                     : TextGrid::Wrap::None,
            .padding = editorPaddingPixels(),
            .language = ToEditorLanguage(config.language),
            .backgroundColorKey = config.backgroundColorKey,
            .textColorKey = "editor_text",
            .lineNumberColorKey = "editor_line_number",
            .gutterSeparatorColorKey = "editor_gutter_separator",
            .selectionColorKey = "editor_selection",
            .selectionMatchColorKey = "editor_selection_match",
            .cursorColorKey = "editor_cursor",
            .scrollbarTrackColorKey = "editor_scrollbar_track",
            .scrollbarThumbColorKey = "editor_scrollbar_thumb",
            .onChange = config.onChange,
            .onSubmit = config.onSubmit,
        });

        const bool consoleOn = visible && consoleExpanded();
        const auto panel = consolePanelRect();
        const auto header = consoleHeaderRect();

        consoleView.setVisible(consoleOn);
        if (consoleOn) {
            consoleView.update({
                .id = config.id + ":console",
                .value = JoinLines(config.consoleOutput),
                .fontSize = fontSizePixels,
                .fontScale = kConsoleFontScale,
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

        consoleChrome.update({
            .id = config.id + ":console-chrome",
            .instances = {
                {.rect = header,
                 .visible = consoleOn,
                 .backgroundColor = ctx.color("editor_console_background")},
                {.rect = {viewRect.x, std::floor(header.y), viewRect.width, outline},
                 .visible = consoleOn,
                 .backgroundColor = ctx.color("editor_console_header_outline")},
            },
        });
        const F32 gripWidth = kConsoleGripWidth * pixelRatio();
        const F32 gripHeight = std::max(2.0f, std::round(kConsoleGripHeight * pixelRatio()));
        consoleGrip.update({
            .id = config.id + ":console-grip",
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

        statusBar.update({
            .id = config.id + ":status",
            .status = config.status,
            .tone = config.statusTone,
            .toggleText = statusConsoleToggleText(),
            .toggleExpanded = !consoleCollapsed,
            .fontSize = fontSizePixels,
            .pixelRatio = pixelRatio(),
            .onToggle = [this]() {
                consoleCollapsed = !consoleCollapsed;
            },
        });

        layoutChild(ctx, backgroundBox, frame());
        layoutChild(ctx, textEditor, editorAreaRect());
        if (consoleOn) {
            layoutChild(ctx, consoleView, panel);
        }
        layoutChild(ctx, consoleChrome, frame());
        layoutChild(ctx, consoleGrip, frame());
        layoutChild(ctx, statusBar, statusBarRect());
    }

    bool event(const MouseEvent& event) override {
        if (consoleResizing && handleChromeMouse(event)) {
            return true;
        }
        if (eventChildren(event)) {
            return true;
        }
        return handleChromeMouse(event);
    }
};

struct CodeEditor::Impl {
    Config config;

    Canvas canvas;
    CodeEditorRoot root;

    Impl() {
        canvas.mount(root);
    }
};

CodeEditor::CodeEditor() {
    this->impl = std::make_unique<Impl>();
}

CodeEditor::~CodeEditor() = default;
CodeEditor::CodeEditor(CodeEditor&&) noexcept = default;
CodeEditor& CodeEditor::operator=(CodeEditor&&) noexcept = default;

bool CodeEditor::update(Config config) {
    impl->config = std::move(config);

    impl->canvas.update({
        .id = impl->config.id + ":canvas",
        .size = impl->config.size,
        .autoHeight = impl->config.autoHeight,
    });

    impl->root.config = impl->config;
    return true;
}

void CodeEditor::render(const Sakura::Context& ctx) {
    impl->canvas.render(ctx);
}

}  // namespace Jetstream::Sakura::Retained
