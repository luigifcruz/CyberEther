#include <jetstream/render/sakura/components/retained/console.hh>

#include <jetstream/render/sakura/components/retained/box.hh>
#include <jetstream/render/sakura/components/retained/canvas.hh>
#include <jetstream/render/sakura/components/retained/label.hh>
#include <jetstream/render/sakura/components/retained/status_bar.hh>
#include <jetstream/render/sakura/components/retained/text_view.hh>

#include "../../helpers.hh"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>

namespace Jetstream::Sakura::Retained {

namespace {

constexpr F32 kReferenceFontSize = Typography::FontSize;
constexpr F32 kOutputFontScale = 0.92f;

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

}  // namespace

struct ConsoleRoot : public Component {
    Console::Config config;

    Box backgroundBox;
    TextView outputView;
    Label emptyLabel;
    StatusBar statusBar;

    Rect viewRect;
    F32 fontSizePixels = kReferenceFontSize;

    ConsoleRoot() {
        setClipsChildren(true);
        add(backgroundBox);
        add(outputView);
        add(emptyLabel);
        add(statusBar);
    }

    F32 pixelRatio() const {
        return config.fontSize > 0.0f ? fontSizePixels / config.fontSize : 1.0f;
    }
    bool statusVisible() const { return !config.status.empty(); }
    F32 statusBarHeightPixels() const {
        return statusVisible() ? StatusBar::HeightPixels(pixelRatio()) : 0.0f;
    }
    Rect outputRect() const {
        return {viewRect.x, viewRect.y, viewRect.width,
                std::max(0.0f, viewRect.height - statusBarHeightPixels())};
    }
    Rect statusBarRect() const {
        const F32 height = statusBarHeightPixels();
        return {viewRect.x, std::max(viewRect.y, viewRect.bottom() - height), viewRect.width, height};
    }

    void layout(const Context& ctx) override {
        fontSizePixels = config.fontSize * ctx.pixelRatio;
        viewRect = frame();

        const bool visible = !viewRect.empty();

        backgroundBox.update({
            .id = config.id + ":background",
            .instances = {{.rect = viewRect, .visible = visible, .backgroundColor = ctx.color(config.backgroundColorKey)}},
        });

        outputView.update({
            .id = config.id + ":output",
            .value = JoinLines(config.output),
            .fontSize = fontSizePixels,
            .fontScale = kOutputFontScale,
            .stickToBottom = true,
            .wrap = TextGrid::Wrap::Character,
            .backgroundColorKey = config.backgroundColorKey,
            .textColorKey = "editor_text",
            .selectionColorKey = "editor_selection",
            .selectionMatchColorKey = "editor_selection_match",
            .scrollbarTrackColorKey = "editor_scrollbar_track",
            .scrollbarThumbColorKey = "editor_scrollbar_thumb",
        });

        const auto output = outputRect();
        emptyLabel.update({
            .id = config.id + ":empty",
            .instances = {{
                .rect = output,
                .str = config.emptyText,
                .visible = visible && config.output.empty() && !config.emptyText.empty(),
                .color = ctx.color("editor_line_number"),
                .fontSize = fontSizePixels * kOutputFontScale,
                .alignment = {1, 1},
            }},
            .fontName = Typography::MonoFont,
        });

        statusBar.update({
            .id = config.id + ":status",
            .status = config.status,
            .tone = config.statusTone,
            .fontSize = fontSizePixels,
            .pixelRatio = pixelRatio(),
        });

        layoutChild(ctx, backgroundBox, viewRect);
        layoutChild(ctx, outputView, output);
        layoutChild(ctx, emptyLabel, output);
        layoutChild(ctx, statusBar, statusBarRect());
    }

    bool event(const MouseEvent& event) override {
        return eventChildren(event);
    }
};

struct Console::Impl {
    Config config;

    Canvas canvas;
    ConsoleRoot root;

    Impl() {
        canvas.mount(root);
    }
};

Console::Console() {
    this->impl = std::make_unique<Impl>();
}

Console::~Console() = default;
Console::Console(Console&&) noexcept = default;
Console& Console::operator=(Console&&) noexcept = default;

bool Console::update(Config config) {
    impl->config = std::move(config);

    impl->canvas.update({
        .id = impl->config.id + ":canvas",
        .size = impl->config.size,
    });

    impl->root.config = impl->config;
    return true;
}

void Console::render(const Sakura::Context& ctx) {
    impl->canvas.render(ctx);
}

}  // namespace Jetstream::Sakura::Retained
