#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <jetstream/render/sakura/components/retained/text_grid.hh>
#include <jetstream/render/tools/imgui.h>

#include "harness.hh"
#include "render/sakura/context.hh"
#include "render/sakura/retained/text_grid_viewport.hh"

#include <limits>
#include <string>

using namespace Jetstream;

namespace {

struct MeasuredTextGrid : Sakura::Retained::TextGrid {
    using TextGrid::measure;
    using Component::frame;
    using Component::clip;
};

struct TextGridHost : Sakura::Component {
    Sakura::Retained::TextGrid grid;

    TextGridHost() {
        add(grid);
    }

    void place(const Sakura::Context& ctx, Jetstream::Rect frame) {
        layoutRoot(ctx, frame);
    }

    bool send(MouseEventType type, F32 x, F32 y) {
        return eventChildren({.type = type, .button = MouseButton::Left, .position = {x, y}});
    }

    bool wheel(F32 dx, F32 dy, F32 x, F32 y) {
        return eventChildren({.type = MouseEventType::Scroll, .position = {x, y}, .scroll = {dx, dy}});
    }

 protected:
    void layout(const Sakura::Context& ctx) override {
        layoutChild(ctx, grid, frame());
    }
};

struct ClippedTextGridHost : Sakura::Component {
    MeasuredTextGrid grid;
    Jetstream::Rect document;

    ClippedTextGridHost() {
        setClipsChildren(true);
        add(grid);
    }

    void place(const Sakura::Context& ctx, Jetstream::Rect frame) {
        layoutRoot(ctx, frame);
    }

 protected:
    void layout(const Sakura::Context& ctx) override {
        layoutChild(ctx, grid, document);
    }
};

std::string Document(U64 lines) {
    std::string text;
    for (U64 i = 0; i < lines; ++i) {
        if (i > 0) {
            text += '\n';
        }
        text += "Thinking through the next step.";
    }
    return text;
}

struct ImGuiContextGuard {
    ImGuiContextGuard() { ImGui::CreateContext(); }
    ~ImGuiContextGuard() { ImGui::DestroyContext(); }
};

}  // namespace

TEST_CASE("Retained text line metrics follow live scale changes",
          "[core][sakura][text-grid][scale]") {
    using TextGrid = Sakura::Retained::TextGrid;
    const auto wrap = GENERATE(TextGrid::Wrap::None, TextGrid::Wrap::Word);
    const Sakura::Context ctx;
    MeasuredTextGrid grid;
    TextGrid::Config config{
        .id = "scaled-note",
        .value = "Heading\nBody text\nList item",
        .wrap = wrap,
        .lineScale = {1.5f, 1.0f, 1.0f},
        .lineTopGap = {0.0f, 9.0f, 3.0f},
        .lineIndent = {0.0f, 0.0f, 21.0f},
    };
    grid.update(config);
    const F32 originalHeight = grid.measure(ctx, {300.0f, 600.0f}).y;
    REQUIRE(originalHeight > 0.0f);

    // Scaling the panel with its font preserves the wrap column count, but
    // cached row heights and paragraph gaps still need to be recalculated.
    for (const F32 scale : {2.0f, 0.5f, 1.25f, 1.0f}) {
        CAPTURE(wrap, scale);
        config.fontSize = 15.0f * scale;
        config.lineTopGap = {0.0f, 9.0f * scale, 3.0f * scale};
        config.lineIndent = {0.0f, 0.0f, 21.0f * scale};
        grid.update(config);
        CHECK(grid.measure(ctx, {300.0f * scale, 600.0f * scale}).y ==
              Catch::Approx(originalHeight * scale));
    }
}

TEST_CASE("Vertical cursor moves stay on short wrapped rows",
          "[core][sakura][text-grid][cursor]") {
    using TextGrid = Sakura::Retained::TextGrid;
    const ImGuiContextGuard imguiContext;
    const Sakura::Context ctx;
    TextGridHost host;
    host.grid.update({
        .id = "wrapped-cursor",
        .value = "aaa bbbbbbbb",
        .editable = true,
        .fontSize = 15.0f,
        .scrollbar = false,
        .wrap = TextGrid::Wrap::Word,
        .padding = Sakura::Padding{0.0f, 0.0f, 0.0f, 0.0f},
    });
    host.place(ctx, {0.0f, 0.0f, 75.0f, 200.0f});

    host.grid.setCursor({0, 10});
    REQUIRE(host.grid.cursor() == TextGrid::Position{0, 10});

    host.grid.moveCursorRows(-1);
    CHECK(host.grid.cursor() == TextGrid::Position{0, 3});

    host.grid.moveCursorRows(-1);
    CHECK(host.grid.cursor() == TextGrid::Position{0, 3});

    host.grid.moveCursorRows(1);
    CHECK(host.grid.cursor() == TextGrid::Position{0, 10});
}

TEST_CASE("Mouse drags past a wrapped row edge select its last character",
          "[core][sakura][text-grid][cursor]") {
    using TextGrid = Sakura::Retained::TextGrid;
    const ImGuiContextGuard imguiContext;
    const Sakura::Context ctx;
    TextGridHost host;
    std::optional<std::pair<TextGrid::Position, TextGrid::Position>> selected;
    host.grid.update({
        .id = "wrapped-drag",
        .value = "abcdefghijk",
        .editable = true,
        .fontSize = 15.0f,
        .scrollbar = false,
        .wrap = TextGrid::Wrap::Character,
        .padding = Sakura::Padding{0.0f, 0.0f, 0.0f, 0.0f},
        .onSelect = [&](TextGrid::Position start, TextGrid::Position end) {
            selected = std::pair{start, end};
        },
    });
    host.place(ctx, {0.0f, 0.0f, 75.0f, 200.0f});

    REQUIRE(host.send(MouseEventType::Click, 1.0f, 5.0f));
    REQUIRE(host.grid.cursor() == TextGrid::Position{0, 0});

    REQUIRE(host.send(MouseEventType::Move, 100.0f, 5.0f));
    REQUIRE(host.send(MouseEventType::Release, 100.0f, 5.0f));

    REQUIRE(selected.has_value());
    CHECK(selected->first == TextGrid::Position{0, 0});
    CHECK(selected->second == TextGrid::Position{0, 10});
}

TEST_CASE("Vertical cursor moves forget their pixel column when metrics change",
          "[core][sakura][text-grid][cursor]") {
    using TextGrid = Sakura::Retained::TextGrid;
    const ImGuiContextGuard imguiContext;
    const Sakura::Context ctx;
    TextGridHost host;
    TextGrid::Config config{
        .id = "rescaled-cursor",
        .value = "aaaaaaaaaa\nbbbbbbbbbb",
        .editable = true,
        .fontSize = 15.0f,
        .scrollbar = false,
        .padding = Sakura::Padding{0.0f, 0.0f, 0.0f, 0.0f},
    };
    host.grid.update(config);
    host.place(ctx, {0.0f, 0.0f, 300.0f, 200.0f});

    host.grid.setCursor({1, 6});
    host.grid.moveCursorRows(-1);
    REQUIRE(host.grid.cursor() == TextGrid::Position{0, 6});

    config.fontSize = 30.0f;
    host.grid.update(config);
    host.place(ctx, {0.0f, 0.0f, 300.0f, 200.0f});

    host.grid.moveCursorRows(1);
    CHECK(host.grid.cursor() == TextGrid::Position{1, 6});
}

TEST_CASE("Monospace wrapping counts characters rather than bytes",
          "[core][sakura][text-grid][wrap]") {
    using TextGrid = Sakura::Retained::TextGrid;
    const auto wrap = GENERATE(TextGrid::Wrap::Character, TextGrid::Wrap::Word);
    const Sakura::Context ctx;
    MeasuredTextGrid ascii;
    MeasuredTextGrid accented;
    TextGrid::Config config{
        .id = "wrap-bytes",
        .value = "eeee eeee",
        .fontSize = 15.0f,
        .wrap = wrap,
        .padding = Sakura::Padding{0.0f, 0.0f, 0.0f, 0.0f},
    };
    ascii.update(config);
    config.value = "\xC3\xA9\xC3\xA9\xC3\xA9\xC3\xA9 \xC3\xA9\xC3\xA9\xC3\xA9\xC3\xA9";
    accented.update(config);

    for (F32 width = 20.0f; width <= 200.0f; width += 5.0f) {
        CAPTURE(wrap, width);
        CHECK(accented.measure(ctx, {width, 600.0f}).y ==
              Catch::Approx(ascii.measure(ctx, {width, 600.0f}).y));
    }
}

TEST_CASE("Vertical cursor moves climb wrapped rows ending in multibyte characters",
          "[core][sakura][text-grid][cursor]") {
    using TextGrid = Sakura::Retained::TextGrid;
    const ImGuiContextGuard imguiContext;
    const Sakura::Context ctx;
    TextGridHost host;
    std::string value;
    for (U64 i = 0; i < 40; ++i) {
        value += "\xC3\xA9";
    }
    value += "\neeeeeeeeee";
    host.grid.update({
        .id = "wrapped-multibyte",
        .value = value,
        .editable = true,
        .fontSize = 15.0f,
        .scrollbar = false,
        .wrap = TextGrid::Wrap::Character,
        .padding = Sakura::Padding{0.0f, 0.0f, 0.0f, 0.0f},
    });
    host.place(ctx, {0.0f, 0.0f, 75.0f, 400.0f});

    host.grid.setCursor({1, 10});
    host.grid.moveCursorRows(-1);
    REQUIRE(host.grid.cursor() == TextGrid::Position{0, 80});

    host.grid.moveCursorRows(-1);
    CHECK(host.grid.cursor() == TextGrid::Position{0, 58});
}

TEST_CASE("Retained text grids keep document geometry independent of parent clipping",
          "[core][sakura][text-grid][viewport]") {
    using TextGrid = Sakura::Retained::TextGrid;
    const ImGuiContextGuard imguiContext;
    const Sakura::Context ctx;
    ClippedTextGridHost host;
    const Jetstream::Rect window{0.0f, 0.0f, 400.0f, 300.0f};

    TextGrid::Config config{
        .id = "long-thought",
        .value = Document(2048),
        .scrollbar = false,
    };
    const auto expand = [&] {
        host.grid.update(config);
        host.document = {0.0f, 0.0f, 400.0f,
            host.grid.measure(ctx, {400.0f, std::numeric_limits<F32>::infinity()}).y};
    };

    expand();
    host.place(ctx, window);
    REQUIRE(host.grid.metrics().contentHeight > 30000.0f);

    for (const F32 scroll : {0.0f, 9.0f, 300.0f, 15000.0f, host.document.height - 300.0f}) {
        CAPTURE(scroll);
        host.document.y = -scroll;
        host.place(ctx, window);
        CHECK(host.grid.frame().height == host.document.height);
        CHECK(host.grid.clip().height == Catch::Approx(300.0f));
        CHECK(host.grid.metrics().contentHeight == host.document.height);
        CHECK(host.grid.metrics().sourceLines.size() == 2048);
    }

    config.value = Document(4096);
    expand();
    host.place(ctx, window);
    CHECK(host.grid.metrics().contentHeight > 60000.0f);
    CHECK(host.grid.clip().height == Catch::Approx(300.0f));

    host.document = {};
    host.place(ctx, window);
    CHECK(Sakura::Retained::Intersect(host.grid.frame(), host.grid.clip()).empty());
    expand();
    host.place(ctx, window);
    CHECK(host.grid.metrics().contentHeight == host.document.height);
}

TEST_CASE("Text grid pools use inherited clipping and survive scrolling and collapse",
          "[core][sakura][text-grid][viewport]") {
    Sakura::Retained::TextGridViewport viewport;
    const Jetstream::Rect clip{0.0f, 0.0f, 400.0f, 300.0f};
    constexpr F32 lineHeight = 17.25f;
    for (const F32 height : {35000.0f, 70000.0f}) {
        for (const F32 top : {0.0f, -9.0f, -1000.0f, 150.0f, 350.0f}) {
            CAPTURE(height, top);
            viewport.update({0.0f, top, 400.0f, height}, clip, lineHeight, 1, 64);
            CHECK(viewport.bounds.height <= clip.height);
            CHECK(viewport.rowCapacity == 64);
        }
    }

    viewport.update({}, clip, lineHeight, 1, 64);
    CHECK(viewport.bounds.empty());
    CHECK(viewport.rowCapacity == 64);
    viewport.update({0.0f, 0.0f, 400.0f, 70000.0f}, clip, lineHeight, 1, 64);
    CHECK(viewport.bounds == clip);
    CHECK(viewport.rowCapacity == 64);
}

TEST_CASE("Text grid pools preserve viewport-sized reserves and same-row columns",
          "[core][sakura][text-grid][viewport]") {
    Sakura::Retained::TextGridViewport viewport;
    const Jetstream::Rect clip{0.0f, 0.0f, 400.0f, 300.0f};
    constexpr F32 lineHeight = 17.25f;

    viewport.update(clip, clip, lineHeight, 1, 64);
    REQUIRE(viewport.rowCapacity == 64);

    viewport.update({0.0f, 0.0f, 400.0f, 35000.0f}, clip, lineHeight, 4, 64);
    REQUIRE(viewport.rowCapacity > 64);
    REQUIRE(viewport.rowCapacity <= 96);
    const U64 capacity = viewport.rowCapacity;

    viewport.update({0.0f, -1000.0f, 400.0f, 35000.0f}, clip, lineHeight, 4, 64);
    CHECK(viewport.rowCapacity == capacity);
    viewport.update({0.0f, 150.0f, 400.0f, 35000.0f}, clip, lineHeight, 4, 64);
    CHECK(viewport.rowCapacity == capacity);
    viewport.update({}, clip, lineHeight, 1, 64);
    CHECK(viewport.rowCapacity == capacity);
}

TEST_CASE("Text grid segment pools grow by visible demand in row sized steps",
          "[core][sakura][text-grid][viewport]") {
    Sakura::Retained::TextGridViewport viewport;
    const Jetstream::Rect clip{0.0f, 0.0f, 400.0f, 300.0f};
    viewport.update(clip, clip, 17.25f, 1, 64);
    REQUIRE(viewport.rowCapacity == 64);

    CHECK(viewport.poolCapacity(0, 0, 32) == 64);
    CHECK(viewport.poolCapacity(0, 64, 32) == 64);
    CHECK(viewport.poolCapacity(0, 65, 32) == 128);
    CHECK(viewport.poolCapacity(128, 3, 32) == 128);
    CHECK(viewport.poolCapacity(0, 5000, 32) == 64 * 32);
    CHECK(viewport.poolCapacity(0, 65, 1) == 64);
}

TEST_CASE("Same-row lines share a band and resolve mouse hits by column",
          "[core][sakura][text-grid][row-group]") {
    const ImGuiContextGuard imguiContext;
    const Sakura::Context ctx;
    TextGridHost host;
    host.grid.update({
        .id = "row-group",
        .value = "aaaa\nbbbb\ncccc\ndddd",
        .editable = true,
        .fontSize = 15.0f,
        .scrollbar = false,
        .padding = Sakura::Padding{0.0f, 0.0f, 0.0f, 0.0f},
        .lineScale = {1.0f, 2.0f, 1.0f, 1.0f},
        .lineIndent = {0.0f, 100.0f, 200.0f, 0.0f},
        .lineSameRow = {0, 1, 1, 0},
    });
    host.place(ctx, {0.0f, 0.0f, 400.0f, 200.0f});

    const auto& metrics = host.grid.metrics();
    REQUIRE(metrics.sourceLines.size() == 4);
    const F32 rowHeight = metrics.sourceLines[0].height;
    REQUIRE(rowHeight > 0.0f);
    CHECK(metrics.sourceLines[1].top == metrics.sourceLines[0].top);
    CHECK(metrics.sourceLines[2].top == metrics.sourceLines[0].top);
    CHECK(metrics.sourceLines[1].height == Catch::Approx(2.0f * rowHeight));
    CHECK(metrics.sourceLines[3].top == Catch::Approx(metrics.sourceLines[0].top + 2.0f * rowHeight));
    CHECK(metrics.contentHeight == Catch::Approx(3.0f * rowHeight));

    REQUIRE(host.send(MouseEventType::Click, 5.0f, 5.0f));
    CHECK(host.grid.cursor().line == 0);
    REQUIRE(host.send(MouseEventType::Release, 5.0f, 5.0f));

    REQUIRE(host.send(MouseEventType::Click, 105.0f, 5.0f));
    CHECK(host.grid.cursor().line == 1);
    REQUIRE(host.send(MouseEventType::Release, 105.0f, 5.0f));

    REQUIRE(host.send(MouseEventType::Click, 205.0f, 5.0f));
    CHECK(host.grid.cursor().line == 2);
    REQUIRE(host.send(MouseEventType::Release, 205.0f, 5.0f));

    REQUIRE(host.send(MouseEventType::Click, 5.0f, 2.0f * rowHeight + 5.0f));
    CHECK(host.grid.cursor().line == 3);
    REQUIRE(host.send(MouseEventType::Release, 5.0f, 2.0f * rowHeight + 5.0f));
}

TEST_CASE("Cursor visibility follows the visual row of a same-row cell",
          "[core][sakura][text-grid][row-group]") {
    const ImGuiContextGuard imguiContext;
    const Sakura::Context ctx;
    TextGridHost host;
    host.grid.update({
        .id = "row-group-cursor",
        .value = "aaaa\nbbbb\ncccc\ndddd",
        .editable = true,
        .fontSize = 15.0f,
        .scrollbar = false,
        .padding = Sakura::Padding{0.0f, 0.0f, 0.0f, 0.0f},
        .lineScale = {1.0f, 1.0f, 1.0f, 1.0f},
        .lineIndent = {0.0f, 100.0f, 200.0f, 0.0f},
        .lineSameRow = {0, 1, 1, 0},
    });
    host.place(ctx, {0.0f, 0.0f, 400.0f, 20.0f});

    host.grid.setCursor({2, 0});
    host.place(ctx, {0.0f, 0.0f, 400.0f, 20.0f});
    CHECK(host.grid.metrics().sourceLines[2].top == Catch::Approx(0.0f));

    host.grid.setCursor({3, 0});
    host.place(ctx, {0.0f, 0.0f, 400.0f, 20.0f});
    const F32 rowHeight = host.grid.metrics().sourceLines[3].height;
    CHECK(host.grid.metrics().sourceLines[3].top == Catch::Approx(20.0f - rowHeight));
}

TEST_CASE("Monospace lines honor per-line wrap widths",
          "[core][sakura][text-grid][row-group]") {
    using TextGrid = Sakura::Retained::TextGrid;
    const Sakura::Context ctx;
    MeasuredTextGrid grid;
    TextGrid::Config config{
        .id = "mono-wrap-width",
        .value = "aaaabbbb",
        .fontSize = 15.0f,
        .wrap = TextGrid::Wrap::Character,
        .padding = Sakura::Padding{0.0f, 0.0f, 0.0f, 0.0f},
    };
    grid.update(config);
    const F32 single = grid.measure(ctx, {400.0f, 600.0f}).y;

    config.lineWrapWidth = {4.0f * 7.5f};
    grid.update(config);
    CHECK(grid.measure(ctx, {400.0f, 600.0f}).y == Catch::Approx(2.0f * single));

    config.lineScale = {2.0f};
    grid.update(config);
    CHECK(grid.measure(ctx, {400.0f, 600.0f}).y == Catch::Approx(8.0f * single));
}

TEST_CASE("Scaled monospace lines place columns with scaled advances",
          "[core][sakura][text-grid][row-group]") {
    using TextGrid = Sakura::Retained::TextGrid;
    const ImGuiContextGuard imguiContext;
    const Sakura::Context ctx;
    TextGridHost host;
    host.grid.update({
        .id = "scaled-mono-hit",
        .value = "aaaa\nbbbb",
        .editable = true,
        .fontSize = 15.0f,
        .scrollbar = false,
        .padding = Sakura::Padding{0.0f, 0.0f, 0.0f, 0.0f},
        .lineScale = {2.0f, 1.0f},
    });
    host.place(ctx, {0.0f, 0.0f, 400.0f, 200.0f});
    const auto& metrics = host.grid.metrics();
    REQUIRE(metrics.sourceLines[0].height == Catch::Approx(2.0f * metrics.sourceLines[1].height));

    REQUIRE(host.send(MouseEventType::Click, 30.0f, 5.0f));
    CHECK(host.grid.cursor() == TextGrid::Position{0, 2});
    REQUIRE(host.send(MouseEventType::Release, 30.0f, 5.0f));

    REQUIRE(host.send(MouseEventType::Click, 30.0f, metrics.sourceLines[1].top + 5.0f));
    CHECK(host.grid.cursor() == TextGrid::Position{1, 4});
    REQUIRE(host.send(MouseEventType::Release, 30.0f, metrics.sourceLines[1].top + 5.0f));

    host.grid.setCursor({1, 4});
    host.grid.moveCursorRows(-1);
    CHECK(host.grid.cursor() == TextGrid::Position{0, 2});
}

TEST_CASE("Clicks on a wrapped row boundary resolve to the row below",
          "[core][sakura][text-grid][cursor]") {
    using TextGrid = Sakura::Retained::TextGrid;
    const ImGuiContextGuard imguiContext;
    const Sakura::Context ctx;
    TextGridHost host;
    host.grid.update({
        .id = "wrapped-boundary",
        .value = "aaaa",
        .editable = true,
        .fontSize = 15.0f,
        .scrollbar = false,
        .wrap = TextGrid::Wrap::Character,
        .padding = Sakura::Padding{0.0f, 0.0f, 0.0f, 0.0f},
        .lineScale = {1.0f},
    });
    host.place(ctx, {0.0f, 0.0f, 15.0f, 200.0f});
    const F32 rowHeight = host.grid.metrics().sourceLines[0].height * 0.5f;
    REQUIRE(rowHeight > 0.0f);

    REQUIRE(host.send(MouseEventType::Click, 1.0f, rowHeight));
    CHECK(host.grid.cursor().column == 2);
    REQUIRE(host.send(MouseEventType::Release, 1.0f, rowHeight));

    host.place(ctx, {0.0f, 0.0f, 22.5f, 200.0f});
    REQUIRE(host.grid.metrics().sourceLines[0].height == Catch::Approx(2.0f * rowHeight));

    REQUIRE(host.send(MouseEventType::Click, 15.0f, rowHeight));
    CHECK(host.grid.cursor().column == 4);
    REQUIRE(host.send(MouseEventType::Release, 15.0f, rowHeight));

    REQUIRE(host.send(MouseEventType::Click, 15.0f, rowHeight - 1.0f));
    CHECK(host.grid.cursor().column == 2);
}

TEST_CASE("Vertical cursor moves cross row groups by horizontal position",
          "[core][sakura][text-grid][row-group]") {
    using TextGrid = Sakura::Retained::TextGrid;
    const ImGuiContextGuard imguiContext;
    const Sakura::Context ctx;
    TextGridHost host;
    host.grid.update({
        .id = "row-group-nav",
        .value = "aaaa\nbbbb\ncccc\ndddddddddddddddddddddddddddddd",
        .editable = true,
        .fontSize = 15.0f,
        .scrollbar = false,
        .padding = Sakura::Padding{0.0f, 0.0f, 0.0f, 0.0f},
        .lineScale = {1.0f, 1.0f, 1.0f, 1.0f},
        .lineIndent = {0.0f, 100.0f, 200.0f, 0.0f},
        .lineSameRow = {0, 1, 1, 0},
    });
    host.place(ctx, {0.0f, 0.0f, 400.0f, 200.0f});

    host.grid.setCursor({0, 0});
    host.grid.moveCursorRows(1);
    CHECK(host.grid.cursor() == TextGrid::Position{3, 0});

    host.grid.moveCursorRows(-1);
    CHECK(host.grid.cursor() == TextGrid::Position{0, 0});

    host.grid.setCursor({1, 2});
    host.grid.moveCursorRows(1);
    CHECK(host.grid.cursor().line == 3);
    CHECK(host.grid.cursor().column == 15);

    host.grid.moveCursorRows(-1);
    CHECK(host.grid.cursor() == TextGrid::Position{1, 2});

    host.grid.setCursor({3, 28});
    host.grid.moveCursorRows(-1);
    CHECK(host.grid.cursor() == TextGrid::Position{2, 1});

    host.grid.moveCursorRows(-1);
    CHECK(host.grid.cursor() == TextGrid::Position{2, 1});

    host.grid.setCursor({1, 2});
    host.grid.moveCursorRows(1);
    REQUIRE(host.grid.cursor().line == 3);
    host.place(ctx, {75.0f, 40.0f, 400.0f, 200.0f});
    host.grid.moveCursorRows(-1);
    CHECK(host.grid.cursor() == TextGrid::Position{1, 2});
}

TEST_CASE("Hit testing deep inside a long wrapped line resolves by row",
          "[core][sakura][text-grid][cursor]") {
    using TextGrid = Sakura::Retained::TextGrid;
    const ImGuiContextGuard imguiContext;
    const Sakura::Context ctx;
    TextGridHost host;
    host.grid.update({
        .id = "long-wrapped-hit",
        .value = std::string(400, 'a'),
        .editable = true,
        .fontSize = 15.0f,
        .scrollbar = false,
        .wrap = TextGrid::Wrap::Character,
        .padding = Sakura::Padding{0.0f, 0.0f, 0.0f, 0.0f},
        .lineScale = {1.0f},
    });
    host.place(ctx, {0.0f, 0.0f, 15.0f, 5000.0f});
    const F32 rowHeight = host.grid.metrics().sourceLines[0].height / 200.0f;
    REQUIRE(rowHeight > 0.0f);

    REQUIRE(host.send(MouseEventType::Click, 1.0f, 100.0f * rowHeight + 1.0f));
    CHECK(host.grid.cursor().column == 200);
    REQUIRE(host.send(MouseEventType::Release, 1.0f, 100.0f * rowHeight + 1.0f));

    host.grid.setCursor({0, 200});
    host.grid.moveCursorRows(1);
    CHECK(host.grid.cursor().column == 202);
    host.grid.moveCursorRows(-2);
    CHECK(host.grid.cursor().column == 198);
}

TEST_CASE("Horizontal scrolling reveals indented same-row cells",
          "[core][sakura][text-grid][row-group]") {
    const ImGuiContextGuard imguiContext;
    const Sakura::Context ctx;
    TextGridHost host;
    host.grid.update({
        .id = "row-group-hscroll",
        .value = "aaaa\nbbbb",
        .editable = true,
        .fontSize = 15.0f,
        .scrollbar = false,
        .padding = Sakura::Padding{0.0f, 0.0f, 0.0f, 0.0f},
        .lineScale = {1.0f, 1.0f},
        .lineIndent = {0.0f, 200.0f},
        .lineSameRow = {0, 1},
    });
    host.place(ctx, {0.0f, 0.0f, 100.0f, 50.0f});

    REQUIRE(host.send(MouseEventType::Click, 95.0f, 5.0f));
    CHECK(host.grid.cursor().line == 0);
    REQUIRE(host.send(MouseEventType::Release, 95.0f, 5.0f));

    host.grid.setCursor({1, 0});
    host.place(ctx, {0.0f, 0.0f, 100.0f, 50.0f});

    REQUIRE(host.send(MouseEventType::Click, 76.0f, 5.0f));
    CHECK(host.grid.cursor() == Sakura::Retained::TextGrid::Position{1, 0});
    REQUIRE(host.send(MouseEventType::Release, 76.0f, 5.0f));

    REQUIRE(host.send(MouseEventType::Click, 91.0f, 5.0f));
    CHECK(host.grid.cursor() == Sakura::Retained::TextGrid::Position{1, 2});
    REQUIRE(host.send(MouseEventType::Release, 91.0f, 5.0f));

    REQUIRE(host.send(MouseEventType::Click, 95.0f, 5.0f));
    CHECK(host.grid.cursor().line == 1);
}

TEST_CASE("Clipped grids with a long wrapped column lay out beside short cells",
          "[core][sakura][text-grid][row-group][viewport]") {
    using TextGrid = Sakura::Retained::TextGrid;
    const ImGuiContextGuard imguiContext;
    const Sakura::Context ctx;
    ClippedTextGridHost host;
    const Jetstream::Rect window{0.0f, 0.0f, 400.0f, 200.0f};
    host.grid.update({
        .id = "long-column",
        .value = std::string(4000, 'a') + "\nshort\nafter",
        .fontSize = 15.0f,
        .scrollbar = false,
        .wrap = TextGrid::Wrap::Character,
        .padding = Sakura::Padding{0.0f, 0.0f, 0.0f, 0.0f},
        .lineScale = {1.0f, 1.0f, 1.0f},
        .lineIndent = {0.0f, 100.0f, 0.0f},
        .lineSameRow = {0, 1, 0},
        .lineWrapWidth = {75.0f, 0.0f, 0.0f},
    });
    host.document = {0.0f, 0.0f, 400.0f,
        host.grid.measure(ctx, {400.0f, std::numeric_limits<F32>::infinity()}).y};
    host.place(ctx, window);

    const auto& metrics = host.grid.metrics();
    REQUIRE(metrics.sourceLines.size() == 3);
    const F32 rowHeight = metrics.sourceLines[1].height;
    CHECK(metrics.sourceLines[0].height == Catch::Approx(400.0f * rowHeight));
    CHECK(metrics.sourceLines[1].top == metrics.sourceLines[0].top);
    CHECK(metrics.sourceLines[2].top == Catch::Approx(metrics.sourceLines[0].top + 400.0f * rowHeight));

    for (const F32 scroll : {0.0f, 100.0f * rowHeight, 399.0f * rowHeight, 400.0f * rowHeight - 50.0f}) {
        CAPTURE(scroll);
        host.document.y = -scroll;
        host.place(ctx, window);
        CHECK(host.grid.metrics().sourceLines[2].top == Catch::Approx(400.0f * rowHeight - scroll));
    }
}

TEST_CASE("Line numbers label only the first cell of a same-row band",
          "[core][sakura][text-grid][row-group]") {
    const ImGuiContextGuard imguiContext;
    const Sakura::Context ctx;
    TextGridHost host;
    host.grid.update({
        .id = "row-group-numbers",
        .value = "aaaa\nbbbb\ncccc\ndddd",
        .editable = true,
        .fontSize = 15.0f,
        .lineNumbers = true,
        .scrollbar = false,
        .padding = Sakura::Padding{0.0f, 0.0f, 0.0f, 0.0f},
        .lineScale = {1.0f, 1.0f, 1.0f, 1.0f},
        .lineIndent = {0.0f, 100.0f, 200.0f, 0.0f},
        .lineSameRow = {0, 1, 1, 0},
    });
    host.place(ctx, {0.0f, 0.0f, 400.0f, 200.0f});

    const auto& metrics = host.grid.metrics();
    REQUIRE(metrics.sourceLines.size() == 4);
    CHECK(metrics.sourceLines[1].top == metrics.sourceLines[0].top);
    CHECK(metrics.sourceLines[2].top == metrics.sourceLines[0].top);
    CHECK(metrics.sourceLines[3].top > metrics.sourceLines[0].top);

    REQUIRE(host.send(MouseEventType::Click, 50.0f, 5.0f));
    CHECK(host.grid.cursor().line == 0);
    REQUIRE(host.send(MouseEventType::Release, 50.0f, 5.0f));

    REQUIRE(host.send(MouseEventType::Click, 150.0f, 5.0f));
    CHECK(host.grid.cursor().line == 1);
    REQUIRE(host.send(MouseEventType::Release, 150.0f, 5.0f));

    REQUIRE(host.send(MouseEventType::Click, 250.0f, 5.0f));
    CHECK(host.grid.cursor().line == 2);
}

TEST_CASE("Clearing the width layout hook restores static wrapping",
          "[core][sakura][text-grid][wrap]") {
    using TextGrid = Sakura::Retained::TextGrid;
    const Sakura::Context ctx;
    MeasuredTextGrid grid;
    TextGrid::Config config{
        .id = "hooked-wrap",
        .value = "aaaaaaaa",
        .fontSize = 15.0f,
        .scrollbar = false,
        .wrap = TextGrid::Wrap::Character,
        .padding = Sakura::Padding{0.0f, 0.0f, 0.0f, 0.0f},
        .widthLayout = [](F32) { return TextGrid::WidthLayout{.lineWrapWidth = {4.0f * 7.5f}}; },
    };
    grid.update(config);
    const F32 hooked = grid.measure(ctx, {400.0f, 600.0f}).y;

    config.widthLayout = nullptr;
    grid.update(config);
    CHECK(grid.measure(ctx, {400.0f, 600.0f}).y == Catch::Approx(hooked * 0.5f));
}

TEST_CASE("Horizontal overflow in an editor does not create vertical scrolling",
          "[core][sakura][text-grid][scroll]") {
    const ImGuiContextGuard imguiContext;
    const Sakura::Context ctx;
    TextGridHost host;
    host.grid.update({
        .id = "editor-overflow",
        .value = "abcdefghijk",
        .editable = true,
        .fontSize = 15.0f,
        .scrollbar = true,
        .padding = Sakura::Padding{0.0f, 0.0f, 0.0f, 0.0f},
    });
    host.place(ctx, {0.0f, 0.0f, 75.0f, 200.0f});

    const auto& metrics = host.grid.metrics();
    CHECK(metrics.contentWidth > 75.0f);
    CHECK(metrics.scrollbarGutter == 0.0f);
    REQUIRE_FALSE(host.wheel(0.0f, -5.0f, 30.0f, 100.0f));
    REQUIRE(host.wheel(-5.0f, 0.0f, 30.0f, 100.0f));
    host.place(ctx, {0.0f, 0.0f, 75.0f, 200.0f});
    CHECK(host.grid.metrics().scrollY == 0.0f);
    CHECK(host.grid.metrics().scrollX > 0.0f);
}

TEST_CASE("Cursor stays visible when a scrollbar appears and rewraps the text",
          "[core][sakura][text-grid][scroll]") {
    using TextGrid = Sakura::Retained::TextGrid;
    const ImGuiContextGuard imguiContext;
    const Sakura::Context ctx;
    TextGridHost host;
    TextGrid::Config config{
        .id = "rewrap-cursor",
        .value = "abcdefghijk",
        .editable = true,
        .fontSize = 15.0f,
        .scrollbar = true,
        .wrap = TextGrid::Wrap::Word,
        .padding = Sakura::Padding{0.0f, 0.0f, 0.0f, 0.0f},
    };
    const Jetstream::Rect frame{0.0f, 0.0f, 90.0f, 34.5f};
    host.grid.update(config);
    host.place(ctx, frame);
    REQUIRE(host.grid.metrics().scrollbarGutter == 0.0f);

    config.value = "abcdefghijk\nx";
    host.grid.update(config);
    host.grid.setCursor({1, 0});
    host.place(ctx, frame);

    const auto& metrics = host.grid.metrics();
    REQUIRE(metrics.scrollbarGutter > 0.0f);
    REQUIRE(metrics.sourceLines.size() == 2);
    CHECK(metrics.sourceLines[1].top >= 0.0f);
    CHECK(metrics.sourceLines[1].top + metrics.sourceLines[1].height <= frame.height + 1e-3f);

    const F32 previousTop = metrics.sourceLines[1].top;
    host.place(ctx, frame);
    CHECK(host.grid.metrics().sourceLines[1].top == Catch::Approx(previousTop));

    config.value = "abcdefghijkx";
    host.grid.update(config);
    host.place(ctx, frame);
    host.place(ctx, frame);
    const auto& joined = host.grid.metrics();
    CHECK(joined.scrollbarGutter == 0.0f);
    REQUIRE(joined.sourceLines.size() == 1);
    CHECK(joined.sourceLines[0].height == Catch::Approx(17.25f));
    CHECK(joined.scrollY == 0.0f);
}

TEST_CASE("Editable wrapped grids measure the height their layout will use",
          "[core][sakura][text-grid][scroll]") {
    using TextGrid = Sakura::Retained::TextGrid;
    const ImGuiContextGuard imguiContext;
    const Sakura::Context ctx;
    struct Host : TextGridHost {
        using TextGridHost::measureChild;
    } host;
    host.grid.update({
        .id = "editor-measure",
        .value = "abcdefghijk\nx",
        .editable = true,
        .fontSize = 15.0f,
        .scrollbar = true,
        .wrap = TextGrid::Wrap::Word,
        .padding = Sakura::Padding{0.0f, 0.0f, 0.0f, 0.0f},
    });
    const F32 measured = host.measureChild(host.grid, ctx, {90.0f, std::numeric_limits<F32>::infinity()}).y;
    CHECK(measured == Catch::Approx(3.0f * 17.25f));

    host.place(ctx, {0.0f, 0.0f, 90.0f, measured});
    const auto& metrics = host.grid.metrics();
    CHECK(metrics.scrollbarGutter > 0.0f);
    CHECK(metrics.contentHeight == Catch::Approx(measured));
}

TEST_CASE("Switching wrap modes re-resolves scrollbar reservations",
          "[core][sakura][text-grid][scroll]") {
    using TextGrid = Sakura::Retained::TextGrid;
    const ImGuiContextGuard imguiContext;
    const Sakura::Context ctx;
    TextGridHost host;
    TextGrid::Config config{
        .id = "wrap-switch",
        .value = "abcdefghijk",
        .editable = true,
        .fontSize = 15.0f,
        .scrollbar = true,
        .padding = Sakura::Padding{0.0f, 0.0f, 0.0f, 0.0f},
    };
    const Jetstream::Rect frame{0.0f, 0.0f, 75.0f, 200.0f};
    host.grid.update(config);
    host.place(ctx, frame);
    REQUIRE(host.grid.metrics().contentWidth > 75.0f);
    REQUIRE(host.grid.metrics().scrollbarGutter == 0.0f);

    config.wrap = TextGrid::Wrap::Word;
    host.grid.update(config);
    host.place(ctx, frame);
    host.place(ctx, frame);
    const auto& wrapped = host.grid.metrics();
    CHECK(wrapped.contentWidth <= 75.0f + 1e-3f);
    CHECK(wrapped.scrollbarGutter > 0.0f);
    REQUIRE(wrapped.sourceLines.size() == 1);
    CHECK(wrapped.sourceLines[0].height == Catch::Approx(2.0f * 17.25f));
    CHECK_FALSE(host.wheel(-5.0f, 0.0f, 30.0f, 10.0f));
}

TEST_CASE("Keyboard input applies once per frame across repeated layouts",
          "[core][sakura][text-grid][input]") {
    using TextGrid = Sakura::Retained::TextGrid;
    SakuraTest::HeadlessUi ui;
    Sakura::Context ctx = ui.sakura();
    ctx.windowFocused = true;
    TextGridHost host;
    host.grid.update({
        .id = "keyboard-once",
        .value = "ab",
        .editable = true,
        .fontSize = 15.0f,
        .padding = Sakura::Padding{0.0f, 0.0f, 0.0f, 0.0f},
    });
    const Jetstream::Rect frame{0.0f, 0.0f, 200.0f, 100.0f};
    ui.frame([&] { host.place(ctx, frame); });
    host.send(MouseEventType::Click, 5.0f, 5.0f);
    host.send(MouseEventType::Release, 5.0f, 5.0f);
    ui.frame([&] { host.place(ctx, frame); });
    REQUIRE(host.grid.metrics().sourceLines.size() == 1);

    ImGui::GetIO().AddKeyEvent(ImGuiKey_Enter, true);
    ui.frame([&] {
        host.place(ctx, frame);
        host.place(ctx, frame);
    });
    ImGui::GetIO().AddKeyEvent(ImGuiKey_Enter, false);
    ui.frame([&] { host.place(ctx, frame); });
    CHECK(host.grid.metrics().sourceLines.size() == 2);

    ImGui::GetIO().AddKeyEvent(ImGuiKey_Enter, true);
    ui.frame([&] { host.place(ctx, frame); });
    ImGui::GetIO().AddKeyEvent(ImGuiKey_Enter, false);
    CHECK(host.grid.metrics().sourceLines.size() == 3);
}
