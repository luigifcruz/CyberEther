#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <jetstream/render/sakura/components/retained/text_grid.hh>
#include <jetstream/render/tools/imgui.h>

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
