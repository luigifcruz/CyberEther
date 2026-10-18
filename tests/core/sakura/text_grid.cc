#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/text_grid.hh>
#include <jetstream/render/tools/imgui.h>

#include "../../../resources/fonts/compressed_inter.hh"
#include "harness.hh"
#include "fonts.hh"
#include "component_harness.hh"
#include "scene.hh"

#include "render/sakura/context.hh"
#include "render/sakura/detail/text_grid_viewport.hh"

#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

using namespace Jetstream;
using namespace SakuraTest;

namespace {

using TextGrid = Sakura::TextGrid;

constexpr Sakura::Padding kNoPadding{0.0f, 0.0f, 0.0f, 0.0f};

struct ExposedTextGrid : TextGrid {
    using Component::clip;
    using Component::frame;
};

struct DocumentHost : Sakura::Component {
    ExposedTextGrid grid;
    Rect document;

    DocumentHost() {
        setClipsChildren(true);
        addChild(grid);
    }

    void place(const Sakura::Context& ctx, Rect bounds) {
        invalidate(Dirty::Layout);
        layoutRoot(ctx, bounds);
    }

    Extent2D<F32> measureOf(const Sakura::Context& ctx, Extent2D<F32> available) {
        return measureChild(grid, ctx, available);
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

std::vector<TextGrid::RowMode> SameRowBand() {
    return {TextGrid::RowMode::Next, TextGrid::RowMode::Same, TextGrid::RowMode::Same, TextGrid::RowMode::Next};
}

}  // namespace

TEST_CASE("Text line metrics rescale cached row heights and gaps when the font scales without rewrapping",
          "[core][sakura][text-grid][scale]") {
    const auto wrap = GENERATE(TextGrid::Wrap::None, TextGrid::Wrap::Word);
    const auto ctx = MakeContext();
    TextGrid grid;
    Host host(grid);
    TextGrid::Config config{
        .id = "scaled-note",
        .value = "Heading\nBody text\nList item",
        .wrap = wrap,
        .lineScale = {1.5f, 1.0f, 1.0f},
        .lineTopGap = {0.0f, 9.0f, 3.0f},
        .lineIndent = {0.0f, 0.0f, 21.0f},
    };
    grid.update(config);
    const F32 originalHeight = host.measureOf(ctx, {300.0f, 600.0f}).y;
    REQUIRE(originalHeight > 0.0f);

    for (const F32 scale : {2.0f, 0.5f, 1.25f, 1.0f}) {
        CAPTURE(wrap, scale);
        config.fontSize = 15.0f * scale;
        config.lineTopGap = {0.0f, 9.0f * scale, 3.0f * scale};
        config.lineIndent = {0.0f, 0.0f, 21.0f * scale};
        grid.update(config);
        CHECK(host.measureOf(ctx, {300.0f * scale, 600.0f * scale}).y == Catch::Approx(originalHeight * scale));
    }
}

TEST_CASE("Vertical cursor moves stay on short wrapped rows", "[core][sakura][text-grid][cursor]") {
    const ImGuiContextGuard imgui;
    const auto ctx = MakeContext();
    TextGrid grid;
    Host host(grid);
    grid.update({
        .id = "wrapped-cursor",
        .value = "aaa bbbbbbbb",
        .editable = true,
        .fontSize = 15.0f,
        .scrollbar = false,
        .wrap = TextGrid::Wrap::Word,
        .padding = kNoPadding,
    });
    host.place(ctx, {0.0f, 0.0f, 75.0f, 200.0f});

    grid.setCursor({0, 10});
    REQUIRE(grid.cursor() == TextGrid::Position{0, 10});

    grid.moveCursorRows(-1);
    CHECK(grid.cursor() == TextGrid::Position{0, 3});

    grid.moveCursorRows(-1);
    CHECK(grid.cursor() == TextGrid::Position{0, 3});

    grid.moveCursorRows(1);
    CHECK(grid.cursor() == TextGrid::Position{0, 10});
}

TEST_CASE("Mouse drags past a wrapped row edge select its last character",
          "[core][sakura][text-grid][cursor]") {
    const ImGuiContextGuard imgui;
    const auto ctx = MakeContext();
    TextGrid grid;
    Host host(grid);
    std::optional<std::pair<TextGrid::Position, TextGrid::Position>> selected;
    grid.update({
        .id = "wrapped-drag",
        .value = "abcdefghijk",
        .editable = true,
        .fontSize = 15.0f,
        .scrollbar = false,
        .wrap = TextGrid::Wrap::Character,
        .padding = kNoPadding,
        .onSelect = [&](TextGrid::Position start, TextGrid::Position end) {
            selected = std::pair{start, end};
        },
    });
    host.place(ctx, {0.0f, 0.0f, 75.0f, 200.0f});

    REQUIRE(host.send(Mouse(MouseEventType::Click, 1.0f, 5.0f)));
    REQUIRE(grid.cursor() == TextGrid::Position{0, 0});

    REQUIRE(host.send(Mouse(MouseEventType::Move, 100.0f, 5.0f)));
    REQUIRE(host.send(Mouse(MouseEventType::Release, 100.0f, 5.0f)));

    REQUIRE(selected.has_value());
    CHECK(selected->first == TextGrid::Position{0, 0});
    CHECK(selected->second == TextGrid::Position{0, 10});
}

TEST_CASE("Vertical cursor moves forget their pixel column when metrics change",
          "[core][sakura][text-grid][cursor]") {
    const ImGuiContextGuard imgui;
    const auto ctx = MakeContext();
    TextGrid grid;
    Host host(grid);
    TextGrid::Config config{
        .id = "rescaled-cursor",
        .value = "aaaaaaaaaa\nbbbbbbbbbb",
        .editable = true,
        .fontSize = 15.0f,
        .scrollbar = false,
        .padding = kNoPadding,
    };
    grid.update(config);
    host.place(ctx, {0.0f, 0.0f, 300.0f, 200.0f});

    grid.setCursor({1, 6});
    grid.moveCursorRows(-1);
    REQUIRE(grid.cursor() == TextGrid::Position{0, 6});

    config.fontSize = 30.0f;
    grid.update(config);
    host.place(ctx, {0.0f, 0.0f, 300.0f, 200.0f});

    grid.moveCursorRows(1);
    CHECK(grid.cursor() == TextGrid::Position{1, 6});
}

TEST_CASE("Monospace wrapping counts characters rather than bytes", "[core][sakura][text-grid][wrap]") {
    const auto wrap = GENERATE(TextGrid::Wrap::Character, TextGrid::Wrap::Word);
    const auto ctx = MakeContext();
    TextGrid ascii;
    TextGrid accented;
    Host asciiHost(ascii);
    Host accentedHost(accented);
    TextGrid::Config config{
        .id = "wrap-bytes",
        .value = "eeee eeee",
        .fontSize = 15.0f,
        .wrap = wrap,
        .padding = kNoPadding,
    };
    ascii.update(config);
    config.value = "\xC3\xA9\xC3\xA9\xC3\xA9\xC3\xA9 \xC3\xA9\xC3\xA9\xC3\xA9\xC3\xA9";
    accented.update(config);

    for (F32 width = 20.0f; width <= 200.0f; width += 5.0f) {
        CAPTURE(wrap, width);
        CHECK(accentedHost.measureOf(ctx, {width, 600.0f}).y ==
              Catch::Approx(asciiHost.measureOf(ctx, {width, 600.0f}).y));
    }
}

TEST_CASE("Vertical cursor moves climb wrapped rows ending in multibyte characters",
          "[core][sakura][text-grid][cursor]") {
    const ImGuiContextGuard imgui;
    const auto ctx = MakeContext();
    TextGrid grid;
    Host host(grid);
    std::string value;
    for (U64 i = 0; i < 40; ++i) {
        value += "\xC3\xA9";
    }
    value += "\neeeeeeeeee";
    grid.update({
        .id = "wrapped-multibyte",
        .value = value,
        .editable = true,
        .fontSize = 15.0f,
        .scrollbar = false,
        .wrap = TextGrid::Wrap::Character,
        .padding = kNoPadding,
    });
    host.place(ctx, {0.0f, 0.0f, 75.0f, 400.0f});

    grid.setCursor({1, 10});
    grid.moveCursorRows(-1);
    REQUIRE(grid.cursor() == TextGrid::Position{0, 80});

    grid.moveCursorRows(-1);
    CHECK(grid.cursor() == TextGrid::Position{0, 58});
}

TEST_CASE("Text grids keep document geometry independent of parent clipping",
          "[core][sakura][text-grid][viewport]") {
    const ImGuiContextGuard imgui;
    const auto ctx = MakeContext();
    DocumentHost host;
    const Rect window{0.0f, 0.0f, 400.0f, 300.0f};

    TextGrid::Config config{
        .id = "long-thought",
        .value = Document(2048),
        .scrollbar = false,
    };
    const auto expand = [&] {
        host.grid.update(config);
        host.document = {0.0f, 0.0f, 400.0f, host.measureOf(ctx, {400.0f, kInfinite}).y};
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
    CHECK(Sakura::Intersect(host.grid.frame(), host.grid.clip()).empty());
    expand();
    host.place(ctx, window);
    CHECK(host.grid.metrics().contentHeight == host.document.height);
}

TEST_CASE("Text grid pools use inherited clipping and survive scrolling and collapse",
          "[core][sakura][text-grid][viewport]") {
    Sakura::TextGridViewport viewport;
    const Rect clip{0.0f, 0.0f, 400.0f, 300.0f};
    constexpr F32 lineHeight = 17.25f;
    for (const F32 height : {35000.0f, 70000.0f}) {
        for (const F32 top : {0.0f, -9.0f, -1000.0f, 150.0f, 350.0f}) {
            CAPTURE(height, top);
            viewport.update({0.0f, top, 400.0f, height}, clip, lineHeight, 64, clip.height);
            CHECK(viewport.bounds.height <= clip.height);
            CHECK(viewport.rowCapacity == 64);
        }
    }

    viewport.update({}, clip, lineHeight, 64, clip.height);
    CHECK(viewport.bounds.empty());
    CHECK(viewport.rowCapacity == 64);
    viewport.update({0.0f, 0.0f, 400.0f, 70000.0f}, clip, lineHeight, 64, clip.height);
    CHECK(viewport.bounds == clip);
    CHECK(viewport.rowCapacity == 64);
}

TEST_CASE("Text grid row capacity jumps once from its minimum to the surface height",
          "[core][sakura][text-grid][viewport]") {
    Sakura::TextGridViewport viewport;
    constexpr F32 lineHeight = 17.25f;
    constexpr F32 surfaceHeight = 3000.0f;
    const auto clipped = [&](F32 height) {
        const Rect frame{0.0f, 0.0f, 400.0f, height};
        viewport.update(frame, frame, lineHeight, 64, surfaceHeight);
    };

    for (const F32 height : {100.0f, 600.0f, 1000.0f}) {
        CAPTURE(height);
        clipped(height);
        CHECK(viewport.rowCapacity == 64);
    }

    clipped(1200.0f);
    const U64 filled = viewport.rowCapacity;
    CHECK(filled == 176);

    for (const F32 height : {1400.0f, 2000.0f, 2600.0f, 3000.0f}) {
        CAPTURE(height);
        clipped(height);
        CHECK(viewport.rowCapacity == filled);
    }
    viewport.update({0.0f, -1000.0f, 400.0f, 35000.0f}, {0.0f, 0.0f, 400.0f, 3000.0f}, lineHeight, 64, surfaceHeight);
    CHECK(viewport.rowCapacity == filled);
    viewport.update({}, {}, lineHeight, 64, surfaceHeight);
    CHECK(viewport.rowCapacity == filled);
}

TEST_CASE("Same-row lines share a band and resolve mouse hits by column",
          "[core][sakura][text-grid][row-group]") {
    const ImGuiContextGuard imgui;
    const auto ctx = MakeContext();
    TextGrid grid;
    Host host(grid);
    grid.update({
        .id = "row-group",
        .value = "aaaa\nbbbb\ncccc\ndddd",
        .editable = true,
        .fontSize = 15.0f,
        .scrollbar = false,
        .padding = kNoPadding,
        .lineScale = {1.0f, 2.0f, 1.0f, 1.0f},
        .lineIndent = {0.0f, 100.0f, 200.0f, 0.0f},
        .lineRowMode = SameRowBand(),
    });
    host.place(ctx, {0.0f, 0.0f, 400.0f, 200.0f});

    const auto& metrics = grid.metrics();
    REQUIRE(metrics.sourceLines.size() == 4);
    const F32 rowHeight = metrics.sourceLines[0].height;
    REQUIRE(rowHeight > 0.0f);
    CHECK(metrics.sourceLines[1].top == metrics.sourceLines[0].top);
    CHECK(metrics.sourceLines[2].top == metrics.sourceLines[0].top);
    CHECK(metrics.sourceLines[1].height == Catch::Approx(2.0f * rowHeight));
    CHECK(metrics.sourceLines[3].top == Catch::Approx(metrics.sourceLines[0].top + 2.0f * rowHeight));
    CHECK(metrics.contentHeight == Catch::Approx(3.0f * rowHeight));

    REQUIRE(host.send(Mouse(MouseEventType::Click, 5.0f, 5.0f)));
    CHECK(grid.cursor().line == 0);
    REQUIRE(host.send(Mouse(MouseEventType::Release, 5.0f, 5.0f)));

    REQUIRE(host.send(Mouse(MouseEventType::Click, 105.0f, 5.0f)));
    CHECK(grid.cursor().line == 1);
    REQUIRE(host.send(Mouse(MouseEventType::Release, 105.0f, 5.0f)));

    REQUIRE(host.send(Mouse(MouseEventType::Click, 205.0f, 5.0f)));
    CHECK(grid.cursor().line == 2);
    REQUIRE(host.send(Mouse(MouseEventType::Release, 205.0f, 5.0f)));

    REQUIRE(host.send(Mouse(MouseEventType::Click, 5.0f, 2.0f * rowHeight + 5.0f)));
    CHECK(grid.cursor().line == 3);
    REQUIRE(host.send(Mouse(MouseEventType::Release, 5.0f, 2.0f * rowHeight + 5.0f)));
}

TEST_CASE("Cursor visibility follows the visual row of a same-row cell",
          "[core][sakura][text-grid][row-group]") {
    const ImGuiContextGuard imgui;
    const auto ctx = MakeContext();
    TextGrid grid;
    Host host(grid);
    grid.update({
        .id = "row-group-cursor",
        .value = "aaaa\nbbbb\ncccc\ndddd",
        .editable = true,
        .fontSize = 15.0f,
        .scrollbar = false,
        .padding = kNoPadding,
        .lineScale = {1.0f, 1.0f, 1.0f, 1.0f},
        .lineIndent = {0.0f, 100.0f, 200.0f, 0.0f},
        .lineRowMode = SameRowBand(),
    });
    host.place(ctx, {0.0f, 0.0f, 400.0f, 20.0f});

    grid.setCursor({2, 0});
    host.place(ctx, {0.0f, 0.0f, 400.0f, 20.0f});
    CHECK(grid.metrics().sourceLines[2].top == Catch::Approx(0.0f));

    grid.setCursor({3, 0});
    host.place(ctx, {0.0f, 0.0f, 400.0f, 20.0f});
    const F32 rowHeight = grid.metrics().sourceLines[3].height;
    CHECK(grid.metrics().sourceLines[3].top == Catch::Approx(20.0f - rowHeight));
}

TEST_CASE("Monospace lines honor per-line wrap widths", "[core][sakura][text-grid][row-group]") {
    const auto ctx = MakeContext();
    TextGrid grid;
    Host host(grid);
    TextGrid::Config config{
        .id = "mono-wrap-width",
        .value = "aaaabbbb",
        .fontSize = 15.0f,
        .wrap = TextGrid::Wrap::Character,
        .padding = kNoPadding,
    };
    grid.update(config);
    const F32 single = host.measureOf(ctx, {400.0f, 600.0f}).y;

    config.lineWrapWidth = {4.0f * 7.5f};
    grid.update(config);
    CHECK(host.measureOf(ctx, {400.0f, 600.0f}).y == Catch::Approx(2.0f * single));

    config.lineScale = {2.0f};
    grid.update(config);
    CHECK(host.measureOf(ctx, {400.0f, 600.0f}).y == Catch::Approx(8.0f * single));
}

TEST_CASE("Scaled monospace lines place columns with scaled advances",
          "[core][sakura][text-grid][row-group]") {
    const ImGuiContextGuard imgui;
    const auto ctx = MakeContext();
    TextGrid grid;
    Host host(grid);
    grid.update({
        .id = "scaled-mono-hit",
        .value = "aaaa\nbbbb",
        .editable = true,
        .fontSize = 15.0f,
        .scrollbar = false,
        .padding = kNoPadding,
        .lineScale = {2.0f, 1.0f},
    });
    host.place(ctx, {0.0f, 0.0f, 400.0f, 200.0f});
    const auto& metrics = grid.metrics();
    REQUIRE(metrics.sourceLines[0].height == Catch::Approx(2.0f * metrics.sourceLines[1].height));

    REQUIRE(host.send(Mouse(MouseEventType::Click, 30.0f, 5.0f)));
    CHECK(grid.cursor() == TextGrid::Position{0, 2});
    REQUIRE(host.send(Mouse(MouseEventType::Release, 30.0f, 5.0f)));

    REQUIRE(host.send(Mouse(MouseEventType::Click, 30.0f, metrics.sourceLines[1].top + 5.0f)));
    CHECK(grid.cursor() == TextGrid::Position{1, 4});
    REQUIRE(host.send(Mouse(MouseEventType::Release, 30.0f, metrics.sourceLines[1].top + 5.0f)));

    grid.setCursor({1, 4});
    grid.moveCursorRows(-1);
    CHECK(grid.cursor() == TextGrid::Position{0, 2});
}

TEST_CASE("Clicks on a wrapped row boundary resolve to the row below", "[core][sakura][text-grid][cursor]") {
    const ImGuiContextGuard imgui;
    const auto ctx = MakeContext();
    TextGrid grid;
    Host host(grid);
    grid.update({
        .id = "wrapped-boundary",
        .value = "aaaa",
        .editable = true,
        .fontSize = 15.0f,
        .scrollbar = false,
        .wrap = TextGrid::Wrap::Character,
        .padding = kNoPadding,
        .lineScale = {1.0f},
    });
    host.place(ctx, {0.0f, 0.0f, 15.0f, 200.0f});
    const F32 rowHeight = grid.metrics().sourceLines[0].height * 0.5f;
    REQUIRE(rowHeight > 0.0f);

    REQUIRE(host.send(Mouse(MouseEventType::Click, 1.0f, rowHeight)));
    CHECK(grid.cursor().column == 2);
    REQUIRE(host.send(Mouse(MouseEventType::Release, 1.0f, rowHeight)));

    host.place(ctx, {0.0f, 0.0f, 22.5f, 200.0f});
    REQUIRE(grid.metrics().sourceLines[0].height == Catch::Approx(2.0f * rowHeight));

    REQUIRE(host.send(Mouse(MouseEventType::Click, 15.0f, rowHeight)));
    CHECK(grid.cursor().column == 4);
    REQUIRE(host.send(Mouse(MouseEventType::Release, 15.0f, rowHeight)));

    REQUIRE(host.send(Mouse(MouseEventType::Click, 15.0f, rowHeight - 1.0f)));
    CHECK(grid.cursor().column == 2);
}

TEST_CASE("Vertical cursor moves cross row groups by horizontal position",
          "[core][sakura][text-grid][row-group]") {
    const ImGuiContextGuard imgui;
    const auto ctx = MakeContext();
    TextGrid grid;
    Host host(grid);
    grid.update({
        .id = "row-group-nav",
        .value = "aaaa\nbbbb\ncccc\ndddddddddddddddddddddddddddddd",
        .editable = true,
        .fontSize = 15.0f,
        .scrollbar = false,
        .padding = kNoPadding,
        .lineScale = {1.0f, 1.0f, 1.0f, 1.0f},
        .lineIndent = {0.0f, 100.0f, 200.0f, 0.0f},
        .lineRowMode = SameRowBand(),
    });
    host.place(ctx, {0.0f, 0.0f, 400.0f, 200.0f});

    grid.setCursor({0, 0});
    grid.moveCursorRows(1);
    CHECK(grid.cursor() == TextGrid::Position{3, 0});

    grid.moveCursorRows(-1);
    CHECK(grid.cursor() == TextGrid::Position{0, 0});

    grid.setCursor({1, 2});
    grid.moveCursorRows(1);
    CHECK(grid.cursor().line == 3);
    CHECK(grid.cursor().column == 15);

    grid.moveCursorRows(-1);
    CHECK(grid.cursor() == TextGrid::Position{1, 2});

    grid.setCursor({3, 28});
    grid.moveCursorRows(-1);
    CHECK(grid.cursor() == TextGrid::Position{2, 1});

    grid.moveCursorRows(-1);
    CHECK(grid.cursor() == TextGrid::Position{2, 1});

    grid.setCursor({1, 2});
    grid.moveCursorRows(1);
    REQUIRE(grid.cursor().line == 3);
    host.place(ctx, {75.0f, 40.0f, 400.0f, 200.0f});
    grid.moveCursorRows(-1);
    CHECK(grid.cursor() == TextGrid::Position{1, 2});
}

TEST_CASE("Hit testing deep inside a long wrapped line resolves by row",
          "[core][sakura][text-grid][cursor]") {
    const ImGuiContextGuard imgui;
    const auto ctx = MakeContext();
    TextGrid grid;
    Host host(grid);
    grid.update({
        .id = "long-wrapped-hit",
        .value = std::string(400, 'a'),
        .editable = true,
        .fontSize = 15.0f,
        .scrollbar = false,
        .wrap = TextGrid::Wrap::Character,
        .padding = kNoPadding,
        .lineScale = {1.0f},
    });
    host.place(ctx, {0.0f, 0.0f, 15.0f, 5000.0f});
    const F32 rowHeight = grid.metrics().sourceLines[0].height / 200.0f;
    REQUIRE(rowHeight > 0.0f);

    REQUIRE(host.send(Mouse(MouseEventType::Click, 1.0f, 100.0f * rowHeight + 1.0f)));
    CHECK(grid.cursor().column == 200);
    REQUIRE(host.send(Mouse(MouseEventType::Release, 1.0f, 100.0f * rowHeight + 1.0f)));

    grid.setCursor({0, 200});
    grid.moveCursorRows(1);
    CHECK(grid.cursor().column == 202);
    grid.moveCursorRows(-2);
    CHECK(grid.cursor().column == 198);
}

TEST_CASE("Horizontal scrolling reveals indented same-row cells", "[core][sakura][text-grid][row-group]") {
    const ImGuiContextGuard imgui;
    const auto ctx = MakeContext();
    TextGrid grid;
    Host host(grid);
    grid.update({
        .id = "row-group-hscroll",
        .value = "aaaa\nbbbb",
        .editable = true,
        .fontSize = 15.0f,
        .scrollbar = false,
        .padding = kNoPadding,
        .lineScale = {1.0f, 1.0f},
        .lineIndent = {0.0f, 200.0f},
        .lineRowMode = {TextGrid::RowMode::Next, TextGrid::RowMode::Same},
    });
    host.place(ctx, {0.0f, 0.0f, 100.0f, 50.0f});

    REQUIRE(host.send(Mouse(MouseEventType::Click, 95.0f, 5.0f)));
    CHECK(grid.cursor().line == 0);
    REQUIRE(host.send(Mouse(MouseEventType::Release, 95.0f, 5.0f)));

    grid.setCursor({1, 0});
    host.place(ctx, {0.0f, 0.0f, 100.0f, 50.0f});

    REQUIRE(host.send(Mouse(MouseEventType::Click, 76.0f, 5.0f)));
    CHECK(grid.cursor() == TextGrid::Position{1, 0});
    REQUIRE(host.send(Mouse(MouseEventType::Release, 76.0f, 5.0f)));

    REQUIRE(host.send(Mouse(MouseEventType::Click, 91.0f, 5.0f)));
    CHECK(grid.cursor() == TextGrid::Position{1, 2});
    REQUIRE(host.send(Mouse(MouseEventType::Release, 91.0f, 5.0f)));

    REQUIRE(host.send(Mouse(MouseEventType::Click, 95.0f, 5.0f)));
    CHECK(grid.cursor().line == 1);
}

TEST_CASE("Clipped grids with a long wrapped column lay out beside short cells",
          "[core][sakura][text-grid][row-group][viewport]") {
    const ImGuiContextGuard imgui;
    const auto ctx = MakeContext();
    DocumentHost host;
    const Rect window{0.0f, 0.0f, 400.0f, 200.0f};
    host.grid.update({
        .id = "long-column",
        .value = std::string(4000, 'a') + "\nshort\nafter",
        .fontSize = 15.0f,
        .scrollbar = false,
        .wrap = TextGrid::Wrap::Character,
        .padding = kNoPadding,
        .lineScale = {1.0f, 1.0f, 1.0f},
        .lineIndent = {0.0f, 100.0f, 0.0f},
        .lineRowMode = {TextGrid::RowMode::Next, TextGrid::RowMode::Same, TextGrid::RowMode::Next},
        .lineWrapWidth = {75.0f, 0.0f, 0.0f},
    });
    host.document = {0.0f, 0.0f, 400.0f, host.measureOf(ctx, {400.0f, kInfinite}).y};
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
    const ImGuiContextGuard imgui;
    const auto ctx = MakeContext();
    TextGrid grid;
    Host host(grid);
    grid.update({
        .id = "row-group-numbers",
        .value = "aaaa\nbbbb\ncccc\ndddd",
        .editable = true,
        .fontSize = 15.0f,
        .lineNumbers = true,
        .scrollbar = false,
        .padding = kNoPadding,
        .lineScale = {1.0f, 1.0f, 1.0f, 1.0f},
        .lineIndent = {0.0f, 100.0f, 200.0f, 0.0f},
        .lineRowMode = SameRowBand(),
    });
    host.place(ctx, {0.0f, 0.0f, 400.0f, 200.0f});

    const auto& metrics = grid.metrics();
    REQUIRE(metrics.sourceLines.size() == 4);
    CHECK(metrics.sourceLines[1].top == metrics.sourceLines[0].top);
    CHECK(metrics.sourceLines[2].top == metrics.sourceLines[0].top);
    CHECK(metrics.sourceLines[3].top > metrics.sourceLines[0].top);

    REQUIRE(host.send(Mouse(MouseEventType::Click, 50.0f, 5.0f)));
    CHECK(grid.cursor().line == 0);
    REQUIRE(host.send(Mouse(MouseEventType::Release, 50.0f, 5.0f)));

    REQUIRE(host.send(Mouse(MouseEventType::Click, 150.0f, 5.0f)));
    CHECK(grid.cursor().line == 1);
    REQUIRE(host.send(Mouse(MouseEventType::Release, 150.0f, 5.0f)));

    REQUIRE(host.send(Mouse(MouseEventType::Click, 250.0f, 5.0f)));
    CHECK(grid.cursor().line == 2);
}

TEST_CASE("Clearing the width layout hook restores static wrapping", "[core][sakura][text-grid][wrap]") {
    const auto ctx = MakeContext();
    TextGrid grid;
    Host host(grid);
    TextGrid::Config config{
        .id = "hooked-wrap",
        .value = "aaaaaaaa",
        .fontSize = 15.0f,
        .scrollbar = false,
        .wrap = TextGrid::Wrap::Character,
        .padding = kNoPadding,
        .widthLayout = [](F32) { return TextGrid::WidthLayout{.lineWrapWidth = {4.0f * 7.5f}}; },
    };
    grid.update(config);
    const F32 hooked = host.measureOf(ctx, {400.0f, 600.0f}).y;

    config.widthLayout = nullptr;
    grid.update(config);
    CHECK(host.measureOf(ctx, {400.0f, 600.0f}).y == Catch::Approx(hooked * 0.5f));
}

TEST_CASE("Horizontal overflow in an editor does not create vertical scrolling",
          "[core][sakura][text-grid][scroll]") {
    const ImGuiContextGuard imgui;
    const auto ctx = MakeContext();
    TextGrid grid;
    Host host(grid);
    grid.update({
        .id = "editor-overflow",
        .value = "abcdefghijk",
        .editable = true,
        .fontSize = 15.0f,
        .scrollbar = true,
        .padding = kNoPadding,
    });
    host.place(ctx, {0.0f, 0.0f, 75.0f, 200.0f});

    const auto& metrics = grid.metrics();
    CHECK(metrics.contentWidth > 75.0f);
    CHECK(metrics.scrollbarGutter == 0.0f);
    REQUIRE_FALSE(host.send(Wheel(30.0f, 100.0f, 0.0f, -5.0f)));
    REQUIRE(host.send(Wheel(30.0f, 100.0f, -5.0f, 0.0f)));
    host.place(ctx, {0.0f, 0.0f, 75.0f, 200.0f});
    CHECK(grid.metrics().scrollY == 0.0f);
    CHECK(grid.metrics().scrollX > 0.0f);
}

TEST_CASE("Cursor stays visible when a scrollbar appears and rewraps the text",
          "[core][sakura][text-grid][scroll]") {
    const ImGuiContextGuard imgui;
    const auto ctx = MakeContext();
    TextGrid grid;
    Host host(grid);
    TextGrid::Config config{
        .id = "rewrap-cursor",
        .value = "abcdefghijk",
        .editable = true,
        .fontSize = 15.0f,
        .scrollbar = true,
        .wrap = TextGrid::Wrap::Word,
        .padding = kNoPadding,
    };
    const Rect frame{0.0f, 0.0f, 90.0f, 34.5f};
    grid.update(config);
    host.place(ctx, frame);
    REQUIRE(grid.metrics().scrollbarGutter == 0.0f);

    config.value = "abcdefghijk\nx";
    grid.update(config);
    grid.setCursor({1, 0});
    host.place(ctx, frame);

    const auto& metrics = grid.metrics();
    REQUIRE(metrics.scrollbarGutter > 0.0f);
    REQUIRE(metrics.sourceLines.size() == 2);
    CHECK(metrics.sourceLines[1].top >= 0.0f);
    CHECK(metrics.sourceLines[1].top + metrics.sourceLines[1].height <= frame.height + 1e-3f);

    const F32 previousTop = metrics.sourceLines[1].top;
    host.place(ctx, frame);
    CHECK(grid.metrics().sourceLines[1].top == Catch::Approx(previousTop));

    config.value = "abcdefghijkx";
    grid.update(config);
    host.place(ctx, frame);
    host.place(ctx, frame);
    const auto& joined = grid.metrics();
    CHECK(joined.scrollbarGutter == 0.0f);
    REQUIRE(joined.sourceLines.size() == 1);
    CHECK(joined.sourceLines[0].height == Catch::Approx(17.25f));
    CHECK(joined.scrollY == 0.0f);
}

TEST_CASE("Editable wrapped grids measure the height their layout will use",
          "[core][sakura][text-grid][scroll]") {
    const ImGuiContextGuard imgui;
    const auto ctx = MakeContext();
    TextGrid grid;
    Host host(grid);
    grid.update({
        .id = "editor-measure",
        .value = "abcdefghijk\nx",
        .editable = true,
        .fontSize = 15.0f,
        .scrollbar = true,
        .wrap = TextGrid::Wrap::Word,
        .padding = kNoPadding,
    });
    const F32 measured = host.measureOf(ctx, {90.0f, kInfinite}).y;
    CHECK(measured == Catch::Approx(3.0f * 17.25f));

    host.place(ctx, {0.0f, 0.0f, 90.0f, measured});
    const auto& metrics = grid.metrics();
    CHECK(metrics.scrollbarGutter > 0.0f);
    CHECK(metrics.contentHeight == Catch::Approx(measured));
}

TEST_CASE("Switching wrap modes re-resolves scrollbar reservations", "[core][sakura][text-grid][scroll]") {
    const ImGuiContextGuard imgui;
    const auto ctx = MakeContext();
    TextGrid grid;
    Host host(grid);
    TextGrid::Config config{
        .id = "wrap-switch",
        .value = "abcdefghijk",
        .editable = true,
        .fontSize = 15.0f,
        .scrollbar = true,
        .padding = kNoPadding,
    };
    const Rect frame{0.0f, 0.0f, 75.0f, 200.0f};
    grid.update(config);
    host.place(ctx, frame);
    REQUIRE(grid.metrics().contentWidth > 75.0f);
    REQUIRE(grid.metrics().scrollbarGutter == 0.0f);

    config.wrap = TextGrid::Wrap::Word;
    grid.update(config);
    host.place(ctx, frame);
    host.place(ctx, frame);
    const auto& wrapped = grid.metrics();
    CHECK(wrapped.contentWidth <= 75.0f + 1e-3f);
    CHECK(wrapped.scrollbarGutter > 0.0f);
    REQUIRE(wrapped.sourceLines.size() == 1);
    CHECK(wrapped.sourceLines[0].height == Catch::Approx(2.0f * 17.25f));
    CHECK_FALSE(host.send(Wheel(30.0f, 10.0f, -5.0f, 0.0f)));
}

TEST_CASE("A key reaches the focused grid once however often it lays out",
          "[core][sakura][text-grid][input]") {
    const ImGuiContextGuard imgui;
    const auto ctx = MakeContext();
    TextGrid grid;
    Host host(grid);
    grid.update({
        .id = "keyboard-once",
        .value = "ab",
        .editable = true,
        .fontSize = 15.0f,
        .padding = kNoPadding,
    });
    const Rect frame{0.0f, 0.0f, 200.0f, 100.0f};
    host.place(ctx, frame);
    CHECK_FALSE(Tap(host, KeyCode::Enter));
    host.click(5.0f, 5.0f);
    host.place(ctx, frame);
    REQUIRE(grid.metrics().sourceLines.size() == 1);

    CHECK(Tap(host, KeyCode::Enter));
    host.place(ctx, frame);
    host.place(ctx, frame);
    CHECK(grid.metrics().sourceLines.size() == 2);

    CHECK(Tap(host, KeyCode::Enter));
    host.place(ctx, frame);
    CHECK(grid.metrics().sourceLines.size() == 3);
}

TEST_CASE("Repeated measures follow content, padding, and width changes",
          "[core][sakura][text-grid][measure]") {
    const ImGuiContextGuard imgui;
    MonoFontWindow window;
    auto ctx = MakeContext();
    ctx.render = &window;
    TextGrid grid;
    Host host(grid);
    TextGrid::Config config{
        .id = "measure-memo",
        .value = "abc",
        .fontSize = 15.0f,
        .wrap = TextGrid::Wrap::Word,
        .padding = kNoPadding,
    };
    grid.update(config);
    const auto measure = [&](F32 width) {
        return host.measureOf(ctx, {width, kInfinite});
    };

    const auto first = measure(400.0f);
    REQUIRE(first.x > 0.0f);
    REQUIRE(first.y > 0.0f);
    CHECK(measure(400.0f).y == Catch::Approx(first.y));

    config.value = "abc\ndef";
    grid.update(config);
    const F32 twoLines = measure(400.0f).y;
    CHECK(twoLines == Catch::Approx(first.y * 2.0f));

    config.padding = Sakura::Padding{0.0f, 10.0f, 0.0f, 0.0f};
    grid.update(config);
    CHECK(measure(400.0f).y == Catch::Approx(twoLines + 10.0f));

    config.padding = kNoPadding;
    config.value = "abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz";
    grid.update(config);
    const auto wide = measure(2000.0f);
    const auto narrow = measure(first.x * 4.0f);
    CHECK(narrow.y > wide.y);
    CHECK(measure(2000.0f).y == Catch::Approx(wide.y));
    CHECK(measure(2000.0f).x == Catch::Approx(wide.x));
}

TEST_CASE("Repeated measures follow fonts registered after a fallback measure",
          "[core][sakura][text-grid][measure]") {
    const ImGuiContextGuard imgui;
    MonoFontWindow window;
    auto ctx = MakeContext();
    ctx.render = &window;
    TextGrid grid;
    Host host(grid);
    grid.update({
        .id = "measure-late-font",
        .value = "iiiiiiiiii",
        .fontSize = 15.0f,
        .fontName = "late_body",
        .monospace = false,
        .padding = kNoPadding,
    });
    const auto measure = [&] {
        return host.measureOf(ctx, {2000.0f, kInfinite}).x;
    };

    const F32 fallback = measure();
    REQUIRE(fallback > 0.0f);
    CHECK(measure() == Catch::Approx(fallback));

    window.load("late_body", inter_regular_compressed_data);
    CHECK(measure() < fallback);
}

TEST_CASE("Repeated measures follow late fonts beyond the first sixty-three styles",
          "[core][sakura][text-grid][measure]") {
    const ImGuiContextGuard imgui;
    MonoFontWindow window;
    auto ctx = MakeContext();
    ctx.render = &window;
    TextGrid grid;
    Host host(grid);
    constexpr U64 kStyles = 70;
    std::vector<std::string> fonts;
    for (U64 i = 1; i <= kStyles; ++i) {
        fonts.push_back("late_style_" + std::to_string(i));
    }
    const std::string value(10, 'i');
    const std::vector<std::vector<TextGrid::StyleId>> styles = {
        std::vector<TextGrid::StyleId>(value.size(), static_cast<TextGrid::StyleId>(kStyles)),
    };
    grid.update({
        .id = "measure-late-style-font",
        .value = value,
        .fontSize = 15.0f,
        .monospace = false,
        .padding = kNoPadding,
        .styleFonts = fonts,
        .styler = [styles](const std::vector<std::string>&, U64) -> const std::vector<std::vector<TextGrid::StyleId>>& {
            return styles;
        },
    });
    const auto measure = [&] {
        return host.measureOf(ctx, {2000.0f, kInfinite}).x;
    };

    const F32 fallback = measure();
    REQUIRE(fallback > 0.0f);
    CHECK(measure() == Catch::Approx(fallback));

    window.load(fonts.back(), inter_regular_compressed_data);
    CHECK(measure() < fallback);
}

TEST_CASE("Text grid font size changes repaint without rebuilding", "[core][sakura][text-grid][retention]") {
    const ImGuiContextGuard imgui;
    MonoFontWindow window;
    TextGrid grid;
    Host host(grid);
    auto ctx = MakeContext();
    ctx.render = &window;
    Frames frames(host, {10.0f, 10.0f, 300.0f, 200.0f}, ctx);
    F32 fontSize = 15.0f;
    const auto update = [&] {
        grid.update({.id = "font-size", .value = "first line\nsecond line", .fontSize = fontSize});
    };
    REQUIRE(frames.settle(update));
    CHECK(frames.run(update).clean());

    fontSize = 18.0f;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(frames.run(update).clean());
    CHECK(frames.rebuilds == 1);
    CHECK(frames.restructures == 0);
}

TEST_CASE("Same-row text grids settle and repaint without rebuilding when a cell moves",
          "[core][sakura][text-grid][row-group][retention]") {
    const ImGuiContextGuard imgui;
    MonoFontWindow window;
    TextGrid grid;
    Host host(grid);
    auto ctx = MakeContext();
    ctx.render = &window;
    Frames frames(host, {0.0f, 0.0f, 400.0f, 200.0f}, ctx);
    F32 indent = 100.0f;
    const auto update = [&] {
        grid.update({
            .id = "row-group-retention",
            .value = "aaaa\nbbbb\ncccc\ndddd",
            .fontSize = 15.0f,
            .scrollbar = false,
            .padding = kNoPadding,
            .lineScale = {1.0f, 1.0f, 1.0f, 1.0f},
            .lineIndent = {0.0f, indent, 200.0f, 0.0f},
            .lineRowMode = SameRowBand(),
        });
    };
    REQUIRE(frames.settle(update));
    CHECK(frames.run(update).clean());

    indent = 120.0f;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(frames.run(update).clean());
    CHECK(frames.rebuilds == 1);
    CHECK(frames.restructures == 0);
}

TEST_CASE("Horizontal wheel scrolling an overflowing text grid repaints without rebuilding",
          "[core][sakura][text-grid][scroll][retention]") {
    const ImGuiContextGuard imgui;
    MonoFontWindow window;
    TextGrid grid;
    Host host(grid);
    auto ctx = MakeContext();
    ctx.render = &window;
    Frames frames(host, {0.0f, 0.0f, 75.0f, 200.0f}, ctx);
    const auto update = [&] {
        grid.update({
            .id = "hscroll-retention",
            .value = "abcdefghijklmnopqrstuvwxyz",
            .fontSize = 15.0f,
            .scrollbar = true,
            .padding = kNoPadding,
        });
    };
    REQUIRE(frames.settle(update));
    REQUIRE(grid.metrics().contentWidth > 75.0f);

    CHECK(host.send(Wheel(30.0f, 10.0f, -5.0f, 0.0f)));
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(grid.metrics().scrollX > 0.0f);
    CHECK(frames.run(update).clean());
    CHECK(frames.rebuilds == 1);
    CHECK(frames.restructures == 0);
}

TEST_CASE("Moving the cursor of a focused text grid repaints without rebuilding",
          "[core][sakura][text-grid][cursor][retention]") {
    const ImGuiContextGuard imgui;
    MonoFontWindow window;
    TextGrid grid;
    Host host(grid);
    auto ctx = MakeContext();
    ctx.render = &window;
    ctx.windowFocused = false;
    Frames frames(host, {0.0f, 0.0f, 300.0f, 200.0f}, ctx);
    const auto update = [&] {
        grid.update({.id = "cursor-retention", .value = "first line\nsecond line", .editable = true});
    };
    REQUIRE(frames.settle(update));
    grid.focus();
    REQUIRE(frames.settle(update));

    grid.setCursor({1, 3});
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(frames.run(update).clean());
    CHECK(frames.rebuilds == 1);
    CHECK(frames.restructures == 0);
    grid.blur();
}

TEST_CASE("A focused grid keeps focus across an id change and gives it up when destroyed",
          "[core][sakura][text-grid][focus]") {
    const ImGuiContextGuard imgui;
    const auto ctx = MakeContext();
    auto grid = std::make_unique<TextGrid>();
    Host host(*grid);
    TextGrid::Config config{
        .id = "renamed-before",
        .value = "value",
        .editable = true,
        .fontSize = 15.0f,
        .padding = kNoPadding,
    };
    grid->update(config);
    host.place(ctx, {0.0f, 0.0f, 200.0f, 100.0f});
    host.click(10.0f, 5.0f);
    host.place(ctx, {0.0f, 0.0f, 200.0f, 100.0f});
    REQUIRE(grid->focused());
    REQUIRE(FocusOwner(host) == grid.get());

    config.id = "renamed-after";
    grid->update(config);
    host.place(ctx, {0.0f, 0.0f, 200.0f, 100.0f});
    CHECK(grid->focused());
    CHECK(FocusOwner(host) == grid.get());

    grid.reset();
    CHECK(FocusOwner(host) == nullptr);
}

TEST_CASE("Scrollbar inset changes alone remeasure a wrapped grid", "[core][sakura][text-grid][scroll]") {
    const Scene scene;
    const auto& ctx = scene.ctx;
    TextGrid::Config config{
        .id = "inset",
        .value = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma",
        .editable = true,
        .fontSize = 15.0f,
        .scrollbar = true,
        .wrap = TextGrid::Wrap::Word,
        .padding = kNoPadding,
    };
    const Rect frame{0.0f, 0.0f, 120.0f, 40.0f};
    TextGrid grid;
    Host host(grid);
    grid.update(config);
    host.place(ctx, frame);
    const F32 before = host.measureOf(ctx, {120.0f, kInfinite}).y;

    config.scrollbarInset = {40.0f, 0.0f};
    grid.update(config);
    host.place(ctx, frame);
    TextGrid fresh;
    Host reference(fresh);
    fresh.update(config);
    reference.place(ctx, frame);
    const F32 expected = reference.measureOf(ctx, {120.0f, kInfinite}).y;
    REQUIRE(expected > before);
    CHECK(host.measureOf(ctx, {120.0f, kInfinite}).y == Catch::Approx(expected));
}

TEST_CASE("Focusing an editor after a clean idle interval shows the caret at once", "[core][sakura][text-grid][cursor]") {
    const Scene scene;
    TextGrid grid;
    Place place(grid, {10.0f, 10.0f, 200.0f, 40.0f});
    Frames frames(place, kScreen, scene.context());
    frames.verify = false;
    const auto update = [&] {
        grid.update({.id = "caret", .value = "abc", .editable = true, .fontSize = 15.0f, .padding = kNoPadding});
    };
    const auto caret = [&] {
        const auto cursor = frames.ctx.color("editor_cursor");
        for (const auto& leaf : Drawn(place)) {
            if (const auto* box = std::get_if<Sakura::Box::Config>(&leaf.config)) {
                for (const auto& instance : box->instances) {
                    if (instance.visible && instance.backgroundColor == cursor && instance.rect.width > 0.0f) {
                        return true;
                    }
                }
            }
        }
        return false;
    };
    REQUIRE(frames.settle(update));
    frames.ctx.time = 10.6;
    REQUIRE(frames.run(update).clean());

    place.send(Mouse(MouseEventType::Click, 40.0f, 20.0f));
    place.send(Mouse(MouseEventType::Release, 40.0f, 20.0f));
    frames.run(update);
    REQUIRE(grid.focused());
    CHECK(caret());
}

TEST_CASE("Arrow navigation in a focused editor restarts the caret blink on that frame",
          "[core][sakura][text-grid][cursor]") {
    const Scene scene;
    TextGrid grid;
    Place place(grid, {10.0f, 10.0f, 200.0f, 40.0f});
    Frames frames(place, kScreen, scene.context());
    frames.verify = false;
    const auto update = [&] {
        grid.update({.id = "caret", .value = "abcdef", .editable = true, .fontSize = 15.0f, .padding = kNoPadding});
    };
    const auto caret = [&] {
        const auto cursor = frames.ctx.color("editor_cursor");
        for (const auto& leaf : Drawn(place)) {
            if (const auto* box = std::get_if<Sakura::Box::Config>(&leaf.config)) {
                for (const auto& instance : box->instances) {
                    if (instance.visible && instance.backgroundColor == cursor && instance.rect.width > 0.0f) {
                        return true;
                    }
                }
            }
        }
        return false;
    };
    REQUIRE(frames.settle(update));
    place.send(Mouse(MouseEventType::Click, 12.0f, 20.0f));
    place.send(Mouse(MouseEventType::Release, 12.0f, 20.0f));
    REQUIRE(frames.settle(update));
    REQUIRE(grid.focused());
    REQUIRE(caret());

    frames.ctx.time = 0.25;
    REQUIRE(frames.run(update).clean());
    CHECK(Tap(place, KeyCode::RightArrow));
    CHECK(frames.run(update).laidOut);
    CHECK(caret());

    frames.ctx.time = 0.85;
    frames.run(update);
    CHECK_FALSE(caret());
}
