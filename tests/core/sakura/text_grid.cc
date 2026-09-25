#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <jetstream/render/sakura/components/retained/text_grid.hh>
#include <jetstream/render/tools/imgui.h>

#include "render/sakura/context.hh"

using namespace Jetstream;

namespace {

struct MeasuredTextGrid : Sakura::Retained::TextGrid {
    using TextGrid::measure;
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
