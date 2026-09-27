#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <jetstream/render/sakura/components/retained/text_markdown.hh>
#include <jetstream/render/tools/imgui.h>

#include "render/sakura/context.hh"

#include <algorithm>
#include <limits>
#include <string>

using namespace Jetstream;

namespace {

struct MarkdownHost : Sakura::Component {
    Sakura::Retained::TextMarkdown markdown;

    MarkdownHost() {
        add(markdown);
    }

    Extent2D<F32> measureAt(const Sakura::Context& ctx, F32 width) {
        return measureChild(markdown, ctx, {width, std::numeric_limits<F32>::infinity()});
    }

    void place(const Sakura::Context& ctx, Jetstream::Rect frame) {
        layoutRoot(ctx, frame);
    }

    bool wheel(F32 dx, F32 dy, F32 x, F32 y) {
        return eventChildren({.type = MouseEventType::Scroll, .position = {x, y}, .scroll = {dx, dy}});
    }

    bool send(MouseEventType type, F32 x, F32 y) {
        return eventChildren({.type = type, .button = MouseButton::Left, .position = {x, y}});
    }

 protected:
    void layout(const Sakura::Context& ctx) override {
        layoutChild(ctx, markdown, frame());
    }
};

struct ImGuiContextGuard {
    ImGuiContextGuard() { ImGui::CreateContext(); }
    ~ImGuiContextGuard() { ImGui::DestroyContext(); }
};

struct Golden {
    const char* name;
    const char* value;
    F32 width;
    F32 height;
};

constexpr Golden kGoldens[] = {
    {"empty", "", 12.0f, 18.75f},
    {"headings", "# H1\n## H2\n### H3\n#### H4\ntext\n# H1 again\n\ntext", 12.0f, 198.75f},
    {"lists", "- a\n- b\n  - c\n    - d\n1. x\n2. y\n10. z\n* s\n+ p\n\n- after blank", 75.0f, 224.0625f},
    {"list after heading", "## H\n- a\n- b\npara\n- c", 33.0f, 128.4375f},
    {"quote then paragraph", "> a\n> b\npara", 18.0f, 67.5f},
    {"rules", "a\n---\nb\n***\nc\n___\n- - -\nd", 12.0f, 191.25f},
    {"fence then text", "```\ncode\n```\nafter", 18.0f, 60.75f},
    {"unterminated fence", "text\n```\ncode line\nmore code", 18.0f, 79.5f},
    {"only code", "```\na\n```", 18.0f, 30.75f},
    {"heading then code", "# H\n```\na\n```\ntext", 18.0f, 92.625f},
    {"inline styles", "plain **b** *i* ***bi*** `c` __b2__ _i2_ [l](http://x) **unterminated *x `y` a_b",
     26.55f, 18.75f},
    {"table 2x2", "| a | b |\n|---|---|\n| c | d |", 108.0f, 52.5f},
    {"table with alignment between text", "before\n| a | b | c |\n|:--|:-:|--:|\n| 1 | 2 | 3 |\n| 4 | 5 | 6 |\nafter",
     156.0f, 137.25f},
    {"table header only", "| only header |\n|---|", 60.0f, 27.75f},
    {"table short row", "| a | b |\n|---|---|\n| c |", 108.0f, 52.5f},
    {"pipes without delimiter row", "| a | b |\n| not a delimiter |\n| c | d |", 12.0f, 56.25f},
    {"table after heading before list", "# H\n| a | b |\n|---|---|\n| `code` | **bold** |\n\n- item",
     108.0f, 114.375f},
    {"escaped pipe", "a \\| b | c\n|---|---|\n| x | y |", 108.0f, 52.5f},
    {"heading with pipe ends a table", "| a | b |\n|---|---|\n# H | x", 108.0f, 74.625f},
    {"unmatched backtick keeps columns", "| a | b |\n|---|---|\n| tick ` | next |", 108.0f, 52.5f},
    {"matched backtick protects a pipe", "| a | b |\n|---|---|\n| `x|y` | z |", 108.0f, 52.5f},
    {"escaped pipe inside code span", "| a | b |\n|---|---|\n| `x\\|y` | z |", 108.0f, 52.5f},
    {"callout with body", "> [!NOTE]\n> body", 30.0f, 57.375f},
    {"callout between text", "text\n> [!WARNING]\n> a\n> b\nafter", 30.0f, 136.125f},
    {"callout title only", "> [!TIP]", 30.0f, 38.625f},
    {"quote with unknown tag stays a quote", "> [!nope]\n> x", 18.0f, 37.5f},
};

}  // namespace

TEST_CASE("Markdown block layout matches recorded measurements",
          "[core][sakura][markdown]") {
    const Sakura::Context ctx;
    for (const auto& golden : kGoldens) {
        CAPTURE(golden.name);
        MarkdownHost host;
        host.markdown.update({.id = "golden", .value = golden.value, .fontSize = 15.0f});
        const auto size = host.measureAt(ctx, 400.0f);
        CHECK(size.x == Catch::Approx(golden.width));
        CHECK(size.y == Catch::Approx(golden.height));
    }
}

TEST_CASE("Markdown measurements scale with the font size",
          "[core][sakura][markdown]") {
    const Sakura::Context ctx;
    MarkdownHost host;
    const std::string value = "# Title\n\nBody text with **bold**.\n\n- one\n- two\n\n> quoted\n\n```\ncode\n```";
    host.markdown.update({.id = "scaled", .value = value, .fontSize = 15.0f});
    const auto base = host.measureAt(ctx, 400.0f);
    host.markdown.update({.id = "scaled", .value = value, .fontSize = 30.0f});
    const auto doubled = host.measureAt(ctx, 800.0f);
    CHECK(doubled.y == Catch::Approx(2.0f * base.y));
    CHECK(doubled.x == Catch::Approx(2.0f * base.x));
}

TEST_CASE("Consecutive code blocks keep their backgrounds apart",
          "[core][sakura][markdown]") {
    const Sakura::Context ctx;
    MarkdownHost single;
    single.markdown.update({.id = "code", .value = "```\na\n```", .fontSize = 15.0f});
    MarkdownHost stacked;
    stacked.markdown.update({.id = "code-code", .value = "```\na\n```\n```\nb\n```", .fontSize = 15.0f});

    const F32 paragraphGap = 15.0f * 1.25f * 0.6f;
    CHECK(stacked.measureAt(ctx, 400.0f).y ==
          Catch::Approx(2.0f * single.measureAt(ctx, 400.0f).y + paragraphGap));
}

TEST_CASE("Table rows share a band and stack below each other",
          "[core][sakura][markdown]") {
    const Sakura::Context ctx;
    MarkdownHost oneRow;
    oneRow.markdown.update({.id = "one", .value = "| a | b | c |\n|---|---|---|", .fontSize = 15.0f});
    MarkdownHost twoRows;
    twoRows.markdown.update({.id = "two", .value = "| a | b | c |\n|---|---|---|\n| d | e | f |", .fontSize = 15.0f});

    const F32 rowHeight = 15.0f * 1.25f;
    const F32 rowPad = 15.0f * 0.4f;
    CHECK(twoRows.measureAt(ctx, 400.0f).y ==
          Catch::Approx(oneRow.measureAt(ctx, 400.0f).y + rowHeight + rowPad));
    CHECK(twoRows.measureAt(ctx, 400.0f).x == Catch::Approx(oneRow.measureAt(ctx, 400.0f).x));
}

TEST_CASE("Callout bodies add rows without changing the box padding",
          "[core][sakura][markdown]") {
    const Sakura::Context ctx;
    MarkdownHost one;
    one.markdown.update({.id = "one", .value = "> [!CAUTION]\n> a", .fontSize = 15.0f});
    MarkdownHost two;
    two.markdown.update({.id = "two", .value = "> [!CAUTION]\n> a\n> b", .fontSize = 15.0f});

    const F32 rowHeight = 15.0f * 1.25f;
    CHECK(two.measureAt(ctx, 400.0f).y == Catch::Approx(one.measureAt(ctx, 400.0f).y + rowHeight));
    CHECK(two.measureAt(ctx, 400.0f).x == Catch::Approx(one.measureAt(ctx, 400.0f).x));
}

TEST_CASE("Narrow viewports keep every table column reachable",
          "[core][sakura][markdown]") {
    const Sakura::Context ctx;
    MarkdownHost host;
    host.markdown.update({.id = "narrow", .value = "| a | b | c |\n|---|---|---|\n| 1 | 2 | 3 |", .fontSize = 15.0f});
    const struct { F32 width; F32 natural; } cases[] = {
        {400.0f, 156.0f},
        {120.0f, 120.0f},
        {60.0f, 60.0f},
        {30.0f, 43.5f},
    };
    for (const auto& c : cases) {
        CAPTURE(c.width);
        CHECK(host.measureAt(ctx, c.width).x == Catch::Approx(std::min(c.width, c.natural)));
        CHECK(host.markdown.naturalWidth() == Catch::Approx(c.natural));
    }
}

TEST_CASE("Overflowing tables scroll horizontally so every column is reachable",
          "[core][sakura][markdown]") {
    const ImGuiContextGuard imguiContext;
    const Sakura::Context ctx;
    MarkdownHost host;
    host.markdown.update({.id = "overflow", .value = "| a | b | c |\n|---|---|---|\n| 1 | 2 | 3 |",
                          .fontSize = 15.0f, .scrollbar = true});
    host.place(ctx, {0.0f, 0.0f, 30.0f, 200.0f});

    const auto& metrics = host.markdown.metrics();
    REQUIRE(metrics.contentWidth > 30.0f);
    REQUIRE(metrics.scrollX == 0.0f);

    REQUIRE(host.wheel(-10.0f, 0.0f, 15.0f, 20.0f));
    host.place(ctx, {0.0f, 0.0f, 30.0f, 200.0f});
    const auto& scrolled = host.markdown.metrics();
    CHECK(scrolled.scrollX > 0.0f);
    CHECK(scrolled.contentWidth - scrolled.scrollX <= 30.0f + 1e-3f);
}

TEST_CASE("Default markdown views scroll overflowing tables without a scrollbar",
          "[core][sakura][markdown]") {
    const ImGuiContextGuard imguiContext;
    const Sakura::Context ctx;
    MarkdownHost host;
    host.markdown.update({.id = "overflow-default", .value = "| a | b | c |\n|---|---|---|\n| 1 | 2 | 3 |",
                          .fontSize = 15.0f});
    host.place(ctx, {0.0f, 0.0f, 30.0f, 200.0f});
    REQUIRE(host.markdown.metrics().contentWidth > 30.0f);

    REQUIRE(host.wheel(-10.0f, 0.0f, 15.0f, 20.0f));
    host.place(ctx, {0.0f, 0.0f, 30.0f, 200.0f});
    const auto& scrolled = host.markdown.metrics();
    CHECK(scrolled.scrollX > 0.0f);
    CHECK(scrolled.contentWidth - scrolled.scrollX <= 30.0f + 1e-3f);
}

TEST_CASE("Scrolling to the bottom of a table note survives repeated layouts",
          "[core][sakura][markdown]") {
    const ImGuiContextGuard imguiContext;
    const Sakura::Context ctx;
    MarkdownHost host;
    std::string value = "| a | b |\n|---|---|\n";
    for (int i = 0; i < 40; ++i) {
        value += "| row | cell |\n";
    }
    host.markdown.update({.id = "tall-table", .value = value, .fontSize = 15.0f, .scrollbar = true});
    const Jetstream::Rect frame{0.0f, 0.0f, 200.0f, 100.0f};
    host.place(ctx, frame);
    REQUIRE(host.markdown.metrics().contentHeight > 100.0f);

    REQUIRE(host.wheel(0.0f, -1000.0f, 100.0f, 50.0f));
    host.place(ctx, frame);
    const F32 bottom = host.markdown.metrics().scrollY;
    REQUIRE(bottom == Catch::Approx(host.markdown.metrics().contentHeight - 100.0f));
    for (int i = 0; i < 3; ++i) {
        host.place(ctx, frame);
        CHECK(host.markdown.metrics().scrollY == Catch::Approx(bottom));
    }
}

TEST_CASE("Aligned cells do not extend the horizontal scroll range",
          "[core][sakura][markdown]") {
    const ImGuiContextGuard imguiContext;
    const Sakura::Context ctx;
    MarkdownHost host;
    host.markdown.update({.id = "aligned", .value = "| a | b | c |\n|:--|:-:|--:|\n| 1 | 2 | 3 |",
                          .fontSize = 15.0f, .scrollbar = true});
    host.place(ctx, {0.0f, 0.0f, 200.0f, 200.0f});
    CHECK(host.markdown.metrics().contentWidth == Catch::Approx(156.0f));

    REQUIRE_FALSE(host.wheel(-10.0f, 0.0f, 100.0f, 20.0f));
    host.place(ctx, {0.0f, 0.0f, 200.0f, 200.0f});
    CHECK(host.markdown.metrics().scrollX == 0.0f);
}

TEST_CASE("A table that fits vertically reserves no scrollbar gutter",
          "[core][sakura][markdown]") {
    const ImGuiContextGuard imguiContext;
    const Sakura::Context ctx;
    MarkdownHost host;
    host.markdown.update({.id = "fits", .value = "| a | b | c |\n|---|---|---|\n| 1 | 2 | 3 |",
                          .fontSize = 15.0f, .scrollbar = true});
    host.place(ctx, {0.0f, 0.0f, 120.0f, 200.0f});
    const auto& metrics = host.markdown.metrics();
    CHECK(metrics.contentHeight < 200.0f);
    CHECK(metrics.contentWidth == Catch::Approx(120.0f));
    REQUIRE_FALSE(host.wheel(-10.0f, 0.0f, 60.0f, 20.0f));
}

TEST_CASE("Horizontal overflow shows a draggable scrollbar without vertical overflow",
          "[core][sakura][markdown]") {
    const ImGuiContextGuard imguiContext;
    const Sakura::Context ctx;
    MarkdownHost host;
    std::string header = "|";
    std::string delimiter = "|";
    std::string row = "|";
    for (int i = 0; i < 12; ++i) {
        header += " h |";
        delimiter += "---|";
        row += " c |";
    }
    host.markdown.update({.id = "wide", .value = header + "\n" + delimiter + "\n" + row,
                          .fontSize = 15.0f, .scrollbar = true});
    const Jetstream::Rect frame{0.0f, 0.0f, 120.0f, 200.0f};
    host.place(ctx, frame);

    const auto& metrics = host.markdown.metrics();
    REQUIRE(metrics.contentHeight < 200.0f);
    REQUIRE(metrics.contentWidth == Catch::Approx(156.0f));
    REQUIRE(metrics.scrollX == 0.0f);

    CHECK(metrics.contentHeight == Catch::Approx(52.5f + 14.0f));
    CHECK(host.measureAt(ctx, 120.0f).y == Catch::Approx(52.5f + 14.0f));

    const F32 thumbY = 200.0f - 4.0f - 3.0f;
    REQUIRE(host.send(MouseEventType::Click, 14.0f, thumbY));
    REQUIRE(host.send(MouseEventType::Move, 120.0f, thumbY));
    REQUIRE(host.send(MouseEventType::Release, 120.0f, thumbY));
    host.place(ctx, frame);
    CHECK(host.markdown.metrics().scrollX == Catch::Approx(36.0f));
}

TEST_CASE("Measuring after layout leaves the live table geometry in place",
          "[core][sakura][markdown]") {
    const ImGuiContextGuard imguiContext;
    const Sakura::Context ctx;
    MarkdownHost host;
    host.markdown.update({.id = "live", .value = "| a | b | c |\n|---|---|---|\n| 1 | 2 | 3 |",
                          .fontSize = 15.0f, .scrollbar = true});
    const Jetstream::Rect frame{0.0f, 0.0f, 120.0f, 200.0f};
    host.place(ctx, frame);
    REQUIRE(host.markdown.metrics().contentWidth == Catch::Approx(120.0f));

    host.measureAt(ctx, 400.0f);
    CHECK(host.markdown.naturalWidth() == Catch::Approx(156.0f));
    REQUIRE_FALSE(host.wheel(-10.0f, 0.0f, 60.0f, 20.0f));
    host.place(ctx, frame);
    CHECK(host.markdown.metrics().scrollX == 0.0f);
    CHECK(host.markdown.metrics().contentWidth == Catch::Approx(120.0f));
}

TEST_CASE("Fractional column widths do not create phantom overflow",
          "[core][sakura][markdown]") {
    const ImGuiContextGuard imguiContext;
    const Sakura::Context ctx;
    MarkdownHost host;
    host.markdown.update({.id = "fractional", .value = "| a | b | c |\n|---|---|---|\n| 1 | 2 | 3 |",
                          .fontSize = 15.0f, .scrollbar = true});
    for (const F32 width : {119.0f, 121.0f, 127.0f, 133.0f}) {
        CAPTURE(width);
        host.place(ctx, {0.0f, 0.0f, width, 200.0f});
        const auto& metrics = host.markdown.metrics();
        CHECK(metrics.contentWidth <= width + 1e-3f);
        CHECK(metrics.contentHeight == Catch::Approx(52.5f));
        CHECK_FALSE(host.wheel(-10.0f, 0.0f, width * 0.5f, 20.0f));
    }
}

TEST_CASE("Fractional padding and a shifted origin do not create phantom overflow",
          "[core][sakura][markdown]") {
    const ImGuiContextGuard imguiContext;
    const Sakura::Context ctx;
    MarkdownHost host;
    host.markdown.update({.id = "fractional-origin", .value = "| a | b | c |\n|---|---|---|\n| 1 | 2 | 3 |",
                          .fontSize = 18.0f, .scrollbar = true});
    const Jetstream::Rect frame{256.0f, 0.0f, 119.0f, 200.0f};
    host.place(ctx, frame);
    const auto& metrics = host.markdown.metrics();
    CHECK(metrics.contentWidth <= 119.0f + 1e-3f);
    CHECK(metrics.contentHeight == Catch::Approx(63.0f));
    CHECK_FALSE(host.wheel(-10.0f, 0.0f, 256.0f + 60.0f, 20.0f));
    host.place(ctx, frame);
    CHECK(host.markdown.metrics().scrollX == 0.0f);
}

TEST_CASE("Laying a callout out at its measured width keeps its height",
          "[core][sakura][markdown]") {
    const ImGuiContextGuard imguiContext;
    const Sakura::Context ctx;
    MarkdownHost host;
    host.markdown.update({.id = "callout-fit", .value = "> [!NOTE]\n> body text", .fontSize = 15.0f});
    const auto measured = host.measureAt(ctx, 400.0f);
    host.place(ctx, {0.0f, 0.0f, measured.x, 400.0f});
    CHECK(host.markdown.metrics().contentHeight == Catch::Approx(measured.y));
}
