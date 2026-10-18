#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <jetstream/render/sakura/components/text_markdown.hh>

#include "harness.hh"
#include "fonts.hh"
#include "component_harness.hh"

#include "render/sakura/context.hh"

#include <algorithm>
#include <string>

using namespace Jetstream;
using namespace SakuraTest;

namespace {

using TextMarkdown = Sakura::TextMarkdown;

Extent2D<F32> MeasureAt(Host& host, const Sakura::Context& ctx, F32 width) {
    return host.measureOf(ctx, {width, kInfinite});
}

struct Golden {
    const char* name;
    const char* value;
    F32 width;
    F32 height;
};

constexpr Golden kGoldens[] = {
    {"empty", "", 12.0f, 18.75f},
    {"headings", "# H1\n## H2\n### H3\n#### H4\ntext\n# H1 again\n\ntext", 12.0f, 198.75f},
    {"lists", "- a\n- b\n  - c\n    - d\n1. x\n2. y\n10. z\n* s\n+ p\n\n- after blank", 82.5f, 224.0625f},
    {"list after heading", "## H\n- a\n- b\npara\n- c", 40.5f, 128.4375f},
    {"quote then paragraph", "> a\n> b\npara", 25.5f, 67.5f},
    {"rules", "a\n---\nb\n***\nc\n___\n- - -\nd", 12.0f, 191.25f},
    {"fence then text", "```\ncode\n```\nafter", 67.05f, 62.25f},
    {"unterminated fence", "text\n```\ncode line\nmore code", 67.05f, 81.0f},
    {"only code", "```\na\n```", 67.05f, 32.25f},
    {"heading then code", "# H\n```\na\n```\ntext", 67.05f, 94.125f},
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
    {"callout with body", "> [!NOTE]\n> body", 37.5f, 59.25f},
    {"callout between text", "text\n> [!WARNING]\n> a\n> b\nafter", 37.5f, 138.0f},
    {"callout title only", "> [!TIP]", 37.5f, 36.75f},
    {"quote with unknown tag stays a quote", "> [!nope]\n> x", 25.5f, 37.5f},
};

}  // namespace

TEST_CASE("Markdown block layout matches recorded measurements", "[core][sakura][markdown]") {
    const auto ctx = MakeContext();
    for (const auto& golden : kGoldens) {
        CAPTURE(golden.name);
        TextMarkdown markdown;
        Host host(markdown);
        markdown.update({.id = "golden", .value = golden.value, .fontSize = 15.0f});
        const auto size = MeasureAt(host, ctx, 400.0f);
        CHECK(size.x == Catch::Approx(golden.width));
        CHECK(size.y == Catch::Approx(golden.height));
    }
}

TEST_CASE("Markdown measurements scale with the font size", "[core][sakura][markdown]") {
    const auto ctx = MakeContext();
    TextMarkdown markdown;
    Host host(markdown);
    const std::string value = "# Title\n\nBody text with **bold**.\n\n- one\n- two\n\n> quoted\n\n```\ncode\n```";
    markdown.update({.id = "scaled", .value = value, .fontSize = 15.0f});
    const auto base = MeasureAt(host, ctx, 400.0f);
    markdown.update({.id = "scaled", .value = value, .fontSize = 30.0f});
    const auto doubled = MeasureAt(host, ctx, 800.0f);
    CHECK(doubled.y == Catch::Approx(2.0f * base.y));
    CHECK(doubled.x == Catch::Approx(2.0f * base.x));
}

TEST_CASE("Consecutive code blocks keep their backgrounds apart", "[core][sakura][markdown]") {
    const auto ctx = MakeContext();
    TextMarkdown singleMarkdown;
    Host single(singleMarkdown);
    singleMarkdown.update({.id = "code", .value = "```\na\n```", .fontSize = 15.0f});
    TextMarkdown stackedMarkdown;
    Host stacked(stackedMarkdown);
    stackedMarkdown.update({.id = "code-code", .value = "```\na\n```\n```\nb\n```", .fontSize = 15.0f});

    const F32 paragraphGap = 15.0f * 1.25f * 0.6f;
    CHECK(MeasureAt(stacked, ctx, 400.0f).y ==
          Catch::Approx(2.0f * MeasureAt(single, ctx, 400.0f).y + paragraphGap));
}

TEST_CASE("Narrow viewports can scroll to code behind the line number gutter", "[core][sakura][markdown]") {
    const ImGuiContextGuard imgui;
    const auto ctx = MakeContext();
    TextMarkdown markdown;
    Host host(markdown);
    markdown.update({.id = "narrow-code", .value = "```\nabc\n```", .fontSize = 15.0f});
    host.place(ctx, {0.0f, 0.0f, 30.0f, 200.0f});
    REQUIRE(markdown.metrics().contentWidth > 30.0f);

    REQUIRE(host.send(Wheel(15.0f, 20.0f, -100.0f, 0.0f)));
    host.place(ctx, {0.0f, 0.0f, 30.0f, 200.0f});
    const auto& scrolled = markdown.metrics();
    CHECK(scrolled.scrollX > 0.0f);
    CHECK(scrolled.contentWidth - scrolled.scrollX == Catch::Approx(30.0f));
}

TEST_CASE("Table rows share a band and stack below each other", "[core][sakura][markdown]") {
    const auto ctx = MakeContext();
    TextMarkdown oneRowMarkdown;
    Host oneRow(oneRowMarkdown);
    oneRowMarkdown.update({.id = "one", .value = "| a | b | c |\n|---|---|---|", .fontSize = 15.0f});
    TextMarkdown twoRowsMarkdown;
    Host twoRows(twoRowsMarkdown);
    twoRowsMarkdown.update({.id = "two", .value = "| a | b | c |\n|---|---|---|\n| d | e | f |", .fontSize = 15.0f});

    const F32 rowHeight = 15.0f * 1.25f;
    const F32 rowPad = 15.0f * 0.4f;
    CHECK(MeasureAt(twoRows, ctx, 400.0f).y ==
          Catch::Approx(MeasureAt(oneRow, ctx, 400.0f).y + rowHeight + rowPad));
    CHECK(MeasureAt(twoRows, ctx, 400.0f).x == Catch::Approx(MeasureAt(oneRow, ctx, 400.0f).x));
}

TEST_CASE("Callout bodies add rows without changing the box padding", "[core][sakura][markdown]") {
    const auto ctx = MakeContext();
    TextMarkdown oneMarkdown;
    Host one(oneMarkdown);
    oneMarkdown.update({.id = "one", .value = "> [!CAUTION]\n> a", .fontSize = 15.0f});
    TextMarkdown twoMarkdown;
    Host two(twoMarkdown);
    twoMarkdown.update({.id = "two", .value = "> [!CAUTION]\n> a\n> b", .fontSize = 15.0f});

    const F32 rowHeight = 15.0f * 1.25f;
    CHECK(MeasureAt(two, ctx, 400.0f).y == Catch::Approx(MeasureAt(one, ctx, 400.0f).y + rowHeight));
    CHECK(MeasureAt(two, ctx, 400.0f).x == Catch::Approx(MeasureAt(one, ctx, 400.0f).x));
}

TEST_CASE("Narrow viewports keep every table column reachable", "[core][sakura][markdown]") {
    const auto ctx = MakeContext();
    TextMarkdown markdown;
    Host host(markdown);
    markdown.update({.id = "narrow", .value = "| a | b | c |\n|---|---|---|\n| 1 | 2 | 3 |", .fontSize = 15.0f});
    const struct { F32 width; F32 natural; } cases[] = {
        {400.0f, 156.0f},
        {120.0f, 120.0f},
        {60.0f, 60.0f},
        {30.0f, 43.5f},
    };
    for (const auto& c : cases) {
        CAPTURE(c.width);
        CHECK(MeasureAt(host, ctx, c.width).x == Catch::Approx(std::min(c.width, c.natural)));
        CHECK(markdown.naturalWidth() == Catch::Approx(c.natural));
    }
}

TEST_CASE("Overflowing tables scroll horizontally so every column is reachable", "[core][sakura][markdown]") {
    const ImGuiContextGuard imgui;
    const auto ctx = MakeContext();
    TextMarkdown markdown;
    Host host(markdown);
    markdown.update({.id = "overflow", .value = "| a | b | c |\n|---|---|---|\n| 1 | 2 | 3 |",
                          .fontSize = 15.0f, .scrollbar = true});
    host.place(ctx, {0.0f, 0.0f, 30.0f, 200.0f});

    const auto& metrics = markdown.metrics();
    REQUIRE(metrics.contentWidth > 30.0f);
    REQUIRE(metrics.scrollX == 0.0f);

    REQUIRE(host.send(Wheel(15.0f, 20.0f, -10.0f, 0.0f)));
    host.place(ctx, {0.0f, 0.0f, 30.0f, 200.0f});
    const auto& scrolled = markdown.metrics();
    CHECK(scrolled.scrollX > 0.0f);
    CHECK(scrolled.contentWidth - scrolled.scrollX <= 30.0f + 1e-3f);
}

TEST_CASE("Default markdown views scroll overflowing tables without a scrollbar",
          "[core][sakura][markdown]") {
    const ImGuiContextGuard imgui;
    const auto ctx = MakeContext();
    TextMarkdown markdown;
    Host host(markdown);
    markdown.update({.id = "overflow-default", .value = "| a | b | c |\n|---|---|---|\n| 1 | 2 | 3 |",
                          .fontSize = 15.0f});
    host.place(ctx, {0.0f, 0.0f, 30.0f, 200.0f});
    REQUIRE(markdown.metrics().contentWidth > 30.0f);

    REQUIRE(host.send(Wheel(15.0f, 20.0f, -10.0f, 0.0f)));
    host.place(ctx, {0.0f, 0.0f, 30.0f, 200.0f});
    const auto& scrolled = markdown.metrics();
    CHECK(scrolled.scrollX > 0.0f);
    CHECK(scrolled.contentWidth - scrolled.scrollX <= 30.0f + 1e-3f);
}

TEST_CASE("Scrolling to the bottom of a table note survives repeated layouts", "[core][sakura][markdown]") {
    const ImGuiContextGuard imgui;
    const auto ctx = MakeContext();
    TextMarkdown markdown;
    Host host(markdown);
    std::string value = "| a | b |\n|---|---|\n";
    for (int i = 0; i < 40; ++i) {
        value += "| row | cell |\n";
    }
    markdown.update({.id = "tall-table", .value = value, .fontSize = 15.0f, .scrollbar = true});
    const Jetstream::Rect frame{0.0f, 0.0f, 200.0f, 100.0f};
    host.place(ctx, frame);
    REQUIRE(markdown.metrics().contentHeight > 100.0f);

    REQUIRE(host.send(Wheel(100.0f, 50.0f, 0.0f, -1000.0f)));
    host.place(ctx, frame);
    const F32 bottom = markdown.metrics().scrollY;
    REQUIRE(bottom == Catch::Approx(markdown.metrics().contentHeight - 100.0f));
    for (int i = 0; i < 3; ++i) {
        host.place(ctx, frame);
        CHECK(markdown.metrics().scrollY == Catch::Approx(bottom));
    }
}

TEST_CASE("Aligned cells do not extend the horizontal scroll range", "[core][sakura][markdown]") {
    const ImGuiContextGuard imgui;
    const auto ctx = MakeContext();
    TextMarkdown markdown;
    Host host(markdown);
    markdown.update({.id = "aligned", .value = "| a | b | c |\n|:--|:-:|--:|\n| 1 | 2 | 3 |",
                          .fontSize = 15.0f, .scrollbar = true});
    host.place(ctx, {0.0f, 0.0f, 200.0f, 200.0f});
    CHECK(markdown.metrics().contentWidth == Catch::Approx(156.0f));

    REQUIRE_FALSE(host.send(Wheel(100.0f, 20.0f, -10.0f, 0.0f)));
    host.place(ctx, {0.0f, 0.0f, 200.0f, 200.0f});
    CHECK(markdown.metrics().scrollX == 0.0f);
}

TEST_CASE("A table that fits vertically reserves no scrollbar gutter", "[core][sakura][markdown]") {
    const ImGuiContextGuard imgui;
    const auto ctx = MakeContext();
    TextMarkdown markdown;
    Host host(markdown);
    markdown.update({.id = "fits", .value = "| a | b | c |\n|---|---|---|\n| 1 | 2 | 3 |",
                          .fontSize = 15.0f, .scrollbar = true});
    host.place(ctx, {0.0f, 0.0f, 120.0f, 200.0f});
    const auto& metrics = markdown.metrics();
    CHECK(metrics.contentHeight < 200.0f);
    CHECK(metrics.contentWidth == Catch::Approx(120.0f));
    REQUIRE_FALSE(host.send(Wheel(60.0f, 20.0f, -10.0f, 0.0f)));
}

TEST_CASE("Horizontal overflow shows a draggable scrollbar without vertical overflow",
          "[core][sakura][markdown]") {
    const ImGuiContextGuard imgui;
    const auto ctx = MakeContext();
    TextMarkdown markdown;
    Host host(markdown);
    std::string header = "|";
    std::string delimiter = "|";
    std::string row = "|";
    for (int i = 0; i < 12; ++i) {
        header += " h |";
        delimiter += "---|";
        row += " c |";
    }
    markdown.update({.id = "wide", .value = header + "\n" + delimiter + "\n" + row,
                          .fontSize = 15.0f, .scrollbar = true});
    const Jetstream::Rect frame{0.0f, 0.0f, 120.0f, 200.0f};
    host.place(ctx, frame);

    const auto& metrics = markdown.metrics();
    REQUIRE(metrics.contentHeight < 200.0f);
    REQUIRE(metrics.contentWidth == Catch::Approx(156.0f));
    REQUIRE(metrics.scrollX == 0.0f);

    CHECK(metrics.contentHeight == Catch::Approx(52.5f + 14.0f));
    CHECK(MeasureAt(host, ctx, 120.0f).y == Catch::Approx(52.5f + 14.0f));

    const F32 thumbY = 200.0f - 4.0f - 3.0f;
    REQUIRE(host.send(Mouse(MouseEventType::Click, 14.0f, thumbY)));
    REQUIRE(host.send(Mouse(MouseEventType::Move, 120.0f, thumbY)));
    REQUIRE(host.send(Mouse(MouseEventType::Release, 120.0f, thumbY)));
    host.place(ctx, frame);
    CHECK(markdown.metrics().scrollX == Catch::Approx(36.0f));
}

TEST_CASE("Measuring after layout leaves the live table geometry in place", "[core][sakura][markdown]") {
    const ImGuiContextGuard imgui;
    const auto ctx = MakeContext();
    TextMarkdown markdown;
    Host host(markdown);
    markdown.update({.id = "live", .value = "| a | b | c |\n|---|---|---|\n| 1 | 2 | 3 |",
                          .fontSize = 15.0f, .scrollbar = true});
    const Jetstream::Rect frame{0.0f, 0.0f, 120.0f, 200.0f};
    host.place(ctx, frame);
    REQUIRE(markdown.metrics().contentWidth == Catch::Approx(120.0f));

    MeasureAt(host, ctx, 400.0f);
    CHECK(markdown.naturalWidth() == Catch::Approx(156.0f));
    REQUIRE_FALSE(host.send(Wheel(60.0f, 20.0f, -10.0f, 0.0f)));
    host.place(ctx, frame);
    CHECK(markdown.metrics().scrollX == 0.0f);
    CHECK(markdown.metrics().contentWidth == Catch::Approx(120.0f));
}

TEST_CASE("Fractional column widths do not create phantom overflow", "[core][sakura][markdown]") {
    const ImGuiContextGuard imgui;
    const auto ctx = MakeContext();
    TextMarkdown markdown;
    Host host(markdown);
    markdown.update({.id = "fractional", .value = "| a | b | c |\n|---|---|---|\n| 1 | 2 | 3 |",
                          .fontSize = 15.0f, .scrollbar = true});
    for (const F32 width : {119.0f, 121.0f, 127.0f, 133.0f}) {
        CAPTURE(width);
        host.place(ctx, {0.0f, 0.0f, width, 200.0f});
        const auto& metrics = markdown.metrics();
        CHECK(metrics.contentWidth <= width + 1e-3f);
        CHECK(metrics.contentHeight == Catch::Approx(52.5f));
        CHECK_FALSE(host.send(Wheel(width * 0.5f, 20.0f, -10.0f, 0.0f)));
    }
}

TEST_CASE("Fractional padding and a shifted origin do not create phantom overflow",
          "[core][sakura][markdown]") {
    const ImGuiContextGuard imgui;
    const auto ctx = MakeContext();
    TextMarkdown markdown;
    Host host(markdown);
    markdown.update({.id = "fractional-origin", .value = "| a | b | c |\n|---|---|---|\n| 1 | 2 | 3 |",
                          .fontSize = 18.0f, .scrollbar = true});
    const Jetstream::Rect frame{256.0f, 0.0f, 119.0f, 200.0f};
    host.place(ctx, frame);
    const auto& metrics = markdown.metrics();
    CHECK(metrics.contentWidth <= 119.0f + 1e-3f);
    CHECK(metrics.contentHeight == Catch::Approx(63.0f));
    CHECK_FALSE(host.send(Wheel(256.0f + 60.0f, 20.0f, -10.0f, 0.0f)));
    host.place(ctx, frame);
    CHECK(markdown.metrics().scrollX == 0.0f);
}

TEST_CASE("Laying a callout out at its measured width keeps its height", "[core][sakura][markdown]") {
    const ImGuiContextGuard imgui;
    const auto ctx = MakeContext();
    TextMarkdown markdown;
    Host host(markdown);
    markdown.update({.id = "callout-fit", .value = "> [!NOTE]\n> body text", .fontSize = 15.0f});
    const auto measured = MeasureAt(host, ctx, 400.0f);
    host.place(ctx, {0.0f, 0.0f, measured.x, 400.0f});
    CHECK(markdown.metrics().contentHeight == Catch::Approx(measured.y));
}

TEST_CASE("Backslash runs before a pipe escape it only when odd", "[core][sakura][markdown]") {
    const auto ctx = MakeContext();
    const std::string tail = "|OK|\n|---|---|\n|x|y|";
    TextMarkdown evenMarkdown;
    Host even(evenMarkdown);
    evenMarkdown.update({.id = "even", .value = "|C:" + std::string(2, '\\') + tail, .fontSize = 15.0f});
    TextMarkdown oddMarkdown;
    Host odd(oddMarkdown);
    oddMarkdown.update({.id = "odd", .value = "|C:" + std::string(3, '\\') + tail, .fontSize = 15.0f});

    const auto evenSize = MeasureAt(even, ctx, 400.0f);
    CHECK(evenSize.x == Catch::Approx(108.0f));
    CHECK(evenSize.y == Catch::Approx(52.5f));

    const auto oddSize = MeasureAt(odd, ctx, 400.0f);
    CHECK(oddSize.x == Catch::Approx(12.0f));
    CHECK(oddSize.y == Catch::Approx(56.25f));
}

TEST_CASE("Unbounded measurement keeps the intrinsic table width before and after layout",
          "[core][sakura][markdown]") {
    const ImGuiContextGuard imgui;
    const auto ctx = MakeContext();
    TextMarkdown markdown;
    Host host(markdown);
    markdown.update({.id = "unbounded", .value = "| a | b |\n|---|---|\n| c | d |", .fontSize = 15.0f});

    const auto fresh = MeasureAt(host, ctx, kInfinite);
    CHECK(fresh.x == Catch::Approx(108.0f));
    CHECK(fresh.y == Catch::Approx(52.5f));

    host.place(ctx, {0.0f, 0.0f, 60.0f, 200.0f});
    const auto after = MeasureAt(host, ctx, kInfinite);
    CHECK(after.x == Catch::Approx(108.0f));
    CHECK(after.y == Catch::Approx(52.5f));
}

TEST_CASE("Repeated markdown measures follow table content changes",
          "[core][sakura][text-markdown][measure]") {
    const ImGuiContextGuard imgui;
    MonoFontWindow window;
    auto ctx = MakeContext();
    ctx.render = &window;
    TextMarkdown markdown;
    Host host(markdown);
    const std::string shortTable = "| a | b |\n|---|---|\n| short | x |";
    const std::string longWord(40, 'w');
    const std::string longTable = "| a | b |\n|---|---|\n| " + longWord + " " + longWord + " | x |";

    markdown.update({.id = "memo-table", .value = shortTable, .fontSize = 15.0f});
    const F32 first = MeasureAt(host, ctx, 300.0f).y;
    REQUIRE(first > 0.0f);
    CHECK(MeasureAt(host, ctx, 300.0f).y == Catch::Approx(first));

    markdown.update({.id = "memo-table", .value = longTable, .fontSize = 15.0f});
    const F32 wrapped = MeasureAt(host, ctx, 300.0f).y;
    CHECK(wrapped > first);
    CHECK(MeasureAt(host, ctx, 300.0f).y == Catch::Approx(wrapped));

    markdown.update({.id = "memo-table", .value = shortTable, .fontSize = 15.0f});
    CHECK(MeasureAt(host, ctx, 300.0f).y == Catch::Approx(first));
}

TEST_CASE("Horizontal wheel scrolling an overflowing markdown table repaints without rebuilding",
          "[core][sakura][markdown][retention]") {
    const ImGuiContextGuard imgui;
    MonoFontWindow window;
    TextMarkdown markdown;
    Host host(markdown);
    auto ctx = MakeContext();
    ctx.render = &window;
    Frames frames(host, {0.0f, 0.0f, 30.0f, 200.0f}, ctx);
    const auto update = [&] {
        markdown.update({.id = "overflow-retention", .value = "| a | b | c |\n|---|---|---|\n| 1 | 2 | 3 |",
                         .fontSize = 15.0f, .scrollbar = true});
    };
    REQUIRE(frames.settle(update));
    REQUIRE(markdown.metrics().contentWidth > 30.0f);

    CHECK(host.send(Wheel(15.0f, 20.0f, -10.0f, 0.0f)));
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(markdown.metrics().scrollX > 0.0f);
    CHECK(frames.run(update).clean());
    CHECK(frames.rebuilds == 1);
    CHECK(frames.restructures == 0);
}

TEST_CASE("Markdown font size changes repaint without rebuilding", "[core][sakura][markdown][retention]") {
    const ImGuiContextGuard imgui;
    MonoFontWindow window;
    TextMarkdown markdown;
    Host host(markdown);
    auto ctx = MakeContext();
    ctx.render = &window;
    Frames frames(host, {10.0f, 10.0f, 400.0f, 300.0f}, ctx);
    F32 fontSize = 15.0f;
    const auto update = [&] {
        markdown.update({.id = "font-retention", .value = "# Title\n\n- one\n- two\n\n> [!TIP]\n> body",
                         .fontSize = fontSize});
    };
    REQUIRE(frames.settle(update));
    CHECK(frames.run(update).clean());

    fontSize = 18.0f;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(frames.run(update).clean());
    CHECK(frames.rebuilds == 1);
    CHECK(frames.restructures == 0);
}
