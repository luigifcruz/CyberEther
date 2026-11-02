#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <jetstream/render/components/text.hh>
#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/label.hh>
#include <jetstream/render/sakura/components/markdown.hh>
#include <jetstream/render/sakura/components/markdown_editor.hh>
#include <jetstream/render/sakura/components/text_markdown.hh>
#include <jetstream/render/tools/icons_lucide.hh>

#include "../../../resources/fonts/compressed_inter.hh"
#include "../../../resources/fonts/compressed_lucide.hh"

#include "harness.hh"
#include "fonts.hh"
#include "component_harness.hh"

#include "render/sakura/context.hh"

#include <algorithm>
#include <limits>
#include <optional>
#include <string>
#include <vector>

using namespace Jetstream;
using namespace SakuraTest;

namespace {

using TextMarkdown = Sakura::TextMarkdown;

Extent2D<F32> MeasureAt(Host& host, const Sakura::Context& ctx, F32 width) {
    return host.measureOf(ctx, {width, kInfinite});
}

constexpr F32 kRowHeight = 15.0f * 1.25f;
const std::string kLongCode = "```python\nvalue = compute(" + std::string(120, 'x') + ")\nshort = 1\n```";
const std::string kLongParagraph = [] {
    std::string words;
    for (int i = 0; i < 40; ++i) {
        words += "word ";
    }
    return words;
}();

F32 RowCenter(const TextMarkdown& markdown, U64 line) {
    const auto& row = markdown.metrics().sourceLines.at(line);
    return row.top + row.height * 0.5f;
}

std::optional<Jetstream::Rect> DrawnRect(Sakura::Component& markdown, const std::string& prefix) {
    for (const auto* label : Find<Sakura::Label>(markdown)) {
        for (const auto& instance : label->config().instances) {
            if (instance.visible && instance.str.starts_with(prefix)) {
                return instance.rect;
            }
        }
    }
    return std::nullopt;
}

struct Golden {
    const char* name;
    const char* value;
    F32 width;
    F32 height;
};

constexpr Golden kGoldens[] = {
    {"empty", "", 12.0f, 18.75f},
    {"headings", "# H1\n## H2\n### H3\n#### H4\ntext\n# H1 again\n\ntext", 12.0f, 210.0f},
    {"lists", "- a\n- b\n  - c\n    - d\n1. x\n2. y\n10. z\n* s\n+ p\n\n- after blank", 82.5f, 224.0625f},
    {"list after heading", "## H\n- a\n- b\npara\n- c", 40.5f, 139.6875f},
    {"quote then paragraph", "> a\n> b\npara", 25.5f, 67.5f},
    {"rules", "a\n---\nb\n***\nc\n___\n- - -\nd", 12.0f, 191.25f},
    {"fence then text", "```\ncode\n```\nafter", 57.0f, 109.5f},
    {"unterminated fence", "text\n```\ncode line\nmore code", 57.0f, 128.25f},
    {"only code", "```\na\n```", 57.0f, 79.5f},
    {"heading then code", "# H\n```\na\n```\ntext", 57.0f, 141.375f},
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
    {"callout with body", "> [!NOTE]\n> body", 49.5f, 59.25f},
    {"callout between text", "text\n> [!WARNING]\n> a\n> b\nafter", 49.5f, 138.0f},
    {"callout title only", "> [!TIP]", 49.5f, 36.75f},
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

TEST_CASE("Repeated Markdown layouts refresh presentation settings without changing text",
          "[core][sakura][markdown][grid-config]") {
    MonoFontWindow window;
    auto ctx = MakeContext();
    ctx.render = &window;
    ctx.palette = DistinctPalette();
    TextMarkdown markdown;
    Host host(markdown);
    TextMarkdown::Config config{.id = "cached-presentation", .value = "Plain **bold** text.", .fontSize = 15.0f};
    markdown.update(config);
    for (int i = 0; i < 4; ++i) {
        host.place(ctx, {0.0f, static_cast<F32>(-i), 300.0f, 120.0f});
    }
    config.textColorKey = "text_secondary";
    config.styleColorKeys[0] = "button_hovered";
    config.padding = Sakura::Padding{24.0f, 4.0f, 12.0f, 4.0f};
    markdown.update(config);
    host.place(ctx, {0.0f, 0.0f, 300.0f, 120.0f});
    bool plain = false;
    bool bold = false;
    for (const auto* label : Find<Sakura::Label>(markdown)) {
        for (const auto& instance : label->config().instances) {
            if (!instance.visible) {
                continue;
            }
            if (instance.str.starts_with("Plain")) {
                plain = true;
                CHECK(instance.color == ctx.color("text_secondary"));
                CHECK(instance.rect.x == Catch::Approx(24.0f));
            }
            if (instance.str == "bold") {
                bold = true;
                CHECK(instance.color == ctx.color("button_hovered"));
            }
        }
    }
    CHECK(plain);
    CHECK(bold);
}

TEST_CASE("Markdown refreshes cached layout when fonts arrive and the width changes",
          "[core][sakura][markdown][grid-config]") {
    MonoFontWindow window;
    auto ctx = MakeContext();
    ctx.render = &window;
    TextMarkdown markdown;
    Host host(markdown);
    const TextMarkdown::Config config{.id = "late-fonts", .value =
        "| Column one | Column two |\n|---|---|\n| **Several words of bold text** | A wrapping table cell |\n\n"
        "```stats\nOne | 123\nTwo | 456\n```\n\n- An indented line with additional words that wrap.", .fontSize = 15.0f};
    markdown.update(config);
    (void)MeasureAt(host, ctx, 240.0f);
    host.place(ctx, {0.0f, 0.0f, 240.0f, 500.0f});
    window.load(Sakura::Typography::BodyFont, inter_regular_compressed_data);
    window.load(Sakura::Typography::BodyBoldFont, inter_bold_compressed_data);
    for (const F32 width : {240.0f, 480.0f, 180.0f, 240.0f}) {
        TextMarkdown fresh;
        Host freshHost(fresh);
        fresh.update(config);
        const auto measured = MeasureAt(host, ctx, width);
        const auto expected = MeasureAt(freshHost, ctx, width);
        CHECK(measured.x == Catch::Approx(expected.x));
        CHECK(measured.y == Catch::Approx(expected.y));
        host.place(ctx, {0.0f, 0.0f, width, 500.0f});
        freshHost.place(ctx, {0.0f, 0.0f, width, 500.0f});
        CHECK(markdown.metrics().contentWidth == Catch::Approx(fresh.metrics().contentWidth));
        CHECK(markdown.metrics().contentHeight == Catch::Approx(fresh.metrics().contentHeight));
    }
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

TEST_CASE("Horizontal rubber banding moves table chrome with its glyphs", "[core][sakura][markdown]") {
    auto ctx = MakeContext();
    ctx.time = 1.0;
    TextMarkdown markdown;
    Host host(markdown);
    markdown.update({.id = "stretch-table", .value = "| a | b | c |\n|---|---|---|\n| 1 | 2 | 3 |",
                     .fontSize = 15.0f});
    const Jetstream::Rect frame{0.0f, 0.0f, 30.0f, 200.0f};
    host.place(ctx, frame);
    REQUIRE(markdown.metrics().contentWidth > 30.0f);

    const auto chromeLeft = [&] {
        F32 left = std::numeric_limits<F32>::max();
        for (const auto* box : Find<Sakura::Box>(markdown)) {
            for (const auto& instance : box->config().instances) {
                if (instance.visible && instance.rect.width < 3.0f && instance.rect.height > instance.rect.width) {
                    left = std::min(left, instance.rect.x);
                }
            }
        }
        return left;
    };
    const auto glyph = DrawnRect(markdown, "a");
    REQUIRE(glyph.has_value());
    const F32 glyphBefore = glyph->x;
    const F32 chromeBefore = chromeLeft();

    auto event = Wheel(15.0f, 20.0f, 0.0f, 0.0f);
    event.phase = ScrollPhase::Began;
    host.send(event);
    event.scroll.x = 5.0f;
    event.phase = ScrollPhase::Changed;
    REQUIRE(host.send(event));
    ctx.time += 1.0 / 60.0;
    host.place(ctx, frame);

    REQUIRE(markdown.metrics().scrollX == 0.0f);
    REQUIRE(markdown.metrics().overscroll.x < 0.0f);
    const auto stretched = DrawnRect(markdown, "a");
    REQUIRE(stretched.has_value());
    const F32 shift = stretched->x - glyphBefore;
    CHECK(shift == Catch::Approx(-markdown.metrics().overscroll.x));
    CHECK(chromeLeft() - chromeBefore == Catch::Approx(shift));
}

TEST_CASE("Scrolling to the bottom of a table note survives repeated layouts", "[core][sakura][markdown]") {
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
    const auto ctx = MakeContext();
    TextMarkdown markdown;
    Host host(markdown);
    markdown.update({.id = "callout-fit", .value = "> [!NOTE]\n> body text", .fontSize = 15.0f});
    const auto measured = MeasureAt(host, ctx, 400.0f);
    host.place(ctx, {0.0f, 0.0f, measured.x, 400.0f});
    CHECK(markdown.metrics().contentHeight == Catch::Approx(measured.y));
}

TEST_CASE("Natural markdown height covers a trailing underline and details box", "[core][sakura][markdown]") {
    MonoFontWindow window;
    auto ctx = MakeContext();
    ctx.render = &window;
    const std::string values[] = {
        "Intro.\n\n## Closing title",
        "Intro.\n\n<details>\n<summary>Closed</summary>\n\nHidden.\n\n</details>",
        "Intro.\n\n<details open>\n<summary>Open</summary>\n\nShown.\n\n</details>",
        "Intro.\n\n<details open>\n<summary>Titled</summary>\n\n## Inner title\n\n</details>",
    };
    for (const auto& value : values) {
        CAPTURE(value);
        TextMarkdown markdown;
        Host host(markdown);
        markdown.update({.id = "trailing", .value = value, .fontSize = 15.0f});
        const auto measured = MeasureAt(host, ctx, 300.0f);
        host.place(ctx, {0.0f, 0.0f, 300.0f, measured.y});
        F32 bottom = 0.0f;
        for (const auto* box : Find<Sakura::Box>(markdown)) {
            for (const auto& instance : box->config().instances) {
                if (instance.visible && !instance.rect.empty()) {
                    bottom = std::max(bottom, instance.rect.bottom());
                }
            }
        }
        CHECK(bottom > 0.0f);
        CHECK(bottom <= measured.y + 0.5f);
    }
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

TEST_CASE("Long code lines stay on one row while paragraphs keep wrapping", "[core][sakura][markdown]") {
    MonoFontWindow window;
    auto ctx = MakeContext();
    ctx.render = &window;
    TextMarkdown markdown;
    Host host(markdown);
    markdown.update({.id = "code-nowrap", .value = kLongCode + "\n\n" + kLongParagraph, .fontSize = 15.0f});
    host.place(ctx, {0.0f, 0.0f, 200.0f, 400.0f});

    const auto& metrics = markdown.metrics();
    REQUIRE(metrics.sourceLines.size() == 4);
    CHECK(metrics.sourceLines[1].height == Catch::Approx(kRowHeight));
    CHECK(metrics.sourceLines[2].height == Catch::Approx(kRowHeight));
    CHECK(metrics.sourceLines[3].height > 2.0f * kRowHeight);
    CHECK(metrics.contentWidth <= 200.0f + 1e-3f);
    CHECK(metrics.scrollX == 0.0f);
}

TEST_CASE("Overflowing code blocks scroll horizontally under the wheel", "[core][sakura][markdown]") {
    MonoFontWindow window;
    auto ctx = MakeContext();
    ctx.render = &window;
    TextMarkdown markdown;
    Host host(markdown);
    markdown.update({.id = "code-scroll", .value = kLongCode + "\n\n" + kLongParagraph, .fontSize = 15.0f});
    const Jetstream::Rect frame{0.0f, 0.0f, 200.0f, 400.0f};
    host.place(ctx, frame);

    REQUIRE(markdown.metrics().regions.size() == 1);
    const F32 maxScroll = markdown.metrics().regions[0].maxScrollX;
    REQUIRE(maxScroll > 0.0f);
    CHECK(markdown.metrics().regions[0].scrollX == 0.0f);
    const auto codeY = RowCenter(markdown, 1);
    const auto start = DrawnRect(markdown, "value");
    REQUIRE(start.has_value());

    REQUIRE(host.send(Wheel(100.0f, codeY, -0.1f, 0.0f)));
    host.place(ctx, frame);
    const F32 scrolled = markdown.metrics().regions[0].scrollX;
    CHECK(scrolled > 0.0f);
    CHECK(markdown.metrics().scrollX == 0.0f);
    const auto moved = DrawnRect(markdown, "value");
    REQUIRE(moved.has_value());
    CHECK(moved->x == Catch::Approx(start->x - scrolled));
    CHECK(moved->y == Catch::Approx(start->y));

    MouseEvent shifted = Wheel(100.0f, codeY, 0.0f, -0.1f);
    shifted.modifiers.shift = true;
    REQUIRE(host.send(shifted));
    host.place(ctx, frame);
    CHECK(markdown.metrics().regions[0].scrollX > scrolled);

    REQUIRE(host.send(Wheel(100.0f, codeY, -1000.0f, 0.0f)));
    host.place(ctx, frame);
    CHECK(markdown.metrics().regions[0].scrollX == Catch::Approx(maxScroll));
    CHECK_FALSE(host.send(Wheel(100.0f, codeY, -1.0f, 0.0f)));

    CHECK_FALSE(host.send(Wheel(100.0f, RowCenter(markdown, 3), -1.0f, 0.0f)));
    host.place(ctx, frame);
    CHECK(markdown.metrics().regions[0].scrollX == Catch::Approx(maxScroll));
    CHECK(markdown.metrics().scrollX == 0.0f);
}

TEST_CASE("Code block headers stay in place while the code scrolls and copy the code on click",
          "[core][sakura][markdown]") {
    MonoFontWindow window;
    auto ctx = MakeContext();
    ctx.render = &window;
    TextMarkdown markdown;
    Host host(markdown);
    markdown.update({.id = "code-header", .value = kLongCode, .fontSize = 15.0f});
    const Jetstream::Rect frame{0.0f, 0.0f, 200.0f, 400.0f};
    host.place(ctx, frame);

    const auto labels = Direct<Sakura::Label>(markdown);
    const auto header = [&](const std::string& str) {
        for (const auto* label : labels) {
            for (const auto& instance : label->config().instances) {
                if (instance.str == str && instance.visible) {
                    return std::optional<Jetstream::Rect>(instance.rect);
                }
            }
        }
        return std::optional<Jetstream::Rect>();
    };
    const auto language = header("PYTHON");
    const auto copy = header("COPY");
    REQUIRE(language.has_value());
    REQUIRE(copy.has_value());
    CHECK_FALSE(header("1").has_value());
    CHECK(language->right() <= copy->x);
    const auto code = DrawnRect(markdown, "value");
    REQUIRE(code.has_value());
    CHECK(code->x == Catch::Approx(language->x));
    CHECK(code->y > language->bottom() - 1e-3f);

    REQUIRE(host.send(Wheel(100.0f, RowCenter(markdown, 1), -0.5f, 0.0f)));
    host.place(ctx, frame);
    REQUIRE(markdown.metrics().regions[0].scrollX > 0.0f);
    CHECK(header("PYTHON") == language);
    CHECK(header("COPY") == copy);
    const auto scrolled = DrawnRect(markdown, "value");
    REQUIRE(scrolled.has_value());
    CHECK(scrolled->x < code->x);

    const Extent2D<F32> pill = {copy->x + copy->width * 0.5f, copy->y + copy->height * 0.5f};
    const Extent2D<F32> label = {language->x + 4.0f, pill.y};
    CHECK(Sakura::Access::cursor(markdown, pill) == SurfaceCursor::Hand);
    CHECK(Sakura::Access::cursor(markdown, label) != SurfaceCursor::Hand);

    window.viewport.platform().clipboard.clear();
    host.send(Mouse(MouseEventType::Click, label.x, label.y));
    host.send(Mouse(MouseEventType::Release, label.x, label.y));
    CHECK(window.viewport.platform().clipboard.empty());

    CHECK(host.send(Mouse(MouseEventType::Click, pill.x, pill.y)));
    CHECK(window.viewport.platform().clipboard.empty());
    host.send(Mouse(MouseEventType::Release, label.x, label.y));
    CHECK(window.viewport.platform().clipboard.empty());

    CHECK(host.send(Mouse(MouseEventType::Click, pill.x, pill.y)));
    CHECK(host.send(Mouse(MouseEventType::Release, pill.x, pill.y)));
    CHECK(window.viewport.platform().clipboard == "value = compute(" + std::string(120, 'x') + ")\nshort = 1");
}

TEST_CASE("Long code fence tags are shortened instead of cut by the header label", "[core][sakura][markdown]") {
    MonoFontWindow window;
    auto ctx = MakeContext();
    ctx.render = &window;
    TextMarkdown markdown;
    Host host(markdown);
    markdown.update({.id = "code-tag", .value = "```javascript\nlet a = 1\n```", .fontSize = 15.0f});
    host.place(ctx, {0.0f, 0.0f, 400.0f, 200.0f});
    CHECK(DrawnRect(markdown, "JAVASCRIPT").has_value());

    markdown.update({.id = "code-tag", .value = "```" + std::string(80, 'a') + "\nlet a = 1\n```", .fontSize = 15.0f});
    host.place(ctx, {0.0f, 0.0f, 2000.0f, 200.0f});
    const auto drawn = [&] {
        for (const auto* label : Find<Sakura::Label>(markdown)) {
            for (const auto& instance : label->config().instances) {
                if (instance.visible && instance.str.starts_with("AAAA")) {
                    return std::make_pair(instance.str, label->config().maxCharacters);
                }
            }
        }
        return std::make_pair(std::string(), U64{0});
    }();
    REQUIRE_FALSE(drawn.first.empty());
    CHECK(drawn.first.ends_with("..."));
    CHECK(drawn.first.size() <= drawn.second);
}

TEST_CASE("Details sections collapse behind their summary and toggle on click", "[core][sakura][markdown]") {
    MonoFontWindow window;
    auto ctx = MakeContext();
    ctx.render = &window;
    TextMarkdown markdown;
    Host host(markdown);
    markdown.update({
        .id = "details",
        .value = "<details>\n<summary>View License</summary>\n\nMIT License text.\n\n</details>\n\nAfter.",
        .fontSize = 15.0f,
    });
    const Jetstream::Rect frame{0.0f, 0.0f, 300.0f, 400.0f};
    host.place(ctx, frame);
    REQUIRE(markdown.metrics().sourceLines.size() == 2);
    CHECK_FALSE(DrawnRect(markdown, "MIT License text.").has_value());

    const auto toggle = [&] {
        const F32 y = RowCenter(markdown, 0);
        CHECK(host.send(Mouse(MouseEventType::Click, 40.0f, y)));
        host.send(Mouse(MouseEventType::Release, 40.0f, y));
        host.place(ctx, frame);
    };
    toggle();
    CHECK(markdown.metrics().sourceLines.size() == 3);
    CHECK(DrawnRect(markdown, "MIT License text.").has_value());
    toggle();
    CHECK(markdown.metrics().sourceLines.size() == 2);
    CHECK_FALSE(DrawnRect(markdown, "MIT License text.").has_value());
}

TEST_CASE("Horizontal wheel scrolling an overflowing code block repaints without rebuilding",
          "[core][sakura][markdown][retention]") {
    MonoFontWindow window;
    TextMarkdown markdown;
    Host host(markdown);
    auto ctx = MakeContext();
    ctx.render = &window;
    Frames frames(host, {0.0f, 0.0f, 200.0f, 300.0f}, ctx);
    const auto update = [&] {
        markdown.update({.id = "code-retention", .value = kLongCode + "\n\n" + kLongParagraph, .fontSize = 15.0f});
    };
    REQUIRE(frames.settle(update));
    REQUIRE(markdown.metrics().regions.size() == 1);
    REQUIRE(markdown.metrics().regions[0].maxScrollX > 0.0f);

    CHECK(host.send(Wheel(100.0f, RowCenter(markdown, 1), -0.2f, 0.0f)));
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(markdown.metrics().regions[0].scrollX > 0.0f);
    CHECK(frames.run(update).clean());
    CHECK(Retains(frames, update));
    CHECK(frames.rebuilds == 1);
    CHECK(frames.restructures == 0);
}

TEST_CASE("Markdown font size changes repaint without rebuilding", "[core][sakura][markdown][retention]") {
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

TEST_CASE("Offscreen list markers leave room for the numbers in view", "[core][sakura][markdown]") {
    const auto ctx = MakeContext();
    TextMarkdown markdown;
    Host host(markdown);
    std::string value;
    for (int i = 0; i < 300; ++i) {
        value += "- item\n";
    }
    value += "\n1. last\n";
    markdown.update({.id = "long-list", .value = value, .fontSize = 15.0f, .scrollbar = true});
    const Jetstream::Rect frame{0.0f, 0.0f, 200.0f, 100.0f};
    host.place(ctx, frame);
    REQUIRE(host.send(Wheel(100.0f, 50.0f, 0.0f, -100000.0f)));
    host.place(ctx, frame);

    bool numbered = false;
    for (auto* label : Find<Sakura::Label>(markdown)) {
        for (const auto& instance : label->config().instances) {
            numbered = numbered || (instance.str == "1." && instance.visible);
        }
    }
    CHECK(numbered);
}

TEST_CASE("Tables far down a note keep their chrome when scrolled into view", "[core][sakura][markdown]") {
    const auto ctx = MakeContext();
    TextMarkdown markdown;
    Host host(markdown);
    std::string value;
    for (int i = 0; i < 100; ++i) {
        value += "text\n\n";
    }
    value += "| a | b |\n|---|---|\n| c | d |\n";
    markdown.update({.id = "late-table", .value = value, .fontSize = 15.0f, .scrollbar = false});
    const Jetstream::Rect frame{0.0f, 0.0f, 200.0f, 100.0f};
    host.place(ctx, frame);
    REQUIRE(host.send(Wheel(100.0f, 50.0f, 0.0f, -100000.0f)));
    host.place(ctx, frame);
    REQUIRE(markdown.metrics().scrollY > 0.0f);

    const auto boxes = Direct<Sakura::Box>(markdown);
    REQUIRE(boxes.size() >= 2);
    CHECK_FALSE(boxes[boxes.size() - 2]->config().instances.empty());
}

TEST_CASE("Markdown turns bare URLs into links and collapses GitHub pull and compare URLs",
          "[core][sakura][markdown]") {
    const auto ctx = MakeContext();
    const auto drawn = [&](const char* value) {
        TextMarkdown markdown;
        Host host(markdown);
        markdown.update({.id = "autolink", .value = value, .fontSize = 15.0f});
        host.place(ctx, {0.0f, 0.0f, 800.0f, 200.0f});
        std::string text;
        for (const auto* label : Find<Sakura::Label>(markdown)) {
            for (const auto& instance : label->config().instances) {
                text += instance.str;
            }
        }
        return text;
    };
    const auto pull = drawn("see https://github.com/acme/repo/pull/400 now");
    CHECK(pull.find("#400") != std::string::npos);
    CHECK(pull.find("github.com") == std::string::npos);
    const auto compare = drawn("diff https://github.com/acme/repo/compare/v1...v2.");
    CHECK(compare.find("v1...v2") != std::string::npos);
    CHECK(compare.find("github.com") == std::string::npos);
    CHECK(drawn("see https://example.com/acme/repo/pull/400 now").find("example.com/acme") != std::string::npos);
    CHECK(drawn("see https://evilgithub.com/acme/repo/pull/400").find("evilgithub.com") != std::string::npos);
    CHECK(drawn("see https://example.com/github.com/a/b/pull/1").find("example.com/github.com") != std::string::npos);
}

TEST_CASE("Markdown autolinks keep balanced parentheses and drop trailing punctuation", "[core][sakura][markdown]") {
    const auto ctx = MakeContext();
    TextMarkdown markdown;
    Host host(markdown);
    markdown.update({.id = "parens",
                     .value = "(see https://github.com/a/b/pull/7) and https://github.com/a/b/compare/v1_(x).",
                     .fontSize = 15.0f});
    host.place(ctx, {0.0f, 0.0f, 800.0f, 200.0f});
    std::string plain;
    std::string links;
    std::string text;
    for (const auto* label : Find<Sakura::Label>(markdown)) {
        for (const auto& instance : label->config().instances) {
            if (!instance.visible) {
                continue;
            }
            (label->config().fontName == Sakura::Typography::BodyBoldFont ? links : plain) += instance.str;
            text += instance.str;
        }
    }
    CHECK(plain.find("(see ") != std::string::npos);
    CHECK(plain.find(") and ") != std::string::npos);
    CHECK(links.find("#7") != std::string::npos);
    CHECK(links.find("v1_(x)") != std::string::npos);
    CHECK(links.find("v1_(x).") == std::string::npos);
    CHECK(plain.back() == '.');
    CHECK(text.find("github.com") == std::string::npos);
}

TEST_CASE("Markdown underlines second level headings only", "[core][sakura][markdown]") {
    const auto ctx = MakeContext();
    const auto underlines = [&](const char* value) {
        TextMarkdown markdown;
        Host host(markdown);
        markdown.update({.id = "underline", .value = value, .fontSize = 15.0f});
        host.place(ctx, {0.0f, 0.0f, 800.0f, 400.0f});
        std::vector<Jetstream::Rect> found;
        for (const auto* box : Find<Sakura::Box>(markdown)) {
            for (const auto& instance : box->config().instances) {
                if (instance.visible && instance.rect.height == 1.0f) {
                    found.push_back(instance.rect);
                }
            }
        }
        return std::make_pair(found, markdown.metrics().sourceLines);
    };

    const auto [h2, lines] = underlines("## Overview\nBody.");
    REQUIRE(h2.size() == 1);
    REQUIRE(lines.size() == 2);
    CHECK(h2.front().y > lines[0].top + lines[0].height);
    CHECK(h2.front().y < lines[1].top);
    CHECK(h2.front().width > 700.0f);
    CHECK(underlines("# Title\nBody.").first.empty());
    CHECK(underlines("### Section\nBody.").first.empty());
}

TEST_CASE("Markdown hides lines that are only an HTML comment", "[core][sakura][markdown]") {
    const auto ctx = MakeContext();
    TextMarkdown markdown;
    Host host(markdown);
    markdown.update({.id = "comments", .value = "## Overview\n\n<!-- hook: preview -->\n\nBody text.", .fontSize = 15.0f});
    host.place(ctx, {0.0f, 0.0f, 800.0f, 200.0f});
    std::string text;
    for (const auto* label : Find<Sakura::Label>(markdown)) {
        for (const auto& instance : label->config().instances) {
            text += instance.str;
        }
    }
    CHECK(text.find("hook") == std::string::npos);
    CHECK(text.find("Body text.") != std::string::npos);
    CHECK(markdown.metrics().sourceLines.size() == 2);
}

TEST_CASE("Markdown footnotes number citations and list the references at the end", "[core][sakura][markdown]") {
    const auto ctx = MakeContext();
    TextMarkdown markdown;
    Host host(markdown);
    markdown.update({.id = "footnotes",
                     .value = "First.[^rtl] Second.[^biq] Again.[^rtl] Missing.[^none]\n\n"
                              "## References\n\n"
                              "[^biq]: BIQ spec.\n"
                              "[^rtl]: Osmocom, *rtl-sdr*.\n"
                              "[^unused]: Never cited.",
                     .fontSize = 15.0f});
    host.place(ctx, {0.0f, 0.0f, 800.0f, 400.0f});

    std::vector<std::pair<std::string, Jetstream::Rect>> drawn;
    std::string text;
    for (const auto* label : Find<Sakura::Label>(markdown)) {
        for (const auto& instance : label->config().instances) {
            if (instance.visible) {
                drawn.push_back({instance.str, instance.rect});
                text += instance.str;
            }
        }
    }
    const auto count = [&](const std::string& needle) {
        return std::count_if(drawn.begin(), drawn.end(), [&](const auto& entry) { return entry.first == needle; });
    };
    const auto find = [&](const std::string& needle) -> std::optional<Jetstream::Rect> {
        for (const auto& [str, rect] : drawn) {
            if (str.find(needle) != std::string::npos) {
                return rect;
            }
        }
        return std::nullopt;
    };

    CHECK(count("[1]") == 2);
    CHECK(count("[2]") == 1);
    CHECK(text.find("[^none]") != std::string::npos);
    CHECK(text.find("[^rtl]") == std::string::npos);
    CHECK(text.find("Never cited") == std::string::npos);
    CHECK(count("1.") == 1);
    CHECK(count("2.") == 1);
    CHECK(count("2") == 1);

    const auto marker = find("[2]");
    const auto body = find("First.");
    REQUIRE(marker.has_value());
    REQUIRE(body.has_value());
    CHECK(marker->y < body->y);

    const auto heading = find("References");
    const auto first = find("Osmocom");
    const auto second = find("BIQ spec");
    REQUIRE(heading.has_value());
    REQUIRE(first.has_value());
    REQUIRE(second.has_value());
    CHECK(first->y > heading->y);
    CHECK(second->y > first->y);
}

TEST_CASE("Markdown footnote citations and back references scroll to each other", "[core][sakura][markdown]") {
    MonoFontWindow window;
    window.load(Sakura::Typography::BodyFont, jbmm_compressed_data);
    auto ctx = MakeContext();
    ctx.render = &window;
    TextMarkdown markdown;
    Host host(markdown);
    std::string value = "Claim.[^a]\n\n";
    for (int i = 0; i < 40; ++i) {
        value += "filler\n";
    }
    value += "\n[^a]: Source.";
    markdown.update({.id = "footnote-jump", .value = value, .fontSize = 15.0f, .scrollbar = true});
    const Jetstream::Rect frame{0.0f, 0.0f, 400.0f, 120.0f};
    host.place(ctx, frame);

    const auto locate = [&](const std::string& needle) -> std::optional<Jetstream::Rect> {
        for (const auto* label : Find<Sakura::Label>(markdown)) {
            for (const auto& instance : label->config().instances) {
                if (instance.visible && instance.str == needle) {
                    return instance.rect;
                }
            }
        }
        return std::nullopt;
    };

    const auto citation = locate("[1]");
    REQUIRE(citation.has_value());
    REQUIRE(markdown.metrics().scrollY == Catch::Approx(0.0f));
    const F32 y = RowCenter(markdown, 0);
    REQUIRE(host.send(Mouse(MouseEventType::Click, citation->x + citation->width * 0.5f, y)));
    host.send(Mouse(MouseEventType::Release, citation->x + citation->width * 0.5f, y));
    host.place(ctx, frame);
    CHECK(markdown.metrics().scrollY > 0.0f);

    const auto backref = locate(ICON_LC_CORNER_DOWN_LEFT);
    REQUIRE(backref.has_value());
    const F32 lastRow = RowCenter(markdown, markdown.metrics().sourceLines.size() - 1);
    REQUIRE(host.send(Mouse(MouseEventType::Click, backref->x + backref->width * 0.5f, lastRow)));
    host.send(Mouse(MouseEventType::Release, backref->x + backref->width * 0.5f, lastRow));
    host.place(ctx, frame);
    CHECK(markdown.metrics().scrollY == Catch::Approx(0.0f));
}

TEST_CASE("Callouts and quotes ignore blank edge lines and turn inner blank lines into paragraph gaps",
          "[core][sakura][markdown]") {
    const auto ctx = MakeContext();
    const auto height = [&](const char* value) {
        TextMarkdown markdown;
        Host host(markdown);
        markdown.update({.id = "callout-blank", .value = value, .fontSize = 15.0f});
        return MeasureAt(host, ctx, 400.0f).y;
    };
    const F32 tight = height("> [!NOTE]\n> Body.");
    CHECK(height("> [!NOTE]\n>\n> Body.\n>") == Catch::Approx(tight));
    CHECK(height("> Quote.") == Catch::Approx(height(">\n> Quote.\n>")));

    const F32 split = height("> [!NOTE]\n> One.\n>\n> Two.");
    const F32 joined = height("> [!NOTE]\n> One.\n> Two.");
    CHECK(split > joined);
    CHECK(split - joined < kRowHeight);
}

TEST_CASE("Details sections honor the open attribute, nest, and stay literal inside code fences",
          "[core][sakura][markdown]") {
    MonoFontWindow window;
    auto ctx = MakeContext();
    ctx.render = &window;
    TextMarkdown markdown;
    Host host(markdown);
    const Jetstream::Rect frame{0.0f, 0.0f, 400.0f, 400.0f};
    const auto place = [&](const std::string& value) {
        markdown.update({.id = "details-forms", .value = value, .fontSize = 15.0f});
        host.place(ctx, frame);
    };

    place("<details open>\n<summary>Shown</summary>\n\nBody text.\n\n</details>");
    CHECK(DrawnRect(markdown, "Body text.").has_value());

    place("<details open>\n<summary>Outer</summary>\n\n<details open>\n<summary>Inner</summary>\n\nDeep text.\n\n"
          "</details>\n\n</details>\n\nAfter.");
    CHECK(DrawnRect(markdown, "Deep text.").has_value());
    CHECK(DrawnRect(markdown, "After.").has_value());
    CHECK_FALSE(DrawnRect(markdown, "<details").has_value());
    CHECK_FALSE(DrawnRect(markdown, "</details>").has_value());

    place("```html\n<details>\n<summary>Hidden</summary>\n</details>\n```\n\nAfter.");
    CHECK(DrawnRect(markdown, "<details>").has_value());
    CHECK(DrawnRect(markdown, "<summary>Hidden</summary>").has_value());
    CHECK(DrawnRect(markdown, "</details>").has_value());
    CHECK_FALSE(DrawnRect(markdown, "\x1f").has_value());
}

TEST_CASE("Inline details mentions inside a section do not swallow the content after it",
          "[core][sakura][markdown]") {
    MonoFontWindow window;
    auto ctx = MakeContext();
    ctx.render = &window;
    TextMarkdown markdown;
    Host host(markdown);
    const Jetstream::Rect frame{0.0f, 0.0f, 400.0f, 400.0f};
    const auto place = [&](const std::string& value) {
        markdown.update({.id = "details-inline", .value = value, .fontSize = 15.0f});
        host.place(ctx, frame);
    };

    place("<details open>\n<summary>Tags</summary>\n\nWrap it in a <details> tag.\n\n</details>\n\nAfter.");
    CHECK(DrawnRect(markdown, "Wrap it in a").has_value());
    CHECK(DrawnRect(markdown, "After.").has_value());

    place("<details>\n<summary>Hidden tags</summary>\n\nWrap it in a <details> tag.\n\n</details>\n\nAfter.");
    CHECK_FALSE(DrawnRect(markdown, "Wrap it in a").has_value());
    CHECK(DrawnRect(markdown, "After.").has_value());

    place("<details open>\n<summary>Closing tags</summary>\n\nEnds with </details> inline.\n\nStill inside.\n\n</details>\n\n"
          "After.");
    CHECK(DrawnRect(markdown, "Still inside.").has_value());
    CHECK(DrawnRect(markdown, "After.").has_value());
    CHECK_FALSE(DrawnRect(markdown, "</details>").has_value());
}

TEST_CASE("Tables and stat grids inside open details keep the details inset", "[core][sakura][markdown]") {
    MonoFontWindow window;
    auto ctx = MakeContext();
    ctx.render = &window;
    const F32 inset = 15.0f;
    const auto drawn = [&](const std::string& value, F32 width, const std::string& prefix) {
        TextMarkdown markdown;
        Host host(markdown);
        markdown.update({.id = "details-table", .value = value, .fontSize = 15.0f});
        host.place(ctx, {0.0f, 0.0f, width, 400.0f});
        const auto rect = DrawnRect(markdown, prefix);
        REQUIRE(rect.has_value());
        return std::make_pair(rect.value(), markdown.metrics().contentWidth);
    };
    const std::string table = "| a | b | c |\n|---|---|---|\n| 1 | 2 | 3 |";
    const std::string stats = "```stats\nLabel | Value\n```";
    const auto open = [](const std::string& body) {
        return "<details open>\n<summary>More</summary>\n\n" + body + "\n\n</details>";
    };

    const auto plainTable = drawn(table, 400.0f, "a");
    const auto nestedTable = drawn(open(table), 400.0f, "a");
    CHECK(nestedTable.first.x - plainTable.first.x == Catch::Approx(inset));

    const auto narrowTable = drawn(open(table), 120.0f, "c");
    CHECK(narrowTable.second <= 120.0f + 1e-3f);
    CHECK(narrowTable.first.right() <= 120.0f - inset + 1e-3f);

    const auto plainStats = drawn(stats, 400.0f, "Value");
    const auto nestedStats = drawn(open(stats), 400.0f, "Value");
    CHECK(nestedStats.first.x - plainStats.first.x == Catch::Approx(inset));
}

TEST_CASE("Opened details stay open while the content updates and reset when the section changes",
          "[core][sakura][markdown]") {
    MonoFontWindow window;
    auto ctx = MakeContext();
    ctx.render = &window;
    TextMarkdown markdown;
    Host host(markdown);
    const Jetstream::Rect frame{0.0f, 0.0f, 300.0f, 400.0f};
    const auto place = [&](const std::string& summary, const std::string& body) {
        markdown.update({
            .id = "details-live",
            .value = "<details>\n<summary>" + summary + "</summary>\n\n" + body + "\n\n</details>",
            .fontSize = 15.0f,
        });
        host.place(ctx, frame);
    };

    place("Stats", "Live one.");
    CHECK_FALSE(DrawnRect(markdown, "Live one.").has_value());
    const F32 y = RowCenter(markdown, 0);
    CHECK(host.send(Mouse(MouseEventType::Click, 40.0f, y)));
    CHECK(host.send(Mouse(MouseEventType::Release, 40.0f, y)));
    host.place(ctx, frame);
    CHECK(DrawnRect(markdown, "Live one.").has_value());

    place("Stats", "Live two.");
    CHECK(DrawnRect(markdown, "Live two.").has_value());

    place("Other", "Live three.");
    CHECK_FALSE(DrawnRect(markdown, "Live three.").has_value());
}

TEST_CASE("Details summaries toggle only on release over the summary box", "[core][sakura][markdown]") {
    MonoFontWindow window;
    auto ctx = MakeContext();
    ctx.render = &window;
    TextMarkdown markdown;
    Host host(markdown);
    markdown.update({
        .id = "details-press",
        .value = "<details>\n<summary>View License</summary>\n\nMIT License text.\n\n</details>\n\nAfter.",
        .fontSize = 15.0f,
    });
    const Jetstream::Rect frame{0.0f, 0.0f, 300.0f, 400.0f};
    host.place(ctx, frame);

    const F32 summary = RowCenter(markdown, 0);
    const F32 after = RowCenter(markdown, 1);
    CHECK(Sakura::Access::cursor(markdown, {40.0f, summary}) == SurfaceCursor::Hand);
    CHECK(Sakura::Access::cursor(markdown, {40.0f, after}) != SurfaceCursor::Hand);

    CHECK(host.send(Mouse(MouseEventType::Click, 40.0f, summary)));
    host.place(ctx, frame);
    CHECK_FALSE(DrawnRect(markdown, "MIT License text.").has_value());
    host.send(Mouse(MouseEventType::Release, 40.0f, after));
    host.place(ctx, frame);
    CHECK_FALSE(DrawnRect(markdown, "MIT License text.").has_value());

    CHECK(host.send(Mouse(MouseEventType::Click, 40.0f, summary)));
    host.send(Mouse(MouseEventType::Cancel, 40.0f, summary));
    host.send(Mouse(MouseEventType::Release, 40.0f, summary));
    host.place(ctx, frame);
    CHECK_FALSE(DrawnRect(markdown, "MIT License text.").has_value());
}

TEST_CASE("Toggling a details section repaints without rebuilding or restructuring",
          "[core][sakura][markdown][retention]") {
    MonoFontWindow window;
    TextMarkdown markdown;
    Host host(markdown);
    auto ctx = MakeContext();
    ctx.render = &window;
    Frames frames(host, {0.0f, 0.0f, 300.0f, 400.0f}, ctx);
    const auto update = [&] {
        markdown.update({
            .id = "details-retention",
            .value = "<details>\n<summary>View License</summary>\n\nMIT License text.\n\n</details>\n\nAfter.",
            .fontSize = 15.0f,
        });
    };
    REQUIRE(frames.settle(update));
    const F32 y = RowCenter(markdown, 0);

    CHECK(host.send(Mouse(MouseEventType::Click, 40.0f, y)));
    CHECK(host.send(Mouse(MouseEventType::Release, 40.0f, y)));
    const auto opened = frames.run(update);
    CHECK(Kept(opened));
    CHECK(opened.repainted);
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
    CHECK(DrawnRect(markdown, "MIT License text.").has_value());

    CHECK(host.send(Mouse(MouseEventType::Click, 40.0f, y)));
    CHECK(host.send(Mouse(MouseEventType::Release, 40.0f, y)));
    CHECK(Kept(frames.run(update)));
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
    CHECK_FALSE(DrawnRect(markdown, "MIT License text.").has_value());
    CHECK(frames.restructures == 0);
}

TEST_CASE("Opening details mounts new text lanes from the toggle and never during layout",
          "[core][sakura][markdown][retention]") {
    MonoFontWindow window;
    TextMarkdown markdown;
    Host host(markdown);
    auto ctx = MakeContext();
    ctx.render = &window;
    Frames frames(host, {0.0f, 0.0f, 300.0f, 400.0f}, ctx);
    const auto update = [&] {
        markdown.update({
            .id = "details-lanes",
            .value = "<details>\n<summary>Code</summary>\n\n```python\nvalue = 1\n```\n\n</details>\n\nAfter.",
            .fontSize = 15.0f,
        });
    };
    REQUIRE(frames.settle(update));
    REQUIRE_FALSE(Sakura::Access::structureDirty(host));
    const F32 y = RowCenter(markdown, 0);
    const auto toggle = [&] {
        CHECK(host.send(Mouse(MouseEventType::Click, 40.0f, y)));
        CHECK(host.send(Mouse(MouseEventType::Release, 40.0f, y)));
    };

    toggle();
    CHECK(Sakura::Access::structureDirty(host));
    const auto opened = frames.run(update);
    CHECK_FALSE(opened.rebuilt);
    CHECK_FALSE(Sakura::Access::structureDirty(host));
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
    CHECK(DrawnRect(markdown, "value").has_value());
    const U64 restructures = frames.restructures;

    toggle();
    CHECK(Kept(frames.run(update)));
    REQUIRE(frames.settle(update));
    CHECK_FALSE(DrawnRect(markdown, "value").has_value());
    toggle();
    CHECK(Kept(frames.run(update)));
    REQUIRE(frames.settle(update));
    CHECK(DrawnRect(markdown, "value").has_value());
    CHECK(frames.restructures == restructures);
}

TEST_CASE("Markdown heroes stack an upper case eyebrow, a large title and a subtitle above the body",
          "[core][sakura][markdown]") {
    const auto ctx = MakeContext();
    TextMarkdown markdown;
    Host host(markdown);
    markdown.update({
        .id = "hero",
        .value = "Body text.",
        .fontSize = 15.0f,
        .hero = TextMarkdown::Hero{.eyebrow = "guide", .title = "Quick Start", .subtitle = "Run a flowgraph."},
    });
    host.place(ctx, {0.0f, 0.0f, 800.0f, 400.0f});

    const auto& lines = markdown.metrics().sourceLines;
    REQUIRE(lines.size() == 4);
    const auto eyebrow = DrawnRect(markdown, "GUIDE");
    const auto title = DrawnRect(markdown, "Quick Start");
    const auto subtitle = DrawnRect(markdown, "Run a flowgraph.");
    const auto body = DrawnRect(markdown, "Body text.");
    REQUIRE(eyebrow.has_value());
    REQUIRE(title.has_value());
    REQUIRE(subtitle.has_value());
    REQUIRE(body.has_value());
    CHECK(eyebrow->y < title->y);
    CHECK(title->y < subtitle->y);
    CHECK(subtitle->y < body->y);
    CHECK(lines[1].height > 2.0f * lines[3].height);
    CHECK(markdown.headings().empty());
}

TEST_CASE("Soft breaks join wrapped source lines in paragraphs and list items", "[core][sakura][markdown]") {
    const auto ctx = MakeContext();
    const std::string value = "First line\nsecond line\n\n- item one\n  continues here\n- item two";
    const auto lines = [&](bool softBreaks) {
        TextMarkdown markdown;
        Host host(markdown);
        markdown.update({.id = "soft", .value = value, .fontSize = 15.0f, .softBreaks = softBreaks});
        host.place(ctx, {0.0f, 0.0f, 800.0f, 400.0f});
        return std::make_pair(markdown.metrics().sourceLines.size(),
                              DrawnRect(markdown, "First line second line").has_value() &&
                                  DrawnRect(markdown, "item one continues here").has_value());
    };
    const auto hard = lines(false);
    const auto soft = lines(true);
    CHECK(soft.first < hard.first);
    CHECK(soft.second);
    CHECK_FALSE(hard.second);
}

TEST_CASE("Markdown links go to the link handler before the system browser", "[core][sakura][markdown]") {
    MonoFontWindow window;
    window.load(Sakura::Typography::BodyFont, jbmm_compressed_data);
    auto ctx = MakeContext();
    ctx.render = &window;
    TextMarkdown markdown;
    Host host(markdown);
    std::vector<std::string> followed;
    markdown.update({
        .id = "links",
        .value = "Read the [Guide](/docs/quick-start) first.",
        .fontSize = 15.0f,
        .onLink = [&](const std::string& url) {
            followed.push_back(url);
            return true;
        },
    });
    host.place(ctx, {0.0f, 0.0f, 600.0f, 200.0f});
    const auto link = DrawnRect(markdown, "Guide");
    REQUIRE(link.has_value());
    const F32 x = link->x + link->width * 0.5f;
    const F32 y = RowCenter(markdown, 0);
    CHECK(Sakura::Access::cursor(markdown, {x, y}) == SurfaceCursor::Hand);
    CHECK(host.send(Mouse(MouseEventType::Click, x, y)));
    host.send(Mouse(MouseEventType::Release, x, y));
    CHECK(followed == std::vector<std::string>{"/docs/quick-start"});
}

TEST_CASE("Markdown parses links and code inside emphasis", "[core][sakura][markdown]") {
    MonoFontWindow window;
    window.load(Sakura::Typography::BodyFont, jbmm_compressed_data);
    auto ctx = MakeContext();
    ctx.render = &window;
    TextMarkdown markdown;
    Host host(markdown);
    std::vector<std::string> followed;
    markdown.update({
        .id = "emphasis-links",
        .value = "*[Physical channels](https://www.etsi.org/ts_136211.pdf)*, ETSI, **see `fft`**.",
        .fontSize = 15.0f,
        .onLink = [&](const std::string& url) {
            followed.push_back(url);
            return true;
        },
    });
    host.place(ctx, {0.0f, 0.0f, 900.0f, 200.0f});
    CHECK_FALSE(DrawnRect(markdown, "](").has_value());
    CHECK_FALSE(DrawnRect(markdown, "`").has_value());
    const auto link = DrawnRect(markdown, "Physical");
    REQUIRE(link.has_value());
    const F32 x = link->x + link->width * 0.5f;
    const F32 y = RowCenter(markdown, 0);
    CHECK(host.send(Mouse(MouseEventType::Click, x, y)));
    host.send(Mouse(MouseEventType::Release, x, y));
    CHECK(followed == std::vector<std::string>{"https://www.etsi.org/ts_136211.pdf"});

    markdown.update({.id = "emphasis-links", .value = "**bold _both_ bold** and *slant __both__*", .fontSize = 15.0f});
    host.place(ctx, {0.0f, 0.0f, 900.0f, 200.0f});
    U64 combined = 0;
    for (const auto* label : Find<Sakura::Label>(markdown)) {
        for (const auto& instance : label->config().instances) {
            if (instance.visible && instance.str.find("both") != std::string::npos) {
                CHECK(label->config().fontName == Sakura::Typography::BodyBoldItalicFont);
                ++combined;
            }
        }
    }
    CHECK(combined == 2);
}

TEST_CASE("Styled markdown hands links to the newest handler after a callback only change",
          "[core][sakura][markdown]") {
    MonoFontWindow window;
    window.load(Sakura::Typography::BodyFont, jbmm_compressed_data);
    auto ctx = MakeContext();
    ctx.render = &window;
    Sakura::Markdown markdown;
    Host host(markdown);
    std::vector<std::string> followed;
    const auto update = [&](const std::string& tag) {
        markdown.update({
            .id = "links",
            .value = "Read the [Guide](/docs/quick-start) first.",
            .fontSize = 15.0f,
            .onLink = [&followed, tag](const std::string& url) {
                followed.push_back(tag + url);
                return true;
            },
        });
        host.place(ctx, {0.0f, 0.0f, 600.0f, 200.0f});
    };
    update("a:");
    update("b:");
    const auto inner = Find<TextMarkdown>(markdown);
    REQUIRE(inner.size() == 1);
    const auto link = DrawnRect(markdown, "Guide");
    REQUIRE(link.has_value());
    const F32 x = link->x + link->width * 0.5f;
    const F32 y = RowCenter(*inner.front(), 0);
    CHECK(host.send(Mouse(MouseEventType::Click, x, y)));
    host.send(Mouse(MouseEventType::Release, x, y));
    CHECK(followed == std::vector<std::string>{"b:/docs/quick-start"});
}

TEST_CASE("Markdown editor previews hand links to the newest handler", "[core][sakura][markdown]") {
    MonoFontWindow window;
    window.load(Sakura::Typography::BodyFont, jbmm_compressed_data);
    auto ctx = MakeContext();
    ctx.render = &window;
    Sakura::MarkdownEditor editor;
    Host host(editor);
    std::vector<std::string> followed;
    const auto update = [&](const std::string& tag) {
        editor.update({
            .id = "notes",
            .value = "Open [QPSK](https://cyberether.org/gallery/flowgraphs/python-qpsk) next.",
            .onLink = [&followed, tag](const std::string& url) {
                followed.push_back(tag + url);
                return true;
            },
        });
        host.place(ctx, {0.0f, 0.0f, 600.0f, 200.0f});
    };
    update("a:");
    update("b:");
    const auto inner = Find<TextMarkdown>(editor);
    REQUIRE(inner.size() == 1);
    const auto link = DrawnRect(editor, "QPSK");
    REQUIRE(link.has_value());
    const F32 x = link->x + link->width * 0.5f;
    const F32 y = RowCenter(*inner.front(), 0);
    CHECK(host.send(Mouse(MouseEventType::Click, x, y)));
    host.send(Mouse(MouseEventType::Release, x, y));
    CHECK(followed == std::vector<std::string>{"b:https://cyberether.org/gallery/flowgraphs/python-qpsk"});
}

TEST_CASE("Markdown lists its headings and scrolls a heading line into view", "[core][sakura][markdown]") {
    MonoFontWindow window;
    auto ctx = MakeContext();
    ctx.render = &window;
    TextMarkdown markdown;
    Host host(markdown);
    std::string value = "# Title\n\n";
    for (int i = 0; i < 40; ++i) {
        value += "filler\n";
    }
    value += "\n## Later\n\nEnd.";
    markdown.update({.id = "headings", .value = value, .fontSize = 15.0f, .scrollbar = true});
    const Jetstream::Rect frame{0.0f, 0.0f, 400.0f, 120.0f};
    host.place(ctx, frame);

    const auto& headings = markdown.headings();
    REQUIRE(headings.size() == 2);
    CHECK(headings[0].text == "Title");
    CHECK(headings[0].level == 1);
    CHECK(headings[0].line == 0);
    CHECK(headings[1].text == "Later");
    CHECK(headings[1].level == 2);
    CHECK_FALSE(DrawnRect(markdown, "Later").has_value());

    markdown.scrollToLine(headings[1].line);
    host.place(ctx, frame);
    CHECK(markdown.metrics().scrollY > 0.0f);
    const auto later = DrawnRect(markdown, "Later");
    REQUIRE(later.has_value());
    CHECK(later->y >= frame.y);
    CHECK(later->bottom() <= frame.bottom());

    markdown.scrollToLine(0);
    host.place(ctx, frame);
    CHECK(markdown.metrics().scrollY == Catch::Approx(0.0f));
}

TEST_CASE("Markdown centers its text column once the frame is wider than the maximum width",
          "[core][sakura][markdown]") {
    const auto left = [](F32 ratio, F32 width) {
        auto ctx = MakeContext();
        ctx.pixelRatio = ratio;
        Sakura::Markdown markdown;
        Host host(markdown);
        markdown.update({
            .id = "centered",
            .value = "Centered body.",
            .fontSize = 15.0f,
            .padding = Sakura::Padding{10.0f, 10.0f, 10.0f, 10.0f},
            .maxWidth = 200.0f,
        });
        host.place(ctx, {0.0f, 0.0f, width, 200.0f});
        const auto body = DrawnRect(markdown, "Centered body.");
        return body.has_value() ? body->x : -1.0f;
    };
    CHECK(left(1.0f, 200.0f) == Catch::Approx(10.0f).margin(1.0f));
    CHECK(left(1.0f, 300.0f) == Catch::Approx(50.0f).margin(1.0f));
    CHECK(left(1.0f, 600.0f) == Catch::Approx(200.0f).margin(1.0f));
    CHECK(left(2.0f, 1200.0f) == Catch::Approx(400.0f).margin(2.0f));
}

TEST_CASE("Code block scroll offsets reset when the code blocks change", "[core][sakura][markdown]") {
    MonoFontWindow window;
    auto ctx = MakeContext();
    ctx.render = &window;
    TextMarkdown markdown;
    Host host(markdown);
    const Jetstream::Rect frame{0.0f, 0.0f, 200.0f, 400.0f};
    const auto place = [&](const std::string& value) {
        markdown.update({.id = "code-reset", .value = value, .fontSize = 15.0f});
        host.place(ctx, frame);
    };

    place(kLongCode + "\n\nTail.");
    REQUIRE(host.send(Wheel(100.0f, RowCenter(markdown, 1), -0.5f, 0.0f)));
    host.place(ctx, frame);
    const F32 scrolled = markdown.metrics().regions.at(0).scrollX;
    REQUIRE(scrolled > 0.0f);

    place(kLongCode + "\n\nTail changed.");
    CHECK(markdown.metrics().regions.at(0).scrollX == Catch::Approx(scrolled));

    place(kLongCode + "\n\n" + kLongCode);
    REQUIRE(markdown.metrics().regions.size() == 2);
    CHECK(markdown.metrics().regions[0].scrollX == 0.0f);
    CHECK(markdown.metrics().regions[1].scrollX == 0.0f);
}

TEST_CASE("A details section without a summary leaves a nested section's summary alone",
          "[core][sakura][markdown]") {
    MonoFontWindow window;
    auto ctx = MakeContext();
    ctx.render = &window;
    TextMarkdown markdown;
    Host host(markdown);
    markdown.update({.id = "details-nested-summary",
                     .value = "<details open>\n<details open>\n<summary>Inner</summary>\n\nDeep text.\n\n</details>\n"
                              "</details>\n\nAfter.",
                     .fontSize = 15.0f});
    host.place(ctx, {0.0f, 0.0f, 400.0f, 400.0f});
    const auto drawn = [&](const std::string& needle) {
        for (const auto* label : Find<Sakura::Label>(markdown)) {
            for (const auto& instance : label->config().instances) {
                if (instance.visible && instance.str.find(needle) != std::string::npos) {
                    return true;
                }
            }
        }
        return false;
    };
    CHECK(drawn("Inner"));
    CHECK(drawn("Deep text."));
    CHECK(drawn("After."));
    CHECK_FALSE(drawn("</details>"));
    CHECK_FALSE(drawn("<summary>"));
}

TEST_CASE("Nested details sections that close together keep separate bottom borders", "[core][sakura][markdown]") {
    MonoFontWindow window;
    auto ctx = MakeContext();
    ctx.render = &window;
    TextMarkdown markdown;
    Host host(markdown);
    markdown.update({.id = "details-nested-tail",
                     .value = "<details open>\n<summary>Outer</summary>\n\n<details open>\n<summary>Inner</summary>\n\n"
                              "Deep text.\n\n</details>\n</details>",
                     .fontSize = 15.0f});
    const auto measured = MeasureAt(host, ctx, 400.0f);
    host.place(ctx, {0.0f, 0.0f, 400.0f, measured.y});
    std::vector<Jetstream::Rect> boxes;
    for (const auto* box : Find<Sakura::Box>(markdown)) {
        if (box->config().borderWidth <= 0.0f) {
            continue;
        }
        for (const auto& instance : box->config().instances) {
            if (instance.visible && !instance.rect.empty()) {
                boxes.push_back(instance.rect);
            }
        }
    }
    REQUIRE(boxes.size() >= 2);
    std::sort(boxes.begin(), boxes.end(), [](const auto& a, const auto& b) { return a.x < b.x; });
    const auto& outer = boxes.front();
    const auto& inner = boxes[1];
    CHECK(inner.x > outer.x);
    CHECK(outer.bottom() > inner.bottom() + 1.0f);
    CHECK(outer.bottom() <= measured.y + 0.5f);
}

TEST_CASE("A footnote back reference scrolls to a citation wrapped deep inside a long paragraph",
          "[core][sakura][markdown]") {
    MonoFontWindow window;
    window.load(Sakura::Typography::BodyFont, jbmm_compressed_data);
    auto ctx = MakeContext();
    ctx.render = &window;
    TextMarkdown markdown;
    Host host(markdown);
    std::string value = "Start ";
    for (int i = 0; i < 6; ++i) {
        value += kLongParagraph;
    }
    value += "Claim.[^a]\n\n";
    for (int i = 0; i < 40; ++i) {
        value += "filler\n";
    }
    value += "\n[^a]: Source.";
    markdown.update({.id = "footnote-wrapped", .value = value, .fontSize = 15.0f, .scrollbar = true});
    const Jetstream::Rect frame{0.0f, 0.0f, 400.0f, 120.0f};
    host.place(ctx, frame);
    const auto locate = [&](const std::string& needle) -> std::optional<Jetstream::Rect> {
        for (const auto* label : Find<Sakura::Label>(markdown)) {
            for (const auto& instance : label->config().instances) {
                if (instance.visible && instance.str == needle) {
                    return instance.rect;
                }
            }
        }
        return std::nullopt;
    };

    markdown.scrollToLine(markdown.metrics().sourceLines.size() - 1);
    host.place(ctx, frame);
    const auto backref = locate(ICON_LC_CORNER_DOWN_LEFT);
    REQUIRE(backref.has_value());
    const F32 y = backref->y + backref->height * 0.5f;
    REQUIRE(host.send(Mouse(MouseEventType::Click, backref->x + backref->width * 0.5f, y)));
    host.send(Mouse(MouseEventType::Release, backref->x + backref->width * 0.5f, y));
    host.place(ctx, frame);

    const auto citation = locate("[1]");
    REQUIRE(citation.has_value());
    CHECK(citation->y >= frame.y);
    CHECK(citation->bottom() <= frame.bottom());
}

TEST_CASE("One-line details sections close on their own line, alone or nested", "[core][sakura][markdown]") {
    MonoFontWindow window;
    auto ctx = MakeContext();
    ctx.render = &window;
    TextMarkdown markdown;
    Host host(markdown);
    const auto drawn = [&](const std::string& needle) {
        for (const auto* label : Find<Sakura::Label>(markdown)) {
            for (const auto& instance : label->config().instances) {
                if (instance.visible && instance.str.find(needle) != std::string::npos) {
                    return true;
                }
            }
        }
        return false;
    };
    const auto place = [&](const std::string& value) {
        markdown.update({.id = "details-one-line", .value = value, .fontSize = 15.0f});
        host.place(ctx, {0.0f, 0.0f, 400.0f, 400.0f});
    };

    place("Before.\n\n<details open><summary>One</summary>Inline body.</details>\n\nAfter.");
    CHECK(drawn("One"));
    CHECK(drawn("Inline body."));
    CHECK(drawn("After."));
    CHECK_FALSE(drawn("</details>"));

    place("<details open>\n<summary>Outer</summary>\n\n<details><summary>Inner</summary>Hidden.</details>\n\n"
          "Outer body.\n\n</details>\n\nAfter.");
    CHECK(drawn("Inner"));
    CHECK_FALSE(drawn("Hidden."));
    CHECK(drawn("Outer body."));
    CHECK(drawn("After."));
    CHECK_FALSE(drawn("</details>"));
}

TEST_CASE("Soft-break list items stop before a table that follows them", "[core][sakura][markdown]") {
    MonoFontWindow window;
    auto ctx = MakeContext();
    ctx.render = &window;
    TextMarkdown markdown;
    Host host(markdown);
    markdown.update({.id = "list-table", .value = "- item\n| Name | Value |\n|---|---|\n| one | two |",
                     .fontSize = 15.0f, .softBreaks = true});
    host.place(ctx, {0.0f, 0.0f, 400.0f, 400.0f});
    bool item = false;
    for (const auto* label : Find<Sakura::Label>(markdown)) {
        for (const auto& instance : label->config().instances) {
            if (!instance.visible) {
                continue;
            }
            CHECK(instance.str.find('|') == std::string::npos);
            item = item || instance.str.find("item") != std::string::npos;
        }
    }
    CHECK(item);
    CHECK(DrawnRect(markdown, "Name").has_value());
    CHECK(DrawnRect(markdown, "two").has_value());
}

TEST_CASE("Natural markdown height covers the underline of hero-only content", "[core][sakura][markdown]") {
    MonoFontWindow window;
    auto ctx = MakeContext();
    ctx.render = &window;
    TextMarkdown markdown;
    Host host(markdown);
    markdown.update({.id = "hero-only", .value = "", .fontSize = 15.0f,
                     .hero = TextMarkdown::Hero{.eyebrow = "Guide", .title = "Title", .subtitle = "Subtitle"}});
    const auto measured = MeasureAt(host, ctx, 300.0f);
    host.place(ctx, {0.0f, 0.0f, 300.0f, measured.y});
    F32 bottom = 0.0f;
    for (const auto* box : Find<Sakura::Box>(markdown)) {
        for (const auto& instance : box->config().instances) {
            if (instance.visible && !instance.rect.empty()) {
                bottom = std::max(bottom, instance.rect.bottom());
            }
        }
    }
    CHECK(bottom > 0.0f);
    CHECK(bottom <= measured.y + 0.5f);
}

TEST_CASE("A details section closed at the end of a content line ends there", "[core][sakura][markdown]") {
    MonoFontWindow window;
    auto ctx = MakeContext();
    ctx.render = &window;
    TextMarkdown markdown;
    Host host(markdown);
    const auto drawn = [&](const std::string& needle) {
        for (const auto* label : Find<Sakura::Label>(markdown)) {
            for (const auto& instance : label->config().instances) {
                if (instance.visible && instance.str.find(needle) != std::string::npos) {
                    return true;
                }
            }
        }
        return false;
    };
    const auto place = [&](const std::string& value) {
        markdown.update({.id = "details-tail-close", .value = value, .fontSize = 15.0f});
        host.place(ctx, {0.0f, 0.0f, 400.0f, 400.0f});
    };

    place("<details>\n<summary>Folded</summary>\n\nHidden.</details>\n\nAfter.");
    CHECK_FALSE(drawn("Hidden."));
    CHECK(drawn("After."));

    place("<details open>\n<summary>Shown</summary>\n\nVisible.</details>\n\nAfter.");
    CHECK(drawn("Visible."));
    CHECK(drawn("After."));
    CHECK_FALSE(drawn("</details>"));
}

TEST_CASE("Long markdown with a details chevron keeps its layout cache when the icon font has zero digit advance",
          "[core][sakura][markdown]") {
    SakuraTest::FontWindow window;
    window.load("default_mono", jbmm_compressed_data);
    window.load("default_body", inter_regular_compressed_data);
    window.load("default_icons", lucide_compressed_data);
    auto ctx = MakeContext();
    ctx.render = &window;
    std::string value = "<details open>\n<summary>License</summary>\n\n";
    for (int i = 0; i < 400; ++i) {
        value += "Line " + std::to_string(i) + " of the license text.\n\n";
    }
    value += "</details>";
    TextMarkdown markdown;
    Host host(markdown);
    markdown.update({.id = "long-details", .value = value, .fontSize = 15.0f});
    host.place(ctx, {0.0f, 0.0f, 800.0f, 600.0f});
    CHECK(markdown.metrics().sourceLines.size() > 400);

    // Exercise the grid's measurement cache directly, counting style requests
    // instead of judging the machine's speed. Loaded icon fonts can have
    // zero-width digit glyphs without invalidating cached measurements.
    Render::Components::Text icons({.font = window.font("default_icons")});
    REQUIRE(icons.advance("0", 1.0f) == 0.0f);
    struct MeasurableGrid : Sakura::TextGrid {
        using Sakura::TextGrid::measure;
    } grid;
    U64 styleRequests = 0;
    const std::vector<std::vector<Sakura::TextGrid::StyleId>> styles;
    grid.update({
        .value = std::string(ICON_LC_CHEVRON_DOWN) + " License\n" + value,
        .fontName = "default_body",
        .monospace = false,
        .wrap = Sakura::TextGrid::Wrap::Word,
        .styler = [&](const std::vector<std::string>&, U64) -> const auto& {
            ++styleRequests;
            return styles;
        },
    });
    const auto measured = grid.measure(ctx, {800.0f, kInfinite});
    REQUIRE(styleRequests > 0);
    const auto requests = styleRequests;
    CHECK(grid.measure(ctx, {800.0f, kInfinite}) == measured);
    CHECK(styleRequests == requests);
}

TEST_CASE("Code block copy buttons brighten while hovered", "[core][sakura][markdown]") {
    MonoFontWindow window;
    auto ctx = MakeContext();
    ctx.render = &window;
    ctx.palette = DistinctPalette();
    TextMarkdown markdown;
    Host host(markdown);
    markdown.update({.id = "copy-hover", .value = kLongCode, .fontSize = 15.0f});
    const Jetstream::Rect frame{0.0f, 0.0f, 200.0f, 400.0f};
    host.place(ctx, frame);
    const auto copy = [&] {
        for (const auto* label : Direct<Sakura::Label>(markdown)) {
            for (const auto& instance : label->config().instances) {
                if (instance.str == "COPY" && instance.visible) {
                    return std::optional<Sakura::Label::Instance>(instance);
                }
            }
        }
        return std::optional<Sakura::Label::Instance>();
    };
    const auto idle = copy();
    REQUIRE(idle.has_value());
    CHECK(idle->color == ctx.color("text_secondary"));

    const F32 x = idle->rect.x + idle->rect.width * 0.5f;
    const F32 y = idle->rect.y + idle->rect.height * 0.5f;
    host.send(Mouse(MouseEventType::Move, x, y));
    host.place(ctx, frame);
    const auto hovered = copy();
    REQUIRE(hovered.has_value());
    CHECK(hovered->color == ctx.color("text_primary"));

    host.send(Mouse(MouseEventType::Leave, x, y));
    host.place(ctx, frame);
    REQUIRE(copy().has_value());
    CHECK(copy()->color == ctx.color("text_secondary"));
}

TEST_CASE("Markdown links render bold in the text color with an underline", "[core][sakura][markdown]") {
    SakuraTest::FontWindow window;
    window.load("default_mono", jbmm_compressed_data);
    window.load("default_body", inter_regular_compressed_data);
    window.load("default_body_bold", inter_bold_compressed_data);
    auto ctx = MakeContext();
    ctx.render = &window;
    ctx.palette = DistinctPalette();
    TextMarkdown markdown;
    Host host(markdown);
    markdown.update({.id = "links", .value = "Read [the docs](https://example.com) now.", .fontSize = 15.0f});
    host.place(ctx, {0.0f, 0.0f, 800.0f, 200.0f});

    std::optional<Sakura::Label::Instance> link;
    for (const auto* label : Find<Sakura::Label>(markdown)) {
        for (const auto& instance : label->config().instances) {
            if (instance.visible && instance.str == "the docs") {
                CHECK(label->config().fontName == Sakura::Typography::BodyBoldFont);
                link = instance;
            }
        }
    }
    REQUIRE(link.has_value());
    CHECK(link->color == ctx.color("text_primary"));

    bool underlined = false;
    for (const auto* box : Find<Sakura::Box>(markdown)) {
        for (const auto& instance : box->config().instances) {
            if (instance.visible && instance.rect.x == Catch::Approx(link->rect.x) &&
                instance.rect.width == Catch::Approx(link->rect.width) && instance.rect.height <= 2.0f &&
                instance.rect.y > link->rect.y + link->rect.height * 0.5f) {
                underlined = underlined || instance.backgroundColor == link->color;
            }
        }
    }
    CHECK(underlined);
}

TEST_CASE("Code block copy buttons drop their hover when scrolled away from a still pointer",
          "[core][sakura][markdown]") {
    MonoFontWindow window;
    auto ctx = MakeContext();
    ctx.render = &window;
    ctx.palette = DistinctPalette();
    TextMarkdown markdown;
    Host host(markdown);
    std::string value = "Intro.\n\nMore.\n\nEven more.\n\nLast.\n\n" + kLongCode + "\n\n";
    for (int line = 0; line < 60; ++line) {
        value += "Line " + std::to_string(line) + ".\n\n";
    }
    markdown.update({.id = "copy-scroll", .value = value, .fontSize = 15.0f, .scrollbar = true});
    const Jetstream::Rect frame{0.0f, 0.0f, 200.0f, 300.0f};
    host.place(ctx, frame);
    const auto copy = [&] {
        for (const auto* label : Direct<Sakura::Label>(markdown)) {
            for (const auto& instance : label->config().instances) {
                if (instance.str == "COPY" && instance.visible) {
                    return std::optional<Sakura::Label::Instance>(instance);
                }
            }
        }
        return std::optional<Sakura::Label::Instance>();
    };
    const auto idle = copy();
    REQUIRE(idle.has_value());
    const F32 x = idle->rect.x + idle->rect.width * 0.5f;
    const F32 y = idle->rect.y + idle->rect.height * 0.5f;
    host.send(Mouse(MouseEventType::Move, x, y));
    host.place(ctx, frame);
    REQUIRE(copy()->color == ctx.color("text_primary"));

    host.send(Wheel(x, y, -1.0f));
    host.place(ctx, frame);
    REQUIRE(markdown.metrics().scrollY > 0.0f);
    const auto scrolled = copy();
    REQUIRE(scrolled.has_value());
    CHECK_FALSE(scrolled->rect.contains(x, y));
    CHECK(scrolled->color == ctx.color("text_secondary"));
}

TEST_CASE("Code block copy buttons drop their hover when a translated parent pans them away",
          "[core][sakura][markdown]") {
    MonoFontWindow window;
    auto ctx = MakeContext();
    ctx.render = &window;
    ctx.palette = DistinctPalette();
    TextMarkdown markdown;
    markdown.update({.id = "copy-pan", .value = kLongCode, .fontSize = 15.0f});
    Place place(markdown, {0.0f, 0.0f, 200.0f, 300.0f});
    const Jetstream::Rect screen{0.0f, 0.0f, 400.0f, 400.0f};
    place.place(ctx, screen);
    const auto copy = [&] {
        for (const auto* label : Direct<Sakura::Label>(markdown)) {
            for (const auto& instance : label->config().instances) {
                if (instance.str == "COPY" && instance.visible) {
                    return std::optional<Sakura::Label::Instance>(instance);
                }
            }
        }
        return std::optional<Sakura::Label::Instance>();
    };
    const auto idle = copy();
    REQUIRE(idle.has_value());
    const F32 x = idle->rect.x + idle->rect.width * 0.5f;
    const F32 y = idle->rect.y + idle->rect.height * 0.5f;
    place.send(Mouse(MouseEventType::Move, x, y));
    place.place(ctx, screen);
    REQUIRE(copy()->color == ctx.color("text_primary"));

    place.pan({0.0f, 80.0f});
    place.place(ctx, screen);
    REQUIRE(copy().has_value());
    CHECK(copy()->color == ctx.color("text_secondary"));

    place.pan({0.0f, 0.0f});
    place.place(ctx, screen);
    REQUIRE(copy().has_value());
    CHECK(copy()->color == ctx.color("text_primary"));
}
