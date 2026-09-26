#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <jetstream/render/sakura/components/retained/text_markdown.hh>

#include "render/sakura/context.hh"

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

 protected:
    void layout(const Sakura::Context& ctx) override {
        layoutChild(ctx, markdown, frame());
    }
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
     22.8f, 18.75f},
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
