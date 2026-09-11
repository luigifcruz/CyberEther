#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <jetstream/render/sakura/components/retained/text_grid.hh>

#include "render/sakura/context.hh"

using namespace Jetstream;

namespace {

struct MeasuredTextGrid : Sakura::Retained::TextGrid {
    using TextGrid::measure;
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
