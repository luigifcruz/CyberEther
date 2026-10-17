#include <catch2/catch_test_macros.hpp>

#include "compositor/default/actions/flowgraph.hh"
#include "compositor/default/model/surface_theme.hh"
#include "compositor/default/themes.hh"

#include <memory>

using namespace Jetstream;

TEST_CASE("Surface theme follows the placement background and the plot keys of each theme",
          "[core][compositor][surface-theme]") {
    for (const auto& [key, palette] : themes) {
        const auto detached = MakeSurfaceTheme(palette, SurfacePlacementType::Detached);
        const auto attached = MakeSurfaceTheme(palette, SurfacePlacementType::Attached);
        CHECK(detached.background == palette.at("tile_bg"));
        CHECK(attached.background.r == palette.at("node_background").r);
        CHECK(attached.background.a == 1.0f);
        CHECK(detached.foreground == palette.at("text_primary"));
        CHECK(detached.secondary == palette.at("text_secondary"));
        CHECK(detached.grid == palette.at("plot_grid"));
        CHECK(detached.majorGrid == palette.at("plot_grid_major"));
        CHECK(detached.accent == palette.at("plot_accent"));
        CHECK(detached.halo == palette.at("plot_halo"));
        CHECK(detached.pill == palette.at("pill_field_bg"));
    }
    CHECK(MakeSurfaceTheme({}, SurfacePlacementType::Detached) == SurfaceTheme{});
}

TEST_CASE("Resized surfaces are tracked and rethemed when the theme changes",
          "[core][compositor][surface-theme]") {
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    FlowgraphActions actions(state, callbacks);
    REQUIRE(themes.size() >= 2);
    const auto first = themes.begin();
    const auto second = std::next(first);
    state.sakura.colorMap = first->second;

    auto live = std::make_shared<Module::Surface>();
    auto gone = std::make_shared<Module::Surface>();
    for (const auto& surface : {live, gone}) {
        actions.handle(MailResizeSurface{
            .surface = surface,
            .placement = SurfacePlacement::Detached,
            .resize = {.framebufferSize = {640, 480}, .scale = 2.0f},
        });
    }
    REQUIRE(state.flowgraph.surfaces.size() == 2);
    const auto& tracked = state.flowgraph.surfaces.at(live.get()).event;
    CHECK(tracked.size == Extent2D<U64>{640, 480});
    CHECK(tracked.scale == 2.0f);
    CHECK(tracked.theme == MakeSurfaceTheme(first->second, SurfacePlacementType::Detached));

    gone.reset();
    RethemeSurfaces(state.flowgraph.surfaces, second->second);
    REQUIRE(state.flowgraph.surfaces.size() == 1);
    const auto& rethemed = state.flowgraph.surfaces.at(live.get()).event;
    CHECK(rethemed.theme == MakeSurfaceTheme(second->second, SurfacePlacementType::Detached));
    CHECK(rethemed.size == Extent2D<U64>{640, 480});
    CHECK(rethemed.placement == SurfacePlacementType::Detached);
}

TEST_CASE("Every theme defines the same palette keys including the surface theme keys",
          "[core][compositor][surface-theme][palette]") {
    REQUIRE(themes.contains("Dark"));
    REQUIRE(themes.contains("Light"));
    const auto& dark = themes.at("Dark");
    for (const auto& [key, palette] : themes) {
        CAPTURE(key);
        for (const auto& [color, _] : dark) {
            CAPTURE(color);
            CHECK(palette.contains(color));
        }
        for (const auto& [color, _] : palette) {
            CAPTURE(color);
            CHECK(dark.contains(color));
        }
        for (const char* color : {"node_background", "tile_bg", "text_primary", "text_secondary", "plot_grid",
                                  "plot_grid_major", "plot_accent", "plot_halo", "pill_field_bg"}) {
            CAPTURE(color);
            CHECK(palette.contains(color));
        }
    }
}
