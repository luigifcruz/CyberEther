#include <catch2/catch_test_macros.hpp>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/node/progress_bar.hh>

#include "component_harness.hh"
#include "compositor/default/views/flowgraph/editor/metrics/progressbar.hh"
#include "harness.hh"
#include "scene.hh"

#include <limits>
#include <optional>
#include <string>
#include <utility>

using namespace Jetstream;
using namespace SakuraTest;

namespace {

const Parser::Map kAbove{{"type", "progressbar"}, {"warning_above", 0.5f}, {"error_above", 0.75f}};
const Parser::Map kBelow{{"type", "progressbar"}, {"warning_below", 0.95f}, {"error_below", 0.8f}};

std::optional<std::string> FillKey(const Parser::Map& format, const F32 value) {
    const Scene scene;
    const auto ctx = scene.context();
    FlowgraphMetricProgressBar metric;
    metric.update({
        .id = "metric",
        .format = format,
        .value = std::pair<std::string, F32>{"overlay", value},
    });
    Host host(metric.component());
    host.place(ctx, {0.0f, 0.0f, 300.0f, 60.0f});
    for (const auto* key : {"button_active", "progress_warning", "progress_error"}) {
        for (const auto* box : Find<Sakura::Box>(metric.component())) {
            for (const auto& instance : box->config().instances) {
                if (instance.visible && instance.backgroundColor == ctx.color(key)) {
                    return key;
                }
            }
        }
    }
    return std::nullopt;
}

}  // namespace

TEST_CASE("Progress bar metrics turn amber and red only past their thresholds",
          "[core][sakura][metrics][progressbar]") {
    CHECK(FillKey(kAbove, 0.4f) == "button_active");
    CHECK(FillKey(kAbove, 0.5f) == "button_active");
    CHECK(FillKey(kAbove, 0.6f) == "progress_warning");
    CHECK(FillKey(kAbove, 0.75f) == "progress_warning");
    CHECK(FillKey(kAbove, 0.9f) == "progress_error");

    CHECK(FillKey(kBelow, 1.0f) == "button_active");
    CHECK(FillKey(kBelow, 0.95f) == "button_active");
    CHECK(FillKey(kBelow, 0.9f) == "progress_warning");
    CHECK(FillKey(kBelow, 0.7f) == "progress_error");

    CHECK(FillKey({{"type", "progressbar"}}, 0.9f) == "button_active");
    CHECK(FillKey({{"type", "progressbar"}, {"warning_above", 0.0f}, {"error_above", 0.01f}}, 0.005f) ==
          "progress_warning");
}

TEST_CASE("Progress bar metrics without a reading stay empty and uncolored", "[core][sakura][metrics][progressbar]") {
    CHECK_FALSE(FillKey(kBelow, std::numeric_limits<F32>::quiet_NaN()).has_value());
    CHECK(FillKey(kBelow, 1.0f) == "button_active");
}

TEST_CASE("Progress bar color changes fade instead of switching", "[core][sakura][metrics][progressbar]") {
    const Scene scene;
    auto ctx = scene.context();
    Sakura::NodeProgressBar bar;
    Host host(bar);
    const Rect bounds = {0.0f, 0.0f, 300.0f, 25.0f};
    const auto fill = [&]() -> std::optional<ColorRGBA<F32>> {
        for (const auto* box : Find<Sakura::Box>(bar)) {
            for (const auto& instance : box->config().instances) {
                if (instance.visible && instance.backgroundColor != ctx.color("card")) {
                    return instance.backgroundColor;
                }
            }
        }
        return std::nullopt;
    };

    ctx.time = 1.0;
    bar.update({.id = "bar", .value = 0.5f, .colorKey = "button_active"});
    host.place(ctx, bounds);
    CHECK(fill() == ctx.color("button_active"));

    bar.update({.id = "bar", .value = 0.5f, .colorKey = "progress_error"});
    host.place(ctx, bounds);
    CHECK(fill() == ctx.color("button_active"));

    ctx.time = 1.125;
    host.place(ctx, bounds);
    const auto midway = fill();
    REQUIRE(midway.has_value());
    CHECK(*midway != ctx.color("button_active"));
    CHECK(*midway != ctx.color("progress_error"));

    ctx.time = 1.5;
    host.place(ctx, bounds);
    CHECK(fill() == ctx.color("progress_error"));
}
