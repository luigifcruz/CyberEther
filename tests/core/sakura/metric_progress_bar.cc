#include <catch2/catch_test_macros.hpp>

#include <imgui_internal.h>

#include "compositor/default/views/flowgraph/editor/metrics/progressbar.hh"
#include "render/sakura/components/node/base.hh"
#include "harness.hh"

#include <cmath>
#include <limits>
#include <optional>
#include <set>
#include <string>

using namespace Jetstream;

namespace {

const Sakura::Palette kPalette{
    {"button_active", {0.2f, 0.4f, 0.6f, 1.0f}},
    {"progress_warning", {0.9f, 0.6f, 0.1f, 1.0f}},
    {"progress_error", {0.8f, 0.1f, 0.1f, 1.0f}},
};

const Parser::Map kAbove{{"type", "progressbar"}, {"warning_above", 0.5f}, {"error_above", 0.75f}};
const Parser::Map kBelow{{"type", "progressbar"}, {"warning_below", 0.95f}, {"error_below", 0.8f}};

ImVec4 PaletteColor(const std::string& key) {
    const auto& color = kPalette.at(key);
    return {color.r, color.g, color.b, color.a};
}

ImU32 Packed(const std::string& key) {
    return ImGui::ColorConvertFloat4ToU32(PaletteColor(key));
}

F32 Distance(const ImVec4& a, const ImVec4& b) {
    return std::abs(a.x - b.x) + std::abs(a.y - b.y) + std::abs(a.z - b.z);
}

struct MetricFixture {
    SakuraTest::HeadlessUi ui{1.0f, ImVec2(800.0f, 600.0f)};
    Sakura::Node node;

    std::optional<ImU32> fill(FlowgraphMetricProgressBar& metric, const Parser::Map& format, const F32 value) {
        node.update({.id = "metric-node", .dimensions = {300.0f, 0.0f}});
        metric.update({
            .id = "metric",
            .format = format,
            .value = std::pair<std::string, F32>{"overlay", value},
        });
        auto ctx = ui.sakura();
        ctx.palette = kPalette;
        std::set<ImU32> colors;
        ui.editorFrame([&] {
            node.render(ctx, [&](const Sakura::Context& ctx) {
                const auto& style = ImGui::GetStyle();
                const ImU32 frame = ImGui::GetColorU32(style.Colors[ImGuiCol_FrameBg]);
                const ImU32 text = ImGui::GetColorU32(style.Colors[ImGuiCol_Text]);
                const auto* drawList = ImGui::GetWindowDrawList();
                const int firstVertex = drawList->VtxBuffer.Size;
                metric.render(ctx);
                for (int i = firstVertex; i < drawList->VtxBuffer.Size; ++i) {
                    const ImU32 color = drawList->VtxBuffer[i].col;
                    if ((color >> IM_COL32_A_SHIFT) == 0xFF && color != frame && color != text) {
                        colors.insert(color);
                    }
                }
            });
        });
        REQUIRE(colors.size() <= 1);
        if (colors.empty()) {
            return std::nullopt;
        }
        return *colors.begin();
    }
};

}  // namespace

TEST_CASE("Progress bar metrics turn amber and red only past their thresholds",
          "[core][sakura][metrics][progressbar]") {
    MetricFixture fixture;
    const auto colorOf = [&](const Parser::Map& format, const F32 value) {
        FlowgraphMetricProgressBar metric;
        return fixture.fill(metric, format, value);
    };

    REQUIRE(colorOf(kAbove, 0.4f) == Packed("button_active"));
    REQUIRE(colorOf(kAbove, 0.5f) == Packed("button_active"));
    REQUIRE(colorOf(kAbove, 0.6f) == Packed("progress_warning"));
    REQUIRE(colorOf(kAbove, 0.75f) == Packed("progress_warning"));
    REQUIRE(colorOf(kAbove, 0.9f) == Packed("progress_error"));

    REQUIRE(colorOf(kBelow, 1.0f) == Packed("button_active"));
    REQUIRE(colorOf(kBelow, 0.95f) == Packed("button_active"));
    REQUIRE(colorOf(kBelow, 0.9f) == Packed("progress_warning"));
    REQUIRE(colorOf(kBelow, 0.7f) == Packed("progress_error"));

    REQUIRE(colorOf({{"type", "progressbar"}}, 0.9f) == Packed("button_active"));
    REQUIRE(colorOf({{"type", "progressbar"}, {"warning_above", 0.0f}, {"error_above", 0.01f}}, 0.001f) ==
            Packed("progress_warning"));
}

TEST_CASE("Progress bar metrics without a reading stay empty and uncolored",
          "[core][sakura][metrics][progressbar]") {
    MetricFixture fixture;
    FlowgraphMetricProgressBar metric;
    REQUIRE_FALSE(fixture.fill(metric, kBelow, std::numeric_limits<F32>::quiet_NaN()).has_value());
    REQUIRE(fixture.fill(metric, kBelow, 1.0f) == Packed("button_active"));
}

TEST_CASE("Progress bar color changes fade instead of switching",
          "[core][sakura][metrics][progressbar]") {
    MetricFixture fixture;
    FlowgraphMetricProgressBar metric;
    REQUIRE(fixture.fill(metric, kAbove, 0.4f) == Packed("button_active"));

    REQUIRE(fixture.fill(metric, kAbove, 0.9f) == Packed("button_active"));
    std::optional<ImU32> midway;
    for (int frame = 0; frame < 7; ++frame) {
        midway = fixture.fill(metric, kAbove, 0.9f);
    }
    REQUIRE(midway.has_value());
    REQUIRE(*midway != Packed("button_active"));
    REQUIRE(*midway != Packed("progress_error"));

    for (int frame = 0; frame < 10; ++frame) {
        midway = fixture.fill(metric, kAbove, 0.9f);
    }
    REQUIRE(midway == Packed("progress_error"));
}

TEST_CASE("Progress bar colors that change every frame still leave the nominal color",
          "[core][sakura][metrics][progressbar]") {
    MetricFixture fixture;
    FlowgraphMetricProgressBar metric;
    REQUIRE(fixture.fill(metric, kAbove, 0.4f) == Packed("button_active"));

    std::optional<ImU32> shown;
    for (int frame = 0; frame < 60; ++frame) {
        shown = fixture.fill(metric, kAbove, frame % 2 == 0 ? 0.6f : 0.9f);
    }
    REQUIRE(shown.has_value());
    const ImVec4 color = ImGui::ColorConvertU32ToFloat4(*shown);
    const ImVec4 nominal = PaletteColor("button_active");
    const F32 nearest = std::min(Distance(nominal, PaletteColor("progress_warning")),
                                 Distance(nominal, PaletteColor("progress_error")));
    REQUIRE(Distance(color, nominal) > 0.25f * nearest);
}
