#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "compositor/default/views/flowgraph/editor/metrics/label.hh"
#include "harness.hh"

#include <algorithm>
#include <map>
#include <string>

using namespace Jetstream;

namespace {

struct MetricGeometry {
    F32 availableWidth = 0.0f;
    F32 width = 0.0f;
    F32 height = 0.0f;
    F32 lineHeight = 0.0f;
    std::map<F32, F32> rightEdges;
};

MetricGeometry RenderMetric(SakuraTest::HeadlessUi& ui, Sakura::Node& node,
                            FlowgraphMetricLabel& metric, const std::string& value,
                            const F32 width) {
    node.update({.id = "metric-node", .dimensions = {width, 0.0f}});
    metric.update({.id = "metric", .format = "label", .value = value});
    MetricGeometry result;
    ui.editorFrame([&] {
        node.render(ui.sakura(), [&](const Sakura::Context& ctx) {
            const auto origin = ImGui::GetCursorScreenPos();
            result.availableWidth = ImGui::GetContentRegionAvail().x;
            result.lineHeight = ImGui::GetTextLineHeight();
            const auto* drawList = ImGui::GetWindowDrawList();
            const int firstVertex = drawList->VtxBuffer.Size;
            metric.render(ctx);
            const auto size = ImGui::GetItemRectSize();
            result.width = size.x;
            result.height = size.y;
            for (int i = firstVertex; i + 3 < drawList->VtxBuffer.Size; i += 4) {
                const auto& topLeft = drawList->VtxBuffer[i].pos;
                const auto& bottomRight = drawList->VtxBuffer[i + 2].pos;
                REQUIRE(topLeft.x >= origin.x - 1.0f);
                REQUIRE(bottomRight.x <= origin.x + result.availableWidth + 1.0f);
                auto& edge = result.rightEdges[topLeft.y - origin.y];
                edge = std::max(edge, bottomRight.x - origin.x);
            }
        });
    });
    return result;
}

}  // namespace

TEST_CASE("Metric label values right-align short text and clip long text to one line",
          "[core][sakura][metrics][label]") {
    SakuraTest::HeadlessUi ui(1.0f, ImVec2(800.0f, 600.0f));
    Sakura::Node node;
    FlowgraphMetricLabel metric;

    const auto shortValue = RenderMetric(ui, node, metric, "AAAA", 80.0f);
    REQUIRE(shortValue.rightEdges.size() == 1);
    REQUIRE(shortValue.height == Catch::Approx(shortValue.lineHeight));
    const F32 rightEdge = shortValue.rightEdges.begin()->second;
    REQUIRE(rightEdge > shortValue.availableWidth - 3.0f);

    std::string value;
    SECTION("words") {
        value = "AAAA AAAA AAAA AAAA AAAA";
    }
    SECTION("unbroken identifier") {
        value = std::string(40, 'A');
    }
    SECTION("explicit lines and surrounding whitespace") {
        value = "AAAA AAAA AAAA   \nAA\n\nAAAA";
    }
    const auto clipped = RenderMetric(ui, node, metric, value, 80.0f);
    REQUIRE(clipped.rightEdges.size() == 1);
    REQUIRE(clipped.width <= clipped.availableWidth + 1.0f);
    REQUIRE(clipped.height == Catch::Approx(clipped.lineHeight));
}

TEST_CASE("Device metric labels stay within the node width without growing taller",
          "[core][sakura][metrics][label]") {
    SakuraTest::HeadlessUi ui(1.0f, ImVec2(800.0f, 600.0f));
    Sakura::Node node;
    FlowgraphMetricLabel metric;
    const std::string value = "LimeSDR Mini [USB 3.0] 1D4249D8AEC2A9";

    const auto narrow = RenderMetric(ui, node, metric, value, 150.0f);
    REQUIRE(narrow.width <= narrow.availableWidth + 1.0f);
    REQUIRE(narrow.height == Catch::Approx(narrow.lineHeight));

    const auto wide = RenderMetric(ui, node, metric, value, 500.0f);
    REQUIRE(wide.width <= wide.availableWidth + 1.0f);
    REQUIRE(wide.height == Catch::Approx(wide.lineHeight));

    const auto narrowedAgain = RenderMetric(ui, node, metric, value, 150.0f);
    REQUIRE(narrowedAgain.width <= narrowedAgain.availableWidth + 1.0f);
    REQUIRE(narrowedAgain.height == Catch::Approx(narrow.height));
}
