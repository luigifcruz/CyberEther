#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "compositor/default/views/flowgraph/editor/metrics/label.hh"
#include "render/sakura/components/node/base.hh"
#include "harness.hh"

#include <algorithm>
#include <limits>
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
    metric.update({.id = "metric", .format = {{"type", "label"}}, .value = value});
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

TEST_CASE("Metric error boxes stay centered and clipped inside nodes",
          "[core][sakura][metrics][error]") {
    for (const F32 scale : {1.0f, 2.0f}) {
        SakuraTest::HeadlessUi ui(scale, ImVec2(1000.0f, 600.0f));
        const Sakura::Palette palette{
            {"text_secondary", {0.0f, 1.0f, 0.0f, 1.0f}},
            {"card", {0.1f, 0.1f, 0.1f, 1.0f}},
            {"table_border_strong", {0.2f, 0.2f, 0.2f, 1.0f}},
        };
        auto ctx = ui.sakura();
        ctx.palette = palette;
        Sakura::Node node;
        FlowgraphMetricLabel metric;
        metric.update({.id = "error-metric", .format = {{"type", "label"}}});

        for (const F32 width : {320.0f, 120.0f, 40.0f, 320.0f}) {
            CAPTURE(scale, width);
            node.update({.id = "error-node", .dimensions = {width, 0.0f}});
            ImVec2 origin;
            F32 availableWidth = 0.0f;
            F32 textWidth = 0.0f;
            F32 height = 0.0f;
            F32 lineHeight = 0.0f;
            const ImDrawList* contentDrawList = nullptr;
            int firstVertex = 0;
            int lastVertex = 0;
            for (int frame = 0; frame < 3; ++frame) {
                ui.editorFrame([&] {
                    node.render(ctx, [&](const Sakura::Context& ctx) {
                        origin = ImGui::GetCursorScreenPos();
                        availableWidth = ImGui::GetContentRegionAvail().x;
                        textWidth = ImGui::CalcTextSize("No metric").x;
                        lineHeight = ImGui::GetTextLineHeight();
                        contentDrawList = ImGui::GetWindowDrawList();
                        firstVertex = contentDrawList->VtxBuffer.Size;
                        metric.render(ctx);
                        lastVertex = contentDrawList->VtxBuffer.Size;
                        height = ImGui::GetCursorScreenPos().y - origin.y;
                    });
                });
            }

            F32 textLeft = std::numeric_limits<F32>::max();
            F32 textRight = std::numeric_limits<F32>::lowest();
            for (int i = firstVertex; i < lastVertex; ++i) {
                const auto& vertex = contentDrawList->VtxBuffer[i];
                if (vertex.col != IM_COL32(0, 255, 0, 255)) {
                    continue;
                }
                textLeft = std::min(textLeft, vertex.pos.x);
                textRight = std::max(textRight, vertex.pos.x);
            }

            REQUIRE(textRight > textLeft);
            REQUIRE(height >= lineHeight + 16.0f * scale);
            REQUIRE(textLeft >= origin.x + 8.0f * scale - 1.0f);
            REQUIRE(textRight <= origin.x + availableWidth - 8.0f * scale + 1.0f);
            if (textWidth <= availableWidth - 16.0f * scale) {
                REQUIRE((textLeft + textRight) * 0.5f ==
                        Catch::Approx(origin.x + availableWidth * 0.5f).margin(2.0f));
            }
        }
    }
}

TEST_CASE("Overlapping metric errors follow node stacking after either node is raised",
          "[core][sakura][metrics][error][stacking]") {
    SakuraTest::HeadlessUi ui(1.0f, ImVec2(800.0f, 600.0f));
    Sakura::Node first;
    Sakura::Node second;
    first.update({.id = "first-error-node", .dimensions = {320.0f, 140.0f},
                  .gridPosition = Extent2D<F32>{60.0f, 60.0f}});
    second.update({.id = "second-error-node", .dimensions = {320.0f, 140.0f},
                   .gridPosition = Extent2D<F32>{90.0f, 60.0f}});
    FlowgraphMetricLabel firstMetric;
    FlowgraphMetricLabel secondMetric;
    firstMetric.update({.id = "first-error", .format = {{"type", "label"}}});
    secondMetric.update({.id = "second-error", .format = {{"type", "label"}}});

    const Sakura::Palette firstPalette{
        {"text_secondary", {0.0f, 1.0f, 0.0f, 1.0f}},
        {"card", {0.1f, 0.1f, 0.1f, 1.0f}},
        {"table_border_strong", {0.2f, 0.2f, 0.2f, 1.0f}},
    };
    auto secondPalette = firstPalette;
    secondPalette["text_secondary"] = {1.0f, 1.0f, 0.0f, 1.0f};
    auto firstContext = ui.sakura();
    firstContext.palette = firstPalette;
    auto secondContext = ui.sakura();
    secondContext.palette = secondPalette;
    const ImU32 firstBackground = IM_COL32(80, 20, 20, 255);
    const ImU32 secondBackground = IM_COL32(20, 20, 80, 255);
    const ImU32 firstText = IM_COL32(0, 255, 0, 255);
    const ImU32 secondText = IM_COL32(255, 255, 0, 255);
    const int firstId = Sakura::Private::NodeEditorObjectId("first-error-node");
    const int secondId = Sakura::Private::NodeEditorObjectId("second-error-node");

    const auto frame = [&] {
        ui.editorFrame([&] {
            const auto render = [&](Sakura::Node& node, FlowgraphMetricLabel& metric,
                                    const Sakura::Context& ctx, ImU32 background) {
                ImNodes::PushColorStyle(ImNodesCol_NodeBackground, background);
                ImNodes::PushColorStyle(ImNodesCol_NodeBackgroundHovered, background);
                ImNodes::PushColorStyle(ImNodesCol_NodeBackgroundSelected, background);
                node.render(ctx, [&](const Sakura::Context& ctx) { metric.render(ctx); });
                ImNodes::PopColorStyle();
                ImNodes::PopColorStyle();
                ImNodes::PopColorStyle();
            };
            // Keep submission order fixed while clicks change the node depth order.
            render(first, firstMetric, firstContext, firstBackground);
            render(second, secondMetric, secondContext, secondBackground);
        });
    };
    const auto checkOrder = [&](ImU32 backText, ImU32 frontBackground, ImU32 frontText) {
        std::map<ImU32, std::pair<U64, U64>> paintOrder;
        U64 index = 0;
        const auto firstPos = ImNodes::GetNodeScreenSpacePos(firstId);
        const auto secondPos = ImNodes::GetNodeScreenSpacePos(secondId);
        const auto firstSize = ImNodes::GetNodeDimensions(firstId);
        const auto secondSize = ImNodes::GetNodeDimensions(secondId);
        const ImRect overlap{
            ImVec2(std::max(firstPos.x, secondPos.x), std::max(firstPos.y, secondPos.y)),
            ImVec2(std::min(firstPos.x + firstSize.x, secondPos.x + secondSize.x),
                   std::min(firstPos.y + firstSize.y, secondPos.y + secondSize.y)),
        };
        // Inspect final submission order, including child-window draw lists.
        for (const auto* drawList : ImGui::GetDrawData()->CmdLists) {
            for (const auto& command : drawList->CmdBuffer) {
                for (U32 i = 0; i < command.ElemCount; ++i, ++index) {
                    const auto& vertex = drawList->VtxBuffer[
                        command.VtxOffset + drawList->IdxBuffer[command.IdxOffset + i]];
                    const auto [it, inserted] = paintOrder.try_emplace(
                        vertex.col, std::make_pair(index, index));
                    it->second.second = index;
                    if (vertex.col == firstText || vertex.col == secondText) {
                        REQUIRE(overlap.Contains(vertex.pos));
                    }
                }
            }
        }
        REQUIRE(paintOrder.contains(backText));
        REQUIRE(paintOrder.contains(frontBackground));
        REQUIRE(paintOrder.contains(frontText));
        REQUIRE(paintOrder.at(backText).second < paintOrder.at(frontBackground).first);
        REQUIRE(paintOrder.at(frontBackground).second < paintOrder.at(frontText).first);
    };

    for (int i = 0; i < 3; ++i) frame();
    checkOrder(firstText, secondBackground, secondText);

    const auto raise = [&](int nodeId, bool leftEdge) {
        const auto pos = ImNodes::GetNodeScreenSpacePos(nodeId);
        const auto size = ImNodes::GetNodeDimensions(nodeId);
        const ImVec2 exposedPoint{leftEdge ? pos.x + 10.0f : pos.x + size.x - 10.0f,
                                  pos.y + 80.0f};
        ui.setMouse(exposedPoint, true);
        frame();
        ui.setMouse(exposedPoint, false);
        frame();
        ui.setMouse(ImVec2(-100.0f, -100.0f), false);
        frame();
    };
    raise(firstId, true);
    checkOrder(secondText, firstBackground, firstText);
    raise(secondId, false);
    checkOrder(firstText, secondBackground, secondText);
}
