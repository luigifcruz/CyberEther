#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <vector>

#include <jetstream/render/sakura/components/workspace_background.hh>

#include "harness.hh"

using namespace Jetstream;

namespace {

class FrameClearWindow : public SakuraTest::FakeWindow {
 public:
    using Render::Window::prepareImgui;
};

}

TEST_CASE("Workspace clear folding respects finalized draw order",
          "[core][sakura][workspace_background]") {
    SakuraTest::HeadlessUi ui;
    FrameClearWindow window;
    auto ctx = ui.sakura();
    ctx.render = &window;
    Sakura::Palette palette{{"background", {0.11f, 0.32f, 0.64f, 1.0f}}};
    ctx.palette = std::cref(palette);
    Sakura::WorkspaceBackground background;
    background.update({.id = "background", .particles = false});
    bool previousDraw = false;
    bool laterDraw = false;
    bool clipped = false;
    bool textured = false;
    bool split = false;
    bool external = false;
    bool callback = false;
    bool backgroundFirst = false;
    bool folded = true;

    SECTION("Opaque background uses the clear") {}
    SECTION("Translucent background keeps its blend") {
        palette["background"].a = 0.5f;
        folded = false;
    }
    SECTION("Earlier geometry keeps its ordering") {
        previousDraw = true;
        folded = false;
    }
    SECTION("Partial clipping keeps the rectangle") {
        clipped = true;
        folded = false;
    }
    SECTION("Custom texture keeps the original draw") {
        textured = true;
        folded = false;
    }
    SECTION("Later geometry shares the command without being removed") {
        laterDraw = true;
    }
    SECTION("Built-in channels can place later geometry before the background") {
        split = laterDraw = true;
        folded = false;
    }
    SECTION("External channels can place later geometry before the background") {
        split = external = laterDraw = true;
        folded = false;
    }
    SECTION("An inactive channel callback keeps its ordering") {
        split = external = callback = true;
        folded = false;
    }
    SECTION("A background that remains first after merging can fold") {
        split = external = laterDraw = backgroundFirst = true;
    }

    ImDrawList* list = nullptr;
    ImDrawListSplitter splitter;
    ui.frame([&] {
        list = ImGui::GetBackgroundDrawList(ImGui::GetMainViewport());
        if (previousDraw) list->AddRectFilled({0, 0}, {10, 10}, IM_COL32_WHITE);
        if (clipped) list->PushClipRect({0, 0}, {100, 100}, true);
        if (textured) list->PushTexture(ImTextureRef(static_cast<ImTextureID>(123)));
        if (split) {
            if (external) {
                splitter.Split(list, 2);
                splitter.SetCurrentChannel(list, backgroundFirst ? 0 : 1);
            } else {
                list->ChannelsSplit(2);
                list->ChannelsSetCurrent(backgroundFirst ? 0 : 1);
            }
        }
        background.render(ctx);
        if (split) {
            if (external) splitter.SetCurrentChannel(list, backgroundFirst ? 1 : 0);
            else list->ChannelsSetCurrent(backgroundFirst ? 1 : 0);
        }
        if (laterDraw) list->AddRectFilled({0, 0}, {10, 10}, IM_COL32(255, 0, 0, 255));
        if (callback) list->AddCallback([](const ImDrawList*, const ImDrawCmd*) {}, nullptr);
        if (split) {
            if (external) splitter.Merge(list);
            else list->ChannelsMerge();
        }
        if (textured) list->PopTexture();
        if (clipped) list->PopClipRect();
    });

    const auto originalIndices = std::vector<ImDrawIdx>(list->IdxBuffer.begin(), list->IdxBuffer.end());
    window.prepareImgui();
    const auto expected = folded ? ImGui::ColorConvertU32ToFloat4(
        ImGui::ColorConvertFloat4ToU32({0.11f, 0.32f, 0.64f, 1.0f})) : ImVec4(0, 0, 0, 1);
    REQUIRE(window.frameClearColor().r == Catch::Approx(expected.x));
    REQUIRE(window.frameClearColor().g == Catch::Approx(expected.y));
    REQUIRE(window.frameClearColor().b == Catch::Approx(expected.z));
    REQUIRE(window.frameClearColor().a == Catch::Approx(expected.w));
    for (int pass = 0; pass < 2; ++pass) {
        U32 indices = 0;
        for (const auto& command : list->CmdBuffer) indices += command.ElemCount;
        REQUIRE(indices == 6 * (1 + previousDraw + laterDraw - folded));
        REQUIRE(std::equal(originalIndices.begin(), originalIndices.end(), list->IdxBuffer.begin()));
        window.prepareImgui();
    }
}
