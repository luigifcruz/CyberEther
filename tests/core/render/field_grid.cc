#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <imgui.h>
#include <imgui_internal.h>

#include <algorithm>
#include <memory>
#include <vector>

#include "jetstream/render/base/window.hh"
#include "jetstream/render/sakura/components/node/field_grid.hh"
#include "render/sakura/context.hh"

using namespace Jetstream;

namespace {

class FieldGridWindow final : public Render::Window {
 public:
    explicit FieldGridWindow(const F32 scale) : Window(Config{scale}) {
        _scalingFactor = scale;
    }

    const Stats& stats() const override { return windowStats; }
    std::string info() const override { return "FieldGridWindow"; }
    constexpr DeviceType device() const override { return DeviceType::None; }

 protected:
    Result bindSurface(const std::shared_ptr<Render::Surface>&) override {
        return Result::SUCCESS;
    }
    Result unbindSurface(const std::shared_ptr<Render::Surface>&) override {
        return Result::SUCCESS;
    }
    Result underlyingCreate() override { return Result::SUCCESS; }
    Result underlyingDestroy() override { return Result::SUCCESS; }
    Result underlyingBegin() override { return Result::SUCCESS; }
    Result underlyingEnd() override { return Result::SUCCESS; }
    Result underlyingSynchronize() override { return Result::SUCCESS; }

 private:
    Stats windowStats{};
};

}  // namespace

TEST_CASE("Node field grids keep column drawing inside the parent clip",
          "[core][render][sakura][field-grid][clipping]") {
    const U64 columns = GENERATE(1, 2, 3);
    const F32 scale = GENERATE(1.0f, 2.0f);
    const bool fullWidth = GENERATE(false, true);
    CAPTURE(columns, scale, fullWidth);

    const std::unique_ptr<ImGuiContext, decltype(&ImGui::DestroyContext)> ui(
        ImGui::CreateContext(), ImGui::DestroyContext);
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.BackendFlags |= ImGuiBackendFlags_RendererHasTextures;
    io.DisplaySize = ImVec2(1200.0f, 900.0f);
    io.DeltaTime = 1.0f / 60.0f;
    io.Fonts->AddFontDefault();
    ImGui::GetStyle().ItemSpacing = ImVec2(0.0f, 0.0f);

    FieldGridWindow renderWindow(scale);
    Sakura::Context ctx;
    ctx.render = &renderWindow;
    Sakura::NodeFieldGrid grid;
    grid.update({.id = "field-grid"});

    std::vector<ImRect> itemClips;
    std::vector<ImVec4> drawClips;
    std::vector<ImVec2> positions;
    std::vector<F32> widths;
    std::vector<Sakura::NodeFieldGrid::Item> items(columns * 2);
    for (auto& item : items) {
        item.fullWidth = fullWidth;
        item.child = [&](const Sakura::Context&) {
            auto* window = ImGui::GetCurrentWindow();
            const ImVec2 position = ImGui::GetCursorScreenPos();
            const F32 width = ImGui::GetContentRegionAvail().x;
            itemClips.push_back(window->ClipRect);
            positions.push_back(position);
            widths.push_back(width);
            window->DrawList->AddRectFilled(
                position, ImVec2(position.x + width, position.y + 40.0f * scale),
                IM_COL32_WHITE);
            drawClips.push_back(window->DrawList->CmdBuffer.back().ClipRect);
            ImGui::Dummy(ImVec2(width, 40.0f * scale));
        };
    }

    ImGui::NewFrame();
    ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
    ImGui::SetNextWindowSize(io.DisplaySize);
    ImGui::Begin("field-grid-host", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings);

    const ImVec2 origin(100.0f, 100.0f);
    const F32 gridWidth = 160.0f * columns * scale;
    // A node may extend past the pane's right edge while its field backgrounds
    // also cross the top/bottom edges. Column layout must not expand this clip.
    const ImRect parentClip(
        ImVec2(origin.x, origin.y + 5.0f * scale),
        ImVec2(origin.x + gridWidth - 20.0f * scale, origin.y + 60.0f * scale));
    ImGui::PushClipRect(parentClip.Min, parentClip.Max, true);
    auto* window = ImGui::GetCurrentWindow();
    const F32 contentMaxX = window->ContentRegionRect.Max.x;
    window->ContentRegionRect.Max.x = origin.x + gridWidth;
    ImGui::SetCursorScreenPos(origin);
    ImGui::BeginGroup();
    grid.render(ctx, items);
    ImGui::EndGroup();
    const ImRect restoredClip = window->ClipRect;
    const F32 restoredContentMaxX = window->ContentRegionRect.Max.x;
    window->ContentRegionRect.Max.x = contentMaxX;
    ImGui::PopClipRect();
    ImGui::End();
    ImGui::Render();

    REQUIRE(itemClips.size() == items.size());
    CHECK(restoredClip.Min.x == parentClip.Min.x);
    CHECK(restoredClip.Min.y == parentClip.Min.y);
    CHECK(restoredClip.Max.x == parentClip.Max.x);
    CHECK(restoredClip.Max.y == parentClip.Max.y);
    CHECK(restoredContentMaxX == origin.x + gridWidth);
    for (U64 i = 0; i < items.size(); ++i) {
        CAPTURE(i);
        const U64 column = fullWidth ? 0 : i % columns;
        const F32 width = fullWidth ? gridWidth : gridWidth / columns;
        CHECK(positions[i].x == origin.x + column * width);
        CHECK(widths[i] == width);
        CHECK(itemClips[i].Min.x >= parentClip.Min.x);
        CHECK(itemClips[i].Min.y >= parentClip.Min.y);
        CHECK(itemClips[i].Max.x <= parentClip.Max.x);
        CHECK(itemClips[i].Max.y <= parentClip.Max.y);
        CHECK(drawClips[i].x >= parentClip.Min.x);
        CHECK(drawClips[i].y >= parentClip.Min.y);
        CHECK(drawClips[i].z <= parentClip.Max.x);
        CHECK(drawClips[i].w <= parentClip.Max.y);
        if (!fullWidth && columns > 1) {
            CHECK(itemClips[i].Max.x ==
                  std::min(parentClip.Max.x, positions[i].x + width));
        }
    }
}
