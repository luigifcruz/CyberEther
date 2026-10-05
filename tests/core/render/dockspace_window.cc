#include <catch2/catch_test_macros.hpp>

#include <imgui.h>
#include <imgui_internal.h>

#include "jetstream/render/base/window.hh"
#include "jetstream/render/sakura/components/dockspace_window.hh"
#include "render/sakura/context.hh"

#include <memory>

using namespace Jetstream;

namespace {

class DockspaceRenderWindow final : public Render::Window {
 public:
    DockspaceRenderWindow() : Window(Config{1.0f}) { _scalingFactor = 1.0f; }
    const Stats& stats() const override { return windowStats; }
    std::string info() const override { return "DockspaceRenderWindow"; }
    constexpr DeviceType device() const override { return DeviceType::None; }

 protected:
    Result bindSurface(const std::shared_ptr<Render::Surface>&) override { return Result::SUCCESS; }
    Result unbindSurface(const std::shared_ptr<Render::Surface>&) override { return Result::SUCCESS; }
    Result underlyingCreate() override { return Result::SUCCESS; }
    Result underlyingDestroy() override { return Result::SUCCESS; }
    Result underlyingBegin() override { return Result::SUCCESS; }
    Result underlyingEnd() override { return Result::SUCCESS; }
    Result underlyingSynchronize() override { return Result::SUCCESS; }

 private:
    Stats windowStats{};
};

}  // namespace

TEST_CASE("Dockspace capture follows window identity when titles and item keys change",
          "[core][render][sakura][dockspace][rename]") {
    const std::unique_ptr<ImGuiContext, decltype(&ImGui::DestroyContext)> ui(
        ImGui::CreateContext(), ImGui::DestroyContext);
    auto& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.BackendFlags |= ImGuiBackendFlags_RendererHasTextures;
    io.DisplaySize = ImVec2(1200.0f, 900.0f);
    io.DeltaTime = 1.0f / 60.0f;
    io.Fonts->AddFontDefault();

    DockspaceRenderWindow renderWindow;
    Sakura::Context ctx;
    ctx.render = &renderWindow;
    Sakura::DockspaceWindow stack;
    std::optional<Sakura::DockspaceWindow::DockLayout> captured;
    std::string label = "Source Config (source)###stable-config";
    Sakura::DockspaceWindow::Config config{
        .id = "stack",
        .title = "Stack",
        .size = {800.0f, 600.0f},
        .restoreLayout = true,
        .layout = Sakura::DockspaceWindow::DockLayout{
            .items = std::vector<Sakura::DockspaceWindow::DockItem>{{"config:source", 0}},
        },
        .dockables = {{"config:source", label}},
        .onLayout = [&](auto layout) { captured = std::move(layout); },
    };
    const auto frame = [&] {
        stack.update(config);
        ImGui::NewFrame();
        // Stack capture precedes the config window's title update in production.
        stack.render(ctx);
        ImGui::Begin(label.c_str());
        ImGui::End();
        ImGui::Render();
    };
    for (int i = 0; i < 4; ++i) {
        frame();
    }
    auto* window = ImGui::FindWindowByName(label.c_str());
    REQUIRE(window);
    const auto dockId = window->DockId;
    REQUIRE(dockId != 0);
    REQUIRE(captured.has_value());
    REQUIRE(captured->items->at(0).key == "config:source");

    label = "Source Config (renamed)###stable-config";
    config.restoreLayout = false;
    config.dockables = {{"config:renamed", label}};
    frame();

    REQUIRE(ImGui::FindWindowByName(label.c_str()) == window);
    REQUIRE(window->DockId == dockId);
    REQUIRE(captured.has_value());
    REQUIRE(captured->items.has_value());
    REQUIRE(captured->items->size() == 1);
    REQUIRE(captured->items->at(0).key == "config:renamed");
}
