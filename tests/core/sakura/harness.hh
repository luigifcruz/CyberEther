#ifndef JETSTREAM_TESTS_CORE_SAKURA_HARNESS_HH
#define JETSTREAM_TESTS_CORE_SAKURA_HARNESS_HH

#include <imgui.h>
#include <imnodes.h>

#include "jetstream/render/base/window.hh"
#include "jetstream/render/sakura/surface.hh"
#include "jetstream/types.hh"

#include "render/sakura/context.hh"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace SakuraTest {

// Minimal Render::Window: only scaling metrics matter; every lifecycle
// hook is a no-op. The scaling factor is set directly (protected member).
class FakeWindow : public Jetstream::Render::Window {
 public:
    explicit FakeWindow(Jetstream::F32 scalingFactor = 1.0f) : Window(Config{scalingFactor}) {
        _scalingFactor = scalingFactor;
        _previousScalingFactor = scalingFactor;
    }

    const Stats& stats() const override {
        static const Stats stats{0, 0};
        return stats;
    }

    std::string info() const override {
        return "FakeWindow";
    }
    constexpr Jetstream::DeviceType device() const override {
        return Jetstream::DeviceType::None;
    }

 protected:
    Jetstream::Result bindSurface(const std::shared_ptr<Jetstream::Render::Surface>&) override {
        return Jetstream::Result::SUCCESS;
    }

    Jetstream::Result unbindSurface(const std::shared_ptr<Jetstream::Render::Surface>&) override {
        return Jetstream::Result::SUCCESS;
    }

    Jetstream::Result underlyingCreate() override {
        return Jetstream::Result::SUCCESS;
    }

    Jetstream::Result underlyingDestroy() override {
        return Jetstream::Result::SUCCESS;
    }

    Jetstream::Result underlyingBegin() override {
        return Jetstream::Result::SUCCESS;
    }

    Jetstream::Result underlyingEnd() override {
        return Jetstream::Result::SUCCESS;
    }

    Jetstream::Result underlyingSynchronize() override {
        return Jetstream::Result::SUCCESS;
    }
};

// Owns an ImGui context styled for deterministic measurement:
// no window padding, no item spacing, a fixed-size root window at the
// origin, and no ini file writes.
class HeadlessUi {
 public:
    explicit HeadlessUi(Jetstream::F32 scalingFactor = 1.0f, ImVec2 displaySize = ImVec2(400.0f, 300.0f)) {
        context = ImGui::CreateContext();
        ImNodes::CreateContext();

        this->displaySize = displaySize;

        ImGuiIO& io = ImGui::GetIO();
        io.IniFilename = nullptr;
        // Declare texture support: skips the font-atlas requirement headlessly.
        io.BackendFlags |= ImGuiBackendFlags_RendererHasTextures;
        io.DisplaySize = displaySize;
        io.DisplayFramebufferScale = ImVec2(1.0f, 1.0f);
        io.DeltaTime = 1.0f / 60.0f;
        io.Fonts->AddFontDefault();

        ImGui::StyleColorsDark();
        ImGuiStyle& style = ImGui::GetStyle();
        style.WindowPadding = ImVec2(0.0f, 0.0f);
        style.ItemSpacing = ImVec2(0.0f, 0.0f);
        style.ItemInnerSpacing = ImVec2(0.0f, 0.0f);

        window = std::make_unique<FakeWindow>(scalingFactor);
    }

    ~HeadlessUi() {
        ImNodes::DestroyContext();
        ImGui::DestroyContext(context);
    }

    HeadlessUi(const HeadlessUi&) = delete;
    HeadlessUi& operator=(const HeadlessUi&) = delete;

    Jetstream::Sakura::Context sakura() const {
        Jetstream::Sakura::Context ctx;
        ctx.render = window.get();
        ctx.pixelRatio = 1.0f;
        return ctx;
    }

    // Positions the synthetic mouse before the next frame.
    void setMouse(ImVec2 position, bool leftDown) {
        mousePosition = position;
        mouseLeftDown = leftDown;
    }

    // One ImGui frame with a fixed-size root window as the content area.
    template<typename Body>
    void frame(Body body) {
        ImGuiIO& io = ImGui::GetIO();
        io.MousePos = mousePosition;
        io.MouseDown[0] = mouseLeftDown;

        ImGui::NewFrame();
        ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
        ImGui::SetNextWindowSize(displaySize);
        ImGui::Begin("sakura-test",
                     nullptr,
                     ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                         ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoScrollbar |
                         ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoNav);
        body();
        ImGui::End();
        ImGui::Render();
    }

    // Same as frame(), wrapped in an imnodes editor canvas so components
    // that call ImNodes::BeginNode/EndNode can render.
    template<typename Body>
    void editorFrame(Body body) {
        frame([&] {
            ImNodes::BeginNodeEditor();
            body();
            ImNodes::EndNodeEditor();
        });
    }

    static constexpr Jetstream::F32 kWindowWidth = 400.0f;
    static constexpr Jetstream::F32 kWindowHeight = 300.0f;

 private:
    ImGuiContext* context = nullptr;
    std::unique_ptr<FakeWindow> window;
    ImVec2 displaySize = ImVec2(400.0f, 300.0f);
    ImVec2 mousePosition = ImVec2(-100.0f, -100.0f);
    bool mouseLeftDown = false;
};

// Records Sakura::SurfaceResize emissions from component callbacks.
struct ResizeLog {
    std::vector<Jetstream::Sakura::SurfaceResize> entries;

    void record(const Jetstream::Sakura::SurfaceResize& resize) {
        entries.push_back(resize);
    }

    std::size_t count() const {
        return entries.size();
    }
};

}  // namespace SakuraTest

#endif  // JETSTREAM_TESTS_CORE_SAKURA_HARNESS_HH
