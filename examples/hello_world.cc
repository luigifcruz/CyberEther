#include <thread>

#include "jetstream/base.hh"

using namespace Jetstream;

// Selects the compute backend to use.
constexpr static Device ComputeDevice = Device::CPU;

// Selects the graphical backend to use.
constexpr static Device RenderDevice  = Device::Vulkan;

// Selects the viewport platform to use.
using ViewportPlatform = Viewport::GLFW<RenderDevice>;

static Extent2D<U64> GetContentRegion() {
    auto [x, y] = ImGui::GetContentRegionAvail();
    return {
        static_cast<U64>(x),
        static_cast<U64>(y)
    };
}

static Extent2D<F32> GetRelativeMouseTranslation(const Extent2D<U64>& dim, const F32& zoom = 1.0f) {
    const auto& [dx, dy] = ImGui::GetMouseDragDelta(0);
    const auto& [x, y] = dim;

    const auto relativeX = (dx / x) * 2.0f;
    const auto relativeY = (dy / y) * 2.0f;

    const auto zoomAdjustedX = relativeX / zoom;
    const auto zoomAdjustedY = relativeY;

    return {
        zoomAdjustedX,
        zoomAdjustedY
    };
}

static Extent2D<F32> GetRelativeMousePos(const Extent2D<U64>& dim, const F32& translation = 0.0f, const F32& zoom = 1.0f) {
    const auto& [x, y] = dim;

    ImVec2 mousePositionAbsolute = ImGui::GetMousePos();
    ImVec2 screenPositionAbsolute = ImGui::GetItemRectMin();

    const auto relativeX = (((mousePositionAbsolute.x - screenPositionAbsolute.x) / x) * 2.0f) - 1.0f;
    const auto relativeY = (((mousePositionAbsolute.y - screenPositionAbsolute.y) / y) * 2.0f) - 1.0f;

    const auto zoomAdjustedX = relativeX / zoom;
    const auto zoomAdjustedY = relativeY;

    const auto translationAdjustedX = zoomAdjustedX - translation;
    const auto translationAdjustedY = zoomAdjustedY;

    return {
        translationAdjustedX,
        translationAdjustedY
    };
}

class UI {
 public:
    UI(Instance& instance) : instance(instance) {}

    Result run() {
        // Add modules to the instance.

        const U64 bins = 2048;

        config.thickness = 1.0f;

        buffer = Tensor<Device::CPU, CF32>({8, bins});
        buffer.set_locale({"main", "main", "main"});

        JST_CHECK(instance.addModule(
            win, "win", {
                .size = bins,
            }, {}
        ));

        JST_CHECK(instance.addModule(
            inv, "inv", {}, {
                .buffer = win->getOutputWindow(),
            }
        ));

        JST_CHECK(instance.addModule(
            win_mul, "win_mul", {}, {
                .factorA = buffer,
                .factorB = inv->getOutputBuffer(),
            }
        ));

        JST_CHECK(instance.addModule(
            fft, "fft", {
                .forward = true,
            }, {
                .buffer = win_mul->getOutputProduct(),
            }
        ));

        JST_CHECK(instance.addModule(
            amp, "amp", {}, {
                .buffer = fft->getOutputBuffer(),
            }
        ));

        JST_CHECK(instance.addModule(
            scl, "scl", {
                .range = {-100.0, 0.0},
            }, {
                .buffer = amp->getOutputBuffer(),
            }
        ));

        JST_CHECK(instance.addModule(
            lpt, "lpt", config, {
                .buffer = scl->getOutputBuffer(),
            }
        ));

        // Start the instance.

        instance.start();

        computeWorker = std::thread([&]{
            while (instance.computing()) {
                this->computeThreadLoop();
            }
        });

        graphicalWorker = std::thread([&]{
            closing = false;
            while (instance.presenting()) {
                this->graphicalThreadLoop();
            }
        });

        // Wait user to close the window.

        while (instance.viewport().keepRunning()) {
            instance.viewport().pollEvents();
        }

        // Stop the instance and wait for threads.

        closing = true;
        instance.reset();
        instance.stop();

        if (computeWorker.joinable()) {
            computeWorker.join();
        }

        if (graphicalWorker.joinable()) {
            graphicalWorker.join();
        }

        // Destroy the instance.

        instance.destroy();

        return Result::SUCCESS;
    }

 private:
    std::thread graphicalWorker;
    std::thread computeWorker;
    Instance& instance;
    bool closing;

    Tensor<Device::CPU, CF32> buffer;

    Lineplot<ComputeDevice>::Config config;

    std::shared_ptr<Window<ComputeDevice>> win;
    std::shared_ptr<Invert<ComputeDevice>> inv;
    std::shared_ptr<Multiply<ComputeDevice>> win_mul;
    std::shared_ptr<FFT<ComputeDevice>> fft;
    std::shared_ptr<Amplitude<ComputeDevice>> amp;
    std::shared_ptr<Scale<ComputeDevice>> scl;
    std::shared_ptr<Lineplot<ComputeDevice>> lpt;

    void computeThreadLoop() {
        // Generate random data for buffer.
        for (U64 i = 0; i < buffer.shape(0); i++) {
            for (U64 j = 0; j < buffer.shape(1); j++) {
                buffer[{i, j}] = std::rand() % 3;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        if (!ImGui::IsKeyDown(ImGuiKey_Space)) {
            JST_CHECK_THROW(instance.compute());
        }
    }

    void graphicalThreadLoop() {
        if (instance.begin() == Result::SKIP) {
            return;
        }

        if (!closing) {
            JST_CHECK_THROW(drawCustomViews());
        }

        JST_CHECK_THROW(instance.present());
        if (instance.end() == Result::SKIP) {
            return;
        }
    }

    Result drawCustomViews() {
        static ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration |
                                        ImGuiWindowFlags_NoMove |
                                        ImGuiWindowFlags_NoSavedSettings;
        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->WorkPos);
        ImGui::SetNextWindowSize(viewport->WorkSize);

        ImGui::Begin("Lineplot", nullptr, flags);

        lpt->scale(ImGui::GetIO().DisplayFramebufferScale.x);
        const auto& viewSize = lpt->viewSize(GetContentRegion());
        ImGui::Image(lpt->getTexture().raw(), ImVec2(viewSize.x, viewSize.y));

        if (ImGui::IsItemHovered()) {
            const auto& mouseRelPos = GetRelativeMousePos(viewSize, lpt->translation(), config.zoom);

            // Handle zoom interaction.

            const auto& scroll = ImGui::GetIO().MouseWheel;
            if (scroll != 0.0f) {
                config.zoom += ((scroll > 0.0f) ? std::max(config.zoom *  0.02f,  0.02f) :
                                                  std::min(config.zoom * -0.02f, -0.02f));
                std::tie(config.zoom, config.translation) = lpt->zoom(mouseRelPos, config.zoom);
            }

            // Handle translation interaction.

            if (ImGui::IsAnyMouseDown()) {
                lpt->translation(GetRelativeMouseTranslation(viewSize, config.zoom).x + config.translation);
            } else {
                config.translation = lpt->translation();
            }

            // Handle reset interaction on right click.

            if (ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
                const auto& [zoom, translation] = lpt->zoom({0.0f, 0.0f}, 1.0f);
                config.zoom = zoom;
                config.translation = translation;
            }

            // Handle cursor position display.

            const auto& currentMouseRelPos = lpt->cursor();
            if (currentMouseRelPos != mouseRelPos) {
                lpt->cursor(mouseRelPos);
            }
        }

        ImGui::End();

        return Result::SUCCESS;
    }
};

int main() {
    std::cout << "Welcome to CyberEther!" << std::endl;

    // Initialize Backends.
    if (Backend::Initialize<ComputeDevice>({}) != Result::SUCCESS) {
        JST_FATAL("Cannot initialize compute backend.");
        return 1;
    }

    if (Backend::Initialize<RenderDevice>({}) != Result::SUCCESS) {
        JST_FATAL("Cannot initialize render backend.");
        return 1;
    }

    // Initialize Instance.
    Instance instance;

    // Initialize Viewport.
    Viewport::Config viewportCfg;
    viewportCfg.vsync = true;
    viewportCfg.size = {1080, 1080};
    viewportCfg.title = "Lineplot";
    JST_CHECK_THROW(instance.buildViewport<ViewportPlatform>(viewportCfg));

    // Initialize Window.
    Render::Window::Config renderCfg;
    renderCfg.scale = 1.0f;
    JST_CHECK_THROW(instance.buildRender<RenderDevice>(renderCfg));

    UI(instance).run();

    Backend::DestroyAll();

    std::cout << "Goodbye from CyberEther!" << std::endl;

    return 0;
}
