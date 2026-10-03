#include <catch2/catch_test_macros.hpp>

#include <span>

#include "jetstream/render/base/window.hh"
#include "jetstream/render/components/shapes.hh"

using namespace Jetstream;

namespace {

// Construct unbound render resources and queue CPU uploads without creating a GPU.
class ShapesWindow final : public Render::Window {
 public:
    ShapesWindow() : Window(Config{}) {}

    const Stats& stats() const override { return windowStats; }
    std::string info() const override { return "ShapesWindow"; }
    constexpr DeviceType device() const override {
#ifdef JETSTREAM_RENDER_VULKAN_AVAILABLE
        return DeviceType::Vulkan;
#elif defined(JETSTREAM_RENDER_WEBGPU_AVAILABLE)
        return DeviceType::WebGPU;
#else
        return DeviceType::None;
#endif
    }

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

TEST_CASE("Disabled shapes stay disabled when nonzero sizes change",
          "[core][render][shapes][visibility]") {
    ShapesWindow window;
    if (window.device() == DeviceType::None) {
        SKIP("Unbound shape resources require Vulkan or WebGPU support.");
    }
    Render::Components::Shapes::Config config;
    config.elements["rect"].size = {2.0f, 2.0f};
    Render::Components::Shapes shapes(config);
    REQUIRE(shapes.create(&window) == Result::SUCCESS);
    Render::Surface::Config surface;
    REQUIRE(shapes.surface(surface) == Result::SUCCESS);
    REQUIRE(surface.programs.size() == 1);
    const auto& program = surface.programs.front();

    SECTION("Disabled before the first present") {}
    SECTION("Disabled after the first present") {
        REQUIRE(shapes.present() == Result::SUCCESS);
        REQUIRE(program->enabled());
    }
    shapes.enabled(false);
    REQUIRE_FALSE(program->enabled());

    std::span<Extent2D<F32>> sizes;
    REQUIRE(shapes.getSizes("rect", sizes) == Result::SUCCESS);
    sizes.front() = {3.0f, 4.0f};
    REQUIRE(shapes.updateSizes("rect") == Result::SUCCESS);
    REQUIRE(shapes.present() == Result::SUCCESS);
    REQUIRE_FALSE(program->enabled());
    REQUIRE(shapes.present() == Result::SUCCESS);
    REQUIRE_FALSE(program->enabled());

    shapes.enabled(true);
    REQUIRE(program->enabled());
    REQUIRE(shapes.destroy(&window) == Result::SUCCESS);
}

TEST_CASE("Shape visibility combines caller state with nonzero batch area",
          "[core][render][shapes][visibility]") {
    ShapesWindow window;
    if (window.device() == DeviceType::None) {
        SKIP("Unbound shape resources require Vulkan or WebGPU support.");
    }
    Render::Components::Shapes::Config config;
    config.elements["rect"].numberOfInstances = 2;
    Render::Components::Shapes shapes(config);
    REQUIRE(shapes.create(&window) == Result::SUCCESS);
    Render::Surface::Config surface;
    REQUIRE(shapes.surface(surface) == Result::SUCCESS);
    REQUIRE(surface.programs.size() == 1);
    const auto& program = surface.programs.front();
    REQUIRE(shapes.present() == Result::SUCCESS);
    REQUIRE_FALSE(program->enabled());
    shapes.enabled(true);
    REQUIRE_FALSE(program->enabled());

    std::span<Extent2D<F32>> sizes;
    REQUIRE(shapes.getSizes("rect", sizes) == Result::SUCCESS);
    sizes[0] = {0.0f, 4.0f};
    sizes[1] = {4.0f, 0.0f};
    REQUIRE(shapes.updateSizes("rect") == Result::SUCCESS);
    REQUIRE(shapes.present() == Result::SUCCESS);
    REQUIRE_FALSE(program->enabled());

    shapes.enabled(false);
    sizes[1] = {4.0f, 4.0f};
    REQUIRE(shapes.updateSizes("rect") == Result::SUCCESS);
    REQUIRE(shapes.present() == Result::SUCCESS);
    REQUIRE_FALSE(program->enabled());
    shapes.enabled(true);
    REQUIRE(program->enabled());

    sizes[1] = {0.0f, 0.0f};
    REQUIRE(shapes.updateSizes("rect") == Result::SUCCESS);
    REQUIRE(shapes.present() == Result::SUCCESS);
    REQUIRE_FALSE(program->enabled());
    sizes[0] = {2.0f, 2.0f};
    REQUIRE(shapes.updateSizes("rect") == Result::SUCCESS);
    REQUIRE(shapes.present() == Result::SUCCESS);
    REQUIRE(program->enabled());
    REQUIRE(shapes.destroy(&window) == Result::SUCCESS);
}
