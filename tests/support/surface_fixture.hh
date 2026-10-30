#ifndef JETSTREAM_TESTS_SUPPORT_SURFACE_FIXTURE_HH
#define JETSTREAM_TESTS_SUPPORT_SURFACE_FIXTURE_HH

#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <string>
#include <utility>

#include "jetstream/config.hh"
#include "jetstream/render/base/surface.hh"
#include "jetstream/render/base/window.hh"
#include "jetstream/render/components/font.hh"
#include "jetstream/scheduler_context.hh"

#include "../../resources/fonts/compressed_jbmm.hh"

inline Jetstream::Result PresentFrame(Jetstream::Scheduler::Context& presenter) {
    const auto captured = presenter.presentCapture();
    if (captured != Jetstream::Result::SUCCESS && captured != Jetstream::Result::RELOAD) {
        return captured;
    }
    return presenter.presentSubmit();
}

#ifdef JETSTREAM_RENDER_VULKAN_AVAILABLE
// Describes render resources without creating a device. Attachments stay in the
// bind queue because the window never begins a frame.
class SurfaceTestWindow final : public Jetstream::Render::Window {
 public:
    SurfaceTestWindow() : Window(Config{}) {
        Jetstream::Render::Components::Font::Config font;
        font.data = jbmm_compressed_data;
        font.size = 32.0f;
        addFont("default_mono", std::make_shared<Jetstream::Render::Components::Font>(font));
    }

    std::string info() const override { return "SurfaceTestWindow"; }
    constexpr Jetstream::DeviceType device() const override {
        return Jetstream::DeviceType::Vulkan;
    }

 protected:
    Jetstream::Result bindSurface(const std::shared_ptr<Jetstream::Render::Surface>&) override {
        return Jetstream::Result::SUCCESS;
    }
    Jetstream::Result unbindSurface(const std::shared_ptr<Jetstream::Render::Surface>&) override {
        return Jetstream::Result::SUCCESS;
    }
    Jetstream::Result underlyingCreate() override { return Jetstream::Result::SUCCESS; }
    Jetstream::Result underlyingDestroy() override { return Jetstream::Result::SUCCESS; }
    Jetstream::Result underlyingBegin() override { return Jetstream::Result::SUCCESS; }
    Jetstream::Result underlyingEnd() override { return Jetstream::Result::SUCCESS; }
    Jetstream::Result underlyingSynchronize() override { return Jetstream::Result::SUCCESS; }
};

struct SurfaceProbe : Jetstream::Render::Surface {
    static bool consumeInvalidation(Jetstream::Render::Surface& surface) {
        return std::exchange(surface.*(&SurfaceProbe::dirty), false);
    }
};

inline bool PresentRedraws(Jetstream::Scheduler::Context& presenter,
                           Jetstream::Render::Surface& surface) {
    REQUIRE(PresentFrame(presenter) == Jetstream::Result::SUCCESS);
    return SurfaceProbe::consumeInvalidation(surface);
}
#endif

#endif  // JETSTREAM_TESTS_SUPPORT_SURFACE_FIXTURE_HH
