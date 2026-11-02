#ifndef JETSTREAM_TESTS_CORE_SAKURA_HARNESS_HH
#define JETSTREAM_TESTS_CORE_SAKURA_HARNESS_HH

#include "jetstream/render/base/window.hh"
#include "jetstream/render/components/font.hh"
#include "jetstream/render/sakura/surface.hh"
#include "jetstream/types.hh"

#include "render/sakura/context.hh"

#include "../render/harness.hh"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace SakuraTest {

class FakeWindow : public RenderTest::Window {
 public:
    explicit FakeWindow(Jetstream::F32 scalingFactor = 1.0f, Jetstream::Extent2D<Jetstream::F32> display = {0.0f, 0.0f})
        : RenderTest::Window(scalingFactor) {
        viewport.display = display;
        (void)open();
    }

    void rescale(Jetstream::F32 scalingFactor) {
        setScale(scalingFactor);
        (void)frame();
    }

    Jetstream::Sakura::Context sakura() {
        Jetstream::Sakura::Context ctx;
        ctx.render = this;
        ctx.pixelRatio = 1.0f;
        return ctx;
    }
};

class FontWindow : public FakeWindow {
 public:
    void load(const std::string& name, const unsigned int* data) {
        Jetstream::Render::Components::Font::Config config;
        config.data = data;
        config.size = 32.0f;
        auto font = std::make_shared<Jetstream::Render::Components::Font>(config);
        (void)font->create(this);
        fonts[name] = font;
        ++_fontRevision;
    }
};

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
