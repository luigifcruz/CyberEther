#ifndef JETSTREAM_TESTS_CORE_SAKURA_SCENE_HH
#define JETSTREAM_TESTS_CORE_SAKURA_SCENE_HH

#include "component_harness.hh"
#include "fonts.hh"

namespace SakuraTest {

struct Scene {
    MonoFontWindow window;
    Jetstream::Sakura::Context ctx;

    explicit Scene(Extent2D<F32> size = {800.0f, 600.0f}) : ctx(MakeContext(size)) {
        ctx.render = &window;
    }

    explicit Scene(const Jetstream::Rect& bounds) : Scene(Extent2D<F32>{bounds.width, bounds.height}) {}

    Jetstream::Sakura::Context context() const {
        auto themed = ctx;
        themed.palette = DistinctPalette();
        return themed;
    }
};

}  // namespace SakuraTest

#endif  // JETSTREAM_TESTS_CORE_SAKURA_SCENE_HH
