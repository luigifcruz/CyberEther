#ifndef JETSTREAM_RENDER_SAKURA_COMPONENTS_RETAINED_CANVAS_HH
#define JETSTREAM_RENDER_SAKURA_COMPONENTS_RETAINED_CANVAS_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <string>

namespace Jetstream::Sakura::Retained {

struct Canvas {
    struct Layout {
        Extent2D<U64> framebufferSize = {0, 0};
        F32 pixelRatio = 1.0f;
    };

    struct Config {
        std::string id;
        Extent2D<F32> size = {0.0f, 0.0f};
        bool autoHeight = false;
        ColorRGBA<F32> clearColor = {0.0f, 0.0f, 0.0f, 1.0f};
        std::function<void(const Layout&)> onLayout;
    };

    Canvas();
    ~Canvas();

    Canvas(Canvas&&) noexcept;
    Canvas& operator=(Canvas&&) noexcept;

    Canvas(const Canvas&) = delete;
    Canvas& operator=(const Canvas&) = delete;

    void mount(Component& root);
    bool update(Config config);
    void render(const Sakura::Context& ctx);

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura::Retained

#endif  // JETSTREAM_RENDER_SAKURA_COMPONENTS_RETAINED_CANVAS_HH
