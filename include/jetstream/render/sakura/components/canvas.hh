#ifndef JETSTREAM_RENDER_SAKURA_CANVAS_HH
#define JETSTREAM_RENDER_SAKURA_CANVAS_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct Canvas {
    struct Config {
        std::string id;
        ColorRGBA<F32> clearColor = {0.0f, 0.0f, 0.0f, 1.0f};

        bool operator==(const Config&) const = default;
    };

    Canvas();
    ~Canvas();


    Canvas(const Canvas&) = delete;
    Canvas& operator=(const Canvas&) = delete;

    void mount(Component& root);
    void unmount(const Component& component);
    void update(Config config);
    void render(const Sakura::Context& ctx);
    void requestCursor(SurfaceCursor cursor);
    bool dispatching() const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_CANVAS_HH
