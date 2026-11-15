#ifndef JETSTREAM_RENDER_SAKURA_CANVAS_HH
#define JETSTREAM_RENDER_SAKURA_CANVAS_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct Canvas {
    struct Config {
        std::string id;
        ColorRGBA<F32> clearColor = {0.0f, 0.0f, 0.0f, 1.0f};
        bool metered = false;

        bool operator==(const Config&) const = default;
    };

    struct Stats {
        U64 revision = 0;
        bool active = false;
        F32 refreshRate = 0.0f;
        F32 presentRate = 0.0f;
        F32 smoothness = 1.0f;
        U64 missedFrames = 0;
        F32 idle = 1.0f;
        F32 workAverage = 0.0f;
        F32 workPeak = 0.0f;
        F32 layoutRate = 0.0f;
        F32 redrawRate = 0.0f;
        F32 uploadRate = 0.0f;
        U64 leaves = 0;
        U64 draws = 0;
        U64 buffers = 0;
        U64 bufferBytes = 0;
        U64 rebuilds = 0;
        std::vector<F32> intervals;

        bool operator==(const Stats&) const = default;
    };

    Canvas();
    ~Canvas();


    Canvas(const Canvas&) = delete;
    Canvas& operator=(const Canvas&) = delete;

    void mount(Component& root);
    void unmount(const Component& component);
    void update(Config config);
    void render(const Sakura::Context& ctx, const Rect& region);
    void requestCursor(SurfaceCursor cursor);
    bool dropFiles(const std::vector<std::string>& paths);
    bool dispatching() const;
    const Stats& stats() const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_CANVAS_HH
