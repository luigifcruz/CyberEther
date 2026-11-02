#ifndef JETSTREAM_RENDER_SAKURA_SURFACE_VIEW_HH
#define JETSTREAM_RENDER_SAKURA_SURFACE_VIEW_HH

#include <jetstream/render/base/present.hh>
#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/surface.hh>
#include <jetstream/surface.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace Jetstream::Sakura {

struct SurfaceView {
    struct Frame {
        Rect rect;
        std::optional<Render::PresentList::Layer> layer;
        Render::PresentList::Blend blend = Render::PresentList::Blend::Straight;
    };

    struct Config {
        std::string id;
        std::shared_ptr<const Render::Texture> textureSource;
        Extent2D<F32> size = {0.0f, 0.0f};
        std::optional<F32> height;
        bool premultiplied = false;
        std::function<std::shared_ptr<const Render::Texture>()> resolveTexture;
        std::function<void(const SurfaceResize&)> onSize;
    };

    SurfaceView();
    ~SurfaceView();

    SurfaceView(SurfaceView&&) noexcept;
    SurfaceView& operator=(SurfaceView&&) noexcept;

    SurfaceView(const SurfaceView&) = delete;
    SurfaceView& operator=(const SurfaceView&) = delete;

    void update(Config config);
    Frame render(const Context& ctx, const Rect& region, const Rect& clip) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_SURFACE_VIEW_HH
