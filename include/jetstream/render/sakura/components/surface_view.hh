#ifndef JETSTREAM_RENDER_SAKURA_SURFACE_VIEW_HH
#define JETSTREAM_RENDER_SAKURA_SURFACE_VIEW_HH

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
    struct Config {
        std::string id;
        U64 texture = 0;
        std::shared_ptr<const Render::Texture> textureSource;
        Extent2D<F32> size = {0.0f, 0.0f};
        std::optional<F32> height;
        F32 rounding = 0.0f;
        bool detachOverlay = false;
        std::function<U64()> onResolveTexture;
        std::function<void(const SurfaceResize&)> onSize;
        std::function<void(MouseEvent)> onMouse;
        std::function<void()> onDetach;
    };

    SurfaceView();
    ~SurfaceView();

    SurfaceView(SurfaceView&&) noexcept;
    SurfaceView& operator=(SurfaceView&&) noexcept;

    SurfaceView(const SurfaceView&) = delete;
    SurfaceView& operator=(const SurfaceView&) = delete;

    bool update(Config config);
    void render(const Context& ctx) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_SURFACE_VIEW_HH
