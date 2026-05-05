#pragma once

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/context.hh>
#include <jetstream/render/sakura/surface_size.hh>
#include <jetstream/surface.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace Jetstream::Sakura {

struct SurfaceView : public Component {
    struct Config {
        std::string id;
        U64 texture = 0;
        std::function<U64()> onResolveTexture;
        Extent2D<F32> size = {0.0f, 0.0f};
        F32 rounding = 0.0f;
        bool detachOverlay = false;
        std::optional<Extent2D<F32>> aspectRatioSize;
        std::function<void(const SurfaceSize&)> onSize;
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
