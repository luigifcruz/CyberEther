#ifndef JETSTREAM_RENDER_COMPONENTS_CURVES_HH
#define JETSTREAM_RENDER_COMPONENTS_CURVES_HH

#include <optional>
#include <span>

#include "jetstream/types.hh"
#include "jetstream/logger.hh"

#include "jetstream/render/types.hh"
#include "jetstream/render/base/surface.hh"
#include "jetstream/render/components/generic.hh"

namespace Jetstream::Render::Components {

class JETSTREAM_API Curves : public Generic {
 public:
    struct Instance {
        Extent2D<F32> p0 = {0.0f, 0.0f};
        Extent2D<F32> p1 = {0.0f, 0.0f};
        Extent2D<F32> p2 = {0.0f, 0.0f};
        Extent2D<F32> p3 = {0.0f, 0.0f};
        ColorRGBA<F32> color = {0.0f, 0.0f, 0.0f, 0.0f};
        F32 thickness = 1.0f;
        F32 segments = 1.0f;
        F32 reserved0 = 0.0f;
        F32 reserved1 = 0.0f;
    };

    struct Config {
        U64 capacity = 1024;
        U64 maxSegments = 256;
        Extent2D<F32> framebufferSize = {1.0f, 1.0f};
    };

    Curves(const Config& config);
    ~Curves();

    Result create(Window* window);

    Result surface(Render::Surface::Config& config);

    Result present();

    std::span<Instance> instances();

    Result update(U64 instanceCount);
    Result updateTransform(const Extent2D<F32>& framebufferSize,
                           const Extent2D<F32>& origin,
                           F32 scale,
                           F32 fringe);
    Result updateScissorRect(const std::optional<Render::ScissorRect>& rect);
    void enabled(bool value);

    constexpr const Config& getConfig() const {
        return config;
    }

 private:
    Config config;

    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

}  // namespace Jetstream::Render::Components

#endif
