#ifndef JETSTREAM_RENDER_COMPONENTS_MESH_HH
#define JETSTREAM_RENDER_COMPONENTS_MESH_HH

#include <optional>
#include <span>

#include "jetstream/types.hh"
#include "jetstream/logger.hh"

#include "jetstream/render/types.hh"
#include "jetstream/render/base/surface.hh"
#include "jetstream/render/components/generic.hh"

namespace Jetstream::Render::Components {

class JETSTREAM_API Mesh : public Generic {
 public:
    struct Config {
        U64 vertexCapacity = 1024;
        U64 indexCapacity = 3072;
        Extent2D<F32> framebufferSize = {1.0f, 1.0f};
    };

    Mesh(const Config& config);
    ~Mesh();

    Result create(Window* window);
    Result destroy(Window* window);

    Result surface(Render::Surface::Config& config);

    Result present();

    std::span<Extent2D<F32>> positions();
    std::span<ColorRGBA<F32>> colors();
    std::span<U32> indices();

    Result update(U64 vertexCount, U64 indexCount);
    Result updateTransform(const Extent2D<F32>& framebufferSize, const Extent2D<F32>& origin, F32 scale);
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
