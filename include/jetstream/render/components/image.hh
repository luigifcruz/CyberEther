#ifndef JETSTREAM_RENDER_COMPONENTS_IMAGE_HH
#define JETSTREAM_RENDER_COMPONENTS_IMAGE_HH

#include <memory>
#include <optional>

#include "jetstream/types.hh"
#include "jetstream/logger.hh"

#include "jetstream/render/types.hh"
#include "jetstream/render/base/surface.hh"
#include "jetstream/render/base/texture.hh"
#include "jetstream/render/components/generic.hh"

namespace Jetstream::Render::Components {

class JETSTREAM_API Image : public Generic {
 public:
    enum class SourceAlpha {
        STRAIGHT,
        PREMULTIPLIED,
    };

    struct Config {
        std::shared_ptr<Render::Texture> texture;
        bool punch = false;
        SourceAlpha sourceAlpha = SourceAlpha::STRAIGHT;
    };

    Image(const Config& config);
    ~Image();

    Result create(Window* window);
    Result destroy(Window* window);

    Result surface(Render::Surface::Config& config);

    Result present();

    Result update(const Extent2D<F32>& framebufferSize,
                  const Extent2D<F32>& translation,
                  const Rect& rect,
                  F32 cornerRadius,
                  F32 opacity);
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
