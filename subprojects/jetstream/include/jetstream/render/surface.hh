#ifndef JETSTREAM_RENDER_SURFACE_HH
#define JETSTREAM_RENDER_SURFACE_HH

#include <memory>
#include <vector>

#include "jetstream/types.hh"
#include "jetstream/logger.hh"
#include "jetstream/render/texture.hh"
#include "jetstream/render/program.hh"
#include "jetstream/render/types.hh"

namespace Jetstream::Render {

template<Device D> class SurfaceImp;

class Surface {
 public:
    struct Config {
        std::shared_ptr<Texture> framebuffer;
        std::vector<std::shared_ptr<Program>> programs;
    };

    explicit Surface(const Config& config) : config(config) {
        JST_DEBUG("Surface initialized.");
    }
    virtual ~Surface() = default;

    constexpr const Size2D<U64> size() const {
        if (config.framebuffer) {
            return config.framebuffer->size();
        }
        return {1, 1};
    }
    virtual const Size2D<U64> size(const Size2D<U64>&) = 0;

    template<Device D> 
    static std::shared_ptr<Surface> Factory(const Config& config) {
        return std::make_shared<SurfaceImp<D>>(config);
    }

 protected:
    Config config;
};

}  // namespace Jetstream::Render

#endif
