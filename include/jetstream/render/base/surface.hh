#ifndef JETSTREAM_RENDER_BASE_SURFACE_HH
#define JETSTREAM_RENDER_BASE_SURFACE_HH

#include <memory>
#include <vector>

#include "jetstream/types.hh"
#include "jetstream/logger.hh"
#include "jetstream/render/base/texture.hh"
#include "jetstream/render/base/program.hh"
#include "jetstream/render/base/kernel.hh"
#include "jetstream/render/types.hh"
#include "jetstream/render/base/implementations.hh"
#include "jetstream/render/base/window_attachment.hh"

namespace Jetstream::Render {

class Surface : public WindowAttachment {
 public:
    struct Config {
        std::shared_ptr<Texture> framebuffer;
        std::vector<std::shared_ptr<Kernel>> kernels;
        std::vector<std::shared_ptr<Program>> programs;
        ColorRGBA<F32> clearColor = {0.0f, 0.0f, 0.0f, 1.0f};
        bool multisampled = false;
    };

    explicit Surface(const Config& config) : config(config) {}
    virtual ~Surface() = default;

    Type type() const override {
        return Type::Surface;
    }

    const Config& getConfig() const {
        return config;
    }

    constexpr const bool& multisampled() const {
        return config.multisampled;
    }

    const Extent2D<U64>& size() const {
        if (config.framebuffer) {
            return config.framebuffer->size();
        }
        return NullSize;
    }
    virtual const Extent2D<U64>& size(const Extent2D<U64>& size) = 0;

    template<Device D>
    static std::shared_ptr<Surface> Factory(const Config& config) {
        return std::make_shared<SurfaceImp<D>>(config);
    }

 protected:
    Config config;
};

}  // namespace Jetstream::Render

#endif
