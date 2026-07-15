#ifndef JETSTREAM_RENDER_BASE_SURFACE_HH
#define JETSTREAM_RENDER_BASE_SURFACE_HH

#include <memory>
#include <unordered_set>
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

class JETSTREAM_API Surface : public WindowAttachment {
 public:
    struct Config {
        std::shared_ptr<Texture> framebuffer;
        std::vector<std::shared_ptr<Kernel>> kernels;
        std::vector<std::shared_ptr<Program>> programs;
        ColorRGBA<F32> clearColor = {0.0f, 0.0f, 0.0f, 1.0f};
        bool multisampled = false;
        bool retained = false;
    };

    explicit Surface(const Config& config);
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

    constexpr const bool& retained() const {
        return config.retained;
    }

    void clearColor(const ColorRGBA<F32>& color);
    void invalidate();

    void commitDraw();

    const Extent2D<U64>& size() const;
    virtual const Extent2D<U64>& size(const Extent2D<U64>& size) = 0;

    template<DeviceType D>
    static std::shared_ptr<Surface> Factory(const Config& config) {
        return std::make_shared<SurfaceImp<D>>(config);
    }

 protected:
    Config config;
    bool dirty = true;

    bool shouldDraw(bool framebufferChanged = false);
    void markDrawn();

 private:
    void prepareFrame();
    void collectTransfers(Transfer::Batch& batch) const;
    bool affectedBy(const Transfer::Batch& batch) const;

    bool drawPending = false;
    std::unordered_set<std::shared_ptr<Buffer>> dependencyBuffers;
    std::unordered_set<std::shared_ptr<Draw>> dependencyDraws;
    std::unordered_set<std::shared_ptr<Texture>> dependencyTextures;

    friend class Window;
};

}  // namespace Jetstream::Render

#endif
