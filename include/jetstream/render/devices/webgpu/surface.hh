#ifndef JETSTREAM_RENDER_WEBGPU_SURFACE_HH
#define JETSTREAM_RENDER_WEBGPU_SURFACE_HH

#include "jetstream/render/base/surface.hh"
#include "jetstream/backend/base.hh"

namespace Jetstream::Render {

template<>
class JETSTREAM_API SurfaceImp<DeviceType::WebGPU> : public Surface {
 public:
    explicit SurfaceImp(const Config& config);

    Result create() override;
    Result destroy() override;

    const Extent2D<U64>& size(const Extent2D<U64>& size) override;

 protected:
    Result reorderPrograms(const std::vector<std::shared_ptr<Program>>& ordered) override;
    bool appendable() const override;
    Result validateAppend(const std::vector<std::shared_ptr<Program>>& programs,
                          const std::vector<std::shared_ptr<Kernel>>& kernels) const override;
    Result createAppended(const std::vector<std::shared_ptr<Program>>& programs,
                          const std::vector<std::shared_ptr<Kernel>>& kernels) override;
    Result prepare();
    Result draw(WGPUCommandEncoder& commandEncoder, bool visible = true);

 private:
    std::shared_ptr<TextureImp<DeviceType::WebGPU>> framebuffer;
    std::vector<std::shared_ptr<ProgramImp<DeviceType::WebGPU>>> programs;
    std::vector<std::shared_ptr<KernelImp<DeviceType::WebGPU>>> kernels;
    std::vector<std::shared_ptr<BufferImp<DeviceType::WebGPU>>> buffers;
    Extent2D<U64> requestedSize;
    bool framebufferChanged = false;

    friend class WindowImp<DeviceType::WebGPU>;
};

}  // namespace Jetstream::Render

#endif
