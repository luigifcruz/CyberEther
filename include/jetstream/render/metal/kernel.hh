#ifndef JETSTREAM_RENDER_METAL_KERNEL_HH
#define JETSTREAM_RENDER_METAL_KERNEL_HH

#include "jetstream/render/base/kernel.hh"
#include "jetstream/backend/base.hh"

namespace Jetstream::Render {

template<>
class KernelImp<Device::Metal> : public Kernel {
 public:
    explicit KernelImp(const Config& config);

 protected:
    Result create();
    Result destroy();
    Result encode(MTL::ComputeCommandEncoder* encoder);

 private:
    MTL::ComputePipelineState* pipelineState = nullptr;

    std::vector<std::shared_ptr<BufferImp<Device::Metal>>> buffers;

    friend class SurfaceImp<Device::Metal>; 
};

}  // namespace Jetstream::Render

#endif
