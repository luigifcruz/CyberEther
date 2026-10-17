#ifndef JETSTREAM_RENDER_WEBGPU_KERNEL_HH
#define JETSTREAM_RENDER_WEBGPU_KERNEL_HH

#include "jetstream/render/base/kernel.hh"
#include "jetstream/backend/base.hh"

namespace Jetstream::Render {

template<>
class JETSTREAM_API KernelImp<DeviceType::WebGPU> : public Kernel {
 public:
    explicit KernelImp(const Config& config);

 protected:
    Result create();
    Result destroy();
    Result encode(WGPUComputePassEncoder& computePassEncoder);

 private:
    WGPUComputePipeline pipeline = nullptr;
    WGPUPipelineLayout pipelineLayout = nullptr;
    WGPUBindGroupLayout bindGroupLayout = nullptr;
    WGPUBindGroup bindGroup = nullptr;
    WGPUShaderModule kernelModule = nullptr;

    std::vector<WGPUBindGroupLayoutEntry> bindings;
    std::vector<WGPUBindGroupEntry> bindGroupEntries;

    std::vector<std::pair<std::shared_ptr<BufferImp<DeviceType::WebGPU>>, Kernel::AccessMode>> buffers;

    static WGPUBufferBindingType BufferDescriptorType(const std::shared_ptr<Buffer>& buffer,
                                                      const Kernel::AccessMode& mode);

    friend class SurfaceImp<DeviceType::WebGPU>;
};

}  // namespace Jetstream::Render

#endif
