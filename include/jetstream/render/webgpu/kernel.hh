#ifndef JETSTREAM_RENDER_WEBGPU_KERNEL_HH
#define JETSTREAM_RENDER_WEBGPU_KERNEL_HH

#include "jetstream/render/base/kernel.hh"
#include "jetstream/backend/base.hh"

namespace Jetstream::Render {

template<>
class KernelImp<Device::WebGPU> : public Kernel {
 public:
    explicit KernelImp(const Config& config);

 protected:
    Result create();
    Result destroy();
    Result encode(wgpu::ComputePassEncoder& computePassEncoder);

 private:
    wgpu::ComputePipeline pipeline;
    wgpu::PipelineLayout pipelineLayout;
    wgpu::BindGroupLayout bindGroupLayout;
    wgpu::BindGroup bindGroup;

    std::vector<wgpu::BindGroupLayoutEntry> bindings;
    std::vector<wgpu::BindGroupEntry> bindGroupEntries;

    std::vector<std::pair<std::shared_ptr<BufferImp<Device::WebGPU>>, Kernel::AccessMode>> buffers;

    static wgpu::BufferBindingType BufferDescriptorType(const std::shared_ptr<Buffer>& buffer, 
                                                        const Kernel::AccessMode& mode);

    friend class SurfaceImp<Device::WebGPU>; 
};

}  // namespace Jetstream::Render

#endif
