#ifndef JETSTREAM_RENDER_WEBGPU_PROGRAM_HH
#define JETSTREAM_RENDER_WEBGPU_PROGRAM_HH

#include "jetstream/render/base/program.hh"
#include "jetstream/backend/base.hh"

namespace Jetstream::Render {

template<>
class ProgramImp<Device::WebGPU> : public Program {
 public:
    explicit ProgramImp(const Config& config);

 protected:
    Result create(const WGPUTextureFormat& pixelFormat);
    Result destroy();
    Result draw(WGPURenderPassEncoder& renderPassEncoder);

 private:
    WGPURenderPipeline pipeline;
    WGPUPipelineLayout pipelineLayout;
    WGPUBindGroupLayout bindGroupLayout;
    WGPUBindGroup bindGroup;

    std::vector<WGPUBindGroupLayoutEntry> bindings;
    std::vector<WGPUBindGroupEntry> bindGroupEntries;

    std::vector<std::shared_ptr<DrawImp<Device::WebGPU>>> draws;
    std::vector<std::shared_ptr<TextureImp<Device::WebGPU>>> textures;
    std::vector<std::pair<std::shared_ptr<BufferImp<Device::WebGPU>>, Program::Target>> buffers;

    static WGPUShaderStage TargetToShaderStage(const Program::Target& target);
    static WGPUBufferBindingType BufferDescriptorType(const std::shared_ptr<Buffer>& buffer);

    friend class SurfaceImp<Device::WebGPU>;
};

}  // namespace Jetstream::Render

#endif
