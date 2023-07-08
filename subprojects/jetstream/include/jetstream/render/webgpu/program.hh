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
    Result create(const wgpu::TextureFormat& pixelFormat);
    Result destroy();
    Result draw(wgpu::RenderPassEncoder& renderPassEncoder);

 private:
    wgpu::RenderPipeline pipeline;
    wgpu::PipelineLayout pipelineLayout;
    wgpu::RenderPipelineDescriptor renderPipelineDescriptor;
    wgpu::BindGroupLayout bindGroupLayout;
    wgpu::BindGroup bindGroup;

    std::vector<wgpu::BindGroupLayoutEntry> bindings;
    std::vector<wgpu::BindGroupEntry> bindGroupEntries;

    std::shared_ptr<DrawImp<Device::WebGPU>> _draw;
    std::vector<std::shared_ptr<TextureImp<Device::WebGPU>>> textures;
    std::vector<std::pair<std::shared_ptr<BufferImp<Device::WebGPU>>, Program::Target>> buffers;

    static Result checkShaderCompilation(U64);
    static Result checkProgramCompilation(U64);
    static wgpu::ShaderStage TargetToWebGPU(const Program::Target& target);

    friend class SurfaceImp<Device::WebGPU>; 
};

}  // namespace Jetstream::Render

#endif
