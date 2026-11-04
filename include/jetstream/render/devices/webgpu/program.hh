#ifndef JETSTREAM_RENDER_WEBGPU_PROGRAM_HH
#define JETSTREAM_RENDER_WEBGPU_PROGRAM_HH

#include <atomic>

#include "jetstream/render/base/program.hh"
#include "jetstream/backend/base.hh"

namespace Jetstream::Render {

template<>
class JETSTREAM_API ProgramImp<DeviceType::WebGPU> : public Program {
 public:
    explicit ProgramImp(const Config& config);

    Result rebindTextures() override;

 protected:
    Result create(const WGPUTextureFormat& pixelFormat);
    Result destroy();
    Result draw(WGPURenderPassEncoder& renderPassEncoder,
                const Extent2D<U64>& framebufferSize);

 private:
    WGPURenderPipeline pipeline = nullptr;
    WGPUPipelineLayout pipelineLayout = nullptr;
    WGPUBindGroupLayout bindGroupLayout = nullptr;
    WGPUBindGroup bindGroup = nullptr;
    WGPUShaderModule vertShaderModule = nullptr;
    WGPUShaderModule fragShaderModule = nullptr;

    std::vector<WGPUBindGroupLayoutEntry> bindings;
    std::vector<WGPUBindGroupEntry> bindGroupEntries;

    std::vector<std::shared_ptr<DrawImp<DeviceType::WebGPU>>> draws;
    std::vector<std::shared_ptr<TextureImp<DeviceType::WebGPU>>> textures;
    std::vector<std::pair<std::shared_ptr<BufferImp<DeviceType::WebGPU>>, Program::Target>> buffers;

    struct BindGroupCandidate {
        ~BindGroupCandidate();
        WGPUBindGroup handle = nullptr;
        std::vector<U64> allocations;
        std::atomic<Result> result = Result::SKIP;
    };

    std::shared_ptr<BindGroupCandidate> pendingBindGroup;

    WGPUBindGroup createBindGroup();

    static WGPUShaderStage TargetToShaderStage(const Program::Target& target);
    static WGPUBufferBindingType BufferDescriptorType(const std::shared_ptr<Buffer>& buffer);

    friend class SurfaceImp<DeviceType::WebGPU>;
};

}  // namespace Jetstream::Render

#endif
