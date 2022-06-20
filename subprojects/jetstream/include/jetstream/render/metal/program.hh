#ifndef JETSTREAM_RENDER_METAL_PROGRAM_HH
#define JETSTREAM_RENDER_METAL_PROGRAM_HH

#include "jetstream/render/base/program.hh"
#include "jetstream/backend/base.hh"

namespace Jetstream::Render {

template<>
class ProgramImp<Device::Metal> : public Program {
 public:
    explicit ProgramImp(const Config& config);

 protected:
    const Result create(const MTL::PixelFormat& pixelFormat);
    const Result destroy();
    const Result draw(MTL::CommandBuffer* commandBuffer,
                      MTL::RenderPassDescriptor* renderPassDescriptor);

 private:
    MTL::RenderPipelineState* renderPipelineState = nullptr;

    std::vector<std::shared_ptr<DrawImp<Device::Metal>>> draws;
    std::vector<std::shared_ptr<TextureImp<Device::Metal>>> textures;
    std::vector<std::shared_ptr<BufferImp<Device::Metal>>> buffers;

    static Result checkShaderCompilation(U64);
    static Result checkProgramCompilation(U64);

    friend class SurfaceImp<Device::Metal>; 
};

}  // namespace Jetstream::Render

#endif
