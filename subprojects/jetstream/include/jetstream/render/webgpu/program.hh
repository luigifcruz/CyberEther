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
    Result create(const MTL::PixelFormat& pixelFormat);
    Result destroy();
    Result draw(MTL::RenderCommandEncoder* renderCmdEncoder);

 private:
    MTL::RenderPipelineState* renderPipelineState = nullptr;

    std::shared_ptr<DrawImp<Device::Metal>> _draw;
    std::vector<std::shared_ptr<TextureImp<Device::Metal>>> textures;
    std::vector<std::pair<std::shared_ptr<BufferImp<Device::Metal>>, Program::Target>> buffers;

    static Result checkShaderCompilation(U64);
    static Result checkProgramCompilation(U64);

    friend class SurfaceImp<Device::Metal>; 
};

}  // namespace Jetstream::Render

#endif
