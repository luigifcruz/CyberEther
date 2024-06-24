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
    Result create(const std::shared_ptr<TextureImp<Device::Metal>>& framebuffer);
    Result destroy();
    Result draw(MTL::RenderCommandEncoder* renderCmdEncoder);

 private:
    MTL::RenderPipelineState* renderPipelineState = nullptr;

    std::shared_ptr<DrawImp<Device::Metal>> _draw;
    std::vector<std::shared_ptr<TextureImp<Device::Metal>>> textures;
    std::vector<std::pair<std::shared_ptr<BufferImp<Device::Metal>>, Program::Target>> buffers;

    friend class SurfaceImp<Device::Metal>; 
};

}  // namespace Jetstream::Render

#endif
