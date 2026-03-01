#ifndef JETSTREAM_RENDER_METAL_PROGRAM_HH
#define JETSTREAM_RENDER_METAL_PROGRAM_HH

#include "jetstream/render/base/program.hh"
#include "jetstream/backend/base.hh"

namespace Jetstream::Render {

template<>
class ProgramImp<DeviceType::Metal> : public Program {
 public:
    explicit ProgramImp(const Config& config);

 protected:
    Result create(const MTL::PixelFormat& pixelFormat, bool multisampled);
    Result destroy();
    Result draw(MTL::RenderCommandEncoder* renderCmdEncoder);

 private:
    MTL::RenderPipelineState* renderPipelineState = nullptr;

    std::vector<std::shared_ptr<DrawImp<DeviceType::Metal>>> draws;
    std::vector<std::shared_ptr<TextureImp<DeviceType::Metal>>> textures;
    std::vector<std::pair<std::shared_ptr<BufferImp<DeviceType::Metal>>, Program::Target>> buffers;

    friend class SurfaceImp<DeviceType::Metal>; 
};

}  // namespace Jetstream::Render

#endif
