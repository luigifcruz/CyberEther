#ifndef JETSTREAM_RENDER_METAL_DRAW_HH
#define JETSTREAM_RENDER_METAL_DRAW_HH

#include "jetstream/render/base/draw.hh"
#include "jetstream/backend/base.hh"

namespace Jetstream::Render {

template<>
class DrawImp<Device::Metal> : public Draw {
 public:
    explicit DrawImp(const Config& config);

 protected:
    Result create(MTL::VertexDescriptor* vertDesc);
    Result destroy();
    Result encode(MTL::RenderCommandEncoder* encode);

 private:
    std::shared_ptr<VertexImp<Device::Metal>> buffer;

    friend class ProgramImp<Device::Metal>;
};

}  // namespace Jetstream::Render

#endif
