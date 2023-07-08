#ifndef JETSTREAM_RENDER_WEBGPU_DRAW_HH
#define JETSTREAM_RENDER_WEBGPU_DRAW_HH

#include "jetstream/render/base/draw.hh"
#include "jetstream/backend/base.hh"

namespace Jetstream::Render {

template<>
class DrawImp<Device::WebGPU> : public Draw {
 public:
    explicit DrawImp(const Config& config);

 protected:
    Result create(wgpu::RenderPipelineDescriptor& renderDescriptor);
    Result destroy();
    Result encode(wgpu::RenderPassEncoder& renderPassEncoder);

 private:
    std::shared_ptr<VertexImp<Device::WebGPU>> buffer;

    friend class ProgramImp<Device::WebGPU>;
};

}  // namespace Jetstream::Render

#endif
