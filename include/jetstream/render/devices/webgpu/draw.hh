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
    struct IndexedDrawCommand {
        U32 indexCount;
        U32 instanceCount;
        U32 firstIndex;
        U32 baseVertex;
        U32 firstInstance;
    };

    struct DrawCommand {
        U32 vertexCount;
        U32 instanceCount;
        U32 firstVertex;
        U32 firstInstance;
    };

    std::shared_ptr<VertexImp<Device::WebGPU>> buffer;

    std::shared_ptr<BufferImp<Device::WebGPU>> indexedIndirectBuffer;
    std::shared_ptr<BufferImp<Device::WebGPU>> indirectBuffer;

    std::vector<IndexedDrawCommand> indexedDrawCommands;
    std::vector<DrawCommand> drawCommands;

    friend class ProgramImp<Device::WebGPU>;
};

}  // namespace Jetstream::Render

#endif
