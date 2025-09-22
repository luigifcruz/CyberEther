#ifndef JETSTREAM_RENDER_WEBGPU_DRAW_HH
#define JETSTREAM_RENDER_WEBGPU_DRAW_HH

#include "jetstream/render/base/draw.hh"
#include "jetstream/backend/base.hh"

namespace Jetstream::Render {

template<>
class DrawImp<DeviceType::WebGPU> : public Draw {
 public:
    explicit DrawImp(const Config& config);

 protected:
    Result create(WGPURenderPipelineDescriptor& renderDescriptor);
    Result destroy();
    Result encode(WGPURenderPassEncoder& renderPassEncoder);
    Result updateVertexCount(U64 vertexCount) override;

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

    std::shared_ptr<VertexImp<DeviceType::WebGPU>> buffer;

    std::shared_ptr<BufferImp<DeviceType::WebGPU>> indexedIndirectBuffer;
    std::shared_ptr<BufferImp<DeviceType::WebGPU>> indirectBuffer;

    std::vector<std::vector<WGPUVertexAttribute>> attributeDescription;
    std::vector<WGPUVertexBufferLayout> vertexLayouts;

    std::vector<IndexedDrawCommand> indexedDrawCommands;
    std::vector<DrawCommand> drawCommands;

    friend class ProgramImp<DeviceType::WebGPU>;
};

}  // namespace Jetstream::Render

#endif
