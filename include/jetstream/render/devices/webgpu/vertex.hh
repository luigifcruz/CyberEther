#ifndef JETSTREAM_RENDER_WEBGPU_VERTEX_HH
#define JETSTREAM_RENDER_WEBGPU_VERTEX_HH

#include "jetstream/render/base/vertex.hh"
#include "jetstream/backend/base.hh"

namespace Jetstream::Render {

template<>
class VertexImp<Device::WebGPU> : public Vertex {
 public:
    explicit VertexImp(const Config& config);

 protected:
    Result create(wgpu::RenderPipelineDescriptor& renderDescriptor);
    Result destroy();
    Result encode(wgpu::RenderPassEncoder& renderPassEncoder);

    constexpr U64 getVertexCount() const {
        return vertexCount;
    }

    constexpr std::vector<wgpu::VertexBufferLayout>& getHandle() {
        return vertexLayouts;
    }

    bool isBuffered() {
        return indices != nullptr;
    }

 private:
    U64 vertexCount;

    std::vector<wgpu::VertexAttribute> vertexAttributes;
    std::vector<wgpu::VertexBufferLayout> vertexLayouts;

    std::vector<std::pair<std::shared_ptr<BufferImp<Device::WebGPU>>, U32>> buffers;
    std::shared_ptr<BufferImp<Device::WebGPU>> indices;

    friend class DrawImp<Device::WebGPU>;
};

}  // namespace Jetstream::Render

#endif
