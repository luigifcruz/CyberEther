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
    Result create(wgpu::RenderPipelineDescriptor& renderDescriptor, 
                  const U64& numberOfDraws,
                  const U64& numberOfInstances);
    Result destroy();
    Result encode(wgpu::RenderPassEncoder& renderPassEncoder);

    constexpr U32 getVertexCount() const {
        return vertexCount;
    }

    constexpr U32 getIndexCount() const {
        return indexCount;
    }

    bool isBuffered() {
        return indices != nullptr;
    }

    constexpr std::vector<wgpu::VertexBufferLayout>& getHandle() {
        return vertexLayouts;
    }

 private:
    U64 vertexCount;
    U32 indexCount;

    std::vector<std::vector<wgpu::VertexAttribute>> attributeDescription;
    std::vector<wgpu::VertexBufferLayout> vertexLayouts;

    std::vector<std::pair<std::shared_ptr<BufferImp<Device::WebGPU>>, U32>> vertices;
    std::vector<std::pair<std::shared_ptr<BufferImp<Device::WebGPU>>, U32>> instances;
    std::shared_ptr<BufferImp<Device::WebGPU>> indices;

    friend class DrawImp<Device::WebGPU>;
};

}  // namespace Jetstream::Render

#endif
