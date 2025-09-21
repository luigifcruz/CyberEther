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
    Result create(std::vector<std::vector<WGPUVertexAttribute>>& attributeDescription,
                  std::vector<WGPUVertexBufferLayout>& vertexLayouts,
                  const U64& numberOfDraws,
                  const U64& numberOfInstances);
    Result destroy();
    Result encode(WGPURenderPassEncoder& renderPassEncoder);

    constexpr const U32& getVertexCount() const {
        return vertexCount;
    }

    constexpr const U32& getIndexCount() const {
        return indexCount;
    }

    bool isBuffered() {
        return indices != nullptr;
    }

 private:
    U32 vertexCount;
    U32 indexCount;

    std::vector<std::pair<std::shared_ptr<BufferImp<Device::WebGPU>>, U32>> vertices;
    std::vector<std::pair<std::shared_ptr<BufferImp<Device::WebGPU>>, U32>> instances;
    std::shared_ptr<BufferImp<Device::WebGPU>> indices;

    friend class DrawImp<Device::WebGPU>;
};

}  // namespace Jetstream::Render

#endif
