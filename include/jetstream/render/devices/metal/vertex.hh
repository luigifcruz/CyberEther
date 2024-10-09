#ifndef JETSTREAM_RENDER_METAL_VERTEX_HH
#define JETSTREAM_RENDER_METAL_VERTEX_HH

#include "jetstream/render/base/vertex.hh"
#include "jetstream/backend/base.hh"

namespace Jetstream::Render {

template<>
class VertexImp<Device::Metal> : public Vertex {
 public:
    explicit VertexImp(const Config& config);

 protected:
    Result create(MTL::VertexDescriptor* vertDesc, 
                  const U64& numberOfDraws, 
                  const U64& numberOfInstances, 
                  const U64& offset);
    Result destroy();
    Result encode(MTL::RenderCommandEncoder* encode);

    const MTL::Buffer* getIndexBuffer();

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
    U32 indexOffset;

    std::vector<std::pair<std::shared_ptr<BufferImp<Device::Metal>>, U32>> vertices;
    std::vector<std::pair<std::shared_ptr<BufferImp<Device::Metal>>, U32>> instances;
    std::shared_ptr<BufferImp<Device::Metal>> indices;

    friend class DrawImp<Device::Metal>;
};

}  // namespace Jetstream::Render

#endif
