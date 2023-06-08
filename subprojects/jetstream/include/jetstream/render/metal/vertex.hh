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
    Result create(MTL::VertexDescriptor* vertDesc);
    Result destroy();
    Result encode(MTL::RenderCommandEncoder* encode,
                  const U64& offset);

    const MTL::Buffer* getIndexBuffer();

    constexpr U64 getVertexCount() const {
        return vertexCount;
    }

    bool isBuffered() {
        return indices != nullptr;
    }

 private:
    U64 vertexCount;
    std::vector<std::pair<std::shared_ptr<BufferImp<Device::Metal>>, U32>> buffers;
    std::shared_ptr<BufferImp<Device::Metal>> indices;

    friend class DrawImp<Device::Metal>;
};

}  // namespace Jetstream::Render

#endif
