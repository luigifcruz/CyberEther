#ifndef JETSTREAM_RENDER_VULKAN_VERTEX_HH
#define JETSTREAM_RENDER_VULKAN_VERTEX_HH

#include "jetstream/render/base/vertex.hh"
#include "jetstream/backend/base.hh"

namespace Jetstream::Render {

template<>
class VertexImp<Device::Vulkan> : public Vertex {
 public:
    explicit VertexImp(const Config& config);

 protected:
    Result create();
    Result encode(VkCommandBuffer& commandBuffer, 
                  const U64& offset);
    Result destroy();

    constexpr U32 getIndicesCount() const {
        return vertexCount;
    }

 private:
    U64 vertexCount;
    std::vector<std::pair<std::shared_ptr<BufferImp<Device::Vulkan>>, U32>> buffers;
    std::shared_ptr<BufferImp<Device::Vulkan>> indices;

    friend class DrawImp<Device::Vulkan>;
};

}  // namespace Jetstream::Render

#endif
