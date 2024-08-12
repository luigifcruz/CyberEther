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
   Result create(std::vector<VkVertexInputBindingDescription>& bindingDescription,
                 std::vector<VkVertexInputAttributeDescription>& attributeDescription,
                 const U64& numberOfDraws,
                 const U64& numberOfInstances);
    Result encode(VkCommandBuffer& commandBuffer);
    Result destroy();

    constexpr const U32& getVertexCount() const {
        return vertexCount;
    }

    constexpr const U32& getIndexCount() const {
        return indexCount;
    }

    bool isBuffered() const {
        return indices != nullptr;
    }

 private:
    U32 vertexCount;
    U32 indexCount;

    std::vector<std::pair<std::shared_ptr<BufferImp<Device::Vulkan>>, U32>> vertices;
    std::vector<std::pair<std::shared_ptr<BufferImp<Device::Vulkan>>, U32>> instances;
    std::shared_ptr<BufferImp<Device::Vulkan>> indices;

    friend class DrawImp<Device::Vulkan>;
};

}  // namespace Jetstream::Render

#endif
