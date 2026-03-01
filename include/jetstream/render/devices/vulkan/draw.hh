#ifndef JETSTREAM_RENDER_VULKAN_DRAW_HH
#define JETSTREAM_RENDER_VULKAN_DRAW_HH

#include "jetstream/render/base/draw.hh"
#include "jetstream/backend/base.hh"

namespace Jetstream::Render {

template<>
class DrawImp<DeviceType::Vulkan> : public Draw {
 public:
    explicit DrawImp(const Config& config);

 protected:
   Result create(std::vector<VkVertexInputBindingDescription>& bindingDescription,
                 std::vector<VkVertexInputAttributeDescription>& attributeDescription,
                 VkPipelineInputAssemblyStateCreateInfo& inputAssembly);
    Result encode(VkCommandBuffer& commandBuffer);
    Result updateVertexCount(U64 vertexCount) override;
    Result destroy();

 private:
    std::shared_ptr<VertexImp<DeviceType::Vulkan>> buffer;

    std::shared_ptr<BufferImp<DeviceType::Vulkan>> indirectBuffer;
    std::shared_ptr<BufferImp<DeviceType::Vulkan>> indexedIndirectBuffer;

    std::vector<VkDrawIndirectCommand> drawCommands;
    std::vector<VkDrawIndexedIndirectCommand> indexedDrawCommands;

    friend class ProgramImp<DeviceType::Vulkan>;
};

}  // namespace Jetstream::Render

#endif
