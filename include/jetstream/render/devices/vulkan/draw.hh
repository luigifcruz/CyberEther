#ifndef JETSTREAM_RENDER_VULKAN_DRAW_HH
#define JETSTREAM_RENDER_VULKAN_DRAW_HH

#include "jetstream/render/base/draw.hh"
#include "jetstream/backend/base.hh"

namespace Jetstream::Render {

template<>
class DrawImp<Device::Vulkan> : public Draw {
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
    std::shared_ptr<VertexImp<Device::Vulkan>> buffer;

    std::shared_ptr<BufferImp<Device::Vulkan>> indirectBuffer;
    std::shared_ptr<BufferImp<Device::Vulkan>> indexedIndirectBuffer;

    std::vector<VkDrawIndirectCommand> drawCommands;
    std::vector<VkDrawIndexedIndirectCommand> indexedDrawCommands;

    friend class ProgramImp<Device::Vulkan>;
};

}  // namespace Jetstream::Render

#endif
