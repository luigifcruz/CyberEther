#ifndef JETSTREAM_RENDER_VULKAN_PROGRAM_HH
#define JETSTREAM_RENDER_VULKAN_PROGRAM_HH

#include "jetstream/render/base/program.hh"
#include "jetstream/backend/base.hh"
#include "jetstream/backend/devices/vulkan/helpers.hh"

namespace Jetstream::Render {

template<>
class ProgramImp<Device::Vulkan> : public Program {
 public:
    explicit ProgramImp(const Config& config);

 protected:
    Result create(VkRenderPass& renderPass,
                  std::shared_ptr<TextureImp<Device::Vulkan>>& texture);
    Result encode(VkCommandBuffer& commandBuffer, VkRenderPass& renderPass);
    Result destroy();

 private:
    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;

    std::vector<std::shared_ptr<DrawImp<Device::Vulkan>>> draws;
    std::vector<std::shared_ptr<TextureImp<Device::Vulkan>>> textures;
    std::vector<std::shared_ptr<BufferImp<Device::Vulkan>>> buffers;

    static Result checkShaderCompilation(U64);
    static Result checkProgramCompilation(U64);

    friend class SurfaceImp<Device::Vulkan>; 
};

}  // namespace Jetstream::Render

#endif
