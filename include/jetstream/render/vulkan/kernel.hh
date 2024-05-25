#ifndef JETSTREAM_RENDER_VULKAN_KERNEL_HH
#define JETSTREAM_RENDER_VULKAN_KERNEL_HH

#include "jetstream/render/base/kernel.hh"
#include "jetstream/backend/base.hh"

namespace Jetstream::Render {

template<>
class KernelImp<Device::Vulkan> : public Kernel {
 public:
    explicit KernelImp(const Config& config);

 protected:
    Result create();
    Result destroy();
    Result encode(VkCommandBuffer& commandBuffer);

 private:
    VkPipeline pipeline;
    VkPipelineLayout pipelineLayout;
    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorSet descriptorSet;

    std::vector<VkDescriptorSetLayoutBinding> bindings;

    std::vector<std::shared_ptr<BufferImp<Device::Vulkan>>> buffers;

    static Result checkShaderCompilation(U64);
    static Result checkProgramCompilation(U64);

    friend class SurfaceImp<Device::Vulkan>; 
};

}  // namespace Jetstream::Render

#endif
