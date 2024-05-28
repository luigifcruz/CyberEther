#include "jetstream/render/vulkan/buffer.hh"
#include "jetstream/render/vulkan/kernel.hh"
#include "jetstream/backend/devices/vulkan/helpers.hh"

namespace Jetstream::Render {

using Implementation = KernelImp<Device::Vulkan>;

Implementation::KernelImp(const Config& config) : Kernel(config) {
    for (auto& buffer : config.buffers) {
        buffers.push_back(
            std::dynamic_pointer_cast<BufferImp<Device::Vulkan>>(buffer)
        );
    }
}

Result Implementation::create() {
    JST_DEBUG("[VULKAN] Creating kernel.");

    auto& backend = Backend::State<Device::Vulkan>();
    auto device = backend->getDevice();

    // Load kernel from buffers. 

    if (config.kernels.contains(Device::Vulkan) == 0) {
        JST_ERROR("[VULKAN] Module doesn't have necessary kernel.");       
        return Result::ERROR;
    }

    const auto& kernels = config.kernels[Device::Vulkan];
    VkShaderModule kernelModule = Backend::LoadShader(kernels[0], device);

    VkPipelineShaderStageCreateInfo kernelStageInfo{};
    kernelStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    kernelStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    kernelStageInfo.module = kernelModule;
    kernelStageInfo.pName = "main";

    // Create descriptor set layout.

    for (U64 i = 0; i < buffers.size(); i++) {
        auto& buffer = buffers[i];

        VkDescriptorSetLayoutBinding binding{};
        binding.binding = i;
        binding.descriptorType = BufferDescriptorType(buffer);
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        bindings.push_back(binding);
    }

    if (!bindings.empty()) {
        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<U32>(bindings.size());
        layoutInfo.pBindings = bindings.data();

        JST_VK_CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout), [&]{
            JST_ERROR("[VULKAN] Failed to create descriptor set layout.");
        });

        // Allocate descriptor set.

        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = backend->getDescriptorPool();
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &descriptorSetLayout;

        JST_VK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet), [&]{
            JST_ERROR("[VULKAN] Failed to allocate descriptor set.");
        });
    }

    // Update descriptor set.

    for (U64 i = 0; i < buffers.size(); i++) {
        auto& buffer = buffers[i];

        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = buffer->getHandle();
        bufferInfo.offset = 0;
        bufferInfo.range = buffer->byteSize();

        VkWriteDescriptorSet descriptorWriteBuffer{};
        descriptorWriteBuffer.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWriteBuffer.dstSet = descriptorSet;
        descriptorWriteBuffer.dstBinding = i;
        descriptorWriteBuffer.dstArrayElement = 0;
        descriptorWriteBuffer.descriptorType = BufferDescriptorType(buffer);
        descriptorWriteBuffer.descriptorCount = 1;
        descriptorWriteBuffer.pBufferInfo = &bufferInfo;

        vkUpdateDescriptorSets(device, 1, &descriptorWriteBuffer, 0, nullptr);
    }

    // Create pipeline layout.

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    if (!bindings.empty()) {
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
    }

    JST_VK_CHECK(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout), [&]{
        JST_ERROR("[VULKAN] Failed to create pipeline layout.");
    });

    // Create compute pipeline.

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.stage = kernelStageInfo;

    JST_VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline), [&]{
        JST_ERROR("[VULKAN] Failed to create compute pipeline.");
    });

    // Clean up.

    vkDestroyShaderModule(device, kernelModule, nullptr);

    // Set update flag.

    this->updated = true;

    return Result::SUCCESS;
}

Result Implementation::destroy() {
    auto& device = Backend::State<Device::Vulkan>()->getDevice();
    auto& descriptorPool = Backend::State<Device::Vulkan>()->getDescriptorPool();

    if (!bindings.empty()) {
        vkFreeDescriptorSets(device, descriptorPool, 1, &descriptorSet);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    }

    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyPipeline(device, pipeline, nullptr);
    bindings.clear();

    return Result::SUCCESS;
}

Result Implementation::encode(VkCommandBuffer& commandBuffer) {
    // Check if data needs to be updated.

    if (!this->updated) {
        return Result::SUCCESS;
    }
    this->updated = false;

    // Bind buffers.

    if (!bindings.empty()) {
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
    }

    // Bind compute pipeline.

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

    // Dispatch kernel.

    const auto& [x, y, z] = config.gridSize;

    if (y != 1 || z != 1) {
        JST_ERROR("[VULKAN] 2D and 3D grid sizes are not implemented.");
        return Result::ERROR;
    }

    vkCmdDispatch(commandBuffer, x, y, z);

    return Result::SUCCESS;
}

VkDescriptorType Implementation::BufferDescriptorType(const std::shared_ptr<Buffer>& buffer) {
    const auto& bufferType = buffer->getConfig().target;

    if ((bufferType & Buffer::Target::UNIFORM) == Buffer::Target::UNIFORM) {
        return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    }

    if ((bufferType & Buffer::Target::STORAGE) == Buffer::Target::STORAGE) {
        return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    }

    JST_ERROR("[VULKAN] Invalid buffer usage.");
    throw Result::ERROR;
}

}  // namespace Jetstream::Render
