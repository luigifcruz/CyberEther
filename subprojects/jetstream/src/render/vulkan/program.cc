#include "jetstream/render/vulkan/buffer.hh"
#include "jetstream/render/vulkan/draw.hh"
#include "jetstream/render/vulkan/texture.hh"
#include "jetstream/render/vulkan/program.hh"

namespace Jetstream::Render {

using Implementation = ProgramImp<Device::Vulkan>;

Implementation::ProgramImp(const Config& config) : Program(config) {
    draw = std::dynamic_pointer_cast<DrawImp<Device::Vulkan>>(config.draw);

    for (auto& texture : config.textures) {
        textures.push_back(
            std::dynamic_pointer_cast<TextureImp<Device::Vulkan>>(texture)
        );
    }

    for (auto& [buffer, target] : config.buffers) {
        buffers.push_back(
            {std::dynamic_pointer_cast<BufferImp<Device::Vulkan>>(buffer), target}
        );
    }
}

Result Implementation::create(VkRenderPass& renderPass,
                              std::shared_ptr<TextureImp<Device::Vulkan>>& framebuffer) {
    JST_DEBUG("[VULKAN] Creating program.");

    auto& backend = Backend::State<Device::Vulkan>();
    auto& device = backend->getDevice();

    // Load shaders from buffers.

    if (config.shaders.count(Device::Vulkan) == 0) {
        JST_FATAL("[VULKAN] Module doesn't have necessary shader.");       
        JST_CHECK(Result::ERROR);
    }

    const auto& shader = config.shaders[Device::Vulkan];
    VkShaderModule vertShaderModule = Backend::LoadShader(shader[0], device);
    VkShaderModule fragShaderModule = Backend::LoadShader(shader[1], device);

    VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = vertShaderModule;
    vertShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = fragShaderModule;
    fragShaderStageInfo.pName = "main";

    // Initiate uniforms and texture.

    for (const auto& texture : textures) {
        JST_CHECK(texture->create());
    }

    for (const auto& [buffer, _] : buffers) {
        JST_CHECK(buffer->create());
    }

    // Create uniforms and texture descriptor buffers.

    bindingOffset = 0;
    for (U64 i = 0; i < buffers.size(); i++) {
        auto& [buffer, target] = buffers[i];

        VkDescriptorSetLayoutBinding binding{};
        binding.descriptorType = buffer->getDescriptorType();
        binding.descriptorCount = 1;
        binding.stageFlags = TargetToVulkan(target);
        binding.binding = bindingOffset++;

        bindings.push_back(binding);
    }

    for (U64 i = 0; i < textures.size(); i++) {
        VkDescriptorSetLayoutBinding binding{};
        binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        binding.binding = bindingOffset++;

        bindings.push_back(binding);
    }

    if (!bindings.empty()) {
        VkDescriptorSetLayoutCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        info.bindingCount = static_cast<U32>(bindings.size());
        info.pBindings = bindings.data();

        JST_VK_CHECK(vkCreateDescriptorSetLayout(device, &info, nullptr, &descriptorSetLayout), [&]{
            JST_FATAL("[VULKAN] Can't create descriptor set layout.");
        });

        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = backend->getDescriptorPool();
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &descriptorSetLayout;

        JST_VK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet), [&]{
            JST_FATAL("[VULKAN] Failed to allocate descriptor sets.");
        });
    }

    bindingOffset = 0;
    for (U64 i = 0; i < buffers.size(); i++) {
        auto& [buffer, target] = buffers[i];

        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = buffer->getHandle();
        bufferInfo.offset = 0;
        bufferInfo.range = buffer->byteSize();

        VkWriteDescriptorSet descriptorWriteBuffer{};
        descriptorWriteBuffer.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWriteBuffer.dstSet = descriptorSet;
        descriptorWriteBuffer.dstBinding = bindingOffset++;
        descriptorWriteBuffer.dstArrayElement = 0;
        descriptorWriteBuffer.descriptorType = buffer->getDescriptorType();
        descriptorWriteBuffer.descriptorCount = 1;
        descriptorWriteBuffer.pBufferInfo = &bufferInfo;

        vkUpdateDescriptorSets(device, 1, &descriptorWriteBuffer, 0, nullptr);
    }

    for (U64 i = 0; i < textures.size(); i++) {
        auto& texture = textures[i];

        VkDescriptorImageInfo imageInfo{};
        imageInfo.imageView = texture->getViewHandle();
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfo.sampler = texture->getSamplerHandler();

        VkWriteDescriptorSet descriptorWriteBuffer{};
        descriptorWriteBuffer.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWriteBuffer.dstSet = descriptorSet;
        descriptorWriteBuffer.dstBinding = bindingOffset++;
        descriptorWriteBuffer.dstArrayElement = 0;
        descriptorWriteBuffer.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWriteBuffer.descriptorCount = 1;
        descriptorWriteBuffer.pImageInfo = &imageInfo;

        vkUpdateDescriptorSets(device, 1, &descriptorWriteBuffer, 0, nullptr);
    }

    // Attach Vertex buffers.

    std::vector<VkVertexInputBindingDescription> bindingDescription;
    std::vector<VkVertexInputAttributeDescription> attributeDescriptions;
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};

    JST_CHECK(draw->create(bindingDescription,
                           attributeDescriptions,
                           inputAssembly));
    JST_ASSERT(bindingDescription.size() == attributeDescriptions.size());

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = static_cast<U32>(bindingDescription.size());
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<U32>(attributeDescriptions.size());
    vertexInputInfo.pVertexBindingDescriptions = bindingDescription.data();
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    // Setup viewport and drawing settings.
    // This is upside down because Vulkan is weird.
    // https://www.saschawillems.de/blog/2019/03/29/flipping-the-vulkan-viewport

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = framebuffer->size().height;
    viewport.width = framebuffer->size().width;
    viewport.height = -static_cast<F32>(framebuffer->size().height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 0.0f;

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = VkExtent2D{
            static_cast<U32>(framebuffer->size().width),
            static_cast<U32>(framebuffer->size().height)
        };

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask =  VK_COLOR_COMPONENT_R_BIT | 
                                           VK_COLOR_COMPONENT_G_BIT |
                                           VK_COLOR_COMPONENT_B_BIT | 
                                           VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.logicOp = VK_LOGIC_OP_COPY;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    colorBlending.blendConstants[0] = 0.0f;
    colorBlending.blendConstants[1] = 0.0f;
    colorBlending.blendConstants[2] = 0.0f;
    colorBlending.blendConstants[3] = 0.0f;

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    if (!bindings.empty()) {
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
    }
    pipelineLayoutInfo.pushConstantRangeCount = 0;

    JST_VK_CHECK(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout), [&]{
        JST_FATAL("[VULKAN] Failed to create pipeline layout.");
    });

    VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

    JST_VK_CHECK(vkCreateGraphicsPipelines(device, nullptr, 1, &pipelineInfo, nullptr, &graphicsPipeline), [&]{
        JST_FATAL("[VULKAN] Can't create graphics pipeline.");    
    });

    vkDestroyShaderModule(device, fragShaderModule, nullptr);
    vkDestroyShaderModule(device, vertShaderModule, nullptr);

    return Result::SUCCESS;
}

Result Implementation::destroy() {
    JST_CHECK(draw->destroy());

    auto& device = Backend::State<Device::Vulkan>()->getDevice();
    auto& descriptorPool = Backend::State<Device::Vulkan>()->getDescriptorPool();

    if (!bindings.empty()) {
        vkFreeDescriptorSets(device, descriptorPool, 1, &descriptorSet);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    }

    for (const auto& texture : textures) {
        JST_CHECK(texture->destroy());
    }

    for (const auto& [buffer, _] : buffers) {
        JST_CHECK(buffer->destroy());
    }

    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyPipeline(device, graphicsPipeline, nullptr);
    bindings.clear();

    return Result::SUCCESS;
}

Result Implementation::encode(VkCommandBuffer& commandBuffer, VkRenderPass& renderPass) {
    if (!bindings.empty()) {
        // Bind uniform and texture buffers.
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
    }

    // Bind graphics pipeline.
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

    // Attach frame encoder.
    JST_CHECK(draw->encode(commandBuffer));

    return Result::SUCCESS;
}

VkShaderStageFlags Implementation::TargetToVulkan(const Program::Target& target) {
    VkShaderStageFlags flags = 0;

    if (static_cast<U8>(target & Program::Target::VERTEX) > 0) {
        flags |= VK_SHADER_STAGE_VERTEX_BIT;
    }

    if (static_cast<U8>(target & Program::Target::FRAGMENT) > 0) {
        flags |= VK_SHADER_STAGE_FRAGMENT_BIT;
    }
        
    return flags;
}

}  // namespace Jetstream::Render
