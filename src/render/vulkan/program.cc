#include "jetstream/render/vulkan/buffer.hh"
#include "jetstream/render/vulkan/draw.hh"
#include "jetstream/render/vulkan/texture.hh"
#include "jetstream/render/vulkan/program.hh"
#include "jetstream/backend/devices/vulkan/helpers.hh"

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
                              const std::shared_ptr<TextureImp<Device::Vulkan>>& framebuffer) {
    JST_DEBUG("[VULKAN] Creating program.");

    auto& backend = Backend::State<Device::Vulkan>();
    auto& device = backend->getDevice();

    // Load shaders from buffers.

    if (config.shaders.contains(Device::Vulkan) == 0) {
        JST_ERROR("[VULKAN] Module doesn't have necessary shader.");       
        return Result::ERROR;
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

    // Initiate textures.

    for (const auto& texture : textures) {
        JST_CHECK(texture->create());
    }

    // Create uniforms and texture descriptor buffers.

    bindingOffset = 0;
    for (U64 i = 0; i < buffers.size(); i++) {
        auto& [buffer, target] = buffers[i];

        VkDescriptorSetLayoutBinding binding{};
        binding.descriptorType = BufferDescriptorType(buffer);
        binding.descriptorCount = 1;
        binding.stageFlags = TargetToShaderStage(target);
        binding.binding = bindingOffset++;

        bindings.push_back(binding);
    }

    for (U64 i = 0; i < textures.size(); i++) {
        {
            VkDescriptorSetLayoutBinding binding{};
            binding.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            binding.descriptorCount = 1;
            binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            binding.binding = bindingOffset++;
            bindings.push_back(binding);
        }

        {
            VkDescriptorSetLayoutBinding binding{};
            binding.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
            binding.descriptorCount = 1;
            binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            binding.binding = bindingOffset++;
            bindings.push_back(binding);
        }
    }

    if (!bindings.empty()) {
        VkDescriptorSetLayoutCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        info.bindingCount = static_cast<U32>(bindings.size());
        info.pBindings = bindings.data();

        JST_VK_CHECK(vkCreateDescriptorSetLayout(device, &info, nullptr, &descriptorSetLayout), [&]{
            JST_ERROR("[VULKAN] Can't create descriptor set layout.");
        });

        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = backend->getDescriptorPool();
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &descriptorSetLayout;

        JST_VK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet), [&]{
            JST_ERROR("[VULKAN] Failed to allocate descriptor sets.");
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
        descriptorWriteBuffer.descriptorType = BufferDescriptorType(buffer);
        descriptorWriteBuffer.descriptorCount = 1;
        descriptorWriteBuffer.pBufferInfo = &bufferInfo;

        vkUpdateDescriptorSets(device, 1, &descriptorWriteBuffer, 0, nullptr);
    }

    for (U64 i = 0; i < textures.size(); i++) {
        auto& texture = textures[i];

        {
            VkDescriptorImageInfo imageInfo{};
            imageInfo.imageView = texture->getViewHandle();
            imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkWriteDescriptorSet descriptorWriteBuffer{};
            descriptorWriteBuffer.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWriteBuffer.dstSet = descriptorSet;
            descriptorWriteBuffer.dstBinding = bindingOffset++;
            descriptorWriteBuffer.dstArrayElement = 0;
            descriptorWriteBuffer.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            descriptorWriteBuffer.descriptorCount = 1;
            descriptorWriteBuffer.pImageInfo = &imageInfo;

            vkUpdateDescriptorSets(device, 1, &descriptorWriteBuffer, 0, nullptr);
        }

        {
            VkDescriptorImageInfo samplerInfo{};
            samplerInfo.sampler = texture->getSamplerHandler();

            VkWriteDescriptorSet descriptorWriteBuffer{};
            descriptorWriteBuffer.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWriteBuffer.dstSet = descriptorSet;
            descriptorWriteBuffer.dstBinding = bindingOffset++;
            descriptorWriteBuffer.dstArrayElement = 0;
            descriptorWriteBuffer.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
            descriptorWriteBuffer.descriptorCount = 1;
            descriptorWriteBuffer.pImageInfo = &samplerInfo;

            vkUpdateDescriptorSets(device, 1, &descriptorWriteBuffer, 0, nullptr);
        }
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
    viewport.y = framebuffer->size().y;
    viewport.width = framebuffer->size().x;
    viewport.height = -static_cast<F32>(framebuffer->size().y);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 0.0f;

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = VkExtent2D{
            static_cast<U32>(framebuffer->size().x),
            static_cast<U32>(framebuffer->size().y)
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
    if (framebuffer->multisampled()) {
        multisampling.rasterizationSamples = Backend::State<Device::Vulkan>()->getMultisampling();
    } else {
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    }

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask =  VK_COLOR_COMPONENT_R_BIT | 
                                           VK_COLOR_COMPONENT_G_BIT |
                                           VK_COLOR_COMPONENT_B_BIT | 
                                           VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;

    if (config.enableAlphaBlending) {
        colorBlendAttachment.blendEnable = VK_TRUE;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
    }

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

    // Create pipeline layout.

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    if (!bindings.empty()) {
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
    }
    pipelineLayoutInfo.pushConstantRangeCount = 0;

    JST_VK_CHECK(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout), [&]{
        JST_ERROR("[VULKAN] Failed to create pipeline layout.");
    });

    // Create graphics pipeline.

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
        JST_ERROR("[VULKAN] Can't create graphics pipeline.");    
    });

    // Clean up.

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

    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyPipeline(device, graphicsPipeline, nullptr);
    bindings.clear();

    return Result::SUCCESS;
}

Result Implementation::encode(VkCommandBuffer& commandBuffer, VkRenderPass&) {
    // Bind uniform and texture buffers.

    if (!bindings.empty()) {
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
    }

    // Bind graphics pipeline.

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

    // Attach frame encoder.

    JST_CHECK(draw->encode(commandBuffer));

    return Result::SUCCESS;
}

VkShaderStageFlags Implementation::TargetToShaderStage(const Program::Target& target) {
    VkShaderStageFlags flags = 0;

    if ((target & Program::Target::VERTEX) == Program::Target::VERTEX) {
        flags |= VK_SHADER_STAGE_VERTEX_BIT;
    }

    if ((target & Program::Target::FRAGMENT) == Program::Target::FRAGMENT) {
        flags |= VK_SHADER_STAGE_FRAGMENT_BIT;
    }
        
    return flags;
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
