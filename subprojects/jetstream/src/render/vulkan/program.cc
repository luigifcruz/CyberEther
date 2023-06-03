#include "jetstream/render/vulkan/buffer.hh"
#include "jetstream/render/vulkan/draw.hh"
#include "jetstream/render/vulkan/texture.hh"
#include "jetstream/render/vulkan/program.hh"

namespace Jetstream::Render {

struct ShaderVertexAttachment {
    F32 pos_x, pos_y;
    F32 col_r, col_g, col_b;
};

using Implementation = ProgramImp<Device::Vulkan>;

Implementation::ProgramImp(const Config& config) : Program(config) {
    for (auto& draw : config.draws) {
        draws.push_back(
            std::dynamic_pointer_cast<DrawImp<Device::Vulkan>>(draw)
        );
    }

    for (auto& texture : config.textures) {
        textures.push_back(
            std::dynamic_pointer_cast<TextureImp<Device::Vulkan>>(texture)
        );
    }

    for (auto& buffer : config.buffers) {
        buffers.push_back(
            std::dynamic_pointer_cast<BufferImp<Device::Vulkan>>(buffer)
        );
    }
}

Result Implementation::create(VkRenderPass& renderPass,
                              std::shared_ptr<TextureImp<Device::Vulkan>>& framebuffer) {
    JST_DEBUG("[VULKAN] Creating program.");

    auto& device = Backend::State<Device::Vulkan>()->getDevice();

    // Load shader data.
    const auto& shader = config.shaders[Device::Vulkan][0];
    VkShaderModule vertShaderModule = Backend::LoadShader(shader[0], sizeof(shader[0]), device);
    VkShaderModule fragShaderModule = Backend::LoadShader(shader[1], sizeof(shader[1]), device);

    // Create Vertex & Fragment shader for the pipeline.
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

    VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

    // Attach Vertex attachments.
    VkVertexInputBindingDescription bindingDescription{};
    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(ShaderVertexAttachment);
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};
    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
    attributeDescriptions[0].offset = 0;
    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[1].offset = 8;
    
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<U32>(attributeDescriptions.size());
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = framebuffer->size().width;
    viewport.height = framebuffer->size().height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

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
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
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
    pipelineLayoutInfo.setLayoutCount = 0;
    pipelineLayoutInfo.pushConstantRangeCount = 0;

    JST_VK_CHECK(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout), [&]{
        JST_FATAL("[VULKAN] Failed to create pipeline layout.");   
    });

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

    JST_VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline), [&]{
        JST_FATAL("[VULKAN] Can't create graphics pipeline.");    
    });

    vkDestroyShaderModule(device, fragShaderModule, nullptr);
    vkDestroyShaderModule(device, vertShaderModule, nullptr);

    for (const auto& draw : draws) {
        JST_CHECK(draw->create());
    }

    for (const auto& texture : textures) {
        JST_CHECK(texture->create());
    }

    for (const auto& buffer : buffers) {
        JST_CHECK(buffer->create());
    }

    return Result::SUCCESS;
}

Result Implementation::destroy() {
    for (const auto& draw : draws) {
        JST_CHECK(draw->destroy());
    }

    for (const auto& texture : textures) {
        JST_CHECK(texture->destroy());
    }

    for (const auto& buffer : buffers) {
        JST_CHECK(buffer->destroy());
    }

    auto& device = Backend::State<Device::Vulkan>()->getDevice();

    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyPipeline(device, graphicsPipeline, nullptr);

    return Result::SUCCESS;
}

Result Implementation::encode(VkCommandBuffer& commandBuffer, VkRenderPass& renderPass) {
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

    // // Attach frame textures.
    // for (U64 i = 0; i < textures.size(); i++) {
    //     renderCmdEncoder->setFragmentTexture(textures[i]->getHandle(), i);
    // }

    // // Attach frame fragment-shader buffers.
    // for (U64 i = 0; i < buffers.size(); i++) {
    //     renderCmdEncoder->setFragmentBuffer(buffers[i]->getHandle(), 0, i);
    //     renderCmdEncoder->setVertexBuffer(buffers[i]->getHandle(), 0, i);
    // }

    drawIndex = 0;
    for (auto& draw : draws) {
        // // Attach drawIndex uniforms.
        // renderCmdEncoder->setVertexBytes(&drawIndex, sizeof(drawIndex), 30);
        // renderCmdEncoder->setFragmentBytes(&drawIndex, sizeof(drawIndex), 30);
        // drawIndex += 1;

        // Attach frame encoder.
        JST_CHECK(draw->encode(commandBuffer, buffers.size()));
    }

    return Result::SUCCESS;
}

}  // namespace Jetstream::Render
