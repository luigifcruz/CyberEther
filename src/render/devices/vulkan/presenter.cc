#include "presenter.hh"

#include "jetstream/backend/devices/vulkan/helpers.hh"
#include "resources/shaders/global_shaders.hh"

#include "buffer_allocation.hh"
#include "creation.hh"
#include "present_layout.hh"
#include "../../present_impl.hh"

#include <array>
#include <cstddef>
#include <cstring>
#include <optional>

namespace Jetstream::Render {

namespace {

constexpr U64 kInitialVertices = 6 * 16;

struct ShaderModule {
    VkDevice device = VK_NULL_HANDLE;
    VkShaderModule handle = VK_NULL_HANDLE;

    ShaderModule(VkDevice device, VkShaderModule handle) : device(device), handle(handle) {}
    ShaderModule(const ShaderModule&) = delete;
    ShaderModule& operator=(const ShaderModule&) = delete;

    ~ShaderModule() {
        if (handle) {
            vkDestroyShaderModule(device, handle, nullptr);
        }
    }
};

Result CreatePipeline(VkDevice device,
                      VkRenderPass renderPass,
                      VkPipelineLayout layout,
                      VkShaderModule vertex,
                      VkShaderModule fragment,
                      PresentList::Blend blend,
                      VkPipeline& pipeline) {
    std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertex;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragment;
    stages[1].pName = "main";

    VkVertexInputBindingDescription binding{};
    binding.binding = 0;
    binding.stride = sizeof(PresentGeometry::Vertex);
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    std::array<VkVertexInputAttributeDescription, 2> attributes{};
    attributes[0].location = 0;
    attributes[0].binding = 0;
    attributes[0].format = VK_FORMAT_R32G32_SFLOAT;
    attributes[0].offset = offsetof(PresentGeometry::Vertex, x);
    attributes[1].location = 1;
    attributes[1].binding = 0;
    attributes[1].format = VK_FORMAT_R32G32_SFLOAT;
    attributes[1].offset = offsetof(PresentGeometry::Vertex, u);

    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount = 1;
    vertexInput.pVertexBindingDescriptions = &binding;
    vertexInput.vertexAttributeDescriptionCount = static_cast<U32>(attributes.size());
    vertexInput.pVertexAttributeDescriptions = attributes.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState attachment{};
    attachment.blendEnable = VK_TRUE;
    attachment.srcColorBlendFactor = blend == PresentList::Blend::Premultiplied ? VK_BLEND_FACTOR_ONE
                                                                                : VK_BLEND_FACTOR_SRC_ALPHA;
    attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    attachment.colorBlendOp = VK_BLEND_OP_ADD;
    attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    attachment.alphaBlendOp = VK_BLEND_OP_ADD;
    attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo blending{};
    blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blending.attachmentCount = 1;
    blending.pAttachments = &attachment;

    std::array<VkDynamicState, 2> dynamicStates = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{};
    dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic.dynamicStateCount = static_cast<U32>(dynamicStates.size());
    dynamic.pDynamicStates = dynamicStates.data();

    VkGraphicsPipelineCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    info.stageCount = static_cast<U32>(stages.size());
    info.pStages = stages.data();
    info.pVertexInputState = &vertexInput;
    info.pInputAssemblyState = &inputAssembly;
    info.pViewportState = &viewportState;
    info.pRasterizationState = &rasterizer;
    info.pMultisampleState = &multisampling;
    info.pColorBlendState = &blending;
    info.pDynamicState = &dynamic;
    info.layout = layout;
    info.renderPass = renderPass;
    info.subpass = 0;

    JST_VK_CHECK(Detail::CreateHandle(pipeline, [&](auto* output) {
        return vkCreateGraphicsPipelines(device, Backend::State<DeviceType::Vulkan>()->getPipelineCache(),
                                         1, &info, nullptr, output);
    }), [&]{
        JST_ERROR("[VULKAN] Failed to create presentation pipeline.");
    });

    return Result::SUCCESS;
}

}  // namespace

Result WindowImp<DeviceType::Vulkan>::Presenter::create(VkRenderPass renderPass, U64 framesInFlight) {
    auto& device = Backend::State<DeviceType::Vulkan>()->getDevice();

    const auto fail = [&]() {
        destroy();
        return Result::ERROR;
    };

    if (Detail::CreatePresentLayout(device, descriptorSetLayout) != Result::SUCCESS) {
        return fail();
    }

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
    if (Detail::CreateHandle(pipelineLayout, [&](auto* output) {
            return vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, output);
        }) != VK_SUCCESS) {
        JST_ERROR("[VULKAN] Failed to create presentation pipeline layout.");
        return fail();
    }

    const auto& shaders = GlobalShadersPackage["present"][DeviceType::Vulkan];
    const ShaderModule vertex(device, Backend::LoadShader(shaders[0], device));
    const ShaderModule fragment(device, Backend::LoadShader(shaders[1], device));

    Result result = (vertex.handle && fragment.handle) ? Result::SUCCESS : Result::ERROR;
    if (result == Result::SUCCESS) {
        result = CreatePipeline(device, renderPass, pipelineLayout, vertex.handle, fragment.handle,
                                PresentList::Blend::Straight, straight);
    }
    if (result == Result::SUCCESS) {
        result = CreatePipeline(device, renderPass, pipelineLayout, vertex.handle, fragment.handle,
                                PresentList::Blend::Premultiplied, premultiplied);
    }

    if (result != Result::SUCCESS) {
        JST_ERROR("[VULKAN] Failed to create presenter.");
        return fail();
    }

    frames.resize(framesInFlight);
    return Result::SUCCESS;
}

void WindowImp<DeviceType::Vulkan>::Presenter::destroy() {
    auto& device = Backend::State<DeviceType::Vulkan>()->getDevice();

    for (auto& frame : frames) {
        release(frame);
    }
    frames.clear();

    if (straight) {
        vkDestroyPipeline(device, straight, nullptr);
        straight = VK_NULL_HANDLE;
    }
    if (premultiplied) {
        vkDestroyPipeline(device, premultiplied, nullptr);
        premultiplied = VK_NULL_HANDLE;
    }
    if (pipelineLayout) {
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        pipelineLayout = VK_NULL_HANDLE;
    }
    if (descriptorSetLayout) {
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        descriptorSetLayout = VK_NULL_HANDLE;
    }
}

void WindowImp<DeviceType::Vulkan>::Presenter::release(Frame& frame) {
    auto& device = Backend::State<DeviceType::Vulkan>()->getDevice();

    if (frame.mapped) {
        vkUnmapMemory(device, frame.memory);
        frame.mapped = nullptr;
    }
    if (frame.buffer) {
        vkDestroyBuffer(device, frame.buffer, nullptr);
        frame.buffer = VK_NULL_HANDLE;
    }
    if (frame.memory) {
        vkFreeMemory(device, frame.memory, nullptr);
        frame.memory = VK_NULL_HANDLE;
    }
    frame.capacity = 0;
}

Result WindowImp<DeviceType::Vulkan>::Presenter::reserve(Frame& frame, U64 vertices) {
    if (frame.capacity >= vertices) {
        return Result::SUCCESS;
    }
    release(frame);

    auto& backend = Backend::State<DeviceType::Vulkan>();
    auto& device = backend->getDevice();
    auto& physicalDevice = backend->getPhysicalDevice();

    U64 capacity = kInitialVertices;
    while (capacity < vertices) {
        capacity *= 2;
    }

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = capacity * sizeof(PresentGeometry::Vertex);
    bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    JST_CHECK(Detail::AllocateBuffer(frame.buffer, frame.memory, [&](VkBuffer& candidate) {
        JST_VK_CHECK(vkCreateBuffer(device, &bufferInfo, nullptr, &candidate), [&]{
            JST_ERROR("[VULKAN] Failed to create presentation vertex buffer.");
        });
        return Result::SUCCESS;
    }, [&](VkBuffer created, VkDeviceMemory& candidate) {
        VkMemoryRequirements requirements;
        vkGetBufferMemoryRequirements(device, created, &requirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = requirements.size;
        allocInfo.memoryTypeIndex = Backend::FindMemoryType(physicalDevice,
                                                            requirements.memoryTypeBits,
                                                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        JST_VK_CHECK(vkAllocateMemory(device, &allocInfo, nullptr, &candidate), [&]{
            JST_ERROR("[VULKAN] Failed to allocate presentation vertex memory.");
        });
        return Result::SUCCESS;
    }, [&](VkBuffer created, VkDeviceMemory allocated) {
        JST_VK_CHECK(vkBindBufferMemory(device, created, allocated, 0), [&]{
            JST_ERROR("[VULKAN] Failed to bind presentation vertex memory.");
        });
        return Result::SUCCESS;
    }, [&](VkBuffer created, VkDeviceMemory allocated) {
        if (created != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, created, nullptr);
        }
        if (allocated != VK_NULL_HANDLE) {
            vkFreeMemory(device, allocated, nullptr);
        }
    }));

    void* mapped = nullptr;
    if (vkMapMemory(device, frame.memory, 0, bufferInfo.size, 0, &mapped) != VK_SUCCESS) {
        JST_ERROR("[VULKAN] Failed to map presentation vertex memory.");
        release(frame);
        return Result::ERROR;
    }

    frame.mapped = mapped;
    frame.capacity = capacity;
    return Result::SUCCESS;
}

Result WindowImp<DeviceType::Vulkan>::Presenter::encode(VkCommandBuffer commandBuffer,
                                                        U64 frameIndex,
                                                        const PresentList& list,
                                                        const WindowMetrics& metrics,
                                                        const Extent2D<U64>& target) {
    const auto size = metrics.presentationSize();
    if (list.quads.empty() || size.x == 0 || size.y == 0 || target.x == 0 || target.y == 0) {
        return Result::SUCCESS;
    }

    auto& frame = frames[frameIndex % frames.size()];
    const U64 vertexCount = list.quads.size() * 6;
    JST_CHECK(reserve(frame, vertexCount));

    auto* vertices = static_cast<PresentGeometry::Vertex*>(frame.mapped);
    for (U64 index = 0; index < list.quads.size(); ++index) {
        const auto triangles = PresentGeometry::Triangles(list.quads[index], metrics.displaySize, true);
        std::memcpy(vertices + index * 6, triangles.data(), sizeof(triangles));
    }

    const VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, &frame.buffer, &offset);

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<F32>(size.x);
    viewport.height = static_cast<F32>(size.y);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

    std::optional<PresentList::Blend> bound;
    for (U64 index = 0; index < list.quads.size(); ++index) {
        const auto& quad = list.quads[index];
        const ScissorRect clamped = PresentGeometry::Clamp(quad.scissor, target);
        if (clamped.width == 0 || clamped.height == 0) {
            continue;
        }
        if (bound != quad.blend) {
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              quad.blend == PresentList::Blend::Premultiplied ? premultiplied : straight);
            bound = quad.blend;
        }

        const auto descriptorSet = reinterpret_cast<VkDescriptorSet>(quad.handle);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1,
                                &descriptorSet, 0, nullptr);

        VkRect2D scissor{};
        scissor.offset = {static_cast<I32>(clamped.x), static_cast<I32>(clamped.y)};
        scissor.extent = {clamped.width, clamped.height};
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        vkCmdDraw(commandBuffer, 6, 1, static_cast<U32>(index * 6), 0);
    }

    return Result::SUCCESS;
}

}  // namespace Jetstream::Render
