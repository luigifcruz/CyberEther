#include "jetstream/render/devices/vulkan/texture.hh"
#include "jetstream/backend/devices/vulkan/helpers.hh"

#include <cstring>
#include <limits>

namespace Jetstream::Render {

using Implementation = TextureImp<DeviceType::Vulkan>;

Implementation::TextureImp(const Config& config) : Texture(config) {
    pixelFormat = ConvertPixelFormat(config.pfmt, config.ptype); 
}

Result Implementation::create() {
    JST_DEBUG("[VULKAN] Creating texture.");

    layout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (config.buffer) {
        JST_CHECK(validateFillRow(0, config.size.y));
    }

    auto& device = Backend::State<DeviceType::Vulkan>()->getDevice();
    auto& physicalDevice = Backend::State<DeviceType::Vulkan>()->getPhysicalDevice();

    // Create extent.

    extent = VkExtent2D {
        static_cast<U32>(config.size.x),
        static_cast<U32>(config.size.y),
    };

    // Create image.

    VkImageCreateInfo imageCreateInfo = {};
    imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    imageCreateInfo.extent.width = extent.width;
    imageCreateInfo.extent.height = extent.height;
    imageCreateInfo.extent.depth = 1;
    imageCreateInfo.mipLevels = 1;
    imageCreateInfo.arrayLayers = 1;
    imageCreateInfo.format = pixelFormat;
    imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    // TODO: Review these.
    imageCreateInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                            VK_IMAGE_USAGE_TRANSFER_DST_BIT | 
                            VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                            VK_IMAGE_USAGE_SAMPLED_BIT;
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (config.multisampled) {
        imageCreateInfo.samples = Backend::State<DeviceType::Vulkan>()->getMultisampling();
    } else {
        imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    }

    JST_VK_CHECK(vkCreateImage(device, &imageCreateInfo, nullptr, &texture), [&]{
        JST_ERROR("[VULKAN] Failed to create texture.");   
    });

    // Allocate backing memory.

    VkMemoryRequirements memoryRequirements;
    vkGetImageMemoryRequirements(device, texture, &memoryRequirements);

    VkMemoryAllocateInfo memoryAllocateInfo = {};
    memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memoryAllocateInfo.allocationSize = memoryRequirements.size;
    memoryAllocateInfo.memoryTypeIndex = Backend::FindMemoryType(physicalDevice,
                                                                 memoryRequirements.memoryTypeBits,
                                                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    JST_VK_CHECK(vkAllocateMemory(device, &memoryAllocateInfo, nullptr, &memory), [&]{
        JST_ERROR("[VULKAN] Failed to allocate texture memory.");
    });

    JST_VK_CHECK(vkBindImageMemory(device, texture, memory, 0), [&]{
        JST_ERROR("[VULKAN] Failed to bind memory to the texture.");
    });

    // Create image view.

    VkImageViewCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    createInfo.image = texture;
    createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    createInfo.format = pixelFormat;

    createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

    createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    createInfo.subresourceRange.baseMipLevel = 0;
    createInfo.subresourceRange.levelCount = 1;
    createInfo.subresourceRange.baseArrayLayer = 0;
    createInfo.subresourceRange.layerCount = 1;

    JST_VK_CHECK(vkCreateImageView(device, &createInfo, nullptr, &imageView), [&]{
        JST_ERROR("[VULKAN] Failed to create image view."); 
    });

    // Create sampler.

    VkFormatProperties formatProperties;
    vkGetPhysicalDeviceFormatProperties(physicalDevice, pixelFormat, &formatProperties);

    VkFilter filter = VK_FILTER_LINEAR;
    VkSamplerMipmapMode mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

    if (!(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) {
        JST_WARN("[VULKAN] The image format does not support linear filtering. Falling back to nearest filtering.");

        filter = VK_FILTER_NEAREST;
        mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    }

    VkSamplerCreateInfo samplerCreateInfo = {};
    samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerCreateInfo.magFilter = filter;
    samplerCreateInfo.minFilter = filter;
    samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCreateInfo.anisotropyEnable = VK_FALSE;
    samplerCreateInfo.maxAnisotropy = 1.0f;
    samplerCreateInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerCreateInfo.unnormalizedCoordinates = VK_FALSE;
    samplerCreateInfo.compareEnable = VK_FALSE;
    samplerCreateInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerCreateInfo.mipmapMode = mipmapMode;
    samplerCreateInfo.mipLodBias = 0.0f;
    samplerCreateInfo.minLod = 0.0f;
    samplerCreateInfo.maxLod = 1.0f;

    JST_VK_CHECK(vkCreateSampler(device, &samplerCreateInfo, nullptr, &sampler), [&]{
        JST_ERROR("[VULKAN] Can't create texture sampler.");
    });

    // Register descriptor for ImGui attachment.

    auto& backend = Backend::State<DeviceType::Vulkan>();

    VkDescriptorSetLayoutBinding binding{};
    binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    binding.binding = 0;

    VkDescriptorSetLayoutCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    info.bindingCount = 1;
    info.pBindings = &binding;

    JST_VK_CHECK(vkCreateDescriptorSetLayout(device, &info, nullptr, &descriptorSetLayout), [&]{
        JST_ERROR("[VULKAN] Can't create descriptor set layout.");
    });

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = backend->getDescriptorPool();
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptorSetLayout;

    JST_VK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet), [&]{
        JST_ERROR("[VULKAN] Failed to allocate descriptor set.");
    });

    VkDescriptorImageInfo descImage{};
    descImage.sampler = sampler;
    descImage.imageView = imageView;
    descImage.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet writeDesc{};
    writeDesc.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDesc.dstSet = descriptorSet;
    writeDesc.descriptorCount = 1;
    writeDesc.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writeDesc.pImageInfo = &descImage;

    vkUpdateDescriptorSets(device, 1, &writeDesc, 0, nullptr);

    // Fill image with initial data.

    if (config.buffer) {
        JST_CHECK(fill());
    }

    return Result::SUCCESS;
}

Result Implementation::destroy() {
    JST_DEBUG("[VULKAN] Destroying texture.");

    auto& device = Backend::State<DeviceType::Vulkan>()->getDevice();
    auto& descriptorPool = Backend::State<DeviceType::Vulkan>()->getDescriptorPool();

    if (descriptorSet) {
        vkFreeDescriptorSets(device, descriptorPool, 1, &descriptorSet);
        descriptorSet = VK_NULL_HANDLE;
    }

    if (descriptorSetLayout) {
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        descriptorSetLayout = VK_NULL_HANDLE;
    }

    if (sampler) {
        vkDestroySampler(device, sampler, nullptr);
        sampler = VK_NULL_HANDLE;
    }

    if (imageView) {
        vkDestroyImageView(device, imageView, nullptr);
        imageView = VK_NULL_HANDLE;
    }

    if (texture) {
        vkDestroyImage(device, texture, nullptr);
        texture = VK_NULL_HANDLE;
    }

    if (memory) {
        vkFreeMemory(device, memory, nullptr);
        memory = VK_NULL_HANDLE;
    }

    layout = VK_IMAGE_LAYOUT_UNDEFINED;
    return Result::SUCCESS;
}

Result Implementation::underlyingDump(uint8_t* output) const {
    if (texture == VK_NULL_HANDLE) {
        JST_ERROR("[VULKAN] Can't dump uninitialized texture.");
        return Result::ERROR;
    }

    if (layout == VK_IMAGE_LAYOUT_UNDEFINED) {
        JST_ERROR("[VULKAN] Can't dump a texture with undefined contents.");
        return Result::ERROR;
    }

    if (config.size.x > std::numeric_limits<U32>::max() ||
        config.size.y > std::numeric_limits<U32>::max()) {
        JST_ERROR("[VULKAN] Invalid texture dimensions for dump.");
        return Result::ERROR;
    }

    auto& backend = Backend::State<DeviceType::Vulkan>();
    auto& device = backend->getDevice();
    const U64 bufferByteSize = config.size.x * config.size.y * pixelByteSize();

    Backend::HostVisibleBuffer readback;
    JST_CHECK(Backend::CreateHostVisibleBuffer(device, backend->getPhysicalDevice(), bufferByteSize,
                                               VK_BUFFER_USAGE_TRANSFER_DST_BIT, readback));

    const Result result = Backend::SubmitOnce(device, backend->getPhysicalDevice(), backend->getGraphicsQueue(),
                                              [&](VkCommandBuffer& cmd) {
        constexpr VkPipelineStageFlags sourceStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        constexpr VkAccessFlags sourceAccess = VK_ACCESS_MEMORY_READ_BIT |
                                               VK_ACCESS_MEMORY_WRITE_BIT;
        VkImageMemoryBarrier before{};
        before.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        before.srcAccessMask = sourceAccess;
        before.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        before.oldLayout = layout;
        before.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        before.image = texture;
        before.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(cmd, sourceStage, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &before);

        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {
            static_cast<U32>(config.size.x),
            static_cast<U32>(config.size.y),
            1,
        };
        vkCmdCopyImageToBuffer(cmd, texture,
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               readback.buffer, 1, &region);

        VkBufferMemoryBarrier hostRead{};
        hostRead.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        hostRead.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        hostRead.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        hostRead.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        hostRead.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        hostRead.buffer = readback.buffer;
        hostRead.offset = 0;
        hostRead.size = VK_WHOLE_SIZE;

        VkImageMemoryBarrier after = before;
        after.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        after.dstAccessMask = sourceAccess;
        after.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        after.newLayout = layout;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             sourceStage | VK_PIPELINE_STAGE_HOST_BIT,
                             0, 0, nullptr, 1, &hostRead, 1, &after);
        return Result::SUCCESS;
    });
    if (result == Result::SUCCESS) {
        std::memcpy(output, readback.mapped, bufferByteSize);
    } else {
        JST_ERROR("[VULKAN] Failed to read back texture.");
    }
    Backend::DestroyHostVisibleBuffer(device, readback);

    return result;
}

VkFormat Implementation::ConvertPixelFormat(const PixelFormat& pfmt,
                                            const PixelType& ptype) {
    if (pfmt == PixelFormat::RED && ptype == PixelType::F32) {
        return VK_FORMAT_R32_SFLOAT;
    }

    if (pfmt == PixelFormat::RED && ptype == PixelType::UI8) {
        return VK_FORMAT_R8_UNORM;
    }

    if (pfmt == PixelFormat::RGBA && ptype == PixelType::F32) {
        return VK_FORMAT_R32G32B32A32_SFLOAT;
    }

    if (pfmt == PixelFormat::RGBA && ptype == PixelType::UI8) {
        return VK_FORMAT_R8G8B8A8_UNORM;
    }

    JST_ERROR("[VULKAN] Can't convert pixel format.");
    return VK_FORMAT_UNDEFINED;
}

}  // namespace Jetstream::Render
