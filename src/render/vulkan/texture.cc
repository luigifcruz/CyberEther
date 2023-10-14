#include "jetstream/render/vulkan/texture.hh"
#include "jetstream/backend/devices/vulkan/helpers.hh"

namespace Jetstream::Render {

using Implementation = TextureImp<Device::Vulkan>;

Implementation::TextureImp(const Config& config) : Texture(config) {
    pixelFormat = ConvertPixelFormat(config.pfmt, config.ptype); 
}

Result Implementation::create() {
    JST_DEBUG("[VULKAN] Creating texture.");

    auto& device = Backend::State<Device::Vulkan>()->getDevice();
    auto& physicalDevice = Backend::State<Device::Vulkan>()->getPhysicalDevice();

    // Create image.

    VkImageCreateInfo imageCreateInfo = {};
    imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    imageCreateInfo.extent.width = config.size.width;
    imageCreateInfo.extent.height = config.size.height;
    imageCreateInfo.extent.depth = 1;
    imageCreateInfo.mipLevels = 1;
    imageCreateInfo.arrayLayers = 1;
    imageCreateInfo.format = pixelFormat;
    imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    // TODO: Review these.
    imageCreateInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                            VK_IMAGE_USAGE_TRANSFER_DST_BIT | 
                            VK_IMAGE_USAGE_SAMPLED_BIT;
    imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

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

    auto& backend = Backend::State<Device::Vulkan>();

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

    auto& device = Backend::State<Device::Vulkan>()->getDevice();
    auto& descriptorPool = Backend::State<Device::Vulkan>()->getDescriptorPool();

    vkFreeDescriptorSets(device, descriptorPool, 1, &descriptorSet);
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    vkDestroySampler(device, sampler, nullptr);
    vkDestroyImageView(device, imageView, nullptr);
    vkDestroyImage(device, texture, nullptr);
    vkFreeMemory(device, memory, nullptr);

    return Result::SUCCESS;
}

bool Implementation::size(const Size2D<U64>& size) {
    if (size <= Size2D<U64>{1, 1}) {
        return false;
    }

    if (config.size != size) {
        config.size = size;
        return true;
    }

    return false;
}

Result Implementation::fill() {
    return fillRow(0, config.size.height);
}

Result Implementation::fillRow(const U64& y, const U64& height) {
    if (height < 1) {
        return Result::SUCCESS;
    }

    // TODO: Implement zero-copy option.

    auto& backend = Backend::State<Device::Vulkan>();

    uint8_t* mappedData = static_cast<uint8_t*>(backend->getStagingBufferMappedMemory());
    const uint8_t* hostData = static_cast<const uint8_t*>(config.buffer);
    const auto rowByteSize = config.size.width * GetPixelByteSize(pixelFormat);
    const auto bufferByteOffset = rowByteSize * y;
    const auto bufferByteSize = rowByteSize * height;

    if (bufferByteSize >= backend->getStagingBufferSize()) {
        JST_ERROR("[VULKAN] Memory copy is larger than the staging buffer.");
        return Result::ERROR;
    }

    memcpy(mappedData, hostData + bufferByteOffset, bufferByteSize);

    // TODO: Maybe worth investigating if creating a command buffer every loop is a good idea.
    JST_CHECK(Backend::ExecuteOnce(backend->getDevice(),
                                   backend->getComputeQueue(),
                                   backend->getTransferCommandPool(),
        [&](VkCommandBuffer& commandBuffer){
            JST_CHECK(Backend::TransitionImageLayout(commandBuffer, 
                                                     texture, 
                                                     VK_IMAGE_LAYOUT_UNDEFINED,
                                                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL));
            VkBufferImageCopy region{};
            region.bufferOffset = 0;
            region.bufferRowLength = 0;
            region.bufferImageHeight = 0;
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.mipLevel = 0;
            region.imageSubresource.baseArrayLayer = 0;
            region.imageSubresource.layerCount = 1;
            region.imageOffset = {
                    0,
                    static_cast<I32>(y),
                    0
                };
            region.imageExtent = {
                    static_cast<U32>(config.size.width),
                    static_cast<U32>(height),
                    1
                }; 

            vkCmdCopyBufferToImage(
                commandBuffer,
                backend->getStagingBuffer(),
                texture,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1,
                &region
            );

            JST_CHECK(Backend::TransitionImageLayout(commandBuffer, 
                                                     texture, 
                                                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));

            return Result::SUCCESS;
        }
    ));

    return Result::SUCCESS;
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

U64 Implementation::GetPixelByteSize(const VkFormat& pfmt) {
    switch (pfmt) {
        case VK_FORMAT_R32_SFLOAT:
            return 4;
        case VK_FORMAT_R8_UNORM:
            return 1;
        case VK_FORMAT_R32G32B32A32_SFLOAT:
            return 16;
        case VK_FORMAT_R8G8B8A8_UNORM:
            return 4;
        default:
            JST_FATAL("[VULKAN] Pixel format not implemented yet.");
            JST_CHECK_THROW(Result::FATAL);
            return 0;
    }
}

}  // namespace Jetstream::Render
