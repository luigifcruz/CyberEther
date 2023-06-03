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
    imageCreateInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    JST_VK_CHECK(vkCreateImage(device, &imageCreateInfo, nullptr, &texture), [&]{
        JST_FATAL("[VULKAN] Failed to create texture.");   
    });

    VkMemoryRequirements memoryRequirements;
    vkGetImageMemoryRequirements(device, texture, &memoryRequirements);

    VkMemoryAllocateInfo memoryAllocateInfo = {};
    memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memoryAllocateInfo.allocationSize = memoryRequirements.size;
    memoryAllocateInfo.memoryTypeIndex = Backend::FindMemoryType(physicalDevice,
                                                                 memoryRequirements.memoryTypeBits,
                                                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    JST_VK_CHECK(vkAllocateMemory(device, &memoryAllocateInfo, nullptr, &memory), [&]{
        JST_FATAL("[VULKAN] Failed to allocate texture memory.");
    });

    JST_VK_CHECK(vkBindImageMemory(device, texture, memory, 0), [&]{
        JST_FATAL("[VULKAN] Failed to bind memory to the texture.");
    });

    if (config.buffer) {
        JST_CHECK(fill());
    }

    return Result::SUCCESS;
}

Result Implementation::destroy() {
    JST_DEBUG("[VULKAN] Destroying texture.");

    auto& device = Backend::State<Device::Vulkan>()->getDevice();
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

    // TODO: Implement this.
    JST_WARN("[VULKAN] Texture fill not implemented.");

    // auto region = MTL::Region::Make2D(0, y, config.size.width, height);
    // auto rowByteSize = config.size.width * GetPixelByteSize(texture->pixelFormat());
    // auto bufferByteOffset = rowByteSize * y;
    // texture->replaceRegion(region, 0, config.buffer + bufferByteOffset, rowByteSize);

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

    JST_FATAL("Can't convert pixel format.");

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
            JST_CHECK_THROW(Result::ERROR);
            return 0;
    }
}

}  // namespace Jetstream::Render
