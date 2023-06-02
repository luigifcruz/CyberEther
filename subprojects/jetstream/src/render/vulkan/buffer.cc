#include "jetstream/render/vulkan/buffer.hh"
#include "jetstream/backend/devices/vulkan/helpers.hh"
#include <vulkan/vulkan_core.h>

namespace Jetstream::Render {

using Implementation = BufferImp<Device::Vulkan>;

Implementation::BufferImp(const Config& config) : Buffer(config) {
}

Result Implementation::create() {
    JST_DEBUG("[VULKAN] Creating buffer.");

    VkBufferUsageFlags bufferUsageFlag = 0;

    // TODO: Implement implicit specification.
    bufferUsageFlag |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufferUsageFlag |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    switch (config.target) {
        case Target::VERTEX:
            bufferUsageFlag |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
            break;
        case Target::VERTEX_INDICES:
            bufferUsageFlag |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
            break;
        case Target::STORAGE:
            bufferUsageFlag |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; 
            break;
    }

    auto& device = Backend::State<Device::Vulkan>()->getDevice();
    auto& physicalDevice = Backend::State<Device::Vulkan>()->getPhysicalDevice();
    const auto& byteSize = config.size * config.elementByteSize;

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = byteSize;
    bufferInfo.usage = bufferUsageFlag;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    JST_VK_CHECK(vkCreateBuffer(device, &bufferInfo, nullptr, &buffer), [&]{
        JST_FATAL("[VULKAN] Can't create memory buffer.");
    });

    VkMemoryRequirements memoryRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memoryRequirements);

    VkMemoryAllocateInfo memoryAllocateInfo = {};
    memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memoryAllocateInfo.allocationSize = memoryRequirements.size;
    memoryAllocateInfo.memoryTypeIndex = Backend::FindMemoryType(physicalDevice,
                                                                 memoryRequirements.memoryTypeBits,
                                                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    JST_VK_CHECK(vkAllocateMemory(device, &memoryAllocateInfo, nullptr, &memory), [&]{
        JST_FATAL("[VULKAN] Failed to allocate buffer memory.");
    });

    JST_VK_CHECK(vkBindBufferMemory(device, buffer, memory, 0), [&]{
        JST_FATAL("[VULKAN] Failed to bind memory to the buffer.");
    });
    
    // TODO: Implement zero-copy option.

    return Result::SUCCESS;
}

Result Implementation::destroy() {
    JST_DEBUG("[VULKAN] Destroying buffer.");

    auto& device = Backend::State<Device::Vulkan>()->getDevice();
    vkDestroyBuffer(device, buffer, nullptr);
    vkFreeMemory(device, memory, nullptr);

    return Result::SUCCESS;
}

Result Implementation::update() {
    return update(0, config.size);
}

Result Implementation::update(const U64& offset, const U64& size) {
//     const auto& byteOffset = offset * config.elementByteSize;
//     const auto& byteSize = size * config.elementByteSize;

//     if (!config.enableZeroCopy) {
//         uint8_t* ptr = static_cast<uint8_t*>(buffer->contents());
//         memcpy(ptr + byteOffset, (uint8_t*)config.buffer + byteOffset, byteSize);
// #if !defined(TARGET_OS_IOS)
//         buffer->didModifyRange(NS::Range(byteOffset, byteOffset + byteSize));
// #endif
//     }

    // TODO: Implement this.
    JST_WARN("[VULKAN] Buffer update not implemented.");

    return Result::SUCCESS;
}

}  // namespace Jetstream::Render
