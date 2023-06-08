#include "jetstream/render/vulkan/buffer.hh"
#include "jetstream/backend/devices/vulkan/helpers.hh"
#include <vulkan/vulkan_core.h>

namespace Jetstream::Render {

using Implementation = BufferImp<Device::Vulkan>;

Implementation::BufferImp(const Config& config) : Buffer(config) {
}

Result Implementation::create() {
    JST_DEBUG("[VULKAN] Creating buffer.");

    auto& device = Backend::State<Device::Vulkan>()->getDevice();
    auto& physicalDevice = Backend::State<Device::Vulkan>()->getPhysicalDevice();

    // Convert usage flags.
    // TODO: Implement implicit specification.

    VkBufferUsageFlags bufferUsageFlag = 0;
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

    // Create buffer.

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = byteSize();
    bufferInfo.usage = bufferUsageFlag;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    JST_VK_CHECK(vkCreateBuffer(device, &bufferInfo, nullptr, &buffer), [&]{
        JST_FATAL("[VULKAN] Can't create memory buffer.");
    });

    // Allocate backing memory.

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

    // Populate memory with initial data.
    
    if (config.buffer) {
        JST_CHECK(update());
    }

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
    // TODO: Implement zero-copy option.

    void* mappedData;
    auto& backend = Backend::State<Device::Vulkan>();
    const auto& byteOffset = offset * config.elementByteSize;
    const auto& byteSize = size * config.elementByteSize;

    if (byteSize >= backend->getStagingBufferSize()) {
        JST_ERROR("[VULKAN] Memory copy is larger than the staging buffer.");
        return Result::ERROR;
    }

    auto& stagingBufferMemory = backend->getStagingBufferMemory();
    JST_VK_CHECK(vkMapMemory(backend->getDevice(), stagingBufferMemory, 0, byteSize, 0, &mappedData), [&]{
        JST_FATAL("[VULKAN] Can't map staging buffer memory.");        
    });
    memcpy(mappedData, (uint8_t*)config.buffer + byteOffset, byteSize);
    vkUnmapMemory(backend->getDevice(), stagingBufferMemory);

    // TODO: Maybe worth investigating if creating a command buffer every loop is a good idea.
    JST_CHECK(Backend::ExecuteOnce(backend->getDevice(),
                                   backend->getComputeQueue(),
                                   backend->getTransferCommandPool(),
        [&](VkCommandBuffer& commandBuffer){
            VkBufferCopy copyRegion{};
            copyRegion.srcOffset = 0;
            copyRegion.dstOffset = byteOffset;
            copyRegion.size = byteSize;
            vkCmdCopyBuffer(commandBuffer, backend->getStagingBuffer(), buffer, 1, &copyRegion);
            return Result::SUCCESS;
        }
    ));

    return Result::SUCCESS;
}

}  // namespace Jetstream::Render
