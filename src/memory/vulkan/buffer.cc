#include "jetstream/memory/devices/vulkan/buffer.hh"
#include "jetstream/backend/devices/vulkan/helpers.hh"

namespace Jetstream {

using Implementation = TensorBuffer<Device::Vulkan>;

Implementation::TensorBuffer(std::shared_ptr<TensorStorageMetadata>& storage,
                             const std::shared_ptr<TensorPrototypeMetadata>& prototype) {
    JST_TRACE("[VULKAN:BUFFER] Allocating new buffer.");

    // Initialize storage.

    storage->root_device = Device::Vulkan;
    storage->compatible_devices = {
        Device::Vulkan
    };

    // Check size.

    if (prototype->size_bytes == 0) {
        return;
    }

    // Get device types.

    auto& device = Backend::State<Device::Vulkan>()->getDevice();
    auto& physicalDevice = Backend::State<Device::Vulkan>()->getPhysicalDevice();
    const auto& unified = Backend::State<Device::Vulkan>()->hasUnifiedMemory();
    const auto& canExport = Backend::State<Device::Vulkan>()->canExportMemory();

    // Add CPU and CUDA support if unified memory is enabled.

    if (unified) {
        storage->compatible_devices.insert(Device::CPU);
    }

#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
    if (canExport) {
        storage->compatible_devices.insert(Device::CUDA);
    }
#endif

    // Sort buffer usage flags.

    VkBufferUsageFlags bufferUsageFlag = 0;
    bufferUsageFlag |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufferUsageFlag |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    // Create buffer object. 

    VkExternalMemoryImageCreateInfo extImageCreateInfo = {};
    extImageCreateInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    extImageCreateInfo.handleTypes |= VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = JST_PAGE_ALIGNED_SIZE(prototype->size_bytes);
    bufferInfo.usage = bufferUsageFlag;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    bufferInfo.pNext = (canExport) ? &extImageCreateInfo : nullptr;

    JST_VK_CHECK_THROW(vkCreateBuffer(device, &bufferInfo, nullptr, &_buffer), [&]{
        JST_ERROR("[VULKAN] Can't create memory buffer.");
    });

    // Allocate backing memory.

    VkMemoryRequirements memoryRequirements;
    vkGetBufferMemoryRequirements(device, _buffer, &memoryRequirements);

    VkMemoryPropertyFlags memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    if (unified) {
        memoryProperties |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        memoryProperties |= VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    }

    VkExportMemoryAllocateInfo exportInfo = {};
    exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkMemoryAllocateInfo memoryAllocateInfo = {};
    memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memoryAllocateInfo.allocationSize = memoryRequirements.size;
    memoryAllocateInfo.memoryTypeIndex = Backend::FindMemoryType(physicalDevice,
                                                                 memoryRequirements.memoryTypeBits,
                                                                 memoryProperties);
    memoryAllocateInfo.pNext = (canExport) ? &exportInfo : nullptr;

    JST_VK_CHECK_THROW(vkAllocateMemory(device, &memoryAllocateInfo, nullptr, &_memory), [&]{
        JST_ERROR("[VULKAN:BUFFER] Failed to allocate buffer memory.");
    });

    JST_VK_CHECK_THROW(vkBindBufferMemory(device, _buffer, _memory, 0), [&]{
        JST_ERROR("[VULKAN:BUFFER] Failed to bind memory to the buffer.");
    });
    owns_data = true;
}

Implementation::~TensorBuffer() {
    JST_TRACE("[VULKAN:BUFFER] Releasing buffer {}.", fmt::ptr(_buffer));

    if (owns_data) {
        auto& device = Backend::State<Device::Vulkan>()->getDevice();

        vkFreeMemory(device, _memory, nullptr);
        vkDestroyBuffer(device, _buffer, nullptr);
    }
}

}  // namespace Jetstream