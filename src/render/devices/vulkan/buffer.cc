#include "jetstream/render/devices/vulkan/buffer.hh"
#include "jetstream/backend/devices/vulkan/helpers.hh"
#include "buffer_allocation.hh"

namespace Jetstream::Render {

using Implementation = BufferImp<DeviceType::Vulkan>;

Implementation::BufferImp(const Config& config) : Buffer(config) {
}

Result Implementation::create() {
    JST_DEBUG("[VULKAN] Creating buffer.");

    if (buffer != VK_NULL_HANDLE || memory != VK_NULL_HANDLE) {
        return Result::ERROR;
    }

    if (config.buffer && !config.enableZeroCopy) {
        JST_CHECK(validateUpdate(0, config.size));
    }

    auto& device = Backend::State<DeviceType::Vulkan>()->getDevice();
    auto& physicalDevice = Backend::State<DeviceType::Vulkan>()->getPhysicalDevice();

    if (config.enableZeroCopy) {
        buffer = reinterpret_cast<VkBuffer>(config.buffer);
    } else {
        // Convert usage flags.

        VkBufferUsageFlags bufferUsageFlag = 0;
        bufferUsageFlag |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bufferUsageFlag |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        if ((config.target & Target::VERTEX) == Target::VERTEX) {
            bufferUsageFlag |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        }

        if ((config.target & Target::VERTEX_INDICES) == Target::VERTEX_INDICES) {
            bufferUsageFlag |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
        }

        if ((config.target & Target::STORAGE) == Target::STORAGE ||
            (config.target & Target::STORAGE_DYNAMIC) == Target::STORAGE_DYNAMIC) {
            bufferUsageFlag |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        }

        if ((config.target & Target::UNIFORM) == Target::UNIFORM ||
            (config.target & Target::UNIFORM_DYNAMIC) == Target::UNIFORM_DYNAMIC) {
            bufferUsageFlag |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        }

        if ((config.target & Target::INDIRECT) == Target::INDIRECT) {
            bufferUsageFlag |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
        }

        if ((config.target & Target::STORAGE_DYNAMIC) == Target::STORAGE_DYNAMIC ||
            (config.target & Target::UNIFORM_DYNAMIC) == Target::UNIFORM_DYNAMIC) {
            JST_ERROR("[VULKAN] Buffer usage type not supported.")
            return Result::ERROR;
        }

        // Create buffer.

        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = byteSize();
        bufferInfo.usage = bufferUsageFlag;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        const Result allocation = Detail::AllocateBuffer(buffer, memory, [&](VkBuffer& candidate) {
            JST_VK_CHECK(vkCreateBuffer(device, &bufferInfo, nullptr, &candidate), [&]{
                JST_ERROR("[VULKAN] Can't create memory buffer.");
            });
            return Result::SUCCESS;
        }, [&](VkBuffer created, VkDeviceMemory& candidate) {

            // Allocate backing memory.

            VkMemoryRequirements memoryRequirements;
            vkGetBufferMemoryRequirements(device, created, &memoryRequirements);

            VkMemoryAllocateInfo memoryAllocateInfo = {};
            memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            memoryAllocateInfo.allocationSize = memoryRequirements.size;
            memoryAllocateInfo.memoryTypeIndex = Backend::FindMemoryType(physicalDevice,
                                                                         memoryRequirements.memoryTypeBits,
                                                                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

            JST_VK_CHECK(vkAllocateMemory(device, &memoryAllocateInfo, nullptr, &candidate), [&]{
                JST_ERROR("[VULKAN] Failed to allocate buffer memory.");
            });
            return Result::SUCCESS;
        }, [&](VkBuffer created, VkDeviceMemory allocated) {
            JST_VK_CHECK(vkBindBufferMemory(device, created, allocated, 0), [&]{
                JST_ERROR("[VULKAN] Failed to bind memory to the buffer.");
            });
            return Result::SUCCESS;
        }, [&](VkBuffer created, VkDeviceMemory allocated) {
            if (created != VK_NULL_HANDLE) {
                vkDestroyBuffer(device, created, nullptr);
            }
            if (allocated != VK_NULL_HANDLE) {
                vkFreeMemory(device, allocated, nullptr);
            }
        });
        JST_CHECK(allocation);

        // Populate memory with initial data.

        if (config.buffer) {
            Result result;
            try {
                result = update();
            } catch (...) {
                destroy();
                throw;
            }
            if (result != Result::SUCCESS && result != Result::RELOAD) {
                destroy();
                return result;
            }
        }
    }

    return Result::SUCCESS;
}

Result Implementation::destroy() {
    JST_DEBUG("[VULKAN] Destroying buffer.");

    if (!config.enableZeroCopy && (buffer != VK_NULL_HANDLE || memory != VK_NULL_HANDLE)) {
        auto& device = Backend::State<DeviceType::Vulkan>()->getDevice();
        if (buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, buffer, nullptr);
        }
        if (memory != VK_NULL_HANDLE) {
            vkFreeMemory(device, memory, nullptr);
        }
    }
    buffer = VK_NULL_HANDLE;
    memory = VK_NULL_HANDLE;

    return Result::SUCCESS;
}

}  // namespace Jetstream::Render
