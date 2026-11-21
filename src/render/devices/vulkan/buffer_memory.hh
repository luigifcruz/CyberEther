#ifndef JETSTREAM_RENDER_VULKAN_BUFFER_MEMORY_HH
#define JETSTREAM_RENDER_VULKAN_BUFFER_MEMORY_HH

#include <vulkan/vulkan.h>

#include "jetstream/types.hh"

#include "creation.hh"

#include <algorithm>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace Jetstream::Render::Detail {

class BufferMemoryRanges {
 public:
    explicit BufferMemoryRanges(VkDeviceSize size) : free{{0, size}} {}

    std::optional<VkDeviceSize> take(VkDeviceSize size, VkDeviceSize alignment) {
        if (!size || !alignment) {
            return std::nullopt;
        }
        for (U64 i = 0; i < free.size(); ++i) {
            const auto range = free[i];
            const VkDeviceSize remainder = range.offset % alignment;
            const VkDeviceSize padding = remainder ? alignment - remainder : 0;
            if (padding > range.size || size > range.size - padding) {
                continue;
            }
            const U64 needed = free.size() + live + 2;
            if (free.capacity() < needed) {
                free.reserve(std::max<U64>(needed, free.capacity() * 2));
            }
            const VkDeviceSize offset = range.offset + padding;
            const VkDeviceSize tail = range.size - padding - size;
            if (padding) {
                free[i].size = padding;
                if (tail) {
                    free.insert(free.begin() + i + 1, {offset + size, tail});
                }
            } else if (tail) {
                free[i] = {offset + size, tail};
            } else {
                free.erase(free.begin() + i);
            }
            ++live;
            return offset;
        }
        return std::nullopt;
    }

    void release(VkDeviceSize offset, VkDeviceSize size) noexcept {
        auto it = std::lower_bound(free.begin(), free.end(), offset,
                                  [](const auto& range, VkDeviceSize value) { return range.offset < value; });
        it = free.insert(it, {offset, size});
        if (it != free.begin() && (it - 1)->offset + (it - 1)->size == it->offset) {
            (it - 1)->size += it->size;
            it = free.erase(it) - 1;
        }
        if (it + 1 != free.end() && it->offset + it->size == (it + 1)->offset) {
            it->size += (it + 1)->size;
            free.erase(it + 1);
        }
        --live;
    }

 private:
    struct Range {
        VkDeviceSize offset = 0;
        VkDeviceSize size = 0;
    };

    std::vector<Range> free;
    U64 live = 0;
};

struct BufferMemoryBlock {
    VkDevice device = VK_NULL_HANDLE;
    U32 type = 0;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    BufferMemoryRanges ranges;
    std::mutex mutex;

    BufferMemoryBlock(VkDevice device, U32 type, VkDeviceSize size)
        : device(device), type(type), ranges(size) {}

    ~BufferMemoryBlock() {
        if (memory != VK_NULL_HANDLE) {
            vkFreeMemory(device, memory, nullptr);
        }
    }
};

struct BufferMemory {
    BufferMemory() = default;
    BufferMemory(const BufferMemory&) = delete;
    BufferMemory& operator=(const BufferMemory&) = delete;

    std::shared_ptr<BufferMemoryBlock> block;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    VkDeviceSize size = 0;

    ~BufferMemory() {
        if (size) {
            std::lock_guard lock(block->mutex);
            block->ranges.release(offset, size);
        }
    }
};

inline VkResult AllocateBufferMemory(VkDevice device, VkBuffer buffer, U32 type,
                                     const VkMemoryRequirements& requirements, bool dedicated,
                                     std::shared_ptr<BufferMemory>& output) {
    constexpr VkDeviceSize PageSize = 8 * 1024 * 1024;
    struct Registry {
        std::mutex mutex;
        std::vector<std::weak_ptr<BufferMemoryBlock>> blocks;
    };
    static Registry registry;
    std::lock_guard registryLock(registry.mutex);
    auto allocation = std::make_shared<BufferMemory>();
    const auto take = [&](const std::shared_ptr<BufferMemoryBlock>& block) {
        std::lock_guard lock(block->mutex);
        const auto offset = block->ranges.take(requirements.size, requirements.alignment);
        if (!offset) {
            return false;
        }
        allocation->block = block;
        allocation->memory = block->memory;
        allocation->offset = *offset;
        allocation->size = requirements.size;
        return true;
    };
    const bool pooled = !dedicated && requirements.size <= PageSize / 2 && requirements.alignment <= PageSize;
    for (auto it = registry.blocks.begin(); it != registry.blocks.end();) {
        auto block = it->lock();
        if (!block) {
            it = registry.blocks.erase(it);
            continue;
        }
        ++it;
        if (pooled && block->device == device && block->type == type && take(block)) {
            output = std::move(allocation);
            return VK_SUCCESS;
        }
    }

    const auto create = [&](VkDeviceSize size, bool dedicatedPage) {
        auto block = std::make_shared<BufferMemoryBlock>(device, type, size);
        VkMemoryDedicatedAllocateInfo dedicatedInfo{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
        dedicatedInfo.buffer = buffer;
        VkMemoryAllocateInfo info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        info.allocationSize = size;
        info.memoryTypeIndex = type;
        if (dedicatedPage) {
            info.pNext = &dedicatedInfo;
        }
        const auto result = CreateHandle(block->memory, [&](auto* memory) {
            return vkAllocateMemory(device, &info, nullptr, memory);
        });
        if (result != VK_SUCCESS) {
            return result;
        }
        if (!take(block)) {
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        }
        if (!dedicatedPage) {
            registry.blocks.push_back(block);
        }
        return VK_SUCCESS;
    };
    auto result = create(pooled ? PageSize : requirements.size, !pooled);
    if (pooled && (result == VK_ERROR_OUT_OF_DEVICE_MEMORY || result == VK_ERROR_OUT_OF_HOST_MEMORY)) {
        result = create(requirements.size, true);
    }
    if (result == VK_SUCCESS) {
        output = std::move(allocation);
    }
    return result;
}

}  // namespace Jetstream::Render::Detail

#endif
