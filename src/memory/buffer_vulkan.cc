#include "buffer_backend.hh"

#include <cstring>

#include "jetstream/backend/base.hh"
#include "jetstream/backend/devices/vulkan/base.hh"
#include "jetstream/backend/devices/vulkan/helpers.hh"
#include "jetstream/logger.hh"
#include "jetstream/memory/macros.hh"
#include "jetstream/memory/devices/vulkan/buffer.hh"

namespace Jetstream::detail {

namespace {

class VulkanBackend final : public VulkanBufferBackend, public Backend {
 public:
    VulkanBackend() = default;
    ~VulkanBackend() override {
        destroy();
    }

    DeviceType device() const override {
        return DeviceType::Vulkan;
    }

    Result create(const U64& bytes, const Buffer::Config& config) override {
        destroy();

        sizeBytes = bytes;
        borrowed = false;
        ownsMemory = true;

        if (bytes == 0) {
            return Result::SUCCESS;
        }

        // Check if Vulkan is available.

        const auto& state = Jetstream::Backend::State<DeviceType::Vulkan>();
        if (!state->isAvailable()) {
            JST_ERROR("[MEMORY:BUFFER:VULKAN] Vulkan is not available.");
            return Result::ERROR;
        }

        // Get device types.

        auto& device = state->getDevice();
        auto& physicalDevice = state->getPhysicalDevice();
        auto& queue = state->getComputeQueue();
        auto& fence = state->getDefaultFence();
        auto& commandBuffer = state->getDefaultCommandBuffer();
        const auto& unified = state->hasUnifiedMemory();
        const auto& canExport = state->canExportDeviceMemory();

        // Create buffer object.

        VkExternalMemoryBufferCreateInfo extBufferCreateInfo = {};
        extBufferCreateInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
        extBufferCreateInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

        VkBufferCreateInfo bufferInfo = {};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = JST_PAGE_ALIGNED_SIZE(bytes);
        bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                           VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        bufferInfo.pNext = canExport ? &extBufferCreateInfo : nullptr;

        JST_VK_CHECK(vkCreateBuffer(device, &bufferInfo, nullptr, &_buffer), [&]{
            JST_ERROR("[MEMORY:BUFFER:VULKAN] Failed to create buffer.");
        });

        // Allocate backing memory.

        VkMemoryRequirements memoryRequirements;
        vkGetBufferMemoryRequirements(device, _buffer, &memoryRequirements);

        VkMemoryPropertyFlags memoryProperties = 0;

        if (unified) {
            memoryProperties |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            memoryProperties |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
            memoryProperties |= VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            locationState = Location::Unified;
            hostAccessible = true;
            JST_TRACE("[MEMORY:BUFFER:VULKAN] Using unified memory (DL, HV, HC).");
        } else if (config.hostAccessible) {
            memoryProperties |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
            memoryProperties |= VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            locationState = Location::Host;
            hostAccessible = true;
            JST_TRACE("[MEMORY:BUFFER:VULKAN] Using host accessible memory (HV, HC).");
        } else {
            memoryProperties |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            locationState = Location::Device;
            hostAccessible = false;
            JST_TRACE("[MEMORY:BUFFER:VULKAN] Using device local memory (DL).");
        }

        VkExportMemoryAllocateInfo exportInfo = {};
        exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
        exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

        VkMemoryAllocateInfo memoryAllocateInfo = {};
        memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        memoryAllocateInfo.allocationSize = memoryRequirements.size;
        memoryAllocateInfo.memoryTypeIndex = Jetstream::Backend::FindMemoryType(physicalDevice,
                                                                     memoryRequirements.memoryTypeBits,
                                                                     memoryProperties);
        memoryAllocateInfo.pNext = canExport ? &exportInfo : nullptr;

        JST_VK_CHECK(vkAllocateMemory(device, &memoryAllocateInfo, nullptr, &_memory), [&]{
            JST_ERROR("[MEMORY:BUFFER:VULKAN] Failed to allocate buffer memory.");
        });

        JST_VK_CHECK(vkBindBufferMemory(device, _buffer, _memory, 0), [&]{
            JST_ERROR("[MEMORY:BUFFER:VULKAN] Failed to bind memory to the buffer.");
        });

        // Map memory for CPU access if host accessible.

        if (hostAccessible) {
            JST_VK_CHECK(vkMapMemory(device, _memory, 0, JST_PAGE_ALIGNED_SIZE(bytes), 0, &mappedPtr), [&]{
                JST_ERROR("[MEMORY:BUFFER:VULKAN] Failed to map buffer memory.");
            });
        }

        // Null out array using command buffer.

        JST_CHECK(Jetstream::Backend::ExecuteOnce(device, queue, fence, commandBuffer, [&](VkCommandBuffer& cmd){
            vkCmdFillBuffer(cmd, _buffer, 0, VK_WHOLE_SIZE, 0);
            return Result::SUCCESS;
        }));

        return Result::SUCCESS;
    }

    Result create(void* pointer, const U64& bytes) override {
        (void)pointer;
        (void)bytes;
        JST_ERROR("[MEMORY:BUFFER:VULKAN] Borrowed host-pointer create is only supported on CPU buffers.");
        return Result::ERROR;
    }

    Result create(const Backend& source) override {
        // Vulkan can mirror CPU buffers by creating a new Vulkan buffer
        // that wraps the same memory (if host-accessible).

        if (source.device() == DeviceType::CPU) {
            JST_TRACE("[MEMORY:BUFFER:VULKAN] Mirroring CPU buffer.");

            // For CPU -> Vulkan, we need to create a new Vulkan buffer
            // that uses the CPU memory. This is not directly supported,
            // so we return an error for now.
            JST_ERROR("[MEMORY:BUFFER:VULKAN] Cannot mirror from CPU device directly.");
            return Result::ERROR;
        }

        JST_ERROR("[MEMORY:BUFFER:VULKAN] Cannot mirror from device {}.", source.device());
        return Result::ERROR;
    }

    Result copyFrom(const Backend& source) override {
        JST_TRACE("[MEMORY:BUFFER:VULKAN] Copying buffer.");

        if (!mappedPtr) {
            JST_ERROR("[MEMORY:BUFFER:VULKAN] Buffer is not host accessible for copy.");
            return Result::ERROR;
        }

        std::memcpy(mappedPtr, source.rawHandle(), source.size());
        return Result::SUCCESS;
    }

    void* rawHandle() override {
        JST_ASSERT_THROW(hostAccessible, "[MEMORY:BUFFER:VULKAN] Buffer is not host accessible.");
        return mappedPtr;
    }

    const void* rawHandle() const override {
        JST_ASSERT_THROW(hostAccessible, "[MEMORY:BUFFER:VULKAN] Buffer is not host accessible.");
        return mappedPtr;
    }

    bool isBorrowed() const override {
        return borrowed;
    }

    Location location() const override {
        return locationState;
    }

    U64 size() const override {
        return sizeBytes;
    }

    bool isHostAccessible() const {
        return hostAccessible;
    }

    VkBuffer buffer() const override {
        return _buffer;
    }

    VkDeviceMemory memory() const override {
        return _memory;
    }

    void destroy() override {
        if (_memory != VK_NULL_HANDLE) {
            auto& device = Jetstream::Backend::State<DeviceType::Vulkan>()->getDevice();

            if (mappedPtr && ownsMemory) {
                vkUnmapMemory(device, _memory);
                mappedPtr = nullptr;
            }

            if (ownsMemory) {
                vkFreeMemory(device, _memory, nullptr);
            }
            _memory = VK_NULL_HANDLE;
        }

        if (_buffer != VK_NULL_HANDLE) {
            auto& device = Jetstream::Backend::State<DeviceType::Vulkan>()->getDevice();

            if (ownsMemory) {
                vkDestroyBuffer(device, _buffer, nullptr);
            }
            _buffer = VK_NULL_HANDLE;
        }

        sizeBytes = 0;
        ownsMemory = true;
        borrowed = false;
        hostAccessible = false;
        locationState = Location::None;
    }

 private:
    VkBuffer _buffer = VK_NULL_HANDLE;
    VkDeviceMemory _memory = VK_NULL_HANDLE;
    void* mappedPtr = nullptr;
    U64 sizeBytes = 0;
    bool ownsMemory = true;
    bool borrowed = false;
    bool hostAccessible = false;
    Location locationState = Location::None;
};

}  // namespace

std::unique_ptr<Backend> CreateVulkanBackend() {
    return std::make_unique<VulkanBackend>();
}

}  // namespace Jetstream::detail
