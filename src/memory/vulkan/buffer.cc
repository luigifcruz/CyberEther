#include "jetstream/memory/devices/vulkan/buffer.hh"
#include "jetstream/backend/devices/vulkan/helpers.hh"

#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
#include "jetstream/memory/devices/cpu/buffer.hh"
#endif

#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
#include "jetstream/memory/devices/metal/buffer.hh"
#endif

#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
#include "jetstream/memory/devices/cuda/buffer.hh"
#endif

namespace Jetstream {

using Implementation = TensorBuffer<Device::Vulkan>;

Implementation::TensorBuffer(std::shared_ptr<TensorStorageMetadata>& storage,
                             const TensorPrototypeMetadata& prototype,
                             const bool& host_accessible,
                             const VkBufferUsageFlags& usage) {
    JST_TRACE("[VULKAN:BUFFER] Allocating new buffer.");

    // Initialize storage.

    storage->root_device = Device::Vulkan;
    storage->compatible_devices = {
        Device::Vulkan
    };

    // Get device types.

    auto& device = Backend::State<Device::Vulkan>()->getDevice();
    auto& physicalDevice = Backend::State<Device::Vulkan>()->getPhysicalDevice();
    const auto& unified = Backend::State<Device::Vulkan>()->hasUnifiedMemory();
    const auto& canExport = Backend::State<Device::Vulkan>()->canExportDeviceMemory();

    // Check size.

    if (prototype.size_bytes == 0) {
        return;
    }

    // Create buffer object. 

    VkExternalMemoryImageCreateInfo extImageCreateInfo = {};
    extImageCreateInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    extImageCreateInfo.handleTypes |= VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = JST_PAGE_ALIGNED_SIZE(prototype.size_bytes);
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    bufferInfo.pNext = (canExport) ? &extImageCreateInfo : nullptr;

    JST_VK_CHECK_THROW(vkCreateBuffer(device, &bufferInfo, nullptr, &_buffer), [&]{
        JST_ERROR("[VULKAN] Can't create memory buffer.");
    });

    // Allocate backing memory.

    VkMemoryRequirements memoryRequirements;
    vkGetBufferMemoryRequirements(device, _buffer, &memoryRequirements);

    VkMemoryPropertyFlags memoryProperties = 0;

    if (unified) {
        memoryProperties |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        memoryProperties |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        memoryProperties |= VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

        _host_accessible = true;
        _device_native = true;
        _host_native = true;

        JST_TRACE("[VULKAN:BUFFER] Using unified memory (DL, HV, HC).");
    } else {
        if (host_accessible) {
            memoryProperties |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
            memoryProperties |= VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

            _host_accessible = true;
            _device_native = false;
            _host_native = true;

            JST_TRACE("[VULKAN:BUFFER] Using host accessible memory (HV, HC).");
        } else {
            memoryProperties |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

            _host_accessible = false;
            _device_native = true;
            _host_native = false;

            JST_TRACE("[VULKAN:BUFFER] Using device local memory (DL).");
        }
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

    // Add compatible devices.

#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
    if (TensorBuffer<Device::CPU>::CanImport(*this)) {
        storage->compatible_devices.insert(Device::CPU);
    }
#endif

#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
    if (TensorBuffer<Device::Metal>::CanImport(*this)) {
        storage->compatible_devices.insert(Device::Metal);
    }
#endif

#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
    if (TensorBuffer<Device::CUDA>::CanImport(*this)) {
        storage->compatible_devices.insert(Device::CUDA);
    }
#endif
}

#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
Implementation::TensorBuffer(std::shared_ptr<TensorStorageMetadata>& storage,
                             const TensorPrototypeMetadata& prototype,
                             const std::shared_ptr<TensorBuffer<Device::CPU>>& root_buffer) {
    throw std::runtime_error("Exporting CPU memory to Vulkan not implemented.");
    // TODO: Add CPU -> Vulkan.
}

bool Implementation::CanImport(const TensorBuffer<Device::CPU>& root_buffer) noexcept {
    return false;
}
#endif

#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
Implementation::TensorBuffer(std::shared_ptr<TensorStorageMetadata>& storage,
                             const TensorPrototypeMetadata& prototype,
                             const std::shared_ptr<TensorBuffer<Device::Metal>>& root_buffer) {
    throw std::runtime_error("Exporting Metal memory to Vulkan not implemented.");
    // TODO: Add Metal -> Vulkan.
}

bool Implementation::CanImport(const TensorBuffer<Device::Metal>& root_buffer) noexcept {
    return false;
}
#endif

#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
Implementation::TensorBuffer(std::shared_ptr<TensorStorageMetadata>& storage,
                             const TensorPrototypeMetadata& prototype,
                             const std::shared_ptr<TensorBuffer<Device::CUDA>>& root_buffer) {
    throw std::runtime_error("Exporting CUDA memory to Vulkan not implemented.");
    // TODO: Add CUDA -> Vulkan.
}

bool Implementation::CanImport(const TensorBuffer<Device::CUDA>& root_buffer) noexcept {
    return false;
}
#endif

Implementation::~TensorBuffer() {
    JST_TRACE("[VULKAN:BUFFER] Releasing buffer {}.", jst::fmt::ptr(_buffer));

    if (owns_data) {
        auto& device = Backend::State<Device::Vulkan>()->getDevice();

        vkFreeMemory(device, _memory, nullptr);
        vkDestroyBuffer(device, _buffer, nullptr);
    }
}

}  // namespace Jetstream