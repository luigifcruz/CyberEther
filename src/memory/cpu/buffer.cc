#include "jetstream/memory/devices/cpu/buffer.hh"

#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
#include "jetstream/memory/devices/metal/buffer.hh"
#endif

#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
#include "jetstream/backend/devices/vulkan/helpers.hh"
#endif

#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
#include "jetstream/memory/devices/cuda/buffer.hh"
#endif

namespace Jetstream {

using Implementation = TensorBuffer<Device::CPU>;

Implementation::TensorBuffer(std::shared_ptr<TensorStorageMetadata>& storage,
                             const std::shared_ptr<TensorPrototypeMetadata>& prototype) {
    JST_TRACE("[CPU:BUFFER] Allocating new buffer.");

    // Initialize storage.

    storage->root_device = Device::CPU;
    storage->compatible_devices = {
        Device::CPU,
    };

    // Check size.

    if (prototype->size_bytes == 0) {
        return;
    }

    // Check alignment.

#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
    if (Backend::State<Device::Metal>()->hasUnifiedMemory()) {
        JST_TRACE("[CPU:BUFFER] Platform is unified. Enabling Metal compatibility.");
        storage->compatible_devices.insert(Device::Metal);
    }
#endif

    // Allocate memory.

    void* memoryAddr = nullptr;
    const auto pageSize = JST_PAGESIZE();
    const auto alignedSizeBytes = JST_PAGE_ALIGNED_SIZE(prototype->size_bytes);
    const auto result = posix_memalign(&memoryAddr, pageSize, alignedSizeBytes);
    if (result < 0 || (buffer = static_cast<void*>(memoryAddr)) == nullptr) {
        JST_ERROR("[CPU:BUFFER] Failed to allocate CPU memory.");
        JST_CHECK_THROW(Result::ERROR);
    }
    owns_data = true;

    // Null out array.
    memset(buffer, 0, prototype->size_bytes);
}

Implementation::TensorBuffer(std::shared_ptr<TensorStorageMetadata>& storage,
                             const std::shared_ptr<TensorPrototypeMetadata>&,
                             void* ptr) {
    JST_TRACE("[CPU:BUFFER] New buffer from raw pointer.");

    // Initialize storage.

    storage->root_device = Device::CPU;
    storage->compatible_devices = {
        Device::CPU,
    };

    // Check alignment and platform.

#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
    if (JST_IS_ALIGNED(ptr) && ptr != nullptr && Backend::State<Device::Metal>()->hasUnifiedMemory()) {
        JST_TRACE("[CPU:BUFFER] Buffer is aligned and platform is unified. Enabling Metal compatibility.");
        storage->compatible_devices.insert(Device::Metal);
    }
#endif

    // Initialize buffer.

    owns_data = false;
    buffer = ptr;
}

#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
Implementation::TensorBuffer(std::shared_ptr<TensorStorageMetadata>& storage,
                             const std::shared_ptr<TensorPrototypeMetadata>& prototype,
                             const std::shared_ptr<TensorBuffer<Device::Metal>>& root_buffer) {
    JST_TRACE("[CPU:BUFFER] Cloning from Metal buffer.");

    // Check platform.

    if (!Backend::State<Device::Metal>()->hasUnifiedMemory()) {
        JST_ERROR("[CPU:BUFFER] Metal buffer is not unified. Cannot share data between CPU and GPU.");
        JST_CHECK_THROW(Result::ERROR);
    }

    // Check size.

    if (prototype->size_bytes == 0) {
        return;
    }

    // Initialize buffer.

    buffer = root_buffer->data()->contents();
    owns_data = false;
    external_memory_device = Device::Metal;
}
#endif

#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
Implementation::TensorBuffer(std::shared_ptr<TensorStorageMetadata>&,
                             const std::shared_ptr<TensorPrototypeMetadata>& prototype,
                             const std::shared_ptr<TensorBuffer<Device::Vulkan>>& root_buffer) {
    JST_TRACE("[CPU:BUFFER] Cloning from Vulkan buffer.");

    // Check platform.

    if (!root_buffer->host_accessible()) {
        JST_ERROR("[CPU:BUFFER] Vulkan buffer is not host accessible. Cannot share data with CPU.");
        JST_CHECK_THROW(Result::ERROR);
    }

    // Check size.

    if (prototype->size_bytes == 0) {
        return;
    }

    // Save variables for later.

    vulkan_memory = root_buffer->memory();

    // Initialize buffer.

    auto& device = Backend::State<Device::Vulkan>()->getDevice();
    const auto size = JST_PAGE_ALIGNED_SIZE(prototype->size_bytes);

    JST_VK_CHECK_THROW(vkMapMemory(device, vulkan_memory, 0, size, 0, &buffer), [&]{
        JST_FATAL("[CPU:BUFFER] Failed to map buffer memory.");
    });

    owns_data = false;
    external_memory_device = Device::Vulkan;
}
#endif

#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
Implementation::TensorBuffer(std::shared_ptr<TensorStorageMetadata>&,
                             const std::shared_ptr<TensorPrototypeMetadata>& prototype,
                             const std::shared_ptr<TensorBuffer<Device::CUDA>>& root_buffer) {
    JST_TRACE("[CPU:BUFFER] Cloning from CUDA buffer.");

    // Check platform.

    if (!root_buffer->host_accessible()) {
        JST_ERROR("[CPU:BUFFER] CUDA buffer is not host accessible. It cannot share data with the CPU.");
        JST_CHECK_THROW(Result::ERROR);
    }

    // Check size.

    if (prototype->size_bytes == 0) {
        return;
    }

    // Initialize buffer.

    buffer = root_buffer->data();
    owns_data = false;
    external_memory_device = Device::CUDA;
}
#endif

Implementation::~TensorBuffer() {
    JST_TRACE("[CPU:BUFFER] Trying to free buffer at {}.", fmt::ptr(buffer));

    if (owns_data) {
        free(buffer);
    }

#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
    if (external_memory_device == Device::Vulkan) {
        auto& device = Backend::State<Device::Vulkan>()->getDevice();
        vkUnmapMemory(device, vulkan_memory);
    }
#endif

    // TODO: Unmap memory if imported from Vulkan.
}

}  // namespace Jetstream