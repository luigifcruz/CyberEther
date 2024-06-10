#include "jetstream/memory/devices/cpu/buffer.hh"

#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
#include "jetstream/memory/devices/vulkan/buffer.hh"
#include "jetstream/backend/devices/vulkan/helpers.hh"
#endif

#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
#include "jetstream/backend/devices/cuda/helpers.hh"
#include "jetstream/memory/devices/cuda/buffer.hh"
#endif

#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
#include "jetstream/memory/devices/metal/buffer.hh"
#endif

#ifdef JST_OS_WINDOWS
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#undef ERROR
#undef FATAL
#endif

namespace Jetstream {

using Implementation = TensorBuffer<Device::CPU>;

Implementation::TensorBuffer(std::shared_ptr<TensorStorageMetadata>& storage,
                             const TensorPrototypeMetadata& prototype) {
    JST_TRACE("[CPU:BUFFER] Allocating new buffer.");

    // Initialize storage.

    storage->root_device = Device::CPU;
    storage->compatible_devices = {
        Device::CPU,
    };

    // Allocate memory.

    if (prototype.size_bytes > 0) {
        void* memoryAddr = nullptr;
        const auto pageSize = JST_PAGESIZE();
        const auto alignedSizeBytes = JST_PAGE_ALIGNED_SIZE(prototype.size_bytes);
#ifdef JST_OS_WINDOWS
        buffer = VirtualAlloc(nullptr, alignedSizeBytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        if (buffer == nullptr) {
            JST_ERROR("[CPU:BUFFER] Failed to allocate CPU memory.");
            JST_CHECK_THROW(Result::ERROR);
        }
#else
        const auto result = posix_memalign(&memoryAddr, pageSize, alignedSizeBytes);
        if (result < 0 || (buffer = static_cast<void*>(memoryAddr)) == nullptr) {
            JST_ERROR("[CPU:BUFFER] Failed to allocate CPU memory.");
            JST_CHECK_THROW(Result::ERROR);
        }
#endif
        // Set buffer flags.

        set_allocated();
        set_host_accessible();

        // Null out array.

        memset(buffer, 0, prototype.size_bytes);
    }

    // Add compatible devices.

#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
    if (TensorBuffer<Device::Metal>::CanImport(*this)) {
        storage->compatible_devices.insert(Device::Metal);
    }
#endif

#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
    if (TensorBuffer<Device::Vulkan>::CanImport(*this)) {
        storage->compatible_devices.insert(Device::Vulkan);
    }
#endif

#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
    if (TensorBuffer<Device::CUDA>::CanImport(*this)) {
        storage->compatible_devices.insert(Device::CUDA);
    }
#endif
}

Implementation::TensorBuffer(std::shared_ptr<TensorStorageMetadata>& storage,
                             const TensorPrototypeMetadata&,
                             void* ptr) {
    JST_TRACE("[CPU:BUFFER] New buffer from raw pointer.");

    // Initialize storage.

    storage->root_device = Device::CPU;
    storage->compatible_devices = {
        Device::CPU,
    };

    // Add compatible devices.

#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
    if (TensorBuffer<Device::Metal>::CanImport(*this)) {
        storage->compatible_devices.insert(Device::Metal);
    }
#endif

    // Initialize buffer.

    buffer = ptr;
}

#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
Implementation::TensorBuffer(std::shared_ptr<TensorStorageMetadata>&,
                             const TensorPrototypeMetadata& prototype,
                             const std::shared_ptr<TensorBuffer<Device::Metal>>& root_buffer) {
    JST_TRACE("[CPU:BUFFER] Cloning from Metal buffer.");

    // Check if root buffer can be imported.

    if (!TensorBuffer<Device::CPU>::CanImport(*root_buffer)) {
        JST_ERROR("[CPU:BUFFER] Metal buffer is not compatible with CPU.");
        JST_CHECK_THROW(Result::ERROR);
    }

    // Check size.

    if (prototype.size_bytes == 0) {
        return;
    }

    // Initialize buffer.

    buffer = root_buffer->data()->contents();

    // Set buffer flags.

    set_host_accessible();
    set_external_memory_device(Device::Metal);
}

bool Implementation::CanImport(const TensorBuffer<Device::Metal>& root_buffer) noexcept {
    JST_TRACE("[CPU:BUFFER] Checking if Metal buffer can be imported.");

    // Allow importing empty buffers.

    if (!root_buffer.allocated()) {
        return true;
    }

    // Check if Metal buffer is host accessible.

    if (!Backend::State<Device::Metal>()->hasUnifiedMemory()) {
        JST_TRACE("[CPU:BUFFER] Metal buffer is not unified.");
        return false;
    }

    return true;
}
#endif

#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
Implementation::TensorBuffer(std::shared_ptr<TensorStorageMetadata>&,
                             const TensorPrototypeMetadata& prototype,
                             const std::shared_ptr<TensorBuffer<Device::Vulkan>>& root_buffer) {
    JST_TRACE("[CPU:BUFFER] Cloning from Vulkan buffer.");

    // Check if root buffer can be imported.

    if (!TensorBuffer<Device::CPU>::CanImport(*root_buffer)) {
        JST_ERROR("[CPU:BUFFER] Vulkan buffer is not compatible with CPU.");
        JST_CHECK_THROW(Result::ERROR);
    }

    // Check size.

    if (prototype.size_bytes == 0) {
        return;
    }

    // Save variables for later.

    vulkan_memory = root_buffer->memory();

    // Initialize buffer.

    auto& device = Backend::State<Device::Vulkan>()->getDevice();
    const auto size = JST_PAGE_ALIGNED_SIZE(prototype.size_bytes);

    JST_VK_CHECK_THROW(vkMapMemory(device, vulkan_memory, 0, size, 0, &buffer), [&]{
        JST_FATAL("[CPU:BUFFER] Failed to map buffer memory.");
    });

    // Set buffer flags.

    set_host_accessible();
    set_external_memory_device(Device::Vulkan);
}

bool Implementation::CanImport(const TensorBuffer<Device::Vulkan>& root_buffer) noexcept {
    JST_TRACE("[CPU:BUFFER] Checking if Vulkan buffer can be imported.");

    // Allow importing empty buffers.

    if (!root_buffer.allocated()) {
        return true;
    }

    // Check if Vulkan buffer is host accessible.

    if (!root_buffer.host_accessible()) {
        JST_TRACE("[CPU:BUFFER] Vulkan buffer is not host accessible.");
        return false;
    }

    return true;
}
#endif

#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
Implementation::TensorBuffer(std::shared_ptr<TensorStorageMetadata>&,
                             const TensorPrototypeMetadata& prototype,
                             const std::shared_ptr<TensorBuffer<Device::CUDA>>& root_buffer) {
    JST_TRACE("[CPU:BUFFER] Cloning from CUDA buffer.");

    // Check if root buffer can be imported.

    if (!TensorBuffer<Device::CPU>::CanImport(*root_buffer)) {
        JST_ERROR("[CPU:BUFFER] CUDA buffer is not compatible with CPU.");
        JST_CHECK_THROW(Result::ERROR);
    }

    // Check size.

    if (prototype.size_bytes == 0) {
        return;
    }

    // Initialize buffer.

    buffer = root_buffer->data();

    // Set buffer flags.

    set_host_accessible();
    set_external_memory_device(Device::CUDA);
}

bool Implementation::CanImport(const TensorBuffer<Device::CUDA>& root_buffer) noexcept {
    JST_TRACE("[CPU:BUFFER] Checking if CUDA buffer can be imported.");

    // Allow importing empty buffers.

    if (!root_buffer.allocated()) {
        return true;
    }

    // Check if CUDA buffer is host accessible.

    if (!root_buffer.host_accessible()) {
        JST_TRACE("[CPU:BUFFER] CUDA buffer is not host accessible.");
        return false;
    }

    return true;
}
#endif

Implementation::~TensorBuffer() {
    JST_TRACE("[CPU:BUFFER] Trying to free buffer at {}.", jst::fmt::ptr(buffer));

    // Unmap memory if imported from Vulkan.

#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
    if (external_memory_device() == Device::Vulkan) {
        auto& device = Backend::State<Device::Vulkan>()->getDevice();
        vkUnmapMemory(device, vulkan_memory);
    }
#endif

    // Free memory.

    if (allocated()) {
#ifdef JST_OS_WINDOWS
        VirtualFree(buffer, 0, MEM_RELEASE);
#else
        free(buffer);
#endif
    }
}

}  // namespace Jetstream