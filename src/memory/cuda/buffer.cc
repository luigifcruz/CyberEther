#include "jetstream/memory/devices/cuda/buffer.hh"
#include "jetstream/backend/devices/cuda/helpers.hh"

#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
#include "jetstream/memory/devices/vulkan/buffer.hh"
#include "jetstream/backend/devices/vulkan/helpers.hh"
#endif

#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
#include "jetstream/memory/devices/cpu/buffer.hh"
#endif

#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
#include "jetstream/memory/devices/metal/buffer.hh"
#endif

namespace Jetstream {

using Implementation = TensorBuffer<Device::CUDA>;

Implementation::TensorBuffer(std::shared_ptr<TensorStorageMetadata>& storage,
                             const TensorPrototypeMetadata& prototype,
                             const bool& host_accessible) {
    JST_TRACE("[CUDA:BUFFER] Allocating new buffer.");

    // Check if CUDA is available.

    if (!Backend::State<Device::CUDA>()->isAvailable()) {
        JST_FATAL("[CUDA:BUFFER] CUDA is not available.");
        JST_CHECK_THROW(Result::ERROR);
    }

    // Initialize storage.

    storage->root_device = Device::CUDA;
    storage->compatible_devices = {
        Device::CUDA
    };

    // Check size.

    if (prototype.size_bytes == 0) {
        return;
    }

    // Allocate memory.

    const auto size_bytes = JST_PAGE_ALIGNED_SIZE(prototype.size_bytes);

    if (host_accessible || Backend::State<Device::CUDA>()->hasUnifiedMemory()) {
        JST_CUDA_CHECK_THROW(cudaMallocManaged(&buffer, size_bytes), [&]{
            JST_FATAL("[CUDA:BUFFER] Failed to allocate managed CUDA memory: {}", err);
        });

        _device_native = true;
        _host_native = true;
        _host_accessible = true;
    } else {
        JST_CUDA_CHECK_THROW(cudaMalloc(&buffer, size_bytes), [&]{
            JST_FATAL("[CUDA:BUFFER] Failed to allocate CUDA memory: {}", err);
        });

        _device_native = true;
        _host_native = false;
        _host_accessible = false;
    }
    owns_data = true;

    // Add compatible devices.

#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
    if (TensorBuffer<Device::CPU>::CanImport(*this)) {
        storage->compatible_devices.insert(Device::CPU);
    }
#endif

#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
    if (TensorBuffer<Device::Vulkan>::CanImport(*this)) {
        storage->compatible_devices.insert(Device::Vulkan);
    }
#endif

#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
    if (TensorBuffer<Device::Metal>::CanImport(*this)) {
        storage->compatible_devices.insert(Device::Metal);
    }
#endif
}

#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
Implementation::TensorBuffer(std::shared_ptr<TensorStorageMetadata>&,
                             const TensorPrototypeMetadata& prototype,
                             const std::shared_ptr<TensorBuffer<Device::Vulkan>>& root_buffer) {
    JST_TRACE("[CUDA:BUFFER] Cloning from Vulkan buffer.");

    // Check if CUDA is available.

    if (!Backend::State<Device::CUDA>()->isAvailable()) {
        JST_ERROR("[CUDA:BUFFER] CUDA is not available.");
        JST_CHECK_THROW(Result::ERROR);
    }

    // Check if root buffer can be imported.

    if (!TensorBuffer<Device::CUDA>::CanImport(*root_buffer)) {
        JST_ERROR("[CUDA:BUFFER] Vulkan buffer is not compatible with CUDA.");
        JST_CHECK_THROW(Result::ERROR);
    }

    // Check size.

    if (prototype.size_bytes == 0) {
        return;
    }

    // Get device types.

    auto& device = Backend::State<Device::Vulkan>()->getDevice();

    // Initialize buffer.

    VkMemoryGetFdInfoKHR fdInfo = {};
    fdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    fdInfo.memory = root_buffer->memory();
    fdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    auto vkGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR");
    if (!vkGetMemoryFdKHR) {
        JST_FATAL("[CUDA:BUFFER] Failed to get vkGetMemoryFdKHR function pointer.");
        JST_CHECK_THROW(Result::ERROR);
    }

    JST_VK_CHECK_THROW(vkGetMemoryFdKHR(device, &fdInfo, &vulkan_file_descriptor), [&]{
        JST_FATAL("[CUDA:BUFFER] Failed to get Vulkan buffer file descriptor.");
    });

    CUDA_EXTERNAL_MEMORY_HANDLE_DESC extMemHandleDesc = {};
    extMemHandleDesc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
    extMemHandleDesc.handle.fd = vulkan_file_descriptor;
    extMemHandleDesc.size = JST_PAGE_ALIGNED_SIZE(prototype.size_bytes);

    JST_CUDA_CHECK_THROW(cuImportExternalMemory(&vulkan_external_memory, &extMemHandleDesc), [&]{
        JST_FATAL("[CUDA:BUFFER] Failed to import Vulkan buffer memory into CUDA: {}", err);
    });

    CUdeviceptr devPtr;

    CUDA_EXTERNAL_MEMORY_BUFFER_DESC bufferDesc = {};
    bufferDesc.offset = 0;
    bufferDesc.size = JST_PAGE_ALIGNED_SIZE(prototype.size_bytes);

    JST_CUDA_CHECK_THROW(cuExternalMemoryGetMappedBuffer(&devPtr, vulkan_external_memory, &bufferDesc), [&]{
        JST_FATAL("[CUDA:BUFFER] Failed to get CUDA buffer from Vulkan buffer memory: {}", err);
    });

    // Initialize storage.

    _device_native = true;
    _host_native = false;
    _host_accessible = false;

    buffer = reinterpret_cast<void*>(devPtr);
    external_memory_device = Device::Vulkan;
    owns_data = false;
}

bool Implementation::CanImport(const TensorBuffer<Device::Vulkan>& root_buffer) noexcept {
    JST_TRACE("[CUDA:BUFFER] Checking if Vulkan buffer can be imported.");

    // Check if CUDA is available.

    if (!Backend::State<Device::CUDA>()->isAvailable()) {
        JST_TRACE("[CUDA:BUFFER] CUDA is not available.");
        return false;
    }

    // Check if Vulkan is available.

    if (!Backend::State<Device::Vulkan>()->isAvailable()) {
        JST_TRACE("[CUDA:BUFFER] Vulkan is not available.");
        return false;
    }

    // Check if Vulkan buffer can export memory.

    if (!Backend::State<Device::Vulkan>()->canExportDeviceMemory()) {
        JST_TRACE("[CUDA:BUFFER] Vulkan buffer cannot export memory.");
        return false;
    }

    // Check if CUDA can import memory.

    if (!Backend::State<Device::CUDA>()->canImportDeviceMemory()) {
        JST_TRACE("[CUDA:BUFFER] CUDA cannot import memory.");
        return false;
    }

    // Check if memory is device native. 

    if (!root_buffer.device_native()) {
        JST_TRACE("[CUDA:BUFFER] Vulkan buffer is not device native.");
        return false;
    }

    return true;
}
#endif

#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
Implementation::TensorBuffer(std::shared_ptr<TensorStorageMetadata>&,
                             const TensorPrototypeMetadata& prototype,
                             const std::shared_ptr<TensorBuffer<Device::CPU>>& root_buffer) {
    JST_TRACE("[CUDA:BUFFER] Cloning from CPU buffer.");

    // Check if CUDA is available.

    if (!Backend::State<Device::CUDA>()->isAvailable()) {
        JST_FATAL("[CUDA:BUFFER] CUDA is not available.");
        JST_CHECK_THROW(Result::ERROR);
    }

    // Check if root buffer can be imported.

    if (!TensorBuffer<Device::CUDA>::CanImport(*root_buffer)) {
        JST_ERROR("[CUDA:BUFFER] CPU buffer is not compatible with CUDA.");
        JST_CHECK_THROW(Result::ERROR);
    }

    // Check size.

    if (prototype.size_bytes == 0) {
        return;
    }

    // Pin memory if needed.

    cudaPointerAttributes attributes;
    JST_CUDA_CHECK_THROW(cudaPointerGetAttributes(&attributes, root_buffer->data()), [&]{
        JST_FATAL("[CUDA:BUFFER] Failed to get CPU buffer attributes: {}", err);
    });

    if (attributes.type == cudaMemoryTypeUnregistered && JST_IS_ALIGNED(root_buffer->data())) {
        const auto size_bytes = JST_PAGE_ALIGNED_SIZE(prototype.size_bytes);
        JST_CUDA_CHECK_THROW(cudaHostRegister(root_buffer->data(), size_bytes, cudaHostRegisterPortable), [&]{
            JST_FATAL("[CUDA:BUFFER] Failed to pin CPU buffer: {}", err);
        });
    }

    // Initialize storage.

    _device_native = Backend::State<Device::CUDA>()->hasUnifiedMemory();
    _host_native = true;
    _host_accessible = true;

    buffer = root_buffer->data();
    external_memory_device = Device::CPU;
    owns_data = false;
}

bool Implementation::CanImport(const TensorBuffer<Device::CPU>& root_buffer) noexcept {
    JST_TRACE("[CUDA:BUFFER] Checking if CPU buffer can be imported.");

    // Check if CUDA is available.

    if (!Backend::State<Device::CUDA>()->isAvailable()) {
        JST_TRACE("[CUDA:BUFFER] CUDA is not available.");
        return false;
    }

    // Check if CUDA can import host memory.

    if (!Backend::State<Device::CUDA>()->canImportHostMemory()) {
        JST_TRACE("[CUDA:BUFFER] CUDA cannot import host memory.");
        return false;
    }

    // Check with CUDA if the CPU buffer is pinned.

    cudaPointerAttributes attributes;
    [&]{
        JST_CUDA_CHECK(cudaPointerGetAttributes(&attributes, root_buffer.data()), [&]{
            JST_FATAL("[CUDA:BUFFER] Failed to get CPU buffer attributes: {}", err);
            return false;
        });
        return Result::SUCCESS;
    }();

    if (attributes.type == cudaMemoryTypeUnregistered && !JST_IS_ALIGNED(root_buffer.data())) {
        JST_TRACE("[CUDA:BUFFER] CPU buffer is not aligned.");
        return false;
    }

    return true;
}
#endif

#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
Implementation::TensorBuffer(std::shared_ptr<TensorStorageMetadata>&,
                             const TensorPrototypeMetadata& prototype,
                             const std::shared_ptr<TensorBuffer<Device::Metal>>& root_buffer) {
    throw std::runtime_error("Metal buffers are not supported on CUDA.");
}

bool Implementation::CanImport(const TensorBuffer<Device::Metal>& root_buffer) noexcept {
    return false;
}
#endif

Implementation::~TensorBuffer() {
    JST_TRACE("[CUDA:BUFFER] Releasing buffer {}.", jst::fmt::ptr(buffer));

    // Close Vulkan file descriptor.

#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
    if (external_memory_device == Device::Vulkan) {
        cuDestroyExternalMemory(vulkan_external_memory);
        close(vulkan_file_descriptor);
    }
#endif

    // Unregister CPU buffer from CUDA.

#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
    if (external_memory_device == Device::CPU) {
        [&]{
            JST_CUDA_CHECK(cudaHostUnregister(buffer), [&]{
                JST_WARN("[CPU:BUFFER] Failed to unregister buffer from CUDA: {}", err);
            });
            return Result::SUCCESS;
        }();
    }
#endif

    // Free memory.

    if (owns_data) {
        cudaFree(buffer);
    }
}

}  // namespace Jetstream