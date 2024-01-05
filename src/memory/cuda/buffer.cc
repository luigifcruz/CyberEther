#include "jetstream/memory/devices/cuda/buffer.hh"
#include "jetstream/backend/devices/cuda/helpers.hh"

#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
#include "jetstream/backend/devices/vulkan/helpers.hh"
#endif

namespace Jetstream {

using Implementation = TensorBuffer<Device::CUDA>;

Implementation::TensorBuffer(std::shared_ptr<TensorStorageMetadata>& storage,
                             const TensorPrototypeMetadata& prototype,
                             const bool& host_accessible) {
    JST_TRACE("[CUDA:BUFFER] Allocating new buffer.");

    // Initialize storage.

    storage->root_device = Device::CUDA;
    storage->compatible_devices = {
        Device::CUDA
    };

    // Get device types.

    const auto& unified = Backend::State<Device::CUDA>()->hasUnifiedMemory();

    // Add CPU compatibility.

    if (host_accessible || unified) {
        storage->compatible_devices.insert(Device::CPU);
    }

    // Check size.

    if (prototype.size_bytes == 0) {
        return;
    }

    // Allocate memory.

    const auto size_bytes = JST_PAGE_ALIGNED_SIZE(prototype.size_bytes);

    if (host_accessible || unified) {
        JST_CUDA_CHECK_THROW(cudaMallocManaged(&_buffer, size_bytes), [&]{
            JST_FATAL("[CUDA:BUFFER] Failed to allocate managed CUDA memory: {}", err);
        });
    } else {
        JST_CUDA_CHECK_THROW(cudaMalloc(&_buffer, size_bytes), [&]{
            JST_FATAL("[CUDA:BUFFER] Failed to allocate CUDA memory: {}", err);
        });
    }
    
    owns_data = true;
    _host_accessible = host_accessible || unified;
}

#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
Implementation::TensorBuffer(std::shared_ptr<TensorStorageMetadata>&,
                             const TensorPrototypeMetadata& prototype,
                             const std::shared_ptr<TensorBuffer<Device::Vulkan>>& root_buffer) {
    JST_TRACE("[CUDA:BUFFER] Cloning from Vulkan buffer.");

    // Check platform.

    if (!Backend::State<Device::CUDA>()->isAvailable()) {
        JST_FATAL("[CUDA:BUFFER] CUDA is not available.");
        JST_CHECK_THROW(Result::ERROR);
    }

    if (!Backend::State<Device::Vulkan>()->canExportMemory()) {
        JST_FATAL("[CUDA:BUFFER] Vulkan buffer cannot export memory. It cannot share data with CUDA.");
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
        JST_FATAL("[CUDA:BUFFER] Failed to import Vulkan buffer memory into CUDA.");
    });

    CUdeviceptr devPtr;

    CUDA_EXTERNAL_MEMORY_BUFFER_DESC bufferDesc = {};
    bufferDesc.offset = 0;
    bufferDesc.size = JST_PAGE_ALIGNED_SIZE(prototype.size_bytes);

    JST_CUDA_CHECK_THROW(cuExternalMemoryGetMappedBuffer(&devPtr, vulkan_external_memory, &bufferDesc), [&]{
        JST_FATAL("[CUDA:BUFFER] Failed to get CUDA buffer from Vulkan buffer memory.");
    });

    _buffer = reinterpret_cast<void*>(devPtr);
    external_memory_device = Device::Vulkan;
    owns_data = false;
}
#endif

#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
Implementation::TensorBuffer(std::shared_ptr<TensorStorageMetadata>&,
                             const TensorPrototypeMetadata& prototype,
                             const std::shared_ptr<TensorBuffer<Device::CPU>>& root_buffer) {
    throw std::runtime_error("Exporting CPU memory to CUDA not implemented.");
    // TODO: Add CPU -> CUDA.
}
#endif

#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
Implementation::TensorBuffer(std::shared_ptr<TensorStorageMetadata>&,
                             const TensorPrototypeMetadata& prototype,
                             const std::shared_ptr<TensorBuffer<Device::Metal>>& root_buffer) {
    throw std::runtime_error("Exporting Metal memory to CUDA not implemented.");
}
#endif

Implementation::~TensorBuffer() {
    JST_TRACE("[CUDA:BUFFER] Releasing buffer {}.", fmt::ptr(_buffer));

    if (owns_data) {
        cudaFree(&_buffer);
    }

#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
    if (external_memory_device == Device::Vulkan) {
        cuDestroyExternalMemory(vulkan_external_memory);
        close(vulkan_file_descriptor);
    }
#endif
}

}  // namespace Jetstream