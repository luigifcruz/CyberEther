#ifndef JETSTREAM_MEMORY_CUDA_BUFFER_HH
#define JETSTREAM_MEMORY_CUDA_BUFFER_HH

#include <memory>

#include "jetstream/memory/devices/base/buffer.hh"

namespace Jetstream {

template<>
class TensorBuffer<Device::CUDA> : public TensorBufferBase {
 public:
    explicit TensorBuffer(std::shared_ptr<TensorStorageMetadata>& storage,
                          const TensorPrototypeMetadata& prototype,
                          const bool& host_accessible = false);

#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
    explicit TensorBuffer(std::shared_ptr<TensorStorageMetadata>& storage,
                          const TensorPrototypeMetadata& prototype,
                          const std::shared_ptr<TensorBuffer<Device::Vulkan>>& root_buffer);
    static bool CanImport(const TensorBuffer<Device::Vulkan>& root_buffer) noexcept;
#endif

#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
    explicit TensorBuffer(std::shared_ptr<TensorStorageMetadata>& storage,
                          const TensorPrototypeMetadata& prototype,
                          const std::shared_ptr<TensorBuffer<Device::CPU>>& root_buffer);
    static bool CanImport(const TensorBuffer<Device::CPU>& root_buffer) noexcept;
#endif

#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
    explicit TensorBuffer(std::shared_ptr<TensorStorageMetadata>& storage,
                          const TensorPrototypeMetadata& prototype,
                          const std::shared_ptr<TensorBuffer<Device::Metal>>& root_buffer);
    static bool CanImport(const TensorBuffer<Device::Metal>& root_buffer) noexcept;
#endif

    ~TensorBuffer();

    TensorBuffer(const TensorBuffer&) = delete;
    TensorBuffer& operator=(const TensorBuffer&) = delete;

    constexpr const void* data() const noexcept {
        return buffer;
    }

    constexpr void* data() noexcept {
        return buffer;
    }

    constexpr const void* data_ptr() const noexcept {
        return &buffer;
    }

    constexpr void* data_ptr() noexcept {
        return &buffer;
    }

    constexpr CUmemGenericAllocationHandle& handle() noexcept {
        return alloc_handle;
    } 

 private:
    void* buffer;
    U64 size_bytes;

    CUdeviceptr device_ptr;
    CUmemGenericAllocationHandle alloc_handle;

#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
    int vulkan_file_descriptor = 0;
    CUexternalMemory vulkan_external_memory = nullptr;
#endif
};

}  // namespace Jetstream

#endif
