#ifndef JETSTREAM_MEMORY_CPU_VECTOR_HH
#define JETSTREAM_MEMORY_CPU_VECTOR_HH

#include "jetstream/memory/types.hh"
#include "jetstream/memory/vector.hh"

namespace Jetstream {

template<typename T>
class JETSTREAM_API Vector<Device::CPU, T> : public VectorImpl<T> {
 public:
    using VectorImpl<T>::VectorImpl;

    explicit Vector(const std::size_t& size) {
        JST_CHECK_THROW(this->resize(size));
    }

    ~Vector() {
        if (this->container.empty() || !this->managed) {
            return;
        }

#ifdef JETSTREAM_CUDA_AVAILABLE
        if (cudaFreeHost(this->container.data()) != cudaSuccess) {
            JST_FATAL("Failed to deallocate host memory.");
        }
#else
        free(this->container.data());
#endif
    }

    // TODO: Implement resize.
    Result resize(const std::size_t& size) override {
        if (!this->container.empty() && !this->managed) {
            return Result::ERROR;
        }

        T* ptr = nullptr;
        const auto sizeBytes = size * sizeof(T);

#ifdef JETSTREAM_CUDA_AVAILABLE
        BL_CUDA_CHECK(cudaMallocHost(&ptr, size_bytes), [&]{
            JST_FATAL("Failed to allocate CPU memory: {}", err);
        });
#else
        void* memoryAddr = nullptr;
        const auto pageSize = JST_PAGESIZE();
        const auto alignedSizeBytes = JST_PAGE_ALIGNED_SIZE(sizeBytes);
        const auto result = posix_memalign(&memoryAddr, 
                                           pageSize,
                                           alignedSizeBytes);
        if (result < 0 || (ptr = static_cast<T*>(memoryAddr)) == nullptr) {
            JST_FATAL("Failed to allocate CPU memory.");
        }
#endif

        this->container = std::span<T>(ptr, size);
        this->managed = true;

        return Result::SUCCESS;
    }
};

}  // namespace Jetstream

#endif
