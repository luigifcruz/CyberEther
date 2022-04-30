#ifndef JETSTREAM_MEMORY_CPU_VECTOR_HH
#define JETSTREAM_MEMORY_CPU_VECTOR_HH

#include "jetstream/types.hh"
#include "jetstream/memory/vector.hh"

namespace Jetstream::Memory {

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

        T* ptr;
        auto size_bytes = size * sizeof(T);

#ifdef JETSTREAM_CUDA_AVAILABLE
        BL_CUDA_CHECK(cudaMallocHost(&ptr, size_bytes), [&]{
            JST_FATAL("Failed to allocate CPU memory: {}", err);
        });
#else
        if ((ptr = static_cast<T*>(malloc(size_bytes))) == nullptr) {
            JST_FATAL("Failed to allocate CPU memory.");
        }
#endif

        this->container = std::span<T>(ptr, size);
        this->managed = true;

        return Result::SUCCESS;
    }
};

}  // namespace Jetstream::Memory

#endif
