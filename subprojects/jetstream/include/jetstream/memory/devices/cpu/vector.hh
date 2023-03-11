#ifndef JETSTREAM_MEMORY_CPU_VECTOR_HH
#define JETSTREAM_MEMORY_CPU_VECTOR_HH

#include "jetstream/memory/types.hh"
#include "jetstream/memory/vector.hh"

namespace Jetstream {

template<typename T>
class JETSTREAM_API Vector<Device::CPU, T> : public VectorImpl<T> {
 public:
    using VectorImpl<T>::VectorImpl;

    explicit Vector(const U64& size) : VectorImpl<T>(size) {
#ifdef JETSTREAM_CUDA_AVAILABLE
        BL_CUDA_CHECK(cudaMallocHost(&this->_data, this->size_bytes()), [&]{
            JST_FATAL("Failed to allocate CPU memory: {}", err);
        });
#else
        void* memoryAddr = nullptr;
        const auto pageSize = JST_PAGESIZE();
        const auto alignedSizeBytes = JST_PAGE_ALIGNED_SIZE(this->size_bytes());
        const auto result = posix_memalign(&memoryAddr, 
                                           pageSize,
                                           alignedSizeBytes);
        if (result < 0 || (this->_data = static_cast<T*>(memoryAddr)) == nullptr) {
            JST_FATAL("Failed to allocate CPU memory.");
        }
#endif
    }

    ~Vector() {
        if (!this->managed || !this->_data) {
            return;
        }

#ifdef JETSTREAM_CUDA_AVAILABLE
        if (cudaFreeHost(this->_data) != cudaSuccess) {
            JST_FATAL("Failed to deallocate host memory.");
        }
#else
        free(this->_data);
#endif
    }

    Vector& operator=(Vector&& other) {
        VectorImpl<T>::operator = (std::move(other));
        return *this;
    }
};

}  // namespace Jetstream

#endif
