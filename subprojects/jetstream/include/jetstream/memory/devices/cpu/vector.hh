#ifndef JETSTREAM_MEMORY_CPU_VECTOR_HH
#define JETSTREAM_MEMORY_CPU_VECTOR_HH

#include "jetstream/memory/types.hh"
#include "jetstream/memory/vector.hh"

namespace Jetstream {

template<typename DataType, U64 Dimensions>
class JETSTREAM_API Vector<Device::CPU, DataType, Dimensions> : public VectorImpl<DataType, Dimensions> {
 public:
    // TODO: Remove. Explicitly add constructors here.
    using VectorType = VectorImpl<DataType, Dimensions>;
    using VectorImpl<DataType, Dimensions>::VectorImpl;

    Vector(const typename VectorType::ShapeType& shape) : VectorType(shape) {
        JST_TRACE("New CPU vector created and allocated: {}", shape);

        // Allocate memory.
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
        if (result < 0 || (this->_data = static_cast<DataType*>(memoryAddr)) == nullptr) {
            JST_FATAL("Failed to allocate CPU memory.");
            JST_CHECK_THROW(Result::ERROR);
        }
#endif
        this->_refs = new U64(1);

        // Register allocated memory. 
        this->_destructorList["_data"] = this->_data;
        this->_destructorList["_refs"] = this->_refs;

        // Register memory destructor.
        this->_destructor = [](std::unordered_map<std::string, void*>& list){
#ifdef JETSTREAM_CUDA_AVAILABLE
            cudaFreeHost(list["_data"]);
#else
            free(list["_data"]);
#endif
            free(list["_refs"]);
        };
    }
};

}  // namespace Jetstream

#endif
