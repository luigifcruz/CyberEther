#ifndef JETSTREAM_MEMORY_METAL_VECTOR_HH
#define JETSTREAM_MEMORY_METAL_VECTOR_HH

#include "jetstream/memory/types.hh"
#include "jetstream/memory/vector.hh"

namespace Jetstream {

template<typename DataType>
class JETSTREAM_API Vector<Device::Metal, DataType> : public VectorImpl<DataType> {
 public:
    using VectorImpl<DataType>::VectorImpl;

    Vector(const U64& size) : VectorImpl<DataType>(size) {
        JST_TRACE("New Metal vector allocated.");

        // Allocate memory.
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

        this->_refs = new U64(1);

        // Register allocated memory. 
        this->_destructorList["_data"] = this->_data;
        this->_destructorList["_refs"] = this->_refs;

        // Register memory destructor.
        this->_destructor = [](std::unordered_map<std::string, void*>& list){
            free(list["_data"]);
            free(list["_refs"]);
        };
    }


};

}  // namespace Jetstream

#endif
