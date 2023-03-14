#ifndef JETSTREAM_MEMORY_CPU_VECTOR_HH
#define JETSTREAM_MEMORY_CPU_VECTOR_HH

#include "jetstream/backend/base.hh"

#include "jetstream/memory/types.hh"
#include "jetstream/memory/vector.hh"

namespace Jetstream {

template<typename DataType>
class JETSTREAM_API Vector<Device::CPU, DataType> : public VectorImpl<DataType> {
 public:
    using VectorImpl<DataType>::VectorImpl;

    Vector(const U64& size) : VectorImpl<DataType>(size) {
        JST_TRACE("New CPU vector allocated.");

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

        // TODO: Move this to the Device::Metal vector.
        auto device = Backend::State<Device::Metal>()->getDevice();
        _buffer = device->newBuffer(this->_data,
                                   JST_PAGE_ALIGNED_SIZE(this->size_bytes()), 
                                   MTL::ResourceStorageModeShared,
                                   nullptr); 
        assert(_buffer);

        // Register memory destructor.
        this->_destructor = [](std::unordered_map<std::string, void*>& list){
#ifdef JETSTREAM_CUDA_AVAILABLE
            cudaFreeHost(list["_data"]);
#else
            free(list["_data"]);
#endif
            free(list["_refs"]);

            // TODO: Add MTL::Buffer destructor.
        };
    }

    // TODO: Move this to the Device::Metal vector.
    // const MTL::Buffer* buffer() const {
    //     return _buffer;
    // }

    MTL::Buffer* buffer() const {
        return _buffer;
    }

 private:
    // TODO: Move this to the Device::Metal vector.
    MTL::Buffer* _buffer;
};

}  // namespace Jetstream

#endif
