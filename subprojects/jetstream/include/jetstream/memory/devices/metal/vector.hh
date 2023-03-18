#ifndef JETSTREAM_MEMORY_METAL_VECTOR_HH
#define JETSTREAM_MEMORY_METAL_VECTOR_HH

#include "jetstream/backend/base.hh"

#include "jetstream/memory/types.hh"
#include "jetstream/memory/vector.hh"

namespace Jetstream {

template<typename DataType, U64 Dimensions>
class JETSTREAM_API Vector<Device::Metal, DataType, Dimensions> : public VectorImpl<DataType, Dimensions> {
 public:
    using VectorType = VectorImpl<DataType, Dimensions>;
    using VectorImpl<DataType, Dimensions>::VectorImpl;

    Vector(const typename VectorType::ShapeType& shape) : VectorType(shape) {
        JST_TRACE("New Metal vector created and allocated: ", shape);

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

        // Create MTL::Buffer.
        auto device = Backend::State<Device::Metal>()->getDevice();
        _buffer = device->newBuffer(this->_data,
                                   JST_PAGE_ALIGNED_SIZE(this->size_bytes()), 
                                   MTL::ResourceStorageModeShared,
                                   nullptr); 
        if (!_buffer) {
            JST_FATAL("Couldn't allocate MTL::Buffer.");
            JST_CHECK_THROW(Result::ERROR);
        }

        // Create unmanaged CPU buffer.
        _cpu = Vector<Device::CPU, DataType, Dimensions>(this->data(), this->shape());

        // Register allocated memory. 
        this->_destructorList["_data"] = this->_data;
        this->_destructorList["_refs"] = this->_refs;
        this->_destructorList["_buffer"] = this->_buffer;
        
        // Register memory destructor.
        this->_destructor = [](std::unordered_map<std::string, void*>& list){
            free(list["_data"]);
            free(list["_refs"]);
            reinterpret_cast<MTL::Buffer*>(list["_buffer"])->release();
        };
    }

    // Overloads for MTL::Buffer.

    operator const MTL::Buffer*() const {
        return _buffer;
    }

    operator MTL::Buffer*() {
        return _buffer;
    }

    // Overloads for Vector<Device::CPU>.

    operator const Vector<Device::CPU, DataType, Dimensions>&() const {
        return _cpu;
    }

    operator Vector<Device::CPU, DataType, Dimensions>&() {
        return _cpu;
    }

 private:
    MTL::Buffer* _buffer;
    Vector<Device::CPU, DataType, Dimensions> _cpu; 
};

}  // namespace Jetstream

#endif
