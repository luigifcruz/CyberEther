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

    Vector() : VectorType() {}

    explicit Vector(const Vector& other)
             : VectorType(other),
               _metal(other._metal),
               _cpu(other._cpu) {
    }

#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
    explicit Vector(const Vector<Device::CPU, DataType, Dimensions>& other)
             : VectorType(other) {
        allocateExtras();     
    }
#endif

    explicit Vector(void* ptr, const typename VectorType::ShapeType& shape)
             : VectorType(ptr, shape) {
        allocateExtras();
    }

    explicit Vector(const typename VectorType::ShapeType& shape) 
             : VectorType(nullptr, shape) {
        JST_TRACE("New Metal vector created and allocated: {}", shape);

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
        this->_destructors->push_back([ptr = this->_data]() { free(ptr); });

        allocateExtras();
    }

    Vector& operator=(const Vector& other) {
        VectorType::operator=(other);
        _metal = other._metal;
        _cpu = other._cpu;
        return *this;
    }

    // Expose overloads for MTL::Buffer.

    operator const MTL::Buffer*() const {
        return this->_metal;
    }

    operator MTL::Buffer*() {
        return this->_metal;
    }

    // Expose overloads for Vector<Device::CPU>.

    operator const Vector<Device::CPU, DataType, Dimensions>&() const {
        return this->_cpu;
    }

    operator Vector<Device::CPU, DataType, Dimensions>&() {
        return this->_cpu;
    }

 private:
    MTL::Buffer* _metal;
    Vector<Device::CPU, DataType, Dimensions> _cpu;

    void allocateExtras() {
        // Check if memory is aligned.
        if (!JST_IS_ALIGNED(this->data())) {
            JST_FATAL("Memory pointer is not aligned. Can't create vector.");
            JST_CHECK_THROW(Result::SUCCESS);
        }

        // Allocate reference counter.
        if (!this->_refs) {
            this->_refs = new U64(1);
            this->_destructors->push_back([ptr = this->_refs]() { free(ptr); });
        }

        // Create MTL::Buffer.
        auto device = Backend::State<Device::Metal>()->getDevice();
        this->_metal = device->newBuffer(this->_data,
                                         JST_PAGE_ALIGNED_SIZE(this->size_bytes()), 
                                         MTL::ResourceStorageModeShared,
                                         nullptr); 
        if (!this->_metal) {
            JST_FATAL("Couldn't allocate MTL::Buffer.");
            JST_CHECK_THROW(Result::ERROR);
        }
        this->_destructors->push_back([ptr = this->_metal]() { reinterpret_cast<MTL::Buffer*>(ptr)->release(); });

        // Create unmanaged CPU buffer.
        this->_cpu = Vector<Device::CPU, DataType, Dimensions>(*this);
    }
};

}  // namespace Jetstream

#endif
