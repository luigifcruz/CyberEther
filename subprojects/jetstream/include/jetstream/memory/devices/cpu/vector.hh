#ifndef JETSTREAM_MEMORY_CPU_VECTOR_HH
#define JETSTREAM_MEMORY_CPU_VECTOR_HH

#include "jetstream/memory/types.hh"
#include "jetstream/memory/vector.hh"

namespace Jetstream {

template<typename DataType, U64 Dimensions>
class JETSTREAM_API Vector<Device::CPU, DataType, Dimensions> : public VectorImpl<DataType, Dimensions> {
 public:
    using VectorType = VectorImpl<DataType, Dimensions>;

    Vector() : VectorType() {}

    Vector(const Vector& other) : VectorType(other) {}

#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
    explicit Vector(const Vector<Device::Metal, DataType, Dimensions>& other)
             : VectorType(other) {
        allocateExtras();
    }
#endif

    explicit Vector(void* ptr, const VectorShape<Dimensions>& shape)
             : VectorType(ptr, shape) {
        allocateExtras();
    }

    explicit Vector(const VectorShape<Dimensions>& shape)
             : VectorType(nullptr, shape) {
        JST_TRACE("New CPU vector created and allocated: {}", shape);

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

        // Null out array.
        std::fill(this->begin(), this->end(), 0.0f);

        allocateExtras();
    }

    Vector& operator=(const Vector& other) {
        VectorType::operator=(other);
        return *this;
    }

    constexpr Device device() const {
        return Device::CPU;
    }

 private:
    void allocateExtras() {
        // Allocate reference counter.
        if (!this->_refs) {
            this->_refs = new U64(1);
            this->_destructors->push_back([ptr = this->_refs]() { free(ptr); });
        }
        if (!this->_pos) {
            this->_pos = new U64(0);
            this->_destructors->push_back([ptr = this->_pos]() { free(ptr); });
        }
    }
};

}  // namespace Jetstream

#endif
