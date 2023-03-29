#ifndef JETSTREAM_MEMORY_VECTOR_HH
#define JETSTREAM_MEMORY_VECTOR_HH

#include <memory>
#include <unordered_map>

#include "jetstream/types.hh"

namespace Jetstream {

template<Device DeviceId, typename DataType, U64 Dimensions = 1> class Vector;

template<typename DataType, U64 Dimensions>
class VectorImpl {
 public:
    using ShapeType = std::array<U64, Dimensions>;

    virtual ~VectorImpl() {
        decreaseRefCount();
    }

    constexpr DataType* data() const noexcept {
        return _data;
    }

    constexpr const U64 size() const noexcept {
        U64 _size = 1;
        for (const auto& dim : _shape) {
            _size *= dim;
        }
        return _size; 
    }

    constexpr const U64 refs() const noexcept {
        if (!_refs) {
            return 0;
        }
        return *_refs;
    }

    constexpr const ShapeType& shape() const noexcept {
        return _shape;
    }

    constexpr const U64& shape(const U64& index) const noexcept {
        return _shape[index];
    }

    constexpr const U64 hash() const noexcept {
        if (!_data) {
            return 0;
        }
        return std::hash<void*>{}(this->_data);
    }

    constexpr const U64 size_bytes() const noexcept {
        return size() * sizeof(DataType);
    }

    [[nodiscard]] constexpr const bool empty() const noexcept {
        return (_data == nullptr);
    }

    constexpr DataType& operator[](const ShapeType& shape) {
        return _data[shapeToOffset(shape)];
    }

    constexpr const DataType& operator[](const ShapeType& shape) const {
        return _data[shapeToOffset(shape)];
    }

    constexpr DataType& operator[](const U64& idx) {
        return _data[idx];
    }

    constexpr const DataType& operator[](const U64& idx) const {
        return _data[idx];
    }

    constexpr auto begin() {
        return _data;
    }

    constexpr auto end() {
        return _data + size();
    }

    constexpr const auto begin() const {
        return _data;
    }

    constexpr const auto end() const {
        return _data + size();
    }

    const U64 shapeToOffset(const ShapeType& shape) const {
        U64 offset = 0;
        for (U64 i = 0; i < shape.size(); i++) {
            U64 product = shape[i];
            for (U64 j = i + 1; j < shape.size(); j++) {
                product *= _shape[j];
            }
            offset += product;
        }
        return offset;
    }

    VectorImpl(VectorImpl&&) = delete;
    VectorImpl& operator=(VectorImpl&&) = delete;

 protected:
    ShapeType _shape;
    DataType* _data;
    U64* _refs;
    std::vector<std::function<void()>>* _destructors;

    VectorImpl()
             : _shape({0}),
               _data(nullptr),
               _refs(nullptr), 
               _destructors(nullptr) {
        JST_TRACE("Empty vector created.");
    }

    explicit VectorImpl(const VectorImpl& other)
             : _shape(other._shape),
               _data(other._data),
               _refs(other._refs),
               _destructors(other._destructors) {
        JST_TRACE("Vector created by copy.");

        increaseRefCount();
    }

    VectorImpl& operator=(const VectorImpl& other) {
        JST_TRACE("Vector copied to existing.");

        decreaseRefCount();
            
        _data = other._data;
        _shape = other._shape;
        _refs = other._refs;
        _destructors = other._destructors;

        increaseRefCount();

        return *this;
    }

    explicit VectorImpl(void* ptr, const ShapeType& shape)
             : _shape(shape),
               _data(static_cast<DataType*>(ptr)),
               _refs(nullptr), 
               _destructors(nullptr) {
        increaseRefCount();
    }
    
    void decreaseRefCount() {
        if (!_refs) {
            return;
        }
        JST_TRACE("Decreasing reference counter to {}.", *_refs - 1);

        if (--(*_refs) == 0) {
            JST_TRACE("Deleting {} pointers.", _destructors->size());

            for (auto& destructor : *_destructors) {
                destructor();                    
            }

            delete _destructors;
            
            reset();
        }
    }

    void increaseRefCount() {
        if (!_refs) {
            JST_TRACE("Creating new destructor list.");
            _destructors = new std::vector<std::function<void()>>();
            return;
        }
        JST_TRACE("Increasing reference counter to {}.", *_refs + 1);
        *_refs += 1;
    }

    void reset() {
        _data = nullptr;
        _refs = nullptr;
        _shape = ShapeType({0});
        _destructors = nullptr;
    }
};

}  // namespace Jetstream

#endif
