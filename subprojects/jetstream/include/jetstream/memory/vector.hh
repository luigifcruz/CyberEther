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
    
    VectorImpl()
             : _shape({0}),
               _data(nullptr),
               _refs(nullptr),
               _destructor(nullptr) {}

    explicit VectorImpl(void* ptr, const ShapeType& shape)
             : _shape(shape),
               _data(static_cast<DataType*>(ptr)),
               _refs(nullptr),
               _destructor(nullptr) {}

    explicit VectorImpl(const VectorImpl& other)
             : _shape(other._shape),
               _data(other._data),
               _refs(other._refs),
               _destructor(other._destructor),
               _destructorList(other._destructorList) {
        increaseRefCount();
    }

    explicit VectorImpl(VectorImpl&& other)
             : _shape({0}),
               _data(nullptr),
               _refs(nullptr),
               _destructor(nullptr) { 
        std::swap(_data, other._data);
        std::swap(_shape, other._shape);
        std::swap(_refs, other._refs);
        std::swap(_destructor, other._destructor);
        std::swap(_destructorList, other._destructorList);
    }

    VectorImpl& operator=(VectorImpl& other) {
        decreaseRefCount();
        _data = other._data;
        _shape = other._shape;
        _refs = other._refs;
        _destructor = other._destructor;
        _destructorList = other._destructorList;

        increaseRefCount();

        return *this;
    }

    VectorImpl& operator=(VectorImpl&& other) {
        decreaseRefCount();
        reset();
        std::swap(_data, other._data);
        std::swap(_shape, other._shape);
        std::swap(_refs, other._refs);
        std::swap(_destructor, other._destructor);
        std::swap(_destructorList, other._destructorList);

        return *this;
    }

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
        return _refs;
    }

    constexpr const ShapeType& shape() const noexcept {
        return _shape;
    }

    constexpr const U64& shape(const U64& index) const noexcept {
        return _shape[index];
    }

    constexpr const U64 hash() const noexcept {
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
        return _data + size_bytes();
    }

    constexpr const auto begin() const {
        return _data;
    }

    constexpr const auto end() const {
        return _data + size_bytes();
    }

 protected:
    ShapeType _shape;
    DataType* _data;
    U64* _refs;

    std::unordered_map<std::string, void*> _destructorList;
    std::function<void(std::unordered_map<std::string, void*>&)> _destructor;

    explicit VectorImpl(const ShapeType& shape)
             : _shape(shape),
               _data(nullptr),
               _refs(nullptr) {}

    void decreaseRefCount() {
        if (_refs) {
            *_refs -= 1;

            if (*_refs == 0) {
                _destructor(_destructorList);
                reset();
            }
        }
    }

    void increaseRefCount() {
        if (_refs) {
            *_refs += 1;
        }
    }

    void reset() {
        _data = nullptr;
        _refs = nullptr;
        _shape = ShapeType({0});
        _destructor = nullptr;
        _destructorList.clear();
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
};

}  // namespace Jetstream

#endif
