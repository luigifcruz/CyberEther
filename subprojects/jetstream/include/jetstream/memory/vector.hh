#ifndef JETSTREAM_MEMORY_VECTOR_HH
#define JETSTREAM_MEMORY_VECTOR_HH

#include <memory>
#include <unordered_map>

#include "jetstream/types.hh"

namespace Jetstream {

template<Device DeviceId, typename DataType, U64 Dimensions = 1> class Vector;

template<typename Type, U64 Dimensions>
class VectorImpl {
 public:
    using DataType = Type;
    using ShapeType = std::array<U64, Dimensions>;

    virtual ~VectorImpl() {
        decreaseRefCount();
    }

    constexpr DataType* data() const noexcept {
        return _data;
    }

    constexpr U64 size() const noexcept {
        U64 _size = 1;
        for (const auto& dim : _shape) {
            _size *= dim;
        }
        return _size; 
    }

    constexpr U64 refs() const noexcept {
        if (!_refs) {
            return 0;
        }
        return *_refs;
    }

    constexpr const ShapeType& shape() const noexcept {
        return _shape;
    }

    constexpr std::vector<U64> shapeVector() const noexcept {
        return std::vector<U64>(_shape.begin(), _shape.end());
    }

    constexpr const U64& shape(const U64& index) const noexcept {
        return _shape[index];
    }

    constexpr U64 hash() const noexcept {
        if (!_data) {
            return 0;
        }

        // For some fucking reason, std::hash is ignoring 32 MSB. 
        // I don't know why, I don't want to know why.
        // Thats why I'm using a custom hash function.
        //
        // Don't believe me? See it for yourself:
        //     uint64_t* a = (uint64_t*)0x118008000;
        //     uint64_t* b = (uint64_t*)0x158008000;
        //     std::cout << (std::hash<void*>{}(a) == std::hash<void*>{}(b)) << std::endl;
        // 
        // This should print 0. But it prints 1 on M1 Pro.
        // https://twitter.com/luigifcruz/status/1670260464058605568

        return HashU64(reinterpret_cast<U64>(this->_data));
    }

    constexpr U64 phash() const noexcept {
        return hash() + *this->_pos;
    }

    constexpr U64 size_bytes() const noexcept {
        return size() * sizeof(DataType);
    }

    [[nodiscard]] constexpr bool empty() const noexcept {
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

    constexpr auto begin() const {
        return _data;
    }

    constexpr auto end() const {
        return _data + size();
    }

    U64 shapeToOffset(const ShapeType& shape) const {
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

    void incrementPositionalIndex() {
        *this->_pos += 1;
    }

    VectorImpl(VectorImpl&&) = delete;
    VectorImpl& operator=(VectorImpl&&) = delete;

 protected:
    U64* _pos;
    ShapeType _shape;
    DataType* _data;
    U64* _refs;
    std::vector<std::function<void()>>* _destructors;

    VectorImpl()
             : _pos(nullptr),
               _shape({0}),
               _data(nullptr),
               _refs(nullptr), 
               _destructors(nullptr) {
        JST_TRACE("Empty vector created.");
    }

    explicit VectorImpl(const VectorImpl& other)
             : _pos(other._pos),
               _shape(other._shape),
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
        _pos = other._pos;
        _shape = other._shape;
        _refs = other._refs;
        _destructors = other._destructors;

        increaseRefCount();

        return *this;
    }

    explicit VectorImpl(void* ptr, const ShapeType& shape)
             : _pos(nullptr),
               _shape(shape),
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
        _pos = nullptr;
        _shape = ShapeType({0});
        _destructors = nullptr;
    }
};

}  // namespace Jetstream

#endif
