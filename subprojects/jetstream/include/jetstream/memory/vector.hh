#ifndef JETSTREAM_MEMORY_VECTOR_HH
#define JETSTREAM_MEMORY_VECTOR_HH

#include <memory>
#include <unordered_map>

#include "jetstream/types.hh"

namespace Jetstream {

template<Device DeviceId, typename DataType> class Vector;

template<typename DataType>
class VectorImpl {
 public:
    VectorImpl()
             : _data(nullptr),
               _size(0),
               _refs(nullptr),
               _destructor(nullptr) {}

    explicit VectorImpl(void* ptr, const U64& size)
             : _data(static_cast<DataType*>(ptr)),
               _size(size),
               _refs(nullptr),
               _destructor(nullptr) {}

    explicit VectorImpl(const VectorImpl& other)
             : _data(other.data()),
               _size(other.size()),
               _refs(other.refs()),
               _destructor(other._destructor),
               _destructorList(other._destructorList) {
        increaseRefCount();
    }

    explicit VectorImpl(VectorImpl&& other)
             : _data(nullptr),
               _size(0),
               _refs(nullptr),
               _destructor(nullptr) { 
        std::swap(_data, other._data);
        std::swap(_size, other._size);
        std::swap(_refs, other._refs);
        std::swap(_destructor, other._destructor);
        std::swap(_destructorList, other._destructorList);
    }

    VectorImpl& operator=(VectorImpl& other) {
        decreaseRefCount();
        _data = other._data;
        _size = other._size;
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
        std::swap(_size, other._size);
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
        return _size; 
    }

    constexpr const U64 refs() const noexcept {
        if (!_refs) {
            return 0;
        }
        return _refs;
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

    constexpr DataType& operator[](U64 idx) {
        return _data[idx];
    }

    constexpr const DataType& operator[](U64 idx) const {
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
    DataType* _data;
    U64 _size;
    U64* _refs;
    std::unordered_map<std::string, void*> _destructorList;
    std::function<void(std::unordered_map<std::string, void*>&)> _destructor;

    explicit VectorImpl(const U64& size)
             : _data(nullptr),
               _size(size),
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
        _size = 0;
        _destructor = nullptr;
        _destructorList.clear();
    }
};

}  // namespace Jetstream

#endif
