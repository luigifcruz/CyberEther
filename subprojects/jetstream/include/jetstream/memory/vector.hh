#ifndef JETSTREAM_MEMORY_VECTOR_HH
#define JETSTREAM_MEMORY_VECTOR_HH

#include "jetstream/types.hh"

namespace Jetstream {

template<Device I, typename T> class Vector;

template<typename T>
class VectorImpl {
 public:
    VectorImpl()
             : _data(nullptr),
               _size(0),
               managed(false) {}
    explicit VectorImpl(const VectorImpl& other)
             : _data(other.data()),
               _size(other.size()),
               managed(false) {}
    explicit VectorImpl(const std::span<T>& other)
             : _data(other.data()),
               _size(other.size()),
               managed(false) {}
    explicit VectorImpl(T* ptr, const U64& size)
             : _data(ptr),
               _size(size),
               managed(false) {}
    explicit VectorImpl(void* ptr, const U64& size)
             : _data(static_cast<T*>(ptr)),
               _size(size),
               managed(false) {}

    VectorImpl& operator=(VectorImpl&& other) {
        if (!empty()) {
            JST_FATAL("This vector is not empty.");
            JST_CHECK_THROW(Result::ERROR);
        }

        std::swap(_data, other._data);
        std::swap(_size, other._size);
        std::swap(managed, other.managed);

        return *this;
    }

    VectorImpl(VectorImpl&&) = delete;
    VectorImpl(VectorImpl&) = delete;
    VectorImpl& operator=(const VectorImpl&) = delete;

    virtual ~VectorImpl() = default;

    constexpr T* data() const noexcept {
        return _data;
    }

    constexpr const U64 size() const noexcept {
        return _size; 
    }

    constexpr const U64 size_bytes() const noexcept {
        return size() * sizeof(T);
    }

    [[nodiscard]] constexpr const bool empty() const noexcept {
        return (_data == nullptr) && (managed == false);
    }

    constexpr T& operator[](U64 idx) {
        return _data[idx];
    }

    constexpr const T& operator[](U64 idx) const {
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
    T* _data;
    U64 _size;
    bool managed;

    VectorImpl(const U64& size)
             : _data(nullptr),
               _size(size),
               managed(true) {}
};

}  // namespace Jetstream

#endif
