#ifndef JETSTREAM_MEMORY_VECTOR_HH
#define JETSTREAM_MEMORY_VECTOR_HH

#include "jetstream/types.hh"

namespace Jetstream {

template<Device I, typename T> class Vector;

template<typename T>
class VectorImpl {
 public:
    VectorImpl()
             : container(),
               managed(false) {}
    explicit VectorImpl(const std::span<T>& other)
             : container(other),
               managed(false) {}
    explicit VectorImpl(T* ptr, const std::size_t& size)
             : container(ptr, size),
               managed(false) {}
    explicit VectorImpl(void* ptr, const std::size_t& size)
             : container(static_cast<T*>(ptr), size),
               managed(false) {}

    VectorImpl(const VectorImpl&) = delete;
    VectorImpl& operator=(const VectorImpl&) = delete;

    virtual ~VectorImpl() {}

    constexpr T* data() const noexcept {
        return container.data();
    }

    constexpr const U64 size() const noexcept {
        return container.size();
    }

    constexpr const U64 size_bytes() const noexcept {
        return container.size_bytes();
    }

    [[nodiscard]] constexpr const bool empty() const noexcept {
        return container.empty();
    }

    constexpr T& operator[](U64 idx) {
        return container[idx];
    }

    constexpr const T& operator[](U64 idx) const {
        return container[idx];
    }

    constexpr auto begin() {
        return container.begin();
    }

    constexpr auto end() {
        return container.end();
    }

    constexpr const auto begin() const {
        return container.begin();
    }

    constexpr const auto end() const {
        return container.end();
    }

    virtual Result resize(const std::size_t& size) = 0;

 protected:
    std::span<T> container;
    bool managed;
};

}  // namespace Jetstream

#endif
