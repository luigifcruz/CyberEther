#ifndef JETSTREAM_MEMORY_PROTOTYPE_HH
#define JETSTREAM_MEMORY_PROTOTYPE_HH

#include <vector>

#include "jetstream/types.hh"
#include "jetstream/memory/metadata.hh"
#include "jetstream/memory/token.hh"

namespace Jetstream {

class TensorPrototype {
 public:
    constexpr const U64& size() const noexcept {
        return prototype.size;
    }

    constexpr const U64& size_bytes() const noexcept {
        return prototype.size_bytes;
    }

    constexpr const U64& element_size() const noexcept {
        return prototype.element_size;
    }

    constexpr const bool& contiguous() const noexcept {
        return prototype.contiguous;
    }

    constexpr const U64& offset() const noexcept {
        return prototype.offset;
    }

    constexpr const U64& offset_bytes() const noexcept {
        return prototype.offset_bytes;
    }

    constexpr const std::vector<U64>& shape() const noexcept {
        return prototype.shape;
    }

    constexpr const std::vector<U64>& stride() const noexcept {
        return prototype.stride;
    }

    constexpr const std::vector<U64>& shape_minus_one() const noexcept {
        return prototype.shape_minus_one;
    }

    constexpr const std::vector<U64>& backstride() const noexcept {
        return prototype.backstride;
    }

    constexpr const U64& hash() const noexcept {
        return prototype.hash;
    }

    constexpr bool empty() const noexcept {
        return prototype.size == 0;
    }

    constexpr bool valid_shape() const noexcept {
        return prototype.size > 0;
    }

    constexpr const Locale& locale() const noexcept {
        return prototype.locale;
    }

    const U64& shape(const U64& idx) const noexcept;

    const U64& stride(const U64& idx) const noexcept;

    U64 rank() const noexcept;

    U64 ndims() const noexcept;

    void set_locale(const Locale& locale) noexcept;

    U64 shape_to_offset(const std::vector<U64>& shape) const;

    void offset_to_shape(U64 index, std::vector<U64>& shape) const;

    void expand_dims(const U64& axis);

    void squeeze_dims(const U64& axis);

    Result permutation(const std::vector<U64>& permutation);

    Result reshape(const std::vector<U64>& shape);

    Result broadcast_to(const std::vector<U64>& shape);

    Result slice(const std::vector<Token>& slice);
    
 protected:
    TensorPrototypeMetadata prototype;

    void initialize(const std::vector<U64>& shape, const U64& element_size);
    void update_cache();
};

}  // namespace Jetstream

#endif
