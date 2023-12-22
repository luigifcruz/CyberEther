#ifndef JETSTREAM_MEMORY_PROTOTYPE_HH
#define JETSTREAM_MEMORY_PROTOTYPE_HH

#include <any>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <cassert>

#include "jetstream/types.hh"
#include "jetstream/memory/metadata.hh"

namespace Jetstream {

class TensorPrototype {
 public:
    const U64& size() const noexcept {
        return prototype.size;
    }

    const U64& size_bytes() const noexcept {
        return prototype.size_bytes;
    }

    const U64& type_size() const noexcept {
        return prototype.type_size;
    }

    const U64& hash() const noexcept {
        return prototype.hash;
    }

    const std::vector<U64>& shape() const noexcept {
        return prototype.shape;
    }

    const std::vector<U64>& strides() const noexcept {
        return prototype.strides;
    }

    const U64& shape(const U64& idx) const noexcept {
        return prototype.shape[idx];
    }

    const U64& strides(const U64& idx) const noexcept {
        return prototype.strides[idx];
    }

    bool empty() const noexcept {
        return prototype.size == 0;
    }

    U64 rank() const noexcept {
        return prototype.shape.size();
    }

    U64 ndims() const noexcept {
        return prototype.shape.size();
    }

    bool valid_shape() const noexcept {
        bool valid = true;
        for (const auto& dim : prototype.shape) {
            valid &= dim > 0;
        }
        return valid;
    }

    const Locale& locale() const noexcept {
        return prototype.locale;
    }

    void set_locale(const Locale& locale) noexcept {
        prototype.locale = locale;
    }

    bool operator==(const TensorPrototype& other) const noexcept {
        return prototype.hash == other.prototype.hash;
    }

    bool operator!=(const TensorPrototype& other) const noexcept {
        return prototype.hash != other.prototype.hash;
    }

    U64 shape_to_offset(const std::vector<U64>& shape) const {
        assert(shape.size() >= prototype.strides.size());
        U64 index = 0;
        U64 pad = shape.size() - prototype.strides.size();
        for (U64 i = 0; i < prototype.strides.size(); i++) {
            // TODO: This is a hack. This should be done by modifiying the strides.
            index += ((shape[pad + i] >= prototype.shape[i]) ? 0 : shape[pad + i]) * prototype.strides[i];
        }
        return index;
    }

    void offset_to_shape(U64 index, std::vector<U64>& shape) const {
        assert(shape.size() >= prototype.strides.size());
        for (U64 i = 0; i < prototype.strides.size(); i++) {
            shape[i] = index / prototype.strides[i];
            index -= shape[i] * prototype.strides[i];
        }
    }

    void expand_dims(const U64& axis) {
        prototype.shape.insert(prototype.shape.begin() + axis, 1);
        prototype.strides.insert(prototype.strides.begin() + axis, 1);
    }

    void squeeze_dims(const U64& axis) {
        assert(prototype.shape[axis] == 1);
        prototype.shape.erase(prototype.shape.begin() + axis);
        prototype.strides.erase(prototype.strides.begin() + axis);
    }
    
 protected:
    TensorPrototypeMetadata prototype;
};

}  // namespace Jetstream

#endif
