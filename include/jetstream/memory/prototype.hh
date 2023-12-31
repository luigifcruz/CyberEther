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

    constexpr const std::vector<U64>& shape_minus_one() const noexcept {
        return prototype.shape_minus_one;
    }

    constexpr const std::vector<U64>& backstride() const noexcept {
        return prototype.backstride;
    }

    constexpr const U64& hash() const noexcept {
        return prototype.hash;
    }

    constexpr const std::vector<U64>& shape() const noexcept {
        return prototype.shape;
    }

    constexpr const std::vector<U64>& stride() const noexcept {
        return prototype.stride;
    }

    constexpr const U64& shape(const U64& idx) const noexcept {
        return prototype.shape[idx];
    }

    constexpr const U64& stride(const U64& idx) const noexcept {
        return prototype.stride[idx];
    }

    constexpr bool empty() const noexcept {
        return prototype.size == 0;
    }

    constexpr U64 rank() const noexcept {
        return prototype.shape.size();
    }

    constexpr U64 ndims() const noexcept {
        return prototype.shape.size();
    }

    constexpr bool valid_shape() const noexcept {
        return prototype.size > 0;
    }

    constexpr const Locale& locale() const noexcept {
        return prototype.locale;
    }

    constexpr void set_locale(const Locale& locale) noexcept {
        prototype.locale = locale;
    }

    constexpr bool operator==(const TensorPrototype& other) const noexcept {
        return prototype.hash == other.prototype.hash;
    }

    constexpr bool operator!=(const TensorPrototype& other) const noexcept {
        return prototype.hash != other.prototype.hash;
    }

    // TODO: Move functions to source file.

    constexpr U64 shape_to_offset(const std::vector<U64>& shape) const {
        U64 index = prototype.offset;
        U64 pad = shape.size() - prototype.stride.size();
        for (U64 i = 0; i < prototype.stride.size(); i++) {
            // TODO: This is a hack. This should be done by modifiying the stride.
            index += ((shape[pad + i] >= prototype.shape[i]) ? 0 : shape[pad + i]) * prototype.stride[i];
        }
        return index;
    }

    constexpr void offset_to_shape(U64 index, std::vector<U64>& shape) const {
        for (U64 i = 0; i < prototype.stride.size(); i++) {
            shape[i] = index / prototype.stride[i];
            index -= shape[i] * prototype.stride[i];
        }
    }

    void expand_dims(const U64& axis) {
        prototype.shape.insert(prototype.shape.begin() + axis, 1);
        prototype.stride.insert(prototype.stride.begin() + axis, 1);
        update_cache();
    }

    void squeeze_dims(const U64& axis) {
        assert(prototype.shape[axis] == 1);
        prototype.shape.erase(prototype.shape.begin() + axis);
        prototype.stride.erase(prototype.stride.begin() + axis);
        update_cache();
    }

    // TODO: Add permutation() function.

    void view(const std::vector<Token>& tokens) {
        std::vector<U64> shape;
        std::vector<U64> stride;
        U64 offset = 0;
        U64 dim = 0;
        bool ellipsis_used = false;

        for (const auto& token : tokens) {
            switch (token.get_type()) {
                case Token::Type::Number: {
                    if (dim >= prototype.shape.size()) {
                        throw std::runtime_error("Index exceeds array dimensions.");
                    }
                    offset += token.get_a() * prototype.stride[dim];
                    dim++;
                    break;
                }
                case Token::Type::Colon: {
                    if (dim >= prototype.shape.size()) {
                        throw std::runtime_error("Index exceeds array dimensions.");
                    }
                    const U64 start = token.get_a();
                    U64 end = token.get_b();
                    const U64 step = token.get_c();

                    if (end == 0) {
                        end = prototype.shape[dim];
                    }

                    shape.push_back((end - start + step - 1) / step);
                    stride.push_back(prototype.stride[dim] * step);
                    offset += start * prototype.stride[dim];
                    dim++;
                    break;
                }
                case Token::Type::Ellipsis: {
                    if (ellipsis_used) {
                        throw std::runtime_error("Ellipsis used more than once.");
                    }
                    ellipsis_used = true;
                    const U64 remaining_dims = prototype.shape.size() - (tokens.size() - 1) + 1;
                    while (dim < remaining_dims) {
                        shape.push_back(prototype.shape[dim]);
                        stride.push_back(prototype.stride[dim]);
                        dim++;
                    }
                    break;
                }
            }
        }

        if (!ellipsis_used) {
            while (dim < prototype.shape.size()) {
                shape.push_back(prototype.shape[dim]);
                stride.push_back(prototype.stride[dim]);
                dim++;
            }
        }

        JST_TRACE("[MEMORY] View shape: {} -> {}.", prototype.shape, shape);
        JST_TRACE("[MEMORY] View stride: {} -> {}.", prototype.stride, stride);
        JST_TRACE("[MEMORY] View offset: {}.", offset);

        prototype.shape = shape;
        prototype.stride = stride;
        prototype.offset = offset;
        prototype.contiguous = false;

        update_cache();

        return Result::SUCCESS;
    }
    
 protected:
    TensorPrototypeMetadata prototype;

    void initialize(const std::vector<U64>& shape, const U64& element_size) {
        prototype.shape = shape;
        prototype.element_size = element_size;

        prototype.stride.resize(prototype.shape.size());
        for (U64 i = 0; i < prototype.shape.size(); i++) {
            prototype.stride[i] = 1;
            for (U64 j = i + 1; j < prototype.shape.size(); j++) {
                prototype.stride[i] *= prototype.shape[j];
            }
        }

        assert(prototype.stride.size() == prototype.shape.size());

        prototype.hash = std::rand() + 1;
        prototype.contiguous = true;

        update_cache();
    }

    void update_cache() {
        prototype.backstride.resize(prototype.shape.size());
        prototype.shape_minus_one.resize(prototype.shape.size());

        for (U64 i = 0; i < prototype.shape.size(); i++) {
            prototype.shape_minus_one[i] = prototype.shape[i] - 1;
            prototype.backstride[i] = prototype.stride[i] * prototype.shape_minus_one[i];
        }

        prototype.size = 1;
        for (const auto& dim : prototype.shape) {
            prototype.size *= dim;
        }

        prototype.size_bytes = prototype.size * 
                               prototype.element_size;
    }
};

}  // namespace Jetstream

#endif
