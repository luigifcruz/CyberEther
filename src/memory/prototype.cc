#include <cassert>

#include "jetstream/memory/prototype.hh" 

namespace Jetstream {

void TensorPrototype::initialize(const std::vector<U64>& shape, const U64& element_size) {
    prototype.element_size = element_size;

    if (shape.empty()) {
        return;
    }

    prototype.shape = shape;

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

void TensorPrototype::update_cache() {
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

    prototype.offset_bytes = prototype.offset * 
                             prototype.element_size;
}

const U64& TensorPrototype::shape(const U64& idx) const noexcept {
    return prototype.shape[idx];
}

const U64& TensorPrototype::stride(const U64& idx) const noexcept {
    return prototype.stride[idx];
}

U64 TensorPrototype::rank() const noexcept {
    return prototype.shape.size();
}

U64 TensorPrototype::ndims() const noexcept {
    return prototype.shape.size();
}

void TensorPrototype::set_locale(const Locale& locale) noexcept {
    prototype.locale = locale;
}

U64 TensorPrototype::shape_to_offset(const std::vector<U64>& shape) const {
    U64 index = prototype.offset;
    U64 pad = shape.size() - prototype.stride.size();
    for (U64 i = 0; i < prototype.stride.size(); i++) {
        index += shape[pad + i] * prototype.stride[i];
    }
    return index;
}

void TensorPrototype::offset_to_shape(U64 index, std::vector<U64>& shape) const {
    index -= prototype.offset;
    for (U64 i = 0; i < prototype.stride.size(); i++) {
        shape[i] = index / prototype.stride[i];
        index -= shape[i] * prototype.stride[i];
    }
}

void TensorPrototype::expand_dims(const U64& axis) {
    prototype.shape.insert(prototype.shape.begin() + axis, 1);
    const U64& stride = (axis == 0) ? prototype.stride[0] : prototype.stride[axis - 1];
    prototype.stride.insert(prototype.stride.begin() + axis, stride);
    update_cache();
}

void TensorPrototype::squeeze_dims(const U64& axis) {
    assert(prototype.shape[axis] == 1);
    prototype.shape.erase(prototype.shape.begin() + axis);
    prototype.stride.erase(prototype.stride.begin() + axis);
    update_cache();
}

Result TensorPrototype::permutation(const std::vector<U64>&) {
    // TODO: Implement permutation.
    throw std::runtime_error("Not implemented");
    return Result::SUCCESS;
}

Result TensorPrototype::reshape(const std::vector<U64>& shape) {
    if (shape.empty()) {
        JST_ERROR("[MEMORY] Cannot reshape to empty shape.");
        return Result::ERROR;
    }

    if (!prototype.contiguous) {
        // TODO: Implement reshape for non-contiguous tensors.
        JST_ERROR("[MEMORY] Cannot reshape non-contiguous tensor.");
        return Result::ERROR;
    }

    const U64& og_size = prototype.size;

    U64 new_size = 1;
    for (const auto& dim : shape) {
        if (dim == 0) {
            JST_ERROR("[MEMORY] Cannot reshape to shape with zero dimension.");
            return Result::ERROR;
        }
        new_size *= dim;
    }

    if (og_size != new_size) {
        JST_ERROR("[MEMORY] Cannot reshape from size {} to size {}.", og_size, new_size);
        return Result::ERROR;
    }

    prototype.shape = shape;

    prototype.stride.resize(prototype.shape.size());
    for (U64 i = 0; i < prototype.shape.size(); i++) {
        prototype.stride[i] = 1;
        for (U64 j = i + 1; j < prototype.shape.size(); j++) {
            prototype.stride[i] *= prototype.shape[j];
        }
    }

    update_cache();

    return Result::SUCCESS;
}

Result TensorPrototype::broadcast_to(const std::vector<U64>& shape) {
    if (shape.size() < prototype.shape.size()) {
        JST_ERROR("[MEMORY] Cannot broadcast shape: {} -> {}.", prototype.shape, shape);
        return Result::ERROR;
    }

    if (shape.size() > prototype.shape.size()) {
        for (U64 i = 0; i < shape.size() - prototype.shape.size(); i++) {
            expand_dims(0);
        }
    }

    bool contiguous = prototype.contiguous;
    std::vector<U64> new_shape(shape.size());
    std::vector<U64> new_stride(shape.size());

    for (U64 i = 0; i < shape.size(); i++) {
        if (prototype.shape[i] != shape[i]) {
            if (prototype.shape[i] == 1) {
                new_shape[i] = shape[i];
                new_stride[i] = 0;
            } else if (shape[i] == 1) {
                new_shape[i] = prototype.shape[i];
                new_stride[i] = prototype.stride[i];
            } else {
                JST_ERROR("[MEMORY] Cannot broadcast shape: {} -> {}.", prototype.shape, shape);
                return Result::ERROR;
            }
        } else {
            new_shape[i] = prototype.shape[i];
            new_stride[i] = prototype.stride[i];
        }
        contiguous &= new_stride[i] != 0;
    }

    JST_TRACE("[MEMORY] Broadcast shape: {} -> {}.", prototype.shape, new_shape);
    JST_TRACE("[MEMORY] Broadcast stride: {} -> {}.", prototype.stride, new_stride);
    JST_TRACE("[MEMORY] Broadcast contiguous: {} -> {}.", prototype.contiguous, contiguous);

    prototype.shape = new_shape;
    prototype.stride = new_stride;
    prototype.contiguous = contiguous;

    update_cache();

    return Result::SUCCESS;
}

Result TensorPrototype::slice(const std::vector<Token>& slice) {
    std::vector<U64> shape;
    std::vector<U64> stride;
    U64 offset = 0;
    U64 dim = 0;
    bool ellipsis_used = false;

    for (const auto& token : slice) {
        switch (token.get_type()) {
            case Token::Type::Number: {
                if (dim >= prototype.shape.size()) {
                    JST_ERROR("[MEMORY] Index exceeds array dimensions.");
                    return Result::ERROR;
                }

                const U64 index = token.get_a();
                if (index >= prototype.shape[dim]) {
                    JST_ERROR("[MEMORY] Index exceeds array dimensions.");
                    return Result::ERROR;
                }

                offset += index * prototype.stride[dim];
                dim++;
                break;
            }
            case Token::Type::Colon: {
                if (dim >= prototype.shape.size()) {
                    JST_ERROR("[MEMORY] Index exceeds array dimensions.");
                    return Result::ERROR;
                }

                const U64 start = token.get_a();
                U64 end = token.get_b();
                const U64 step = token.get_c();

                if (end == 0) {
                    end = prototype.shape[dim];
                }

                if (step == 0) {
                    JST_ERROR("[MEMORY] Slice step cannot be zero.");
                    return Result::ERROR;
                }

                if (start >= prototype.shape[dim] || end > prototype.shape[dim]) {
                    JST_ERROR("[MEMORY] Slice index exceeds array dimensions.");
                    return Result::ERROR;
                }

                if (start >= end) {
                    JST_ERROR("[MEMORY] Slice start index must be less than end index.");
                    return Result::ERROR;
                }

                shape.push_back((end - start + step - 1) / step);
                stride.push_back(prototype.stride[dim] * step);
                offset += start * prototype.stride[dim];
                dim++;
                break;
            }
            case Token::Type::Ellipsis: {
                if (ellipsis_used) {
                    JST_ERROR("[MEMORY] Ellipsis used more than once.");
                    return Result::ERROR;
                }
                ellipsis_used = true;

                const U64 remaining_dims = prototype.shape.size() - (slice.size() - 1);
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

    JST_TRACE("[MEMORY] Slice shape: {} -> {}.", prototype.shape, shape);
    JST_TRACE("[MEMORY] Slice stride: {} -> {}.", prototype.stride, stride);
    JST_TRACE("[MEMORY] Slice offset: {}.", offset);

    prototype.shape = shape;
    prototype.stride = stride;
    prototype.offset = offset;
    // TODO: Actually check if contiguous.
    prototype.contiguous = true;

    update_cache();

    return Result::SUCCESS;
}

}  // namespace Jetstream