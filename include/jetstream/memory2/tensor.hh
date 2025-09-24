#ifndef JETSTREAM_MEMORY2_TENSOR_HH
#define JETSTREAM_MEMORY2_TENSOR_HH

#include <initializer_list>
#include <memory>
#include <type_traits>
#include <vector>

#include "jetstream/types.hh"
#include "jetstream/memory2/buffer.hh"
#include "jetstream/memory2/token.hh"
#include "jetstream/memory2/types.hh"

namespace Jetstream::mem2 {

class Tensor;

template<typename T>
class View;

class Tensor : public std::enable_shared_from_this<Tensor> {
 public:
    Tensor();
    Tensor(const Device& device, const DataType& dtype, const Shape& shape);
    Tensor(const Device& device, const Tensor& source);

    Tensor(const Tensor& other) = default;
    Tensor(Tensor&& other) noexcept = default;
    Tensor& operator=(const Tensor& other) = default;
    Tensor& operator=(Tensor&& other) noexcept = default;

    ~Tensor();

    Result create(const Device& device, const DataType& dtype, const Shape& shape);
    Result create(const Device& device, const Tensor& source);

    Result has_device(const Device& device);
    Result copy_from(const Tensor& source);

    const Device& device() const;
    const Device& native_device() const;
    const DataType& dtype() const;

    const U64& size() const;
    const U64& size_bytes() const;
    const U64& element_size() const;

    bool contiguous() const;
    bool empty() const;
    bool valid_shape() const;

    const U64& offset() const;
    const U64& offset_bytes() const;

    const Shape& shape() const;
    const Shape& stride() const;
    const Shape& shape_minus_one() const;
    const Shape& backstride() const;

    const U64& shape(Index idx) const;
    const U64& stride(Index idx) const;

    Index rank() const;
    Index ndims() const;

    U64 shape_to_offset(const std::initializer_list<U64>& coordinates) const;

    Result expand_dims(Index axis);
    Result squeeze_dims(Index axis);
    Result reshape(const Shape& new_shape);
    Result broadcast_to(const Shape& new_shape);
    Result slice(const std::vector<Token>& tokens);

    void* data();
    const void* data() const;

    const Buffer& buffer() const;
    Buffer& buffer();

    const Index& id() const;

 private:
    struct Impl;
    std::shared_ptr<Impl> impl;

    void ensure_impl();
};

template<typename T>
class View {
 public:
    explicit View(Tensor& tensor) : tensor_(&tensor) {}
    explicit View(const Tensor& tensor) : tensor_(const_cast<Tensor*>(&tensor)) {}

    T& operator[](std::initializer_list<U64> indices) {
        return data()[offset(indices)];
    }

    const std::remove_const_t<T>& operator[](std::initializer_list<U64> indices) const {
        return data()[offset(indices)];
    }

    Tensor& tensor() {
        return *tensor_;
    }

    const Tensor& tensor() const {
        return *tensor_;
    }

 private:
    using value_type = std::remove_const_t<T>;

    value_type* data() {
        return static_cast<value_type*>(tensor_->data());
    }

    const value_type* data() const {
        return static_cast<const value_type*>(tensor_->data());
    }

    U64 offset(std::initializer_list<U64> indices) const {
        if (indices.size() != tensor_->rank()) {
            throw std::invalid_argument("Invalid number of indices.");
        }
        return tensor_->shape_to_offset(indices);
    }

    Tensor* tensor_ = nullptr;
};

}  // namespace Jetstream::mem2

#endif  // JETSTREAM_MEMORY2_TENSOR_HH
