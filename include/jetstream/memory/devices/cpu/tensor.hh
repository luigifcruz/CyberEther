#ifndef JETSTREAM_MEMORY_CPU_TENSOR_HH
#define JETSTREAM_MEMORY_CPU_TENSOR_HH

#include <vector>

#include "jetstream/memory/devices/base/tensor.hh"

namespace Jetstream {

template<typename T>
class Tensor<Device::CPU, T> : public TensorBase<Device::CPU, T> {
 public:
    using TensorBase<Device::CPU, T>::TensorBase;
    using TensorBase<Device::CPU, T>::operator=;

    Tensor(const TensorBase<Device::CPU, T>& base) : TensorBase<Device::CPU, T>(base) {}

    constexpr const T* data() const noexcept {
        // TODO: This disregards the offset.
        return reinterpret_cast<T*>(this->buffer->data());
    }

    constexpr T* data() noexcept {
        // TODO: This disregards the offset.
        return reinterpret_cast<T*>(this->buffer->data());
    }

    constexpr const T& operator[](const U64& idx) const noexcept {
        return data()[idx];
    }

    constexpr T& operator[](const U64& idx) noexcept {
        return data()[idx];
    }

    constexpr const T& operator[](const std::vector<U64>& idx) const noexcept {
        return data()[this->shape_to_offset(idx)];
    }

    constexpr T& operator[](const std::vector<U64>& idx) noexcept {
        return data()[this->shape_to_offset(idx)];
    }

    constexpr auto begin() {
        return data();
    }

    constexpr auto end() {
        return data() + this->size();
    }

    constexpr auto begin() const {
        return data();
    }

    constexpr auto end() const {
        return data() + this->size();
    }
};

}  // namespace Jetstream

#endif
