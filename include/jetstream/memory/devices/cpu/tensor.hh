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

    const T* data() const noexcept {
        return static_cast<T*>(this->buffer->data());
    }

    T* data() noexcept {
        return static_cast<T*>(this->buffer->data());
    }

    const T& operator[](const U64& idx) const noexcept {
        return data()[idx];
    }

    T& operator[](const U64& idx) noexcept {
        return data()[idx];
    }

    const T& operator[](const std::vector<U64>& idx) const noexcept {
        return data()[this->shapeToOffset(idx)];
    }

    T& operator[](const std::vector<U64>& idx) noexcept {
        return data()[this->shapeToOffset(idx)];
    }

    auto begin() {
        return data();
    }

    auto end() {
        return data() + this->size();
    }

    auto begin() const {
        return data();
    }

    auto end() const {
        return data() + this->size();
    }
};

}  // namespace Jetstream

#endif
