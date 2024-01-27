#ifndef JETSTREAM_MEMORY_CUDA_TENSOR_HH
#define JETSTREAM_MEMORY_CUDA_TENSOR_HH

#include "jetstream/backend/base.hh"
#include "jetstream/memory/devices/base/tensor.hh"

namespace Jetstream {

template<typename T>
class Tensor<Device::CUDA, T> : public TensorBase<Device::CUDA, T> {
 public:
    using TensorBase<Device::CUDA, T>::TensorBase;
    using TensorBase<Device::CUDA, T>::operator=;

    Tensor(const TensorBase<Device::CUDA, T>& base) : TensorBase<Device::CUDA, T>(base) {}

    constexpr const bool& device_native() const noexcept {
        return this->buffer->device_native();
    }

    constexpr const bool& host_native() const noexcept {
        return this->buffer->host_native();
    }

    constexpr const void* data() const noexcept {
        // TODO: This disregards the offset.
        return this->buffer->data();
    }

    constexpr void* data() noexcept {
        // TODO: This disregards the offset.
        return this->buffer->data();
    }

    constexpr const void* data_ptr() const noexcept {
        return this->buffer->data_ptr();
    }

    constexpr void* data_ptr() noexcept {
        return this->buffer->data_ptr();
    }
};

}  // namespace Jetstream

#endif
