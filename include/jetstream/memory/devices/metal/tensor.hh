#ifndef JETSTREAM_MEMORY_METAL_TENSOR_HH
#define JETSTREAM_MEMORY_METAL_TENSOR_HH

#include "jetstream/backend/base.hh"
#include "jetstream/memory/devices/base/tensor.hh"

namespace Jetstream {

template<typename T>
class Tensor<Device::Metal, T> : public TensorBase<Device::Metal, T> {
 public:
    using TensorBase<Device::Metal, T>::TensorBase;
    using TensorBase<Device::Metal, T>::operator=;

    Tensor(const TensorBase<Device::Metal, T>& base) : TensorBase<Device::Metal, T>(base) {}

    constexpr const MTL::Buffer* data() const noexcept {
        return this->buffer->data();
    }

    constexpr MTL::Buffer* data() noexcept {
        return this->buffer->data();
    }
};

}  // namespace Jetstream

#endif
