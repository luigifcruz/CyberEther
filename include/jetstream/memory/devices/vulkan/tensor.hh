#ifndef JETSTREAM_MEMORY_VULKAN_TENSOR_HH
#define JETSTREAM_MEMORY_VULKAN_TENSOR_HH

#include "jetstream/backend/base.hh"
#include "jetstream/memory/devices/base/tensor.hh"

namespace Jetstream {

template<typename T>
class Tensor<Device::Vulkan, T> : public TensorBase<Device::Vulkan, T> {
 public:
    using TensorBase<Device::Vulkan, T>::TensorBase;
    using TensorBase<Device::Vulkan, T>::operator=;

    Tensor(const TensorBase<Device::Vulkan, T>& base) : TensorBase<Device::Vulkan, T>(base) {}

    const VkBuffer& data() const noexcept {
        return this->buffer->data();
    }

    VkBuffer& data() noexcept {
        return this->buffer->data();
    }
};

}  // namespace Jetstream

#endif
