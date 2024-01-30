#ifndef JETSTREAM_MEMORY_BASE_TENSOR_HH
#define JETSTREAM_MEMORY_BASE_TENSOR_HH

#include <vector>
#include <memory>

#include "jetstream/memory/storage.hh"

namespace Jetstream {

template<Device D, typename T>
class Tensor;

template<Device D, typename T>
class TensorBase : public TensorStorage<T> {
 public:
    using DataType = T;

    TensorBase() : TensorStorage<T>() {
        buffer = this->template create_buffer<D>();
    }

    TensorBase(const std::vector<U64>& shape) : TensorStorage<T>(shape) {
        buffer = this->template create_buffer<D>();
    }

    TensorBase(void* ptr, const std::vector<U64>& shape) : TensorStorage<T>(shape) {
        buffer = this->template create_buffer<D>(ptr);
    }

    template<typename... Args>
    TensorBase(const std::vector<U64>& shape, Args... args) : TensorStorage<T>(shape) {
        buffer = this->template create_buffer<D>(args...);
    }

    template<Device RootDevice>
    TensorBase(const TensorBase<RootDevice, T>& other) {
        buffer = this->template clone_buffer<D, RootDevice>(other);
    }

    template<Device RootDevice>
    TensorBase& operator=(TensorBase<RootDevice, T>& other) {
        buffer = this->template clone_buffer<D, RootDevice>(other);
        return *this;
    }

    constexpr Device device() const noexcept {
        return D;
    }

    constexpr bool host_acessible() const noexcept {
        return buffer->host_accessible();
    }

    constexpr bool device_native() const noexcept {
        return buffer->device_native();
    }

    constexpr bool host_native() const noexcept {
        return buffer->host_native();
    }

#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
    Tensor<Device::CPU, T>& cpu() {
        if (!cpu_tensor_cache) {
            cpu_tensor_cache = std::make_shared<Tensor<Device::CPU, T>>(*this);
        }
        return *cpu_tensor_cache;
    }
#endif

#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
    Tensor<Device::Metal, T>& metal() {
        if (!metal_tensor_cache) {
            metal_tensor_cache = std::make_shared<Tensor<Device::Metal, T>>(*this);
        }
        return *metal_tensor_cache;
    }
#endif

#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
    Tensor<Device::Vulkan, T>& vulkan() {
        if (!vulkan_tensor_cache) {
            vulkan_tensor_cache = std::make_shared<Tensor<Device::Vulkan, T>>(*this);
        }
        return *vulkan_tensor_cache;
    }
#endif

#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
    Tensor<Device::CUDA, T>& cuda() {
        if (!cuda_tensor_cache) {
            cuda_tensor_cache = std::make_shared<Tensor<Device::CUDA, T>>(*this);
        }
        return *cuda_tensor_cache;
    }
#endif

 protected:
    std::shared_ptr<TensorBuffer<D>> buffer;

#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
    std::shared_ptr<Tensor<Device::CPU, T>> cpu_tensor_cache;
#endif
#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
    std::shared_ptr<Tensor<Device::Metal, T>> metal_tensor_cache;
#endif
#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
    std::shared_ptr<Tensor<Device::Vulkan, T>> vulkan_tensor_cache;
#endif
#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
    std::shared_ptr<Tensor<Device::CUDA, T>> cuda_tensor_cache;
#endif
};

template<Device CloneDevice>
inline auto& MapOn(auto& tensor) {
#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
    if constexpr (CloneDevice == Device::CPU) {
        return tensor.cpu();
    }
#endif

#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
    if constexpr (CloneDevice == Device::Metal) {
        return tensor.metal();
    }
#endif

#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
    if constexpr (CloneDevice == Device::Vulkan) {
        return tensor.vulkan();
    }
#endif

#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
    if constexpr (CloneDevice == Device::CUDA) {
        return tensor.cuda();
    }
#endif

    JST_ERROR("[TENSOR] Device not supported.");
    JST_CHECK_THROW(Result::ERROR);
}

inline auto& MapOn(const Device& device, auto& tensor) {
    switch (device) {
#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
        case Device::CPU:
            return tensor.cpu();
#endif
#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
        case Device::Metal:
            return tensor.metal();
#endif
#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
        case Device::Vulkan:
            return tensor.vulkan();
#endif
#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
        case Device::CUDA:
            return tensor.cuda();
#endif
        default:
            JST_ERROR("[TENSOR] Device not supported.");
            JST_CHECK_THROW(Result::ERROR);
    }
}

template<Device D, typename T>
class Tensor : public TensorBase<D, T> {};

template <typename T>
struct IsTensor : std::false_type {};

template <Device D, typename T>
struct IsTensor<Tensor<D, T>> : std::true_type {};

}  // namespace Jetstream

#endif
