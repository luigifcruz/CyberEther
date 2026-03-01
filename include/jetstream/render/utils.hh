#ifndef JETSTREAM_RENDER_UTILS_HH
#define JETSTREAM_RENDER_UTILS_HH

#include "jetstream/render/base.hh"
#include "jetstream/memory/tensor.hh"

namespace Jetstream {

inline Result ConvertToOptimalStorage(auto& window,
                                      Tensor& tensor,
                                      Tensor& storage) {
    const auto optimalDevice = tensor.hasDevice(window->device()) ? tensor.device() : DeviceType::CPU;

    if (optimalDevice == tensor.device()) {
        storage = tensor.clone();
        const bool enableZeroCopy = storage.device() == window->device();

        JST_TRACE("[RENDER] Tensor Device: {} | Render Device: {} | Optimal Device: {} | Zero-Copy: {}",
                  tensor.device(), window->device(), storage.device(),
                  enableZeroCopy ? "YES" : "NO");

        return Result::SUCCESS;
    }

    JST_CHECK(storage.create(optimalDevice, tensor));

    JST_TRACE("[RENDER] Tensor Device: {} | Render Device: {} | Optimal Device: {} | Zero-Copy: {}",
              tensor.device(), window->device(), storage.device(),
              (storage.device() == window->device()) ? "YES" : "NO");

    return Result::SUCCESS;
}

}  // namespace Jetstream

#endif
