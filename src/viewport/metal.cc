#include "jetstream/viewport/adapters/metal.hh"

#include <utility>

namespace Jetstream::Viewport {

void Adapter<DeviceType::Metal>::setCaptureCallback(CaptureCallback callback) {
    setCaptureEnabled(static_cast<bool>(callback));
    captureCallback = std::move(callback);
}

Result Adapter<DeviceType::Metal>::encodeCapture(MTL::CommandBuffer* commandBuffer, MTL::Texture* texture) {
    if (captureCallback) {
        return captureCallback(commandBuffer, texture);
    }
    return Result::SUCCESS;
}

void Adapter<DeviceType::Metal>::setCaptureEnabled(bool) {}

}  // namespace Jetstream::Viewport
