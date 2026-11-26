#ifndef JETSTREAM_VIEWPORT_CAPTURE_METAL_HH
#define JETSTREAM_VIEWPORT_CAPTURE_METAL_HH

#include "jetstream/viewport/capture.hh"

#include <functional>
#include <memory>

namespace Jetstream::Viewport {

class FrameCaptureMetal : public FrameCapture {
 public:
    using Gate = std::function<bool(bool available)>;

    explicit FrameCaptureMetal(Gate gate = {});
    ~FrameCaptureMetal() override;

    Result create(Generic* viewport, const DeviceType& outputDevice = DeviceType::CPU) override;
    Result destroy() override;
    Result stop() override;
    Result captureFrame() override;
    Result getFrameData(Tensor& tensor) override;
    Result releaseFrame() override;

 private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

}  // namespace Jetstream::Viewport

#endif  // JETSTREAM_VIEWPORT_CAPTURE_METAL_HH
