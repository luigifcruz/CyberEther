#ifndef JETSTREAM_VIEWPORT_CAPTURE_VULKAN_HH
#define JETSTREAM_VIEWPORT_CAPTURE_VULKAN_HH

#include "jetstream/viewport/capture.hh"

#include <memory>

namespace Jetstream::Viewport {

class FrameCaptureVulkan : public FrameCapture {
 public:
    FrameCaptureVulkan();
    ~FrameCaptureVulkan() override;

    Result create(Generic* viewport) override;
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

#endif  // JETSTREAM_VIEWPORT_CAPTURE_VULKAN_HH
