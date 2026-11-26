#ifndef JETSTREAM_VIEWPORT_CAPTURE_VULKAN_HH
#define JETSTREAM_VIEWPORT_CAPTURE_VULKAN_HH

#include "jetstream/viewport/capture.hh"

#include <memory>

namespace Jetstream::Viewport {

class FrameCaptureVulkan : public FrameCapture {
 public:
    explicit FrameCaptureVulkan(Export exporting = Export::None);
    ~FrameCaptureVulkan() override;

    static bool SupportsExport(Export exporting);
    static SharedDevice Device();

    Result create(Generic* viewport, const DeviceType& outputDevice = DeviceType::CPU) override;
    Result destroy() override;
    Result stop() override;
    Result captureFrame() override;
    Result getFrameData(Tensor& tensor) override;
    Result getSharedFrame(SharedFrame& frame) override;
    Result releaseFrame() override;
    Extent2D<U64> extent() const override;

 private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

}  // namespace Jetstream::Viewport

#endif  // JETSTREAM_VIEWPORT_CAPTURE_VULKAN_HH
