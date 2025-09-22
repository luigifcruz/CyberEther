#ifndef JETSTREAM_VIEWPORT_CAPTURE_HH
#define JETSTREAM_VIEWPORT_CAPTURE_HH

#include "jetstream/types.hh"
#include "jetstream/memory/tensor.hh"
#include "jetstream/viewport/adapters/generic.hh"

namespace Jetstream::Viewport {

class FrameCapture {
 public:
    virtual ~FrameCapture() = default;

    virtual Result create(Generic* viewport) = 0;
    virtual Result destroy() = 0;

    virtual Result stop() = 0;
    virtual Result captureFrame() = 0;
    virtual Result getFrameData(Tensor& tensor) = 0;
    virtual Result releaseFrame() = 0;
};

}  // namespace Jetstream::Viewport

#endif  // JETSTREAM_VIEWPORT_CAPTURE_HH
