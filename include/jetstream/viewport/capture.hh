#ifndef JETSTREAM_VIEWPORT_CAPTURE_HH
#define JETSTREAM_VIEWPORT_CAPTURE_HH

#include "jetstream/types.hh"
#include "jetstream/memory/tensor.hh"
#include "jetstream/viewport/adapters/generic.hh"

#include <array>

namespace Jetstream::Viewport {

class FrameCapture {
 public:
    enum class Export {
        None,
        OpaqueFd,
        DmaBuf,
        D3D11,
    };

    struct SharedDevice {
        U64 luid = 0;
        U32 render = 0;
        std::array<U8, 16> uuid = {};
    };

    struct SharedFrame {
        U64 handle = 0;
        U64 pitch = 0;
        U64 size = 0;
    };

    virtual ~FrameCapture() = default;

    virtual Result create(Generic* viewport, const DeviceType& outputDevice = DeviceType::CPU) = 0;
    virtual Result destroy() = 0;

    virtual Result stop() = 0;
    virtual Result captureFrame() = 0;
    virtual Result getFrameData(Tensor& tensor) = 0;
    virtual Result getSharedFrame(SharedFrame&) { return Result::ERROR; }
    virtual Result releaseFrame() = 0;
};

}  // namespace Jetstream::Viewport

#endif  // JETSTREAM_VIEWPORT_CAPTURE_HH
