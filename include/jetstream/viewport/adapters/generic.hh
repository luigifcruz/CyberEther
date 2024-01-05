#ifndef JETSTREAM_ADAPTER_GENERIC_HH
#define JETSTREAM_ADAPTER_GENERIC_HH

#include "jetstream/logger.hh"
#include "jetstream/types.hh"
#include "jetstream/macros.hh"
#include "jetstream/parser.hh"
#include "jetstream/render/types.hh"
#include "jetstream/render/tools/imgui.h"

namespace Jetstream::Viewport {

struct Config {
    /// @brief Whether vertical synchronization (vsync) is enabled. Only applicable to headed viewports.
    bool vsync = true;

    /// @brief The title of the application or window.
    std::string title = "CyberEther";

    /// @brief The size of the application or window.
    Size2D<U64> size = {1920, 1080};

    /// @brief The framerate of the headless viewport.
    U64 framerate = 60;

    /// @brief The endpoint of the headless viewport.
    std::string endpoint = "/tmp/cyberether";

    /// @brief The video codec of the headless viewport.
    Render::VideoCodec codec = Render::VideoCodec::FFV1;

    /// @brief Whether hardware acceleration is enabled.
    bool hardwareAcceleration = true;

    JST_SERDES(vsync, title, size, framerate, endpoint, codec, hardwareAcceleration);
};

class Generic {
 public:
    explicit Generic(const Config& config);   
    virtual ~Generic() = default;

    virtual std::string id() const = 0;
    virtual std::string name() const = 0;
    virtual Device device() const = 0;

    virtual Result create() = 0;
    virtual Result destroy() = 0;

    virtual Result createImgui() = 0;
    virtual Result destroyImgui() = 0;

    virtual F32 scale(const F32& scale) const = 0;

    virtual Result pollEvents() = 0;
    virtual bool keepRunning() = 0;

    Result addMousePosEvent(F32 x, F32 y);
    Result addMouseButtonEvent(U64 button, bool down);

 protected:
    Config config;
};

template<Device DeviceId>
class Adapter : public Generic {};

}  // namespace Jetstream::Viewport

#endif
