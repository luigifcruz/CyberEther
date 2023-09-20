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
    bool vsync = true;
    std::string title = "Render";
    Size2D<U64> size = {1280, 720};

    JST_SERDES(
        JST_SERDES_VAL("vsync", vsync);
        JST_SERDES_VAL("title", title);
        JST_SERDES_VAL("size", size);
    );
};

class Generic {
 public:
    explicit Generic(const Config& config);   
    virtual ~Generic() = default;

    virtual std::string_view prettyName() const = 0;
    virtual std::string_view name() const = 0;
    virtual Device device() const = 0;

    virtual Result create() = 0;
    virtual Result destroy() = 0;

    virtual Result createImgui() = 0;
    virtual Result destroyImgui() = 0;
    virtual F32 calculateScale(const F32& scale) = 0;

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
