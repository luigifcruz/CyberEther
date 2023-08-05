#ifndef JETSTREAM_INTERFACE_HH
#define JETSTREAM_INTERFACE_HH

#include "jetstream/types.hh"
#include "jetstream/macros.hh"
#include "jetstream/logger.hh"
#include "jetstream/parser.hh"

namespace Jetstream {

class JETSTREAM_API Interface {
 public:
    struct Config {
        F32 nodeWidth = 0.0f;
        bool viewEnabled = false;
        bool previewEnabled = false;
        Size2D<F32> nodePos = {0.0f, 0.0f};

        JST_SERDES(
            JST_SERDES_VAL("nodeWidth", nodeWidth);
            JST_SERDES_VAL("viewEnabled", viewEnabled);
            JST_SERDES_VAL("previewEnabled", previewEnabled);
            JST_SERDES_VAL("nodePos", nodePos);
        );
    };

    virtual constexpr Device device() const = 0;
    virtual constexpr std::string name() const = 0;
    virtual constexpr std::string prettyName() const = 0;

    virtual void drawPreview(const F32&) {}
    virtual constexpr bool shouldDrawPreview() const {
        return false;
    }

    virtual void drawView() {}
    virtual constexpr bool shouldDrawView() const {
        return false;
    }
    
    virtual void drawControl() {}
    virtual constexpr bool shouldDrawControl() const {
        return false;
    }

    virtual void drawInfo() {}
    virtual constexpr bool shouldDrawInfo() const {
        return false;
    }

 protected:
    Config config;

    friend class Instance;
};

}  // namespace Jetstream

#endif
