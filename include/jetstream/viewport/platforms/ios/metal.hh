#ifndef JETSTREAM_VIEWPORT_PLATFORM_IOS_METAL_HH
#define JETSTREAM_VIEWPORT_PLATFORM_IOS_METAL_HH

#include "jetstream/viewport/adapters/metal.hh"
#include "jetstream/viewport/platforms/ios/generic.hh"

namespace Jetstream::Viewport {

template<>
class iOS<Device::Metal> : public Adapter<Device::Metal> {
 public:
    explicit iOS(const Config& config, CA::MetalLayer* layer);
    virtual ~iOS();

    std::string_view prettyName() const {
        return "iOS (Native)";
    }

    std::string_view name() const {
        return "ios";
    }

    constexpr Device device() const {
        return Device::Metal;
    };

    Result create();
    Result destroy();

    Result createImgui();
    Result destroyImgui();
    F32 scale(const F32& scale);

    void* nextDrawable();
    
    Result pollEvents();
    bool keepRunning();

 private:
    CA::MetalLayer* swapchain;
};

}  // namespace Jetstream::Viewport

#endif
