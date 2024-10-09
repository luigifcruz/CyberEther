#ifndef JETSTREAM_VIEWPORT_PLATFORM_IOS_METAL_HH
#define JETSTREAM_VIEWPORT_PLATFORM_IOS_METAL_HH

#include <chrono>

#include "jetstream/viewport/adapters/metal.hh"
#include "jetstream/viewport/platforms/ios/generic.hh"

namespace Jetstream::Viewport {

template<>
class iOS<Device::Metal> : public Adapter<Device::Metal> {
 public:
    explicit iOS(const Config& config, CA::MetalLayer* layer);
    virtual ~iOS();

    std::string id() const {
        return "ios";
    }

    std::string name() const {
        return "iOS (Native)";
    }

    constexpr Device device() const {
        return Device::Metal;
    };

    Result create();
    Result destroy();

    Result createImgui();
    Result destroyImgui();
    F32 scale(const F32& scale) const;

    void* nextDrawable();
    
    Result waitEvents();
    Result pollEvents();
    bool keepRunning();

 private:
    CA::MetalLayer* swapchain;

    std::chrono::high_resolution_clock::time_point lastTime;
};

}  // namespace Jetstream::Viewport

#endif
