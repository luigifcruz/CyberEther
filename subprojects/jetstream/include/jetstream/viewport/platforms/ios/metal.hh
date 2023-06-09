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

    const std::string name() const {
        return "iOS (Native)";
    }

    constexpr Device device() const {
        return Device::Metal;
    };

    Result create();
    Result destroy();

    Result createImgui();
    Result destroyImgui();

    void* nextDrawable();
    
    Result pollEvents();
    bool keepRunning();

    static std::shared_ptr<iOS> Factory(const Config& config, 
                                        CA::MetalLayer* layer) {
        return std::make_shared<iOS>(config, layer);
    }

 private:
    CA::MetalLayer* swapchain;
};

}  // namespace Jetstream::Viewport

#endif
