#ifndef JETSTREAM_VIEWPORT_PLATFORM_IOS_METAL_HH
#define JETSTREAM_VIEWPORT_PLATFORM_IOS_METAL_HH

#include "jetstream/viewport/adapters/metal.hh"
#include "jetstream/viewport/platforms/ios/generic.hh"

namespace Jetstream::Viewport {

template<>
class iOS<DeviceType::Metal> : public Adapter<DeviceType::Metal> {
 public:
    explicit iOS(const Config& config, CA::MetalLayer* layer);
    virtual ~iOS();

    std::string id() const {
        return "ios";
    }

    std::string name() const {
        return "iOS (Native)";
    }

    constexpr DeviceType device() const {
        return DeviceType::Metal;
    };

    Result create();
    Result destroy();

    Extent2D<F32> displaySize() const;
    Extent2D<F32> framebufferSize() const;

    MTL::Texture* nextDrawable(CA::MetalDrawable*& drawable);
    Extent2D<U64> getSwapchainExtent() const;
    
    Result waitEvents();
    Result pollEvents();
    bool keepRunning();

 private:
    void setCaptureEnabled(bool enabled);

    CA::MetalLayer* swapchain;
};

}  // namespace Jetstream::Viewport

#endif
