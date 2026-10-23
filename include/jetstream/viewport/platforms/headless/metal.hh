#ifndef JETSTREAM_VIEWPORT_PLATFORM_HEADLESS_METAL_HH
#define JETSTREAM_VIEWPORT_PLATFORM_HEADLESS_METAL_HH

#include <chrono>

#include "jetstream/viewport/adapters/metal.hh"
#include "jetstream/viewport/platforms/headless/generic.hh"

namespace Jetstream::Viewport {

template<>
class Headless<DeviceType::Metal> : public Adapter<DeviceType::Metal> {
 public:
    explicit Headless(const Config& config);
    virtual ~Headless();

    std::string id() const {
        return "headless";
    }

    std::string name() const {
        return "Headless (Metal)";
    }

    constexpr DeviceType device() const {
        return DeviceType::Metal;
    }

    Result create();
    Result destroy();

    Result createImgui();
    Result destroyImgui();
    F32 scale(const F32& scale) const;

    MTL::Texture* nextDrawable(CA::MetalDrawable*& drawable);
    Extent2D<U64> getSwapchainExtent() const;

    Result waitEvents();
    Result pollEvents();
    bool keepRunning();

 private:
    MTL::Texture* texture = nullptr;
    std::chrono::steady_clock::time_point lastTime;
};

}  // namespace Jetstream::Viewport

#endif  // JETSTREAM_VIEWPORT_PLATFORM_HEADLESS_METAL_HH
