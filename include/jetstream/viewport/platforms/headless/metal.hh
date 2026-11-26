#ifndef JETSTREAM_VIEWPORT_PLATFORM_HEADLESS_METAL_HH
#define JETSTREAM_VIEWPORT_PLATFORM_HEADLESS_METAL_HH

#include <atomic>
#include <chrono>
#include <mutex>

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

    Extent2D<F32> displaySize() const;
    Extent2D<F32> framebufferSize() const;

    MTL::Texture* nextDrawable(CA::MetalDrawable*& drawable);
    Extent2D<U64> getSwapchainExtent() const;

    Result waitEvents();
    Result pollEvents();
    bool keepRunning();
    bool retainedPresent() const;
    Result resize(const Extent2D<U64>& size, F32 scale);

 private:
    Result createTexture(const Extent2D<U64>& size);

    MTL::Texture* texture = nullptr;
    std::atomic<bool> running{false};
    Extent2D<U64> extent = {};
    F32 scale = 1.0f;
    Extent2D<U64> requestedExtent = {};
    F32 requestedScale = 1.0f;
    mutable std::mutex extentMutex;
    std::chrono::steady_clock::time_point lastTime;
};

}  // namespace Jetstream::Viewport

#endif  // JETSTREAM_VIEWPORT_PLATFORM_HEADLESS_METAL_HH
