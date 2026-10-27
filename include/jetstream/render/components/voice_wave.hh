#ifndef JETSTREAM_RENDER_COMPONENTS_VOICE_WAVE_HH
#define JETSTREAM_RENDER_COMPONENTS_VOICE_WAVE_HH

#include <memory>
#include <optional>

#include "jetstream/types.hh"
#include "jetstream/logger.hh"

#include "jetstream/render/types.hh"
#include "jetstream/render/base/surface.hh"
#include "jetstream/render/components/generic.hh"

namespace Jetstream::Render::Components {

class JETSTREAM_API VoiceWave : public Generic {
 public:
    struct Config {
        Extent2D<F32> framebufferSize = {1.0f, 1.0f};
    };

    struct State {
        Rect rect;
        F32 pitch = 0.0f;
        F32 dot = 0.0f;
        F32 cornerRadius = 0.0f;
        F32 margin = 0.0f;
        F32 time = 0.0f;
        F32 busy = 0.0f;
        F32 activity = 0.0f;
        F32 level = 0.0f;
        ColorRGBA<F32> listeningColor = {0.0f, 0.0f, 0.0f, 0.0f};
        ColorRGBA<F32> busyColor = {0.0f, 0.0f, 0.0f, 0.0f};
        ColorRGBA<F32> dimColor = {0.0f, 0.0f, 0.0f, 0.0f};
    };

    VoiceWave(const Config& config);
    ~VoiceWave();

    Result create(Window* window);

    Result surface(Render::Surface::Config& config);

    Result present();

    Result update(const Extent2D<F32>& framebufferSize, const State& state);
    Result updateScissorRect(const std::optional<Render::ScissorRect>& rect);
    void enabled(bool value);

    constexpr const Config& getConfig() const {
        return config;
    }

 private:
    Config config;

    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

}  // namespace Jetstream::Render::Components

#endif
