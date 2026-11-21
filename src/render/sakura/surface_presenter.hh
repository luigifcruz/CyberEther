#ifndef JETSTREAM_RENDER_SAKURA_SURFACE_PRESENTER_HH
#define JETSTREAM_RENDER_SAKURA_SURFACE_PRESENTER_HH

#include <jetstream/render/base/texture.hh>
#include <jetstream/render/sakura/surface.hh>

#include "context.hh"
#include "metrics.hh"

#include <functional>
#include <optional>

namespace Jetstream::Sakura::Private {

struct SurfacePresenter {
    static constexpr U64 kSettleFrames = 30;

    std::optional<SurfaceResize> lastResize;
    std::optional<U64> lastFrame;
    bool settled = false;
    std::optional<U64> settleStart;

    void resetResize() {
        lastResize.reset();
    }

    void resetSettle() {
        settled = false;
        settleStart.reset();
    }

    bool beginFrame(U64 frame) {
        const bool resumed = lastFrame.has_value() && frame > *lastFrame + 1;
        if (resumed) {
            lastResize.reset();
        }
        lastFrame = frame;
        return resumed;
    }

    void emitResize(const std::optional<SurfaceResize>& resize,
                    const std::function<void(const SurfaceResize&)>& onSize) {
        if (resize.has_value() && onSize && (!lastResize.has_value() || !(*lastResize == *resize))) {
            lastResize = *resize;
            onSize(*resize);
        }
    }

    bool ready(const Render::Texture* texture, U64 frame) const {
        if (settled || !texture) {
            return true;
        }
        const bool matched = lastResize.has_value() && texture->size() == lastResize->framebufferSize;
        return matched || (settleStart.has_value() && frame >= *settleStart && frame - *settleStart >= kSettleFrames);
    }

    bool settle(const Render::Texture* texture, const std::optional<SurfaceResize>& resize, U64 frame) {
        if (!texture) {
            settled = true;
            return true;
        }
        if (!settled) {
            if (!settleStart.has_value()) {
                settleStart = frame;
            }
            const bool matched = resize.has_value() && texture->size() == resize->framebufferSize;
            settled = matched || (frame >= *settleStart && frame - *settleStart >= kSettleFrames);
        }
        return settled;
    }
};

}  // namespace Jetstream::Sakura::Private

#endif  // JETSTREAM_RENDER_SAKURA_SURFACE_PRESENTER_HH
