#ifndef JETSTREAM_RENDER_SAKURA_CONTEXT_HH
#define JETSTREAM_RENDER_SAKURA_CONTEXT_HH

#include <jetstream/render/base/surface.hh>
#include <jetstream/render/base/window.hh>
#include <jetstream/render/sakura/palette.hh>
#include <jetstream/types.hh>

#include "input.hh"

#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace Jetstream::Render::Components {
class Generic;
}  // namespace Jetstream::Render::Components

namespace Jetstream::Sakura {

const Palette& EmptyPalette();

struct Drawable;
struct PopupLayer;
struct DockDrag;
struct Component;

struct Context {
    std::reference_wrapper<const Palette> palette = EmptyPalette();
    Render::Window* render = nullptr;
    F32 pixelRatio = 1.0f;
    Extent2D<U64> framebufferSize = {0, 0};
    bool windowFocused = false;
    std::optional<Extent2D<F32>> mouse;
    const InputFrame* input = nullptr;
    PopupLayer* popups = nullptr;
    DockDrag* dockDrag = nullptr;
    Extent2D<F32> translation = {0.0f, 0.0f};
    F32 fontSize = 0.0f;
    F64 time = 0.0;
    F32 deltaTime = 0.0f;
    U64 frame = 0;
    F32 hoverDelay = 0.0f;
    bool underlays = false;
    std::string backdropColorKey;
    std::string backdropTopColorKey;
    std::function<void(Drawable*, std::shared_ptr<Render::Components::Generic>)> release;
    std::function<Result(const Component&, Drawable*)> attach;

    ColorRGBA<F32> color(const std::string& key, ColorRGBA<F32> fallback = {}) const {
        if (key.empty()) {
            return fallback;
        }
        const auto& activePalette = palette.get();
        const auto it = activePalette.find(key);
        if (it == activePalette.end()) {
            return fallback;
        }
        return it->second;
    }

    Extent2D<F32> pixelSize() const {
        return {
            framebufferSize.x > 0 ? 2.0f / static_cast<F32>(framebufferSize.x) : 0.0f,
            framebufferSize.y > 0 ? 2.0f / static_cast<F32>(framebufferSize.y) : 0.0f,
        };
    }

    Context translated(Extent2D<F32> offset) const {
        Context next = *this;
        next.translation = {translation.x + offset.x, translation.y + offset.y};
        if (next.mouse.has_value()) {
            next.mouse = Extent2D<F32>{next.mouse->x - offset.x, next.mouse->y - offset.y};
        }
        return next;
    }
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_CONTEXT_HH
