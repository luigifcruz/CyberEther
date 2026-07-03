#ifndef JETSTREAM_RENDER_SAKURA_CONTEXT_HH
#define JETSTREAM_RENDER_SAKURA_CONTEXT_HH

#include <jetstream/render/base/surface.hh>
#include <jetstream/render/base/window.hh>
#include <jetstream/render/sakura/palette.hh>
#include <jetstream/types.hh>

#include <functional>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct FontHandle {
    void* native = nullptr;
};

struct Fonts {
    FontHandle body;
    FontHandle h1;
    FontHandle h2;
    FontHandle bold;
    FontHandle display;
};

const Palette& EmptyPalette();

namespace Retained {
struct Drawable;
}  // namespace Retained

struct Context {
    std::reference_wrapper<const Palette> palette = EmptyPalette();
    Render::Window* render = nullptr;
    Fonts fonts;
    F32 pixelRatio = 1.0f;
    Extent2D<U64> framebufferSize = {0, 0};
    bool hovered = false;
    bool active = false;
    bool windowFocused = false;
    std::function<void()> invalidate;
    std::function<void(Retained::Drawable*)> release;
    Render::Surface::Config* surface = nullptr;
    std::vector<Retained::Drawable*>* drawables = nullptr;

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

    void invalidateSurface() const {
        if (invalidate) {
            invalidate();
        }
    }
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_CONTEXT_HH
