#pragma once

#include <jetstream/render/sakura/color.hh>
#include <jetstream/types.hh>

#include <functional>
#include <string>

namespace Jetstream::Render {

class Window;

}  // namespace Jetstream::Render

namespace Jetstream::Sakura {

struct FontHandle {
    void* native = nullptr;
};

struct Fonts {
    FontHandle body;
    FontHandle h1;
    FontHandle h2;
    FontHandle bold;
};

struct MarkdownConfigHandle {
    const void* native = nullptr;
};

struct NodeContextHandle {
    void* native = nullptr;
};

struct Context {
    std::reference_wrapper<const Palette> palette = EmptyPalette();
    const Render::Window* render = nullptr;
    Fonts fonts;
    MarkdownConfigHandle markdownConfig;
    std::function<NodeContextHandle(const std::string&)> nodeContextResolver;

    NodeContextHandle nodeContext(const std::string& id) const;
};

const Render::Window& RenderWindow(const Context& ctx);
F32 ScalingFactor(const Context& ctx);
F32 Scale(const Context& ctx, F32 value);
F32 Unscale(const Context& ctx, F32 value);
Extent2D<F32> Scale(const Context& ctx, Extent2D<F32> value);
Extent2D<F32> Unscale(const Context& ctx, Extent2D<F32> value);
Extent2D<F32> FramebufferScale(const Context& ctx);
Extent2D<U64> FramebufferSize(const Context& ctx, const Extent2D<F32>& displaySize);
Extent2D<U64> LogicalFramebufferSize(const Context& ctx, const Extent2D<F32>& logicalSize);
F32 SurfaceScale(const Context& ctx);
F32 FrameRate();

}  // namespace Jetstream::Sakura
