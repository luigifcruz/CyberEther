#include <jetstream/render/sakura/components/retained/canvas.hh>
#include <jetstream/render/sakura/components/surface_view.hh>

#include <jetstream/render/base.hh>

#include "../../helpers.hh"
#include "../../retained/component.hh"
#include "../../retained/drawable.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace Jetstream::Sakura::Retained {

namespace {

constexpr U64 kDefaultFramebufferWidth = 512;
constexpr U64 kDefaultFramebufferHeight = 512;
constexpr const char* kRequiredFontName = "default_mono";

bool SameSurfaceResize(const SurfaceResize& lhs, const SurfaceResize& rhs) {
    return lhs.logicalSize.x == rhs.logicalSize.x &&
           lhs.logicalSize.y == rhs.logicalSize.y &&
           lhs.framebufferSize.x == rhs.framebufferSize.x &&
           lhs.framebufferSize.y == rhs.framebufferSize.y &&
           std::abs(lhs.scale - rhs.scale) <= 1e-6f;
}

MouseEvent ConvertMouse(const MouseEvent& event, const Extent2D<U64>& framebufferSize) {
    MouseEvent out = event;
    if (out.type == MouseEventType::Enter) {
        out.type = MouseEventType::Move;
    }
    out.position = {
        event.position.x * static_cast<F32>(framebufferSize.x),
        event.position.y * static_cast<F32>(framebufferSize.y),
    };
    return out;
}

}  // namespace

struct Canvas::Impl {
    static void frame(Component& root, Rect viewport, const Context& ctx) {
        root.layoutRoot(ctx, viewport);
    }

    static Result build(Component& root, Context& ctx) {
        return root.impl->buildTree(ctx);
    }

    static Result paint(Component& root) {
        return root.impl->paintTree();
    }

    static bool event(Component& root, const MouseEvent& event) {
        return root.event(event);
    }

    static bool resourceDirty(Component& root) {
        return root.impl->treeResourceDirty();
    }

    static bool paintDirty(Component& root) {
        return root.impl->isPaintDirty();
    }

    static Extent2D<F32> measure(Component& root, const Context& ctx, Extent2D<F32> available) {
        return root.measure(ctx, available);
    }

    static bool hitTest(const Component& root, const Extent2D<F32>& point) {
        return root.hitTest(point);
    }

    static bool hoverBlocked(const ImGuiContext& g, const ImGuiIO& io) {
        if (g.HoveredWindow != g.HoveredWindowBeforeClear || ImGui::GetTopMostPopupModal() ||
            (io.ConfigFlags & ImGuiConfigFlags_NoMouse)) {
            return true;
        }
        for (int i = 0; i < IM_ARRAYSIZE(io.MouseDown); ++i) {
            if (io.MouseDown[i] && !io.MouseClicked[i] && !io.MouseDownOwned[i]) {
                return true;
            }
        }
        return false;
    }

    void routeMouse() {
        ImGuiContext& g = *ImGui::GetCurrentContext();
        ImGuiWindow* window = ImGui::GetCurrentWindow();
        if (!root || surfaceRect.empty() || context.framebufferSize.x == 0 || context.framebufferSize.y == 0 ||
            (g.ActiveId != 0 && g.ActiveIdWindow == window)) {
            return;
        }
        ImGuiIO& io = ImGui::GetIO();
        const ImVec2 mouse = io.MousePos;
        const bool inside = ImGui::IsMousePosValid(&mouse) && hitTest(*root, {
            (mouse.x - surfaceRect.x) / surfaceRect.width * static_cast<F32>(context.framebufferSize.x),
            (mouse.y - surfaceRect.y) / surfaceRect.height * static_cast<F32>(context.framebufferSize.y),
        });
        if (!inside) {
            window->Flags |= ImGuiWindowFlags_NoMouseInputs;
        }
        const bool hovered = g.HoveredWindow && g.HoveredWindow->RootWindow == window->RootWindow;
        if (hovered == inside || hoverBlocked(g, io)) {
            return;
        }
        ImGui::FindHoveredWindowEx(mouse, false, &g.HoveredWindow, &g.HoveredWindowUnderMovingWindow);
        g.HoveredWindowBeforeClear = g.HoveredWindow;
        for (int i = 0; i < IM_ARRAYSIZE(io.MouseDown); ++i) {
            if (io.MouseClicked[i]) {
                io.MouseDownOwned[i] = g.HoveredWindow != nullptr || g.OpenPopupStack.Size > 0;
                io.MouseDownOwnedUnlessPopupClose[i] = g.HoveredWindow != nullptr;
            }
        }
    }

    Config config;
    Component* root = nullptr;
    Rect surfaceRect;

    Render::Window* renderWindow = nullptr;
    Render::Window* boundWindow = nullptr;
    std::shared_ptr<Render::Texture> framebuffer;
    std::shared_ptr<Render::Surface> surface;
    SurfaceView surfaceView;
    Context context;
    std::vector<Drawable*> attachedDrawables;
    std::optional<SurfaceResize> lastResize;
    bool hovered = false;
    bool active = false;
    bool windowFocused = false;
    bool bound = false;
    bool surfaceDirty = true;

    ~Impl() {
        root = nullptr;
        surfaceView.update({});
        (void)destroySurface();
    }

    void invalidateSurface() {
        surfaceDirty = true;
        if (surface) {
            surface->invalidate();
        }
    }

    void runLayout() {
        if (config.onLayout) {
            config.onLayout({
                .framebufferSize = context.framebufferSize,
                .pixelRatio = context.pixelRatio,
            });
        }
    }

    F32 currentPixelRatio() const {
        if (lastResize.has_value() && lastResize->scale > 0.0f) {
            return lastResize->scale * 2.0f;
        }
        return 1.0f;
    }

    Extent2D<U64> resolveFramebufferSize() const {
        if (lastResize.has_value()) {
            return {
                std::max<U64>(2, lastResize->framebufferSize.x),
                std::max<U64>(2, lastResize->framebufferSize.y),
            };
        }
        return {kDefaultFramebufferWidth, kDefaultFramebufferHeight};
    }

    Context retainedContext(const Sakura::Context& ctx) const {
        return {
            .palette = ctx.palette,
            .render = renderWindow,
            .fonts = ctx.fonts,
            .pixelRatio = context.pixelRatio,
            .framebufferSize = context.framebufferSize,
            .hovered = hovered,
            .active = active,
            .windowFocused = windowFocused,
        };
    }

    Result destroySurface() {
        if (bound && surface && renderWindow) {
            JST_CHECK(renderWindow->unbind(surface));
        }
        surface.reset();

        for (auto* drawable : attachedDrawables) {
            if (drawable) {
                JST_CHECK(drawable->detach(renderWindow));
            }
        }
        attachedDrawables.clear();

        framebuffer.reset();
        boundWindow = nullptr;
        bound = false;
        surfaceDirty = true;
        return Result::SUCCESS;
    }

    Result ensureSurface() {
        Render::Window* window = renderWindow;
        if (!window || !root) {
            return Result::SUCCESS;
        }

        if (bound && (boundWindow != window || resourceDirty(*root))) {
            JST_CHECK(destroySurface());
        }

        if (bound) {
            surface->clearColor(config.clearColor);
            return Result::SUCCESS;
        }

        if (!window->hasFont(kRequiredFontName)) {
            return Result::SUCCESS;
        }

        const Extent2D<U64> framebufferSize = resolveFramebufferSize();

        JST_CHECK(window->build(framebuffer, Render::Texture::Config{
            .size = framebufferSize,
        }));

        context.render = window;
        context.framebufferSize = framebufferSize;
        context.pixelRatio = currentPixelRatio();
        context.invalidate = [this]() {
            invalidateSurface();
        };
        context.release = [this](Drawable*) {
            (void)destroySurface();
        };

        runLayout();

        Render::Surface::Config surfaceConfig;
        surfaceConfig.framebuffer = framebuffer;
        surfaceConfig.clearColor = config.clearColor;
        surfaceConfig.multisampled = false;
        surfaceConfig.retained = true;

        attachedDrawables.clear();
        context.surface = &surfaceConfig;
        context.drawables = &attachedDrawables;
        const Result built = build(*root, context);
        context.surface = nullptr;
        context.drawables = nullptr;
        if (built != Result::SUCCESS) {
            (void)destroySurface();
            return Result::ERROR;
        }

        JST_CHECK(window->build(surface, surfaceConfig));
        JST_CHECK(window->bind(surface));

        boundWindow = window;
        bound = true;
        invalidateSurface();
        return Result::SUCCESS;
    }

    void applyContext(const Extent2D<U64>& framebufferSize, const F32 pixelRatio) {
        const bool changed = context.framebufferSize != framebufferSize || context.pixelRatio != pixelRatio;
        context.framebufferSize = framebufferSize;
        context.pixelRatio = pixelRatio;
        if (root && changed) {
            root->impl->invalidatePaintTree();
        }
        runLayout();
        invalidateSurface();
    }

    bool handleResize(const SurfaceResize& resize) {
        if (lastResize.has_value() && SameSurfaceResize(*lastResize, resize)) {
            return false;
        }
        lastResize = resize;
        if (surface) {
            surface->size(resize.framebufferSize);
        }
        applyContext(surface ? surface->size() : resize.framebufferSize, currentPixelRatio());
        return true;
    }

    void handleMouse(const MouseEvent& rawEvent) {
        const MouseEvent event = ConvertMouse(rawEvent, context.framebufferSize);

        if (!root) {
            return;
        }

        Impl::event(*root, event);
        if (resourceDirty(*root) || paintDirty(*root)) {
            invalidateSurface();
        }
    }

    Result paintTree() {
        if (!surfaceDirty || !root || resourceDirty(*root)) {
            return Result::SUCCESS;
        }

        JST_CHECK(paint(*root));

        surfaceDirty = false;
        return Result::SUCCESS;
    }
};

Canvas::Canvas() {
    this->impl = std::make_unique<Impl>();
}

Canvas::~Canvas() = default;
Canvas::Canvas(Canvas&&) noexcept = default;
Canvas& Canvas::operator=(Canvas&&) noexcept = default;

void Canvas::mount(Component& root) {
    this->impl->root = &root;
    root.impl->attachTo(this);
}

bool Canvas::update(Config config) {
    this->impl->config = std::move(config);

    if (this->impl->ensureSurface() != Result::SUCCESS) {
        JST_ERROR("[SAKURA] Canvas '{}' failed to create render surface.", this->impl->config.id);
    }
    return true;
}

void Canvas::render(const Sakura::Context& ctx) {
    impl->renderWindow = ctx.render;

    if (impl->surface) {
        const auto framebufferSize = impl->surface->size();
        const F32 pixelRatio = impl->currentPixelRatio();
        if (impl->context.framebufferSize != framebufferSize || impl->context.pixelRatio != pixelRatio) {
            impl->applyContext(framebufferSize, pixelRatio);
        }
    }

    Extent2D<F32> surfaceSize = impl->config.size;
    const Extent2D<F32> availableLogicalSize =
        Unscale(ctx, Private::ToExtent2D(ImGui::GetContentRegionAvail()));
    if (surfaceSize.x <= 0.0f) {
        surfaceSize.x = std::max(0.0f, availableLogicalSize.x);
    }
    if (surfaceSize.y <= 0.0f) {
        surfaceSize.y = std::max(0.0f, availableLogicalSize.y);
    }

    // Resolve the actual panel width and DPI before allocating the first
    // retained surface instead of laying it out against the 512px fallback.
    if (!impl->bound) {
        Extent2D<F32> preflightSize = surfaceSize;
        if (impl->config.autoHeight && impl->lastResize.has_value()) {
            preflightSize.y = static_cast<F32>(impl->lastResize->logicalSize.y);
        }
        if (const auto resize = ResolveSurfaceResize(ctx, preflightSize)) {
            impl->handleResize(*resize);
        }
    }

    Extent2D<U64> laidOutFramebufferSize = impl->context.framebufferSize;
    F32 laidOutPixelRatio = impl->context.pixelRatio;

    if (impl->root) {
        if (impl->context.framebufferSize.x == 0 || impl->context.framebufferSize.y == 0) {
            impl->context.render = impl->renderWindow;
            impl->context.framebufferSize = impl->resolveFramebufferSize();
            impl->context.pixelRatio = impl->currentPixelRatio();
            impl->context.invalidate = [this]() {
                impl->invalidateSurface();
            };
            impl->context.release = [this](Drawable*) {
                (void)impl->destroySurface();
            };
            impl->runLayout();
        }

        const Context rctx = impl->retainedContext(ctx);
        const Rect viewport = {
            0.0f, 0.0f,
            static_cast<F32>(impl->context.framebufferSize.x),
            static_cast<F32>(impl->context.framebufferSize.y),
        };
        Impl::frame(*impl->root, viewport, rctx);
        laidOutFramebufferSize = impl->context.framebufferSize;
        laidOutPixelRatio = impl->context.pixelRatio;

        if (impl->config.autoHeight && impl->context.framebufferSize.x > 0) {
            const Extent2D<F32> available = {
                static_cast<F32>(impl->context.framebufferSize.x),
                std::numeric_limits<F32>::infinity(),
            };
            const F32 desiredPx = Impl::measure(*impl->root, rctx, available).y;
            surfaceSize.y = desiredPx / std::max(1e-3f, impl->context.pixelRatio);

            if (!impl->bound) {
                if (const auto resize = ResolveSurfaceResize(ctx, surfaceSize);
                    resize.has_value() && impl->handleResize(*resize)) {
                    const Context resizedContext = impl->retainedContext(ctx);
                    const Rect resizedViewport = {
                        0.0f, 0.0f,
                        static_cast<F32>(impl->context.framebufferSize.x),
                        static_cast<F32>(impl->context.framebufferSize.y),
                    };
                    Impl::frame(*impl->root, resizedViewport, resizedContext);
                    laidOutFramebufferSize = impl->context.framebufferSize;
                    laidOutPixelRatio = impl->context.pixelRatio;
                }
            }
        }

        if (Impl::resourceDirty(*impl->root) || Impl::paintDirty(*impl->root)) {
            impl->invalidateSurface();
        }
    }

    if (!impl->bound) {
        ImGui::Dummy(Private::ToImVec2({Scale(ctx, surfaceSize.x), Scale(ctx, surfaceSize.y)}));
        return;
    }

    impl->surfaceView.update({
        .id = impl->config.id + ":surface",
        .size = surfaceSize,
        .premultiplied = impl->config.clearColor.a < 1.0f,
        .detachOverlay = false,
        .onResolveTexture = [impl = impl.get()]() {
            return impl->framebuffer ? impl->framebuffer->raw() : 0;
        },
        .onSize = [impl = impl.get()](const SurfaceResize& resize) {
            impl->handleResize(resize);
        },
        .onInput = [impl = impl.get()](InputEvent event) {
            if (const auto mouse = SurfaceMouseEvent(event)) {
                impl->handleMouse(*mouse);
            }
        },
    });
    if (impl->config.passthrough) {
        impl->routeMouse();
    }
    impl->surfaceView.render(ctx);
    const ImVec2 itemMin = ImGui::GetItemRectMin();
    const ImVec2 itemMax = ImGui::GetItemRectMax();
    impl->surfaceRect = {itemMin.x, itemMin.y, itemMax.x - itemMin.x, itemMax.y - itemMin.y};
    impl->hovered = ImGui::IsItemHovered();
    impl->active = ImGui::IsItemActive();
    impl->windowFocused = ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows);

    if (impl->root && (laidOutFramebufferSize != impl->context.framebufferSize ||
                       laidOutPixelRatio != impl->context.pixelRatio)) {
        const Context rctx = impl->retainedContext(ctx);
        const Rect viewport = {
            0.0f, 0.0f,
            static_cast<F32>(impl->context.framebufferSize.x),
            static_cast<F32>(impl->context.framebufferSize.y),
        };
        Impl::frame(*impl->root, viewport, rctx);
    }

    (void)impl->paintTree();
}

}  // namespace Jetstream::Sakura::Retained
