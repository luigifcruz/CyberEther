#include <jetstream/render/sakura/components/retained/canvas.hh>
#include <jetstream/render/sakura/components/surface_view.hh>

#include <jetstream/render/base.hh>

#include "../../helpers.hh"
#include "../../retained/component.hh"
#include "../../retained/drawable.hh"

#include <algorithm>
#include <limits>
#include <utility>

namespace Jetstream::Sakura::Retained {

namespace {

constexpr U64 kDefaultFramebufferWidth = 512;
constexpr U64 kDefaultFramebufferHeight = 512;
constexpr const char* kRequiredFontName = "default_mono";

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
        root.impl->setFrame(viewport);
        root.impl->setClip(viewport);
        root.layout(ctx);
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

    Config config;
    Component* root = nullptr;

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
        if (lastResize.has_value() && lastResize->logicalSize.x > 0) {
            return static_cast<F32>(lastResize->framebufferSize.x) /
                   static_cast<F32>(lastResize->logicalSize.x);
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

    void handleResize(const SurfaceResize& resize) {
        lastResize = resize;
        if (surface) {
            surface->size(resize.framebufferSize);
        }
        context.framebufferSize = resize.framebufferSize;
        context.pixelRatio = currentPixelRatio();
        runLayout();
        invalidateSurface();
    }

    void handleMouse(const MouseEvent& rawEvent) {
        const MouseEvent event = ConvertMouse(rawEvent, context.framebufferSize);

        const bool handled = root && Impl::event(*root, event);
        if (root && (handled || resourceDirty(*root) || paintDirty(*root))) {
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

    Extent2D<F32> surfaceSize = impl->config.size;

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

        if (impl->config.autoHeight && impl->context.framebufferSize.x > 0) {
            const Extent2D<F32> available = {
                static_cast<F32>(impl->context.framebufferSize.x),
                std::numeric_limits<F32>::infinity(),
            };
            const F32 desiredPx = Impl::measure(*impl->root, rctx, available).y;
            surfaceSize.y = desiredPx / std::max(1e-3f, impl->currentPixelRatio());
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
        .detachOverlay = false,
        .onResolveTexture = [impl = impl.get()]() {
            return impl->framebuffer ? impl->framebuffer->raw() : 0;
        },
        .onSize = [impl = impl.get()](const SurfaceResize& resize) {
            impl->handleResize(resize);
        },
        .onMouse = [impl = impl.get()](MouseEvent event) {
            impl->handleMouse(event);
        },
    });
    impl->surfaceView.render(ctx);
    impl->hovered = ImGui::IsItemHovered();
    impl->active = ImGui::IsItemActive();
    impl->windowFocused = ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows);

    (void)impl->paintTree();
}

}  // namespace Jetstream::Sakura::Retained
