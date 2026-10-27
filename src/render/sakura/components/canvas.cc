#include <jetstream/render/sakura/components/canvas.hh>

#include <jetstream/render/base.hh>
#include <jetstream/render/components/generic.hh>

#include "../context.hh"
#include "../metrics.hh"
#include "../input.hh"
#include "../detail/access.hh"
#include "../detail/component.hh"
#include "../detail/drawable.hh"
#include "../detail/helpers.hh"
#include "../detail/input_router.hh"
#include "../surface_view.hh"

#include <algorithm>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Jetstream::Sakura {

namespace {

constexpr F32 kHoverDelay = 0.15f;
constexpr U64 kDefaultFramebufferWidth = 512;
constexpr U64 kDefaultFramebufferHeight = 512;
constexpr const char* kRequiredFontName = Typography::MonoFont;
constexpr U64 kCompactionMinimumPrograms = 64;
constexpr U64 kMaximumDeferredBuilds = 2;

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
    Config config;
    Component* root = nullptr;

    Render::Window* renderWindow = nullptr;
    Render::Window* boundWindow = nullptr;
    std::shared_ptr<Render::Texture> framebuffer;
    std::shared_ptr<Render::Surface> surface;
    SurfaceView surfaceView;
    Context context;
    struct Attachment {
        const Component* owner = nullptr;
        Drawable* drawable = nullptr;
        std::vector<std::shared_ptr<Render::Program>> programs;
    };
    std::vector<Attachment> attachments;
    Render::Surface::Config* building = nullptr;
    std::vector<std::shared_ptr<Render::Components::Generic>> releasedComponents;
    std::vector<std::shared_ptr<Render::Program>> releasedPrograms;
    bool releasePending = false;
    std::optional<Extent2D<F32>> mouse;
    Rect screenRect;
    std::optional<SurfaceResize> lastResize;
    bool hovered = false;
    bool active = false;
    bool windowFocused = false;
    bool bound = false;
    bool surfaceDirty = true;
    SurfaceCursor cursor = SurfaceCursor::Default;
    bool middleHeld = false;
    U64 dispatchDepth = 0;
    InputRouter router;
    bool inputDirtied = false;
    U64 deferredBuilds = 0;
    std::string surfaceId;
    std::string viewId;
    Extent2D<F32> viewSize = {-1.0f, -1.0f};
    bool viewPremultiplied = false;

    struct Environment {
        Palette palette;
        Render::Window* render = nullptr;
        Extent2D<U64> framebufferSize = {0, 0};
        F32 pixelRatio = 0.0f;
        Render::WindowMetrics metrics;
        bool windowFocused = false;

        bool operator==(const Environment&) const = default;
    };
    std::optional<Environment> environment;

    void refreshEnvironment(const Context& ctx) {
        Environment next{
            .palette = ctx.palette.get(),
            .render = ctx.render,
            .framebufferSize = ctx.framebufferSize,
            .pixelRatio = ctx.pixelRatio,
            .metrics = ctx.render ? ctx.render->metrics() : Render::WindowMetrics{},
            .windowFocused = ctx.windowFocused,
        };
        if (environment == next) {
            return;
        }
        environment = std::move(next);
        Access::invalidateLayout(*root);
    }

    bool treeDirtied() const {
        return Access::resourceDirty(*root) || Access::paintDirty(*root) || Access::layoutDirty(*root);
    }

    Impl() {
        context.release = [this](Drawable* drawable, std::shared_ptr<Render::Components::Generic> component) {
            release(drawable, std::move(component));
        };
        context.attach = [this](const Component& owner, Drawable* drawable) {
            return attach(owner, drawable);
        };
    }

    ~Impl() {
        if (root) {
            root->impl->attachTo(nullptr);
        }
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

    F32 currentPixelRatio() const {
        return lastResize.has_value() ? DisplayMetrics::PixelRatio(*lastResize) : 1.0f;
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

    Context canvasContext(const Sakura::Context& ctx) {
        return {
            .palette = ctx.palette,
            .render = renderWindow,
            .pixelRatio = context.pixelRatio,
            .framebufferSize = context.framebufferSize,
            .windowFocused = windowFocused,
            .mouse = mouse,
            .time = ctx.input ? ctx.input->time : 0.0,
            .deltaTime = ctx.input ? ctx.input->deltaTime : 0.0f,
            .frame = renderWindow->frameIndex(),
            .hoverDelay = kHoverDelay,
        };
    }

    Result attach(const Component& owner, Drawable* drawable) {
        if (!building) {
            return Result::ERROR;
        }
        const auto it = std::find_if(attachments.begin(), attachments.end(), [drawable](const auto& attachment) {
            return attachment.drawable == drawable;
        });
        if (it != attachments.end()) {
            it->owner = &owner;
            return Result::SUCCESS;
        }
        const U64 first = building->programs.size();
        attachments.push_back({.owner = &owner, .drawable = drawable});
        JST_CHECK(drawable->attach(&context, *building));
        attachments.back().programs.assign(building->programs.begin() + static_cast<I64>(first),
                                           building->programs.end());
        return Result::SUCCESS;
    }

    void collectOrder(const Component& component,
                      std::unordered_map<const Component*, std::vector<Attachment*>>& owned,
                      std::vector<Attachment>& ordered) {
        if (const auto it = owned.find(&component); it != owned.end()) {
            for (auto* attachment : it->second) {
                ordered.push_back(*attachment);
            }
            owned.erase(it);
        }
        for (const auto* child : component.impl->children) {
            if (child) {
                collectOrder(*child, owned, ordered);
            }
        }
    }

    void retireAttachment(const Attachment& attachment, std::shared_ptr<Render::Components::Generic> component) {
        for (const auto& program : attachment.programs) {
            program->setEnabled(false);
            releasedPrograms.push_back(program);
        }
        if (component) {
            releasedComponents.push_back(std::move(component));
        }
        U64 livePrograms = 0;
        for (const auto& live : attachments) {
            livePrograms += live.programs.size();
        }
        if (releasedPrograms.size() > std::max<U64>(kCompactionMinimumPrograms, livePrograms / 4)) {
            releasePending = true;
        }
    }

    bool restructureSurface() {
        if (root->impl->hasUnbuilt()) {
            return false;
        }
        std::unordered_map<const Component*, std::vector<Attachment*>> owned;
        for (auto& attachment : attachments) {
            owned[attachment.owner].push_back(&attachment);
        }
        std::vector<Attachment> ordered;
        ordered.reserve(attachments.size());
        collectOrder(*root, owned, ordered);
        for (const auto& [_, orphans] : owned) {
            for (auto* orphan : orphans) {
                retireAttachment(*orphan, orphan->drawable ? orphan->drawable->takeComponent() : nullptr);
            }
        }
        std::vector<std::shared_ptr<Render::Program>> programs;
        for (const auto& attachment : ordered) {
            programs.insert(programs.end(), attachment.programs.begin(), attachment.programs.end());
        }
        programs.insert(programs.end(), releasedPrograms.begin(), releasedPrograms.end());
        if (surface->reorder(programs) != Result::SUCCESS) {
            return false;
        }
        attachments = std::move(ordered);
        root->impl->clearStructureDirty();
        invalidateSurface();
        return true;
    }

    Result extendSurface() {
        Render::Surface::Config extension;
        building = &extension;
        const Result built = root->impl->buildPending(context);
        building = nullptr;
        if (built != Result::SUCCESS) {
            return built;
        }
        const Result extended = renderWindow->extend(surface, extension.programs, extension.kernels);
        if (extended != Result::SUCCESS) {
            return extended;
        }
        return restructureSurface() ? Result::SUCCESS : Result::ERROR;
    }

    void release(Drawable* drawable, std::shared_ptr<Render::Components::Generic> component) {
        const auto it = std::find_if(attachments.begin(), attachments.end(), [drawable](const auto& attachment) {
            return attachment.drawable == drawable;
        });
        if (it == attachments.end()) {
            if (component) {
                releasedComponents.push_back(std::move(component));
            }
            return;
        }
        const Attachment attachment = *it;
        attachments.erase(it);
        retireAttachment(attachment, std::move(component));
        invalidateSurface();
    }

    Result destroySurface() {
        if (bound && surface && renderWindow) {
            JST_CHECK(renderWindow->unbind(surface));
        }
        surface.reset();

        for (const auto& component : releasedComponents) {
            if (renderWindow) {
                JST_CHECK(renderWindow->unbind(component));
            }
        }
        releasedComponents.clear();
        releasedPrograms.clear();
        releasePending = false;

        for (auto& attachment : attachments) {
            if (attachment.drawable) {
                JST_CHECK(attachment.drawable->detach(renderWindow));
            }
        }
        attachments.clear();

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

        const bool underlays = UnderlaysAvailable(window->supportsEraseBlending(), config.clearColor);
        const bool underlaysChanged = bound && underlays != context.underlays;
        const bool rebuildNeeded = !bound || boundWindow != window || releasePending || underlaysChanged ||
                                   Access::resourceDirty(*root) ||
                                   (Access::structureDirty(*root) && root->impl->hasUnbuilt());
        if (rebuildNeeded && root->impl->awaitingLayout() && deferredBuilds < kMaximumDeferredBuilds) {
            ++deferredBuilds;
            return Result::SUCCESS;
        }
        deferredBuilds = 0;

        if (bound && (boundWindow != window || releasePending || underlaysChanged || Access::resourceDirty(*root))) {
            JST_CHECK(destroySurface());
        } else if (bound && Access::structureDirty(*root) && root->impl->hasUnbuilt()) {
            if (extendSurface() != Result::SUCCESS) {
                JST_CHECK(destroySurface());
            }
        } else if (bound && Access::structureDirty(*root) && !restructureSurface()) {
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
        context.underlays = underlays;

        Render::Surface::Config surfaceConfig;
        surfaceConfig.framebuffer = framebuffer;
        surfaceConfig.clearColor = config.clearColor;
        surfaceConfig.multisampled = false;
        surfaceConfig.retained = true;
        surfaceConfig.composite = true;

        attachments.clear();
        building = &surfaceConfig;
        const Result built = Access::build(*root, context);
        building = nullptr;
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

    bool handleResize(const SurfaceResize& resize) {
        if (lastResize.has_value() && *lastResize == resize) {
            return false;
        }
        lastResize = resize;
        if (surface) {
            surface->size(resize.framebufferSize);
        }
        const Extent2D<U64> previousFramebufferSize = context.framebufferSize;
        const F32 previousPixelRatio = context.pixelRatio;
        context.framebufferSize = resize.framebufferSize;
        context.pixelRatio = currentPixelRatio();
        if (root && (context.framebufferSize != previousFramebufferSize ||
                     context.pixelRatio != previousPixelRatio)) {
            root->impl->invalidatePaintTree();
        }
        invalidateSurface();
        return true;
    }

    void handleMouse(const MouseEvent& rawEvent) {
        MouseEvent event = ConvertMouse(rawEvent, context.framebufferSize);
        if (event.type == MouseEventType::Release && event.cancel) {
            event.type = MouseEventType::Cancel;
        }
        if (event.type == MouseEventType::Leave) {
            mouse.reset();
        } else {
            mouse = event.position;
        }

        if (!root) {
            return;
        }

        Access::mouseEvent(*root, event);
        if (root && treeDirtied()) {
            inputDirtied = true;
            invalidateSurface();
        }
    }

    void guardUnderlays() {
        std::vector<LiveRegion> regions;
        std::vector<Drawable*> owners;
        for (U64 index = 0; index < attachments.size(); ++index) {
            auto* drawable = attachments[index].drawable;
            if (const auto region = drawable ? drawable->liveRegion() : std::nullopt) {
                regions.push_back({.order = index, .rect = *region});
                owners.push_back(drawable);
            }
        }
        const auto allowed = UnderlayEligibility(regions, [&](U64 earlier, U64 later, const Rect& overlap) {
            for (U64 index = earlier + 1; index < later; ++index) {
                if (attachments[index].drawable && attachments[index].drawable->covers(overlap)) {
                    return true;
                }
            }
            return false;
        });
        for (U64 index = 0; index < owners.size(); ++index) {
            owners[index]->allowUnderlay(allowed[index]);
        }
    }

    void consumeInput(const InputFrame* input) {
        if (!input) {
            hovered = false;
            active = false;
            middleHeld = false;
            return;
        }
        windowFocused = input->windowFocused;
        const auto routed = router.route(*input, screenRect);
        if (root && routed.outsideClick) {
            Access::clickFocus(*root, std::nullopt, false, Access::focusGeneration(*root));
        }
        for (const auto& event : routed.events) {
            if (const auto* pointer = std::get_if<MouseEvent>(&event)) {
                const bool focusing = pointer->type == MouseEventType::Click && pointer->button != MouseButton::Middle;
                const U64 generation = root ? Access::focusGeneration(*root) : 0;
                handleMouse(*pointer);
                if (root && focusing) {
                    Access::clickFocus(*root, mouse, true, generation);
                }
                continue;
            }
            if (!root) {
                continue;
            }
            bool handled = false;
            if (const auto* key = std::get_if<KeyEvent>(&event)) {
                handled = Access::key(*root, *key);
            } else if (const auto* text = std::get_if<TextEvent>(&event)) {
                handled = Access::text(*root, *text);
            }
            if (handled || (root && treeDirtied())) {
                inputDirtied = true;
                invalidateSurface();
            }
        }
        hovered = routed.hovered;
        active = routed.active;
        middleHeld = routed.middle;
    }

    std::vector<Render::PresentList::Layer> collectUnderlays(const Rect& windowClip) const {
        std::vector<Render::PresentList::Layer> layers;
        if (screenRect.width <= 0.0f || screenRect.height <= 0.0f || !framebuffer) {
            return layers;
        }
        for (const auto& attachment : attachments) {
            const auto underlay = attachment.drawable ? attachment.drawable->underlay() : std::nullopt;
            if (!underlay || underlay->texture->raw() == 0) {
                continue;
            }
            layers.push_back({
                .texture = underlay->texture,
                .rect = MapToScreen(underlay->rect, screenRect, underlay->frame),
                .clip = Intersect(MapToScreen(underlay->clip, screenRect, framebuffer->size()), windowClip),
            });
        }
        return layers;
    }

    Result occludeHidden() {
        std::vector<Rect> later;
        for (auto it = attachments.rbegin(); it != attachments.rend(); ++it) {
            if (!it->drawable) {
                continue;
            }
            JST_CHECK(it->drawable->occlude(later));
            it->drawable->opaqueRects(later);
        }
        return Result::SUCCESS;
    }

    Result paintTree() {
        if (!surfaceDirty || !root || !bound) {
            return Result::SUCCESS;
        }

        JST_CHECK(Access::paint(*root));
        JST_CHECK(occludeHidden());

        if (!Access::resourceDirty(*root)) {
            surfaceDirty = false;
        }
        return Result::SUCCESS;
    }
};

Canvas::Canvas() {
    this->impl = std::make_unique<Impl>();
}

Canvas::~Canvas() = default;

void Canvas::unmount(const Component& component) {
    if (impl->root == &component) {
        impl->root = nullptr;
    }
}

void Canvas::mount(Component& root) {
    this->impl->root = &root;
    root.impl->attachTo(this);
}

void Canvas::update(Config config) {
    if (this->impl->config.id != config.id || this->impl->surfaceId.empty()) {
        this->impl->surfaceId = config.id + ":surface";
    }
    this->impl->config = std::move(config);

    if (this->impl->ensureSurface() != Result::SUCCESS) {
        JST_ERROR("[SAKURA] Canvas '{}' failed to create render surface.", this->impl->config.id);
    }
}

void Canvas::render(const Sakura::Context& ctx, const Rect& region) {
    impl->renderWindow = ctx.render;
    impl->cursor = SurfaceCursor::Default;
    impl->inputDirtied = false;

    const Extent2D<F32> surfaceSize = DisplayMetrics::LogicalSize(Metrics(ctx), {region.width, region.height});
    const Rect clip = DisplayMetrics::RootClip(Metrics(ctx), region);

    if (!impl->bound) {
        if (const auto resize = ResolveSurfaceResize(ctx, surfaceSize)) {
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
        }

        const Context canvasCtx = impl->canvasContext(ctx);
        const Rect viewport = {
            0.0f, 0.0f,
            static_cast<F32>(impl->context.framebufferSize.x),
            static_cast<F32>(impl->context.framebufferSize.y),
        };
        impl->refreshEnvironment(canvasCtx);
        Access::advance(canvasCtx.time);
        Access::layout(*impl->root, canvasCtx, viewport);
        Access::present(*impl->root, canvasCtx.frame);
        ++impl->dispatchDepth;
        const bool settled = Access::settleFocus(*impl->root);
        --impl->dispatchDepth;
        if (settled) {
            impl->inputDirtied = true;
        }
        laidOutFramebufferSize = impl->context.framebufferSize;
        laidOutPixelRatio = impl->context.pixelRatio;

        if (impl->root && (Access::resourceDirty(*impl->root) || Access::paintDirty(*impl->root))) {
            impl->invalidateSurface();
        }
    }

    if (!impl->bound) {
        return;
    }

    if (impl->viewId != impl->surfaceId || impl->viewSize != surfaceSize ||
        impl->viewPremultiplied != impl->context.underlays) {
        impl->viewId = impl->surfaceId;
        impl->viewSize = surfaceSize;
        impl->viewPremultiplied = impl->context.underlays;
        impl->surfaceView.update({
            .id = impl->surfaceId,
            .size = surfaceSize,
            .premultiplied = impl->context.underlays,
            .resolveTexture = [impl = impl.get()]() -> std::shared_ptr<const Render::Texture> {
                return impl->framebuffer;
            },
            .onSize = [impl = impl.get()](const SurfaceResize& resize) {
                impl->handleResize(resize);
            },
        });
    }
    const bool layered = impl->context.underlays;
    const auto presented = impl->surfaceView.render(ctx, region, clip);
    impl->screenRect = presented.rect;
    ++impl->dispatchDepth;
    impl->consumeInput(ctx.input);
    if (!impl->hovered && !impl->active && !impl->middleHeld && impl->mouse.has_value()) {
        MouseEvent leave{};
        leave.type = MouseEventType::Leave;
        impl->handleMouse(leave);
    }
    --impl->dispatchDepth;
    std::optional<Component::Impl::Hit> hit;
    if (impl->root && impl->mouse.has_value() && (impl->hovered || impl->active)) {
        hit = impl->root->impl->hit(*impl->mouse);
    }
    if (impl->cursor == SurfaceCursor::Default && hit.has_value()) {
        impl->cursor = hit->cursor;
    }
    if (impl->hovered || impl->active) {
        Private::ApplyCursor(ctx, impl->cursor);
    }
    if (!impl->hovered && !impl->active && !impl->middleHeld) {
        impl->mouse.reset();
    }

    if (impl->root && (impl->inputDirtied || laidOutFramebufferSize != impl->context.framebufferSize ||
                       laidOutPixelRatio != impl->context.pixelRatio)) {
        Context canvasCtx = impl->canvasContext(ctx);
        canvasCtx.deltaTime = 0.0f;
        const Rect viewport = {
            0.0f, 0.0f,
            static_cast<F32>(impl->context.framebufferSize.x),
            static_cast<F32>(impl->context.framebufferSize.y),
        };
        impl->refreshEnvironment(canvasCtx);
        Access::layout(*impl->root, canvasCtx, viewport);
        if (Access::resourceDirty(*impl->root) || Access::paintDirty(*impl->root)) {
            impl->invalidateSurface();
        }
    }

    (void)impl->paintTree();

    std::vector<Render::PresentList::Layer> underlays;
    if (layered) {
        impl->guardUnderlays();
        underlays = impl->collectUnderlays(clip);
    }
    if (presented.layer.has_value() || !underlays.empty()) {
        ctx.render->present(Render::PresentList::Build(Metrics(ctx), underlays, presented.layer, presented.blend));
    }
}

void Canvas::requestCursor(SurfaceCursor cursor) {
    impl->cursor = cursor;
}

bool Canvas::dispatching() const {
    return impl->dispatchDepth > 0;
}

}  // namespace Jetstream::Sakura
