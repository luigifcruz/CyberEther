#include <jetstream/render/sakura/components/surface_image.hh>

#include <jetstream/render/sakura/components/image.hh>
#include <jetstream/render/base.hh>

#include "../context.hh"
#include "../metrics.hh"
#include "../surface_presenter.hh"
#include "../detail/helpers.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>

namespace Jetstream::Sakura {

struct SurfaceImage::Impl {
    Config config;
    Image image;
    Private::SurfacePresenter presenter;
    Rect bounds;
    Extent2D<U64> textureSize = {0, 0};
    bool hovered = false;
    std::array<bool, 3> pressed{};

    bool pressing() const {
        return std::any_of(pressed.begin(), pressed.end(), [](bool held) { return held; });
    }

    bool& held(MouseButton button) {
        return pressed[std::min<U64>(static_cast<U64>(button), pressed.size() - 1)];
    }

    Extent2D<F32> normalized(const Extent2D<F32>& position) const {
        if (bounds.width <= 0.0f || bounds.height <= 0.0f) {
            return {0.0f, 0.0f};
        }
        return {(position.x - bounds.x) / bounds.width, (position.y - bounds.y) / bounds.height};
    }

    void emit(InputEvent event) const {
        if (config.onInput) {
            config.onInput(std::move(event));
        }
    }
};

SurfaceImage::SurfaceImage() {
    impl = std::make_unique<Impl>();
    addChild(impl->image);
}

SurfaceImage::~SurfaceImage() {
    if (hasFocus()) {
        impl->emit(FocusEvent{false});
    }
}

void SurfaceImage::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    if (impl->config.texture != config.texture) {
        cancelMouse(*this);
        releaseFocus();
        impl->presenter.resetResize();
        impl->presenter.resetSettle();
    } else if (impl->config.onInput && !config.onInput) {
        cancelMouse(*this);
        releaseFocus();
    }
    impl->config = std::move(config);
}

Extent2D<F32> SurfaceImage::measure(const Context&, Extent2D<F32> available) {
    return {FillLength(available.x), FillLength(available.y)};
}

void SurfaceImage::layout(const Context& ctx) {
    const Rect bounds = frame();
    const Rect clipRect = Intersect(bounds, clip());
    impl->bounds = bounds;

    const F32 scaling = ScalingFactor(ctx);
    const Extent2D<F32> framebufferScale = FramebufferScale(ctx);
    const F32 screenRatioX = std::max(1e-3f, scaling * framebufferScale.x);
    const F32 screenRatioY = std::max(1e-3f, scaling * framebufferScale.y);
    const F32 zoom = ctx.pixelRatio / screenRatioX;
    const Extent2D<F32> logical = {bounds.width / screenRatioX, bounds.height / screenRatioY};
    auto resize = ResolveSurfaceResize(ctx, logical);
    if (resize.has_value() && std::abs(zoom - 1.0f) > 1e-3f) {
        resize->logicalSize = {
            static_cast<U64>(std::max(1.0f, logical.x / zoom)),
            static_cast<U64>(std::max(1.0f, logical.y / zoom)),
        };
        resize->scale *= zoom;
    }
    impl->presenter.emitResize(resize, impl->config.onSize);

    const auto& texture = impl->config.texture;
    const bool visible = texture && !clipRect.empty() && this->visible();
    const bool settled = impl->presenter.settle(texture.get(), resize, ctx.frame);

    Rect imageRect = bounds;
    impl->textureSize = texture ? texture->size() : Extent2D<U64>{0, 0};
    if (texture) {
        imageRect.width = static_cast<F32>(impl->textureSize.x);
        imageRect.height = static_cast<F32>(impl->textureSize.y);
    }
    impl->image.update({
        .texture = texture,
        .rect = imageRect,
        .clip = bounds,
        .cornerRadius = impl->config.rounding * ctx.pixelRatio,
        .visible = visible && settled,
        .live = true,
    });
    layoutChild(ctx, impl->image, bounds);
}

bool SurfaceImage::mouseEvent(const MouseEvent& event) {
    if (!impl->config.onInput) {
        return false;
    }
    const bool inside = impl->bounds.contains(event.position.x, event.position.y);
    MouseEvent forwarded = event;
    forwarded.position = impl->normalized(event.position);

    switch (event.type) {
        case MouseEventType::Click:
            if (!inside) {
                return false;
            }
            if (!requestFocus()) {
                return true;
            }
            impl->held(event.button) = true;
            impl->emit(forwarded);
            return true;
        case MouseEventType::Release: {
            bool& pressed = impl->held(event.button);
            if (!pressed) {
                return false;
            }
            pressed = false;
            impl->emit(forwarded);
            return true;
        }
        case MouseEventType::Move:
            if (inside || impl->pressing()) {
                impl->hovered = inside;
                impl->emit(forwarded);
                return impl->pressing();
            }
            if (impl->hovered) {
                impl->hovered = false;
                impl->emit(MouseEvent{.type = MouseEventType::Leave, .position = forwarded.position});
            }
            return false;
        case MouseEventType::Scroll:
            if (!inside) {
                return false;
            }
            impl->emit(forwarded);
            return true;
        case MouseEventType::Leave:
        case MouseEventType::Cancel:
            if (impl->hovered || impl->pressing()) {
                impl->hovered = false;
                impl->pressed.fill(false);
                impl->emit(forwarded);
            }
            return false;
        case MouseEventType::Enter:
            return false;
    }
    return false;
}

bool SurfaceImage::keyEvent(const KeyEvent& event) {
    impl->emit(event);
    return !event.modifiers.control && !event.modifiers.super;
}

void SurfaceImage::focusChanged(bool focused) {
    impl->emit(FocusEvent{focused});
}

std::optional<SurfaceCursor> SurfaceImage::cursorAt(const Extent2D<F32>& point) const {
    if (!impl->bounds.contains(point.x, point.y)) {
        return std::nullopt;
    }
    if (impl->config.resolveCursor) {
        return impl->config.resolveCursor();
    }
    if (impl->config.onInput) {
        return SurfaceCursor::Default;
    }
    return std::nullopt;
}

void SurfaceImage::wentOnscreen(U64 frame) {
    const bool resumed = impl->presenter.beginFrame(frame);
    const auto& texture = impl->config.texture;
    if (!texture) {
        return;
    }
    texture->presentationHint(frame, !Intersect(impl->bounds, clip()).empty());
    const bool settling = !impl->presenter.settled && impl->presenter.ready(texture.get(), frame);
    if (resumed || settling || texture->size() != impl->textureSize) {
        invalidate(Dirty::Layout);
    }
}

void SurfaceImage::wentOffscreen(U64 frame) {
    if (impl->config.texture) {
        impl->config.texture->presentationHint(frame, false);
    }
}

}  // namespace Jetstream::Sakura
