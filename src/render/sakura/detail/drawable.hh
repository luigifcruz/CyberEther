#ifndef JETSTREAM_RENDER_SAKURA_DETAIL_DRAWABLE_HH
#define JETSTREAM_RENDER_SAKURA_DETAIL_DRAWABLE_HH

#include "../context.hh"
#include "helpers.hh"
#include "underlay.hh"

#include <jetstream/render/base/window.hh>
#include <jetstream/render/components/generic.hh>

#include <memory>
#include <optional>

namespace Jetstream::Sakura {

struct Drawable {
    virtual ~Drawable() {
        if (context && context->release) {
            context->release(this, component);
        }
    }

    virtual Result attach(Context* context, Render::Surface::Config& surfaceConfig) = 0;
    virtual Result upload() = 0;
    virtual void conceal() = 0;

    virtual std::optional<Underlay> underlay() const {
        return std::nullopt;
    }

    virtual std::optional<Rect> liveRegion() const {
        return std::nullopt;
    }

    virtual void allowUnderlay(bool) {}

    virtual bool covers(const Rect&) const {
        return false;
    }

    bool attached() const {
        return component != nullptr;
    }

    std::shared_ptr<Render::Components::Generic> takeComponent() {
        context = nullptr;
        return std::move(component);
    }

    Result detach(Render::Window* window) {
        if (component && window) {
            JST_CHECK(window->unbind(component));
        }
        component.reset();
        context = nullptr;
        return Result::SUCCESS;
    }

    Result paint() {
        JST_CHECK(upload());
        if (component) {
            JST_CHECK(component->present());
        }
        return Result::SUCCESS;
    }

 protected:
    template<typename T>
    Result bind(Context* target, std::shared_ptr<T> typed, Render::Surface::Config& surfaceConfig) {
        JST_CHECK(target->render->bind(typed));
        JST_CHECK(typed->surface(surfaceConfig));
        component = std::move(typed);
        return Result::SUCCESS;
    }

    template<typename T>
    T* as() const {
        return static_cast<T*>(component.get());
    }

    Extent2D<F32> translationNdc() const {
        const auto& size = context->framebufferSize;
        if (size.x == 0 || size.y == 0) {
            return {0.0f, 0.0f};
        }
        return {
            2.0f * translation.x / static_cast<F32>(size.x),
            -2.0f * translation.y / static_cast<F32>(size.y),
        };
    }

    std::optional<Render::ScissorRect> scissor(const std::optional<Rect>& clip) const {
        std::optional<Rect> bounds = clip;
        if (inherited.has_value()) {
            bounds = bounds.has_value() ? Intersect(*bounds, *inherited) : *inherited;
        }
        if (!bounds.has_value()) {
            return std::nullopt;
        }
        Rect rect = *bounds;
        rect.x += translation.x;
        rect.y += translation.y;
        return RectToScissor(rect, context->framebufferSize);
    }

 public:
    Extent2D<F32> translation = {0.0f, 0.0f};
    std::optional<Rect> inherited;

 protected:
    Context* context = nullptr;
    std::shared_ptr<Render::Components::Generic> component;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_DETAIL_DRAWABLE_HH
