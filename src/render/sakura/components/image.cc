#include <jetstream/render/sakura/components/image.hh>

#include <jetstream/render/base.hh>
#include <jetstream/render/components/image.hh>

#include "../context.hh"
#include "../metrics.hh"
#include "../detail/drawable.hh"
#include "../detail/helpers.hh"

#include <utility>

namespace Jetstream::Sakura {

struct Image::Impl : public Drawable {
    Config config;

    Render::Components::Image* image() const {
        return as<Render::Components::Image>();
    }

    Result attach(Context* context, Render::Surface::Config& surfaceConfig) override {
        this->context = context;
        if (!config.texture) {
            return Result::SUCCESS;
        }

        punch = config.live && context->underlays && underlayAllowed;
        std::shared_ptr<Render::Components::Image> image;
        JST_CHECK(context->render->build(image, Render::Components::Image::Config{
            .texture = std::const_pointer_cast<Render::Texture>(config.texture),
            .punch = punch,
        }));
        return bind(context, std::move(image), surfaceConfig);
    }

    void conceal() override {
        active = false;
        if (component) {
            image()->enabled(false);
        }
    }

    std::optional<Underlay> underlay() const override {
        if (!punching() || !active || !clip) {
            return std::nullopt;
        }
        return MakeUnderlay(config.texture, translation, config.rect, *clip, uploadedFrame);
    }

    std::optional<Rect> liveRegion() const override {
        if (!config.live || !active || !component || !clip || !context || !context->underlays) {
            return std::nullopt;
        }
        const Rect rect = {translation.x + config.rect.x, translation.y + config.rect.y, config.rect.width,
                           config.rect.height};
        const Rect bounds = {static_cast<F32>(clip->x), static_cast<F32>(clip->y), static_cast<F32>(clip->width),
                             static_cast<F32>(clip->height)};
        return Intersect(rect, bounds);
    }

    void allowUnderlay(bool allowed) override {
        if (allowed == underlayAllowed) {
            return;
        }
        underlayAllowed = allowed;
        if (punch && component) {
            image()->enabled(active && punching());
        }
        if (owner) {
            owner->invalidate(Dirty::Resource);
        }
    }

    bool punching() const {
        return punch && underlayAllowed && component;
    }

    Result upload() override {
        if (!component || !context) {
            return Result::SUCCESS;
        }
        auto* image = this->image();

        active = config.visible && !config.rect.empty();
        image->enabled(active && (!punch || punching()));
        if (!active) {
            return Result::SUCCESS;
        }

        const Extent2D<F32> framebufferSize = {
            static_cast<F32>(context->framebufferSize.x),
            static_cast<F32>(context->framebufferSize.y),
        };
        JST_CHECK(image->update(framebufferSize, translation, config.rect, config.cornerRadius, config.opacity));
        uploadedFrame = context->framebufferSize;
        clip = scissor(config.clip).value_or(Render::ScissorRect{
            0, 0, static_cast<U32>(context->framebufferSize.x), static_cast<U32>(context->framebufferSize.y)});
        JST_CHECK(image->updateScissorRect(clip));
        return Result::SUCCESS;
    }

    Image* owner = nullptr;
    bool punch = false;
    bool active = false;
    bool underlayAllowed = true;
    Extent2D<U64> uploadedFrame = {0, 0};
    std::optional<Render::ScissorRect> clip;
};

Image::Image() {
    impl = std::make_unique<Impl>();
    impl->owner = this;
}

Image::~Image() = default;

const Image::Config& Image::config() const {
    return impl->config;
}

void Image::update(Config config) {
    markConfigured();
    const bool textureChanged = impl->config.texture != config.texture || impl->config.live != config.live;
    const bool visualsChanged = impl->config.rect != config.rect || impl->config.clip != config.clip ||
                                impl->config.cornerRadius != config.cornerRadius ||
                                impl->config.opacity != config.opacity || impl->config.visible != config.visible;
    impl->config = std::move(config);
    if (textureChanged) {
        invalidate(Dirty::Resource);
    } else if (visualsChanged) {
        invalidate(Dirty::Paint);
    }
}

Result Image::build(Context& ctx) {
    return ctx.attach(*this, impl.get());
}

Result Image::paint() {
    impl->translation = translation();
    impl->inherited = inheritedClip();
    return impl->paint();
}

void Image::conceal() {
    impl->conceal();
}

}  // namespace Jetstream::Sakura
