#include <jetstream/render/sakura/components/retained/label.hh>

#include <jetstream/render/base.hh>
#include <jetstream/render/components/text.hh>

#include "../../retained/drawable.hh"
#include "../../retained/helpers.hh"

#include <algorithm>
#include <utility>

namespace Jetstream::Sakura::Retained {

namespace {

constexpr const char* kFallbackFontName = "default_mono";

std::string LabelElementId(U64 index) {
    return jst::fmt::format("label{:03}", index);
}

}  // namespace

struct Label::Impl : public Drawable {
    Config config;
    Context* context = nullptr;
    std::shared_ptr<Render::Components::Text> text;
    U64 capacity = 0;

    ~Impl() override {
        if (context && context->release) {
            context->release(this);
        }
    }

    bool sameVisuals(const Config& other) const {
        return config.instances == other.instances &&
               config.clip == other.clip;
    }

    bool sameResources(const Config& other) const {
        return requiredCapacity(config) == requiredCapacity(other) &&
               config.fontName == other.fontName &&
               config.sharpness == other.sharpness &&
               config.maxCharacters == other.maxCharacters;
    }

    U64 requiredCapacity(const Config& other) const {
        return other.capacity > 0 ? other.capacity : std::max<U64>(1, other.instances.size());
    }

    F32 fontPixelSize() const {
        if (!text || !text->getConfig().font) {
            return 32.0f;
        }
        return text->getConfig().font->getConfig().size;
    }

    static Extent2D<F32> AnchorPixels(const Instance& instance) {
        const auto& rect = instance.rect;
        const F32 x = instance.alignment.x == 1 ? rect.x + rect.width * 0.5f
                    : instance.alignment.x == 2 ? rect.right()
                    : rect.x;
        const F32 y = instance.alignment.y == 1 ? rect.y + rect.height * 0.5f
                    : instance.alignment.y == 2 ? rect.bottom()
                    : rect.y;
        return {x, y};
    }

    Result attach(Context* context, Render::Surface::Config& surfaceConfig) override {
        this->context = context;
        capacity = requiredCapacity(config);

        const auto& fontName = context->render->hasFont(config.fontName)
            ? config.fontName
            : std::string(kFallbackFontName);
        if (!context->render->hasFont(fontName)) {
            JST_ERROR("[SAKURA] Label '{}' could not resolve font '{}'.", config.id, config.fontName);
            return Result::ERROR;
        }

        Render::Components::Text::Config textConfig;
        textConfig.maxCharacters = std::max<U64>(1, config.maxCharacters);
        textConfig.font = context->render->font(fontName);
        textConfig.pixelSize = context->pixelSize();
        textConfig.sharpness = config.sharpness;
        for (U64 i = 0; i < capacity; ++i) {
            textConfig.elements[LabelElementId(i)] = {
                .scale = 1.0f,
                .position = {-2.0f, -2.0f},
                .fill = "",
            };
        }

        JST_CHECK(context->render->build(text, textConfig));
        JST_CHECK(context->render->bind(text));
        JST_CHECK(text->surface(surfaceConfig));

        return Result::SUCCESS;
    }

    Result detach(Render::Window* window) override {
        if (text && window) {
            JST_CHECK(window->unbind(text));
        }
        text.reset();
        context = nullptr;
        return Result::SUCCESS;
    }

    Result upload() override {
        if (!text || !context) {
            return Result::SUCCESS;
        }

        const auto& framebufferSize = context->framebufferSize;
        const U64 characterCapacity = std::max<U64>(1, text->getConfig().maxCharacters);

        JST_CHECK(text->updatePixelSize(context->pixelSize()));

        for (U64 i = 0; i < capacity; ++i) {
            const bool on = i < config.instances.size() &&
                            config.instances[i].visible &&
                            !config.instances[i].str.empty();
            if (!on) {
                JST_CHECK(text->update(LabelElementId(i), {
                    .scale = 1.0f,
                    .position = {-2.0f, -2.0f},
                    .fill = "",
                }));
                continue;
            }

            const auto& instance = config.instances[i];
            const auto anchor = AnchorPixels(instance);
            JST_CHECK(text->update(LabelElementId(i), {
                .scale = instance.fontSize / fontPixelSize(),
                .position = PixelToNdc(framebufferSize, anchor.x, anchor.y),
                .alignment = instance.alignment,
                .fill = instance.str.substr(0, characterCapacity),
                .color = instance.color,
            }));
        }

        JST_CHECK(text->updateScissorRect(config.clip.has_value()
            ? std::optional<Render::ScissorRect>(RectToScissor(*config.clip, framebufferSize))
            : std::nullopt));

        return Result::SUCCESS;
    }

    Result present() override {
        if (text) {
            JST_CHECK(text->present());
        }
        return Result::SUCCESS;
    }
};

Label::Label() {
    this->impl = std::make_unique<Impl>();
}

Label::~Label() = default;

bool Label::update(Config config) {
    const bool visualsChanged = !this->impl->sameVisuals(config);
    const bool resourcesChanged = !this->impl->sameResources(config);
    this->impl->config = std::move(config);

    if (resourcesChanged) {
        invalidate(Dirty::Resource);
    } else if (visualsChanged) {
        invalidate(Dirty::Paint);
    }
    return true;
}

Result Label::build(Context& ctx) {
    ctx.drawables->push_back(this->impl.get());
    return this->impl->attach(&ctx, *ctx.surface);
}

Result Label::paint() {
    JST_CHECK(this->impl->upload());
    JST_CHECK(this->impl->present());
    return Result::SUCCESS;
}

}  // namespace Jetstream::Sakura::Retained
