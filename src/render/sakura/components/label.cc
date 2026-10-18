#include <jetstream/render/sakura/components/label.hh>

#include <jetstream/render/base.hh>
#include <jetstream/render/components/text.hh>

#include "../detail/drawable.hh"
#include "../detail/helpers.hh"

#include <algorithm>
#include <utility>

using Unicode = Jetstream::Render::Components::Text::Unicode;

namespace Jetstream::Sakura {

namespace {

constexpr const char* kFallbackFontName = "default_mono";

std::string LabelElementId(U64 index) {
    return jst::fmt::format("label{:06}", index);
}

}  // namespace

struct Label::Impl : public Drawable {
    Config config;
    U64 capacity = 0;
    U64 liveSlots = 0;
    std::string attachedFontName;
    bool instancesDirty = true;
    Extent2D<U64> uploadedFramebuffer = {0, 0};

    Render::Components::Text* text() const {
        return as<Render::Components::Text>();
    }

    std::string resolveFontName(const std::string& fontName) const {
        return context->render->hasFont(fontName) ? fontName : std::string(kFallbackFontName);
    }

    bool resolvedFontChanged(const Config& other) const {
        return context && context->render && !attachedFontName.empty() &&
               resolveFontName(other.fontName) != attachedFontName;
    }

    bool sameResources(const Config& other) const {
        return requiredCapacity(other) == capacity &&
               config.fontName == other.fontName &&
               config.sharpness == other.sharpness &&
               config.maxCharacters == other.maxCharacters;
    }

    U64 requiredCapacity(const Config& other) const {
        if (other.capacity > 0) {
            return other.capacity;
        }
        return std::max<U64>({1, capacity, other.instances.size()});
    }

    F32 fontPixelSize() const {
        if (!component || !text()->getConfig().font) {
            return 32.0f;
        }
        return text()->getConfig().font->getConfig().size;
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
        liveSlots = 0;
        instancesDirty = true;

        const std::string fontName = resolveFontName(config.fontName);
        if (!context->render->hasFont(fontName)) {
            JST_ERROR("[SAKURA] Label could not resolve font '{}'.", config.fontName);
            return Result::ERROR;
        }

        Render::Components::Text::Config textConfig;
        textConfig.maxCharacters = std::max<U64>(1, config.maxCharacters);
        textConfig.font = context->render->font(fontName);
        attachedFontName = fontName;
        textConfig.pixelSize = context->pixelSize();
        textConfig.sharpness = config.sharpness;
        for (U64 i = 0; i < capacity; ++i) {
            textConfig.elements[LabelElementId(i)] = {
                .scale = 1.0f,
                .position = {-2.0f, -2.0f},
                .fill = "",
            };
        }

        std::shared_ptr<Render::Components::Text> created;
        JST_CHECK(context->render->build(created, textConfig));
        return bind(context, std::move(created), surfaceConfig);
    }

    void conceal() override {
        if (component) {
            text()->enabled(false);
        }
    }

    Result upload() override {
        if (!component || !context) {
            return Result::SUCCESS;
        }
        auto* text = this->text();

        const bool active = std::any_of(
            config.instances.begin(), config.instances.end(), [](const auto& instance) {
                return instance.visible && !instance.str.empty();
            });
        text->enabled(active);
        if (!active) {
            return Result::SUCCESS;
        }

        const auto& framebufferSize = context->framebufferSize;
        const U64 characterCapacity = std::max<U64>(1, text->getConfig().maxCharacters);

        JST_CHECK(text->updatePixelSize(context->pixelSize()));
        JST_CHECK(text->updateTranslation(translationNdc()));
        JST_CHECK(text->updateScissorRect(scissor(config.clip)));

        if (!instancesDirty && uploadedFramebuffer == framebufferSize) {
            return Result::SUCCESS;
        }
        instancesDirty = false;
        uploadedFramebuffer = framebufferSize;

        const U64 slots = std::min(capacity, std::max<U64>(config.instances.size(), liveSlots));
        U64 nextLiveSlots = 0;
        for (U64 i = 0; i < slots; ++i) {
            const bool on = i < config.instances.size() &&
                            config.instances[i].visible &&
                            !config.instances[i].str.empty();
            if (!on) {
                if (i < liveSlots) {
                    JST_CHECK(text->update(LabelElementId(i), {
                        .scale = 1.0f,
                        .position = {-2.0f, -2.0f},
                        .fill = "",
                    }));
                }
                continue;
            }
            nextLiveSlots = i + 1;

            const auto& instance = config.instances[i];
            const auto anchor = AnchorPixels(instance);
            JST_CHECK(text->update(LabelElementId(i), {
                .scale = instance.fontSize / fontPixelSize(),
                .position = PixelToNdc(framebufferSize, anchor.x, anchor.y),
                .alignment = instance.alignment,
                .fill = instance.str.substr(0, Unicode::Advance(instance.str, 0, characterCapacity)),
                .color = instance.color,
            }));
        }

        liveSlots = nextLiveSlots;

        return Result::SUCCESS;
    }
};

Label::Label() {
    this->impl = std::make_unique<Impl>();
}

Label::~Label() = default;

const Label::Config& Label::config() const {
    return impl->config;
}

void Label::update(Config config) {
    markConfigured();
    const bool instancesChanged = this->impl->config.instances != config.instances;
    const bool visualsChanged = instancesChanged || this->impl->config.clip != config.clip;
    this->impl->instancesDirty |= instancesChanged;
    const bool resourcesChanged = !this->impl->sameResources(config) || this->impl->resolvedFontChanged(config);
    this->impl->config = std::move(config);

    if (resourcesChanged && this->impl->attached()) {
        invalidate(Dirty::Resource);
    } else if (visualsChanged) {
        invalidate(Dirty::Paint);
    }
}

Result Label::build(Context& ctx) {
    return ctx.attach(*this, this->impl.get());
}

Result Label::paint() {
    this->impl->translation = translation();
    this->impl->inherited = inheritedClip();
    return this->impl->paint();
}

void Label::conceal() {
    this->impl->conceal();
}

}  // namespace Jetstream::Sakura
