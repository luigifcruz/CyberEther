#include <jetstream/render/sakura/components/label.hh>

#include <jetstream/render/base.hh>
#include <jetstream/render/components/text.hh>

#include "../detail/drawable.hh"
#include "../detail/helpers.hh"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <utility>

using Unicode = Jetstream::Render::Components::Text::Unicode;

namespace Jetstream::Sakura {

namespace {

constexpr const char* kFallbackFontName = Typography::MonoFont;

constexpr U64 kQuadVertices = 4;
constexpr U64 kQuadIndices = 6;

}  // namespace

struct Label::Impl : public Drawable {
    Config config;
    U64 capacity = 0;
    U64 glyphs = 0;
    bool instancesDirty = true;
    bool active = false;
    Extent2D<U64> shapedFramebuffer = {0, 0};
    U64 uploadedVertex = 0;
    bool uploaded = false;
    std::optional<Render::ScissorRect> clipRect;
    std::shared_ptr<Render::Components::Font> font;
    std::vector<F32> geometry;
    std::vector<Render::Components::Text::Quad> quads;
    Painter::Slot slot;

    std::string resolveFontName(const std::string& fontName) const {
        return context->render->hasFont(fontName) ? fontName : std::string(kFallbackFontName);
    }

    std::shared_ptr<Render::Components::Font> resolveFont() const {
        const std::string fontName = resolveFontName(config.fontName);
        return context->render->hasFont(fontName) ? context->render->font(fontName) : nullptr;
    }

    bool resolvedFontChanged(const Config& other) const {
        if (!painter || !context || !context->render || !font) {
            return false;
        }
        const std::string fontName = resolveFontName(other.fontName);
        return context->render->hasFont(fontName) && context->render->font(fontName) != font;
    }

    U64 requiredCapacity(const Config& other) const {
        if (other.capacity > 0) {
            return other.capacity;
        }
        return std::max<U64>({1, capacity, other.instances.size()});
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

    void shape() {
        geometry.clear();
        glyphs = 0;
        const auto& framebufferSize = context->framebufferSize;
        const auto pixel = context->pixelSize();
        const U64 characterCapacity = std::max<U64>(1, config.maxCharacters);
        const F32 fontPixelSize = font->getConfig().size;
        const U64 count = std::min<U64>(capacity, config.instances.size());
        for (U64 i = 0; i < count; ++i) {
            const auto& instance = config.instances[i];
            if (!instance.visible || instance.str.empty()) {
                continue;
            }
            quads.clear();
            const auto bounds = Render::Components::Text::Shape(
                *font, instance.str.substr(0, Unicode::Advance(instance.str, 0, characterCapacity)),
                instance.fontSize / fontPixelSize, quads);

            Extent2D<F32> alignment = {0.0f, 0.0f};
            if (instance.alignment.x == 1) {
                alignment.x = -static_cast<F32>(bounds.x) / 2.0f;
            } else if (instance.alignment.x == 2) {
                alignment.x = -static_cast<F32>(bounds.x);
            }
            if (instance.alignment.y == 1) {
                alignment.y = static_cast<F32>(bounds.y) / 2.0f;
            } else if (instance.alignment.y == 2) {
                alignment.y = static_cast<F32>(bounds.y);
            }

            const auto anchor = AnchorPixels(instance);
            Extent2D<F32> origin = anchor;
            if (instance.rotation == 0.0f && pixel.x > 0.0f && pixel.y > 0.0f) {
                const auto ndc = PixelToNdc(framebufferSize, anchor.x, anchor.y);
                origin = {std::round((ndc.x + 1.0f) / pixel.x + alignment.x),
                          std::round((1.0f - ndc.y) / pixel.y - alignment.y)};
                alignment = {0.0f, 0.0f};
            }

            const F32 angle = instance.rotation * std::numbers::pi_v<F32> / 180.0f;
            const F32 cosine = std::cos(angle);
            const F32 sine = std::sin(angle);
            const auto& color = instance.color;
            const auto emit = [&](F32 x, F32 y, F32 u, F32 v) {
                const F32 localX = x + alignment.x;
                const F32 localY = y + alignment.y;
                geometry.insert(geometry.end(), {
                    origin.x + localX * cosine - localY * sine,
                    origin.y - (localX * sine + localY * cosine),
                    u,
                    v,
                    color.r,
                    color.g,
                    color.b,
                    color.a,
                    -y,
                    0.0f,
                    0.0f,
                    0.0f,
                });
            };
            for (const auto& quad : quads) {
                emit(quad.origin.x, quad.origin.y, quad.texcoordOrigin.x, quad.texcoordOrigin.y);
                emit(quad.corner.x, quad.origin.y, quad.texcoordCorner.x, quad.texcoordOrigin.y);
                emit(quad.corner.x, quad.corner.y, quad.texcoordCorner.x, quad.texcoordCorner.y);
                emit(quad.origin.x, quad.corner.y, quad.texcoordOrigin.x, quad.texcoordCorner.y);
            }
            glyphs += quads.size();
        }
    }

    Result attach(Context* context, Render::Surface::Config&) override {
        JST_CHECK(join(context));
        capacity = requiredCapacity(config);
        instancesDirty = true;
        uploaded = false;

        font = resolveFont();
        if (!font) {
            JST_ERROR("[SAKURA] Label could not resolve font '{}'.", config.fontName);
            return Result::ERROR;
        }

        U64 characters = 0;
        const U64 limit = std::max<U64>(1, config.maxCharacters);
        const U64 count = std::min<U64>(capacity, config.instances.size());
        for (U64 i = 0; i < count; ++i) {
            const auto& instance = config.instances[i];
            characters += instance.visible ? std::min<U64>(instance.str.size(), limit) : 0;
        }
        painter->demand(Painter::Kind::Glyphs, {.vertices = characters * kQuadVertices, .instances = 1}, font);
        return Result::SUCCESS;
    }

    void conceal() override {
        active = false;
    }

    void abandon() override {
        slot.reset();
        uploaded = false;
    }

    void steps(std::vector<Render::Surface::Step>& out) const override {
        if (!active || !slot.held() || glyphs == 0) {
            return;
        }
        out.push_back({
            .program = painter->program(*slot),
            .range = Render::Draw::Range{
                .count = glyphs * kQuadIndices,
                .baseVertex = static_cast<I64>(slot->vertex),
                .firstInstance = slot->instance,
                .instanceCount = 1,
            },
            .scissor = clipRect,
        });
    }

    Result upload() override {
        if (!painter || !context) {
            return Result::SUCCESS;
        }

        active = std::any_of(
            config.instances.begin(), config.instances.end(), [](const auto& instance) {
                return instance.visible && !instance.str.empty();
            });
        if (!active) {
            return Result::SUCCESS;
        }

        const auto resolved = resolveFont();
        if (!resolved) {
            active = false;
            return Result::SUCCESS;
        }

        if (instancesDirty || resolved != font || shapedFramebuffer != context->framebufferSize) {
            font = resolved;
            shapedFramebuffer = context->framebufferSize;
            instancesDirty = false;
            uploaded = false;
            shape();
        }
        if (glyphs == 0) {
            active = false;
            return Result::SUCCESS;
        }

        if (!slot.reserve(*painter, Painter::Kind::Glyphs, {.vertices = glyphs * kQuadVertices, .instances = 1},
                          font)) {
            active = false;
            return Result::SUCCESS;
        }
        clipRect = scissor(config.clip);

        if (!uploaded || uploadedVertex != slot->vertex) {
            auto vertices = painter->vertices(*slot);
            std::copy(geometry.begin(), geometry.end(), vertices.begin());
            painter->touch(*slot);
            uploaded = true;
            uploadedVertex = slot->vertex;
        }

        auto place = painter->instances(*slot);
        const F32 next[Painter::GlyphInstanceStride] = {
            translation.x, translation.y, config.sharpness, config.weight,
            config.emboss, config.embossBackdrop, 0.0f, 0.0f,
        };
        if (!std::equal(next, next + Painter::GlyphInstanceStride, place.begin())) {
            std::copy(next, next + Painter::GlyphInstanceStride, place.begin());
            painter->touchInstances(*slot);
        }

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
    const bool visualsChanged = instancesChanged || this->impl->config.clip != config.clip ||
                                this->impl->config.sharpness != config.sharpness ||
                                this->impl->config.weight != config.weight ||
                                this->impl->config.emboss != config.emboss ||
                                this->impl->config.embossBackdrop != config.embossBackdrop;
    const bool resourcesChanged = this->impl->requiredCapacity(config) != this->impl->capacity ||
                                  this->impl->config.fontName != config.fontName ||
                                  this->impl->config.maxCharacters != config.maxCharacters ||
                                  this->impl->resolvedFontChanged(config);
    this->impl->instancesDirty |= instancesChanged || resourcesChanged;
    this->impl->config = std::move(config);

    if (resourcesChanged && this->impl->attached()) {
        this->impl->capacity = this->impl->requiredCapacity(this->impl->config);
        invalidate(Dirty::Paint);
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
