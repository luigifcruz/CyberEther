#include <jetstream/render/sakura/components/box.hh>

#include <jetstream/render/base.hh>

#include "../detail/drawable.hh"
#include "../detail/helpers.hh"
#include "../detail/occlusion.hh"

#include <algorithm>
#include <span>
#include <utility>
#include <vector>

namespace Jetstream::Sakura {

namespace {

constexpr U8 kStrokeFringe = 16;
constexpr U8 kSoft = 32;
constexpr U8 kTopLit = 64;
constexpr F32 kCoverFringe = 1.0f;
constexpr F32 kRectType = 1.0f;
constexpr U64 kFill = 16;
constexpr U64 kBorder = 20;
constexpr U64 kParams = 24;
constexpr U64 kHole = 28;

}  // namespace

struct Box::Impl : public Drawable {
    Config config;
    U64 capacity = 0;
    U64 drawn = 0;
    bool instancesDirty = true;
    Extent2D<U64> uploadedFramebuffer = {0, 0};
    Extent2D<F32> uploadedTranslation = {0.0f, 0.0f};
    U64 uploadedInstance = 0;
    std::optional<Render::ScissorRect> clipRect;
    std::vector<Rect> holes;
    Painter::Slot slot;

    bool sameInstances(const Config& other) const {
        return config.instances == other.instances &&
               config.cornerRadius == other.cornerRadius &&
               config.borderWidth == other.borderWidth &&
               config.borderColor == other.borderColor &&
               config.strokeFringe == other.strokeFringe &&
               config.softness == other.softness &&
               config.topLit == other.topLit;
    }

    U8 flags() const {
        return (config.strokeFringe ? kStrokeFringe : 0) | (config.softness > 0.0f ? kSoft : 0) |
               (config.topLit ? kTopLit : 0);
    }

    F32 edgeWidth() const {
        return config.softness > 0.0f ? config.softness : config.borderWidth;
    }

    U64 requiredCapacity(const Config& other) const {
        if (other.capacity > 0) {
            return other.capacity;
        }
        return std::max<U64>({1, capacity, other.instances.size()});
    }

    Result attach(Context* context, Render::Surface::Config&) override {
        JST_CHECK(join(context));
        capacity = requiredCapacity(config);
        drawn = 0;
        instancesDirty = true;
        holes.assign(capacity, Rect{});
        painter->demand(Painter::Kind::Shapes, {.instances = rendered().size()});
        return Result::SUCCESS;
    }

    void conceal() override {
        active = false;
    }

    void abandon() override {
        slot.reset();
        drawn = 0;
    }

    void steps(std::vector<Render::Surface::Step>& out) const override {
        if (!active || !slot.held() || drawn == 0) {
            return;
        }
        out.push_back({
            .program = painter->program(*slot),
            .range = Render::Draw::Range{
                .count = Painter::ShapeIndices,
                .firstInstance = slot->instance,
                .instanceCount = drawn,
            },
            .scissor = clipRect,
        });
    }

    std::span<const Instance> rendered() const {
        return std::span<const Instance>(config.instances).first(std::min<U64>(config.instances.size(), capacity));
    }

    std::optional<Rect> screenClip() const {
        if (const auto bounds = scissor(config.clip)) {
            return Rect{static_cast<F32>(bounds->x), static_cast<F32>(bounds->y), static_cast<F32>(bounds->width),
                        static_cast<F32>(bounds->height)};
        }
        return std::nullopt;
    }

    bool covers(const Rect& target) const override {
        if (!active || !painter || !context || config.softness > 0.0f) {
            return false;
        }
        const auto clip = screenClip();
        const auto instances = rendered();
        const F32 ring = (config.borderColor.a < 1.0f ? std::max(0.0f, config.borderWidth) : 0.0f) + kCoverFringe;
        const F32 radius = std::max(0.0f, config.cornerRadius - ring);
        return std::any_of(instances.begin(), instances.end(), [&](const auto& instance) {
            const Rect shape = Inset({instance.rect.x + translation.x, instance.rect.y + translation.y,
                                      instance.rect.width, instance.rect.height},
                                     ring);
            return instance.visible && OpaqueCovers(shape, radius, instance.backgroundColor, clip, target);
        });
    }

    bool active = false;

    void opaqueRects(std::vector<Rect>& rects) const override {
        if (!active || !painter || !context) {
            return;
        }
        const auto clip = screenClip();
        for (const auto& instance : rendered()) {
            if (!instance.visible) {
                continue;
            }
            const Rect interior = OpaqueInterior(instance.rect.translated(translation.x, translation.y),
                                                 config.cornerRadius, config.borderWidth, config.softness,
                                                 instance.backgroundColor, clip);
            if (!interior.empty()) {
                rects.push_back(interior);
            }
        }
    }

    Result occlude(const std::vector<Rect>& later) override {
        if (!active || !painter || !context) {
            return Result::SUCCESS;
        }
        const auto instances = rendered();
        std::span<F32> data;
        if (slot.held()) {
            data = painter->instances(*slot);
        }
        bool changed = false;
        for (U64 i = 0; i < holes.size(); ++i) {
            Rect hole;
            if (i < instances.size() && instances[i].visible && !instances[i].rect.empty()) {
                const auto& instance = instances[i];
                const Rect screen = instance.rect.translated(translation.x, translation.y);
                const Rect own = HollowInterior(screen, config.cornerRadius, config.borderWidth,
                                                config.softness, instance.backgroundColor);
                hole = LocalHole(screen, LargestHole(screen, own, later));
            }
            if (holes[i] != hole) {
                holes[i] = hole;
                if ((i + 1) * Painter::ShapeStride <= data.size()) {
                    F32* out = data.data() + i * Painter::ShapeStride + kHole;
                    out[0] = hole.x;
                    out[1] = hole.y;
                    out[2] = hole.width;
                    out[3] = hole.height;
                }
                changed = true;
            }
        }
        if (changed && slot.held()) {
            painter->touch(*slot);
        }
        return Result::SUCCESS;
    }

    Result upload() override {
        if (!painter || !context) {
            return Result::SUCCESS;
        }

        const Rect visible = screenClip().value_or(Rect{0.0f, 0.0f, static_cast<F32>(context->framebufferSize.x),
                                                        static_cast<F32>(context->framebufferSize.y)});
        const F32 fringe = 2.0f + 2.0f * std::max(config.borderWidth, config.softness);
        const auto instances = rendered();
        active = std::any_of(
            instances.begin(), instances.end(), [&](const auto& instance) {
                const Rect bounds = Inset(instance.rect.translated(translation.x, translation.y), -fringe);
                return instance.visible && !instance.rect.empty() && !Intersect(bounds, visible).empty();
            });
        if (!active) {
            return Result::SUCCESS;
        }

        if (!slot.reserve(*painter, Painter::Kind::Shapes, {.instances = instances.size()})) {
            active = false;
            return Result::SUCCESS;
        }

        const auto& framebufferSize = context->framebufferSize;
        clipRect = scissor(config.clip);

        if (!instancesDirty && uploadedFramebuffer == framebufferSize && uploadedTranslation == translation &&
            uploadedInstance == slot->instance) {
            return Result::SUCCESS;
        }
        instancesDirty = false;
        uploadedFramebuffer = framebufferSize;
        uploadedTranslation = translation;
        uploadedInstance = slot->instance;

        const auto pixel = context->pixelSize();
        const auto shift = translationNdc();
        const F32 edge = edgeWidth();
        auto data = painter->instances(*slot);
        std::fill(data.begin(), data.end(), 0.0f);
        drawn = 0;
        for (U64 i = 0; i < instances.size(); ++i) {
            const auto& instance = instances[i];
            F32* out = data.data() + i * Painter::ShapeStride;
            out[10] = 1.0f;
            out[15] = 1.0f;
            if (!instance.visible || instance.rect.empty()) {
                out[12] = -2.0f;
                out[13] = -2.0f;
                continue;
            }
            drawn = i + 1;

            const auto center = PixelToNdc(framebufferSize, instance.rect.center().x, instance.rect.center().y);
            out[0] = instance.rect.width * pixel.x;
            out[5] = instance.rect.height * pixel.y;
            out[12] = center.x + shift.x;
            out[13] = center.y + shift.y;
            out[kFill + 0] = instance.backgroundColor.r;
            out[kFill + 1] = instance.backgroundColor.g;
            out[kFill + 2] = instance.backgroundColor.b;
            out[kFill + 3] = instance.backgroundColor.a;
            out[kBorder + 0] = config.borderColor.r;
            out[kBorder + 1] = config.borderColor.g;
            out[kBorder + 2] = config.borderColor.b;
            out[kBorder + 3] = config.borderColor.a;
            out[kParams + 0] = kRectType;
            out[kParams + 1] = edge;
            out[kParams + 2] = config.cornerRadius;
            out[kParams + 3] = static_cast<F32>(instance.corners | flags());
            if (i < holes.size()) {
                out[kHole + 0] = holes[i].x;
                out[kHole + 1] = holes[i].y;
                out[kHole + 2] = holes[i].width;
                out[kHole + 3] = holes[i].height;
            }
        }
        painter->touch(*slot);

        return Result::SUCCESS;
    }
};

Box::Box() {
    this->impl = std::make_unique<Impl>();
}

Box::~Box() = default;

const Box::Config& Box::config() const {
    return impl->config;
}

void Box::update(Config config) {
    markConfigured();
    const bool instancesChanged = !this->impl->sameInstances(config);
    const bool visualsChanged = instancesChanged || this->impl->config.clip != config.clip;
    const bool capacityChanged = this->impl->requiredCapacity(config) != this->impl->capacity;
    this->impl->instancesDirty |= instancesChanged || capacityChanged;
    this->impl->config = std::move(config);

    if (capacityChanged && this->impl->attached()) {
        this->impl->capacity = this->impl->requiredCapacity(this->impl->config);
        this->impl->holes.resize(this->impl->capacity);
        invalidate(Dirty::Paint);
    } else if (visualsChanged) {
        invalidate(Dirty::Paint);
    }
}

Result Box::build(Context& ctx) {
    return ctx.attach(*this, this->impl.get());
}

Result Box::paint() {
    this->impl->translation = translation();
    this->impl->inherited = inheritedClip();
    return this->impl->paint();
}

void Box::conceal() {
    this->impl->conceal();
}

}  // namespace Jetstream::Sakura
