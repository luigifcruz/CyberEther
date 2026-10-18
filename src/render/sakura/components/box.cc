#include <jetstream/render/sakura/components/box.hh>

#include <jetstream/render/base.hh>
#include <jetstream/render/components/shapes.hh>

#include "../detail/drawable.hh"
#include "../detail/helpers.hh"

#include <algorithm>
#include <span>
#include <utility>
#include <vector>

namespace Jetstream::Sakura {

namespace {

constexpr U8 kStrokeFringe = 16;
constexpr U8 kSoft = 32;
constexpr U8 kTopLit = 64;

constexpr const char* kRectElementId = "rect";

}  // namespace

struct Box::Impl : public Drawable {
    Config config;
    U64 capacity = 0;
    U64 liveSlots = 0;
    bool instancesDirty = true;
    Extent2D<U64> uploadedFramebuffer = {0, 0};
    std::vector<U8> corners;

    Render::Components::Shapes* shape() const {
        return as<Render::Components::Shapes>();
    }

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

    Result attach(Context* context, Render::Surface::Config& surfaceConfig) override {
        this->context = context;
        capacity = requiredCapacity(config);
        liveSlots = 0;
        instancesDirty = true;

        Render::Components::Shapes::Config shapeConfig;
        shapeConfig.pixelSize = context->pixelSize();
        shapeConfig.elements[kRectElementId] = {
            .type = Render::Components::Shapes::Type::RECT,
            .numberOfInstances = capacity,
            .position = {-2.0f, -2.0f},
            .size = {0.0f, 0.0f},
            .cornerRadius = config.cornerRadius,
            .borderWidth = edgeWidth(),
            .borderColor = config.borderColor,
        };

        std::shared_ptr<Render::Components::Shapes> shapes;
        JST_CHECK(context->render->build(shapes, shapeConfig));
        return bind(context, std::move(shapes), surfaceConfig);
    }

    void conceal() override {
        active = false;
        if (component) {
            shape()->enabled(false);
        }
    }

    bool covers(const Rect& target) const override {
        if (!active || !component || !context || config.softness > 0.0f) {
            return false;
        }
        std::optional<Rect> clip;
        if (const auto bounds = scissor(config.clip)) {
            clip = Rect{static_cast<F32>(bounds->x), static_cast<F32>(bounds->y), static_cast<F32>(bounds->width),
                        static_cast<F32>(bounds->height)};
        }
        return std::any_of(config.instances.begin(), config.instances.end(), [&](const auto& instance) {
            const Rect shape = {instance.rect.x + translation.x, instance.rect.y + translation.y, instance.rect.width,
                                instance.rect.height};
            return instance.visible && OpaqueCovers(shape, config.cornerRadius, instance.backgroundColor, clip, target);
        });
    }

    bool active = false;

    Result upload() override {
        if (!component || !context) {
            return Result::SUCCESS;
        }
        auto* shape = this->shape();

        active = std::any_of(
            config.instances.begin(), config.instances.end(), [](const auto& instance) {
                return instance.visible && !instance.rect.empty();
            });
        shape->enabled(active);
        if (!active) {
            return Result::SUCCESS;
        }

        const auto& framebufferSize = context->framebufferSize;

        JST_CHECK(shape->updatePixelSize(context->pixelSize()));
        JST_CHECK(shape->updateTranslation(translationNdc()));
        JST_CHECK(shape->updateScissorRect(scissor(config.clip)));

        if (!instancesDirty && uploadedFramebuffer == framebufferSize) {
            return Result::SUCCESS;
        }
        instancesDirty = false;
        uploadedFramebuffer = framebufferSize;

        std::span<Extent2D<F32>> positions;
        JST_CHECK(shape->getPositions(kRectElementId, positions));
        std::span<Extent2D<F32>> sizes;
        JST_CHECK(shape->getSizes(kRectElementId, sizes));
        std::span<ColorRGBA<F32>> colors;
        JST_CHECK(shape->getColors(kRectElementId, colors));
        corners.assign(capacity, AllCorners | flags());

        const U64 slots = std::min(capacity, std::max<U64>(config.instances.size(), liveSlots));
        U64 nextLiveSlots = 0;
        for (U64 i = 0; i < slots; ++i) {
            const bool on = i < config.instances.size() &&
                            config.instances[i].visible &&
                            !config.instances[i].rect.empty();
            if (!on) {
                positions[i] = {-2.0f, -2.0f};
                sizes[i] = {0.0f, 0.0f};
                continue;
            }
            nextLiveSlots = i + 1;

            const auto& instance = config.instances[i];
            positions[i] = PixelToNdc(framebufferSize, instance.rect.center().x, instance.rect.center().y);
            sizes[i] = {instance.rect.width, instance.rect.height};
            colors[i] = instance.backgroundColor;
            corners[i] = instance.corners | flags();
        }

        liveSlots = nextLiveSlots;

        JST_CHECK(shape->updatePositions(kRectElementId));
        JST_CHECK(shape->updateSizes(kRectElementId));
        JST_CHECK(shape->updateColors(kRectElementId));

        JST_CHECK(shape->updateProperties(kRectElementId, config.cornerRadius, edgeWidth(), config.borderColor));
        JST_CHECK(shape->updateCorners(kRectElementId, corners));

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
    this->impl->instancesDirty |= instancesChanged;
    this->impl->config = std::move(config);

    if (capacityChanged && this->impl->attached()) {
        invalidate(Dirty::Resource);
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
