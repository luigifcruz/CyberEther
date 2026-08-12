#include <jetstream/render/sakura/components/retained/box.hh>

#include <jetstream/render/base.hh>
#include <jetstream/render/components/shapes.hh>

#include "../../retained/drawable.hh"
#include "../../retained/helpers.hh"

#include <algorithm>
#include <span>
#include <utility>

namespace Jetstream::Sakura::Retained {

namespace {

constexpr const char* kRectElementId = "rect";

}  // namespace

struct Box::Impl : public Drawable {
    Config config;
    Context* context = nullptr;
    std::shared_ptr<Render::Components::Shapes> shape;
    U64 capacity = 0;

    ~Impl() override {
        if (context && context->release) {
            context->release(this);
        }
    }

    bool sameVisuals(const Config& other) const {
        return config.instances == other.instances &&
               config.clip == other.clip &&
               config.cornerRadius == other.cornerRadius &&
               config.borderWidth == other.borderWidth &&
               config.borderColor == other.borderColor;
    }

    U64 requiredCapacity(const Config& other) const {
        return other.capacity > 0 ? other.capacity : std::max<U64>(1, other.instances.size());
    }

    Result attach(Context* context, Render::Surface::Config& surfaceConfig) override {
        this->context = context;
        capacity = requiredCapacity(config);

        Render::Components::Shapes::Config shapeConfig;
        shapeConfig.pixelSize = context->pixelSize();
        shapeConfig.elements[kRectElementId] = {
            .type = Render::Components::Shapes::Type::RECT,
            .numberOfInstances = capacity,
            .position = {-2.0f, -2.0f},
            .size = {0.0f, 0.0f},
            .cornerRadius = config.cornerRadius,
            .borderWidth = config.borderWidth,
            .borderColor = config.borderColor,
        };

        JST_CHECK(context->render->build(shape, shapeConfig));
        JST_CHECK(context->render->bind(shape));
        JST_CHECK(shape->surface(surfaceConfig));

        return Result::SUCCESS;
    }

    Result detach(Render::Window* window) override {
        if (shape && window) {
            JST_CHECK(window->unbind(shape));
        }
        shape.reset();
        context = nullptr;
        return Result::SUCCESS;
    }

    Result upload() override {
        if (!shape || !context) {
            return Result::SUCCESS;
        }

        const auto& framebufferSize = context->framebufferSize;

        JST_CHECK(shape->updatePixelSize(context->pixelSize()));

        std::span<Extent2D<F32>> positions;
        JST_CHECK(shape->getPositions(kRectElementId, positions));
        std::span<Extent2D<F32>> sizes;
        JST_CHECK(shape->getSizes(kRectElementId, sizes));
        std::span<ColorRGBA<F32>> colors;
        JST_CHECK(shape->getColors(kRectElementId, colors));

        for (U64 i = 0; i < capacity; ++i) {
            const bool on = i < config.instances.size() &&
                            config.instances[i].visible &&
                            !config.instances[i].rect.empty();
            if (!on) {
                positions[i] = {-2.0f, -2.0f};
                sizes[i] = {0.0f, 0.0f};
                continue;
            }

            const auto& instance = config.instances[i];
            positions[i] = PixelToNdc(framebufferSize, instance.rect.center().x, instance.rect.center().y);
            sizes[i] = {instance.rect.width, instance.rect.height};
            colors[i] = instance.backgroundColor;
        }

        JST_CHECK(shape->updatePositions(kRectElementId));
        JST_CHECK(shape->updateSizes(kRectElementId));
        JST_CHECK(shape->updateColors(kRectElementId));

        JST_CHECK(shape->updateProperties(kRectElementId, config.cornerRadius,
                                          config.borderWidth, config.borderColor));

        JST_CHECK(shape->updateScissorRect(config.clip.has_value()
            ? std::optional<Render::ScissorRect>(RectToScissor(*config.clip, framebufferSize))
            : std::nullopt));

        return Result::SUCCESS;
    }

    Result present() override {
        if (shape) {
            JST_CHECK(shape->present());
        }
        return Result::SUCCESS;
    }
};

Box::Box() {
    this->impl = std::make_unique<Impl>();
}

Box::~Box() = default;

bool Box::update(Config config) {
    const bool visualsChanged = !this->impl->sameVisuals(config);
    const bool capacityChanged = this->impl->requiredCapacity(config) != this->impl->capacity;
    this->impl->config = std::move(config);

    if (capacityChanged) {
        invalidate(Dirty::Resource);
    } else if (visualsChanged) {
        invalidate(Dirty::Paint);
    }
    return true;
}

Result Box::build(Context& ctx) {
    ctx.drawables->push_back(this->impl.get());
    return this->impl->attach(&ctx, *ctx.surface);
}

Result Box::paint() {
    JST_CHECK(this->impl->upload());
    JST_CHECK(this->impl->present());
    return Result::SUCCESS;
}

}  // namespace Jetstream::Sakura::Retained
