#include <jetstream/render/sakura/components/curves.hh>

#include <jetstream/render/base.hh>
#include <jetstream/render/components/curves.hh>

#include "../detail/drawable.hh"
#include "../detail/helpers.hh"

#include <algorithm>
#include <utility>

namespace Jetstream::Sakura {

struct Curves::Impl : public Drawable {
    Config config;
    bool instancesDirty = true;

    Render::Components::Curves* curves() const {
        return as<Render::Components::Curves>();
    }

    Result attach(Context* context, Render::Surface::Config& surfaceConfig) override {
        this->context = context;

        Render::Components::Curves::Config curvesConfig;
        curvesConfig.capacity = std::max<U64>(1, config.capacity);
        curvesConfig.maxSegments = std::max<U64>(1, config.maxSegments);
        curvesConfig.framebufferSize = {
            static_cast<F32>(context->framebufferSize.x),
            static_cast<F32>(context->framebufferSize.y),
        };

        std::shared_ptr<Render::Components::Curves> created;
        JST_CHECK(context->render->build(created, curvesConfig));
        instancesDirty = true;
        return bind(context, std::move(created), surfaceConfig);
    }

    void conceal() override {
        if (component) {
            curves()->enabled(false);
        }
    }

    Result upload() override {
        if (!component || !context) {
            return Result::SUCCESS;
        }
        auto* curves = this->curves();
        curves->enabled(true);

        JST_CHECK(curves->updateTransform(
            {static_cast<F32>(context->framebufferSize.x), static_cast<F32>(context->framebufferSize.y)},
            {config.origin.x + translation.x, config.origin.y + translation.y},
            config.scale,
            config.fringe));

        if (instancesDirty) {
            const auto& capacity = curves->getConfig();
            auto destination = curves->instances();
            const U64 count = std::min<U64>(config.instances.size(), capacity.capacity);
            const F32 maxSegments = static_cast<F32>(capacity.maxSegments);
            for (U64 i = 0; i < count; ++i) {
                const auto& source = config.instances[i];
                destination[i] = {
                    .p0 = source.p0,
                    .p1 = source.p1,
                    .p2 = source.p2,
                    .p3 = source.p3,
                    .color = source.color,
                    .thickness = source.thickness,
                    .segments = std::clamp(static_cast<F32>(source.segments), 1.0f, maxSegments),
                };
            }
            JST_CHECK(curves->update(count));
            instancesDirty = false;
        }

        JST_CHECK(curves->updateScissorRect(scissor(config.clip)));

        return Result::SUCCESS;
    }
};

Curves::Curves() {
    this->impl = std::make_unique<Impl>();
}

Curves::~Curves() = default;

const Curves::Config& Curves::config() const {
    return impl->config;
}

void Curves::update(Config config) {
    markConfigured();
    const bool resized = impl->config.capacity != config.capacity ||
                         impl->config.maxSegments != config.maxSegments;
    const bool instancesChanged = impl->config.instances != config.instances;
    const bool changed = instancesChanged ||
                         impl->config.origin != config.origin ||
                         impl->config.scale != config.scale ||
                         impl->config.clip != config.clip ||
                         impl->config.fringe != config.fringe;
    impl->config = std::move(config);
    if (instancesChanged) {
        impl->instancesDirty = true;
    }
    if (resized && impl->attached()) {
        invalidate(Dirty::Resource);
    } else if (changed) {
        invalidate(Dirty::Paint);
    }
}

Result Curves::build(Context& ctx) {
    return ctx.attach(*this, this->impl.get());
}

Result Curves::paint() {
    this->impl->translation = translation();
    this->impl->inherited = inheritedClip();
    return this->impl->paint();
}

void Curves::conceal() {
    this->impl->conceal();
}

}  // namespace Jetstream::Sakura
