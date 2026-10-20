#include "orb.hh"

#include <jetstream/render/base.hh>
#include <jetstream/render/components/orb.hh>

#include "drawable.hh"
#include "helpers.hh"

#include <utility>

namespace Jetstream::Sakura {

struct Orb::Impl : public Drawable {
    Config config;

    Render::Components::Orb* orb() const {
        return as<Render::Components::Orb>();
    }

    Result attach(Context* context, Render::Surface::Config& surfaceConfig) override {
        this->context = context;

        std::shared_ptr<Render::Components::Orb> created;
        JST_CHECK(context->render->build(created, Render::Components::Orb::Config{
            .framebufferSize = {static_cast<F32>(context->framebufferSize.x),
                                static_cast<F32>(context->framebufferSize.y)},
        }));
        return bind(context, std::move(created), surfaceConfig);
    }

    void conceal() override {
        if (component) {
            orb()->enabled(false);
        }
    }

    Result upload() override {
        if (!component || !context) {
            return Result::SUCCESS;
        }
        auto* orb = this->orb();
        orb->enabled(true);

        const Rect& rect = config.rect;
        JST_CHECK(orb->update(
            {static_cast<F32>(context->framebufferSize.x), static_cast<F32>(context->framebufferSize.y)},
            {
                .rect = {rect.x + translation.x, rect.y + translation.y, rect.width, rect.height},
                .center = {config.center.x + translation.x, config.center.y + translation.y},
                .radius = config.radius,
                .time = config.time,
                .busy = config.busy,
                .activity = config.activity,
                .level = config.level,
                .listeningColor = config.listeningColor,
                .busyColor = config.busyColor,
                .listeningAccent = config.listeningAccent,
                .busyAccent = config.busyAccent,
            }));
        JST_CHECK(orb->updateScissorRect(scissor(config.clip)));

        return Result::SUCCESS;
    }
};

Orb::Orb() {
    impl = std::make_unique<Impl>();
}

Orb::~Orb() = default;

const Orb::Config& Orb::config() const {
    return impl->config;
}

void Orb::update(Config config) {
    markConfigured();
    const bool changed = config != impl->config;
    impl->config = std::move(config);
    if (changed) {
        invalidate(Dirty::Paint);
    }
}

Result Orb::build(Context& ctx) {
    return ctx.attach(*this, impl.get());
}

Result Orb::paint() {
    impl->translation = translation();
    impl->inherited = inheritedClip();
    return impl->paint();
}

void Orb::conceal() {
    impl->conceal();
}

}  // namespace Jetstream::Sakura
