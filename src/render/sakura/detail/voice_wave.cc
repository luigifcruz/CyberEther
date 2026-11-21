#include "voice_wave.hh"

#include <jetstream/render/base.hh>
#include <jetstream/render/components/voice_wave.hh>

#include "drawable.hh"
#include "helpers.hh"

#include <utility>

namespace Jetstream::Sakura {

struct VoiceWave::Impl : public Drawable {
    Config config;

    Render::Components::VoiceWave* wave() const {
        return as<Render::Components::VoiceWave>();
    }

    Result attach(Context* context, Render::Surface::Config& surfaceConfig) override {
        this->context = context;

        std::shared_ptr<Render::Components::VoiceWave> created;
        JST_CHECK(context->render->build(created, Render::Components::VoiceWave::Config{
            .framebufferSize = {static_cast<F32>(context->framebufferSize.x),
                                static_cast<F32>(context->framebufferSize.y)},
        }));
        return bind(context, std::move(created), surfaceConfig);
    }

    void conceal() override {
        if (component) {
            wave()->enabled(false);
        }
    }

    Result upload() override {
        if (!component || !context) {
            return Result::SUCCESS;
        }
        auto* wave = this->wave();
        wave->enabled(true);

        const Rect& rect = config.rect;
        JST_CHECK(wave->update(
            {static_cast<F32>(context->framebufferSize.x), static_cast<F32>(context->framebufferSize.y)},
            {
                .rect = {rect.x + translation.x, rect.y + translation.y, rect.width, rect.height},
                .pitch = config.pitch,
                .dot = config.dot,
                .cornerRadius = config.cornerRadius,
                .margin = config.margin,
                .time = config.time,
                .busy = config.busy,
                .activity = config.activity,
                .level = config.level,
                .listeningColor = config.listeningColor,
                .busyColor = config.busyColor,
                .dimColor = config.dimColor,
            }));
        JST_CHECK(wave->updateScissorRect(scissor(config.clip)));

        return Result::SUCCESS;
    }
};

VoiceWave::VoiceWave() {
    impl = std::make_unique<Impl>();
}

VoiceWave::~VoiceWave() = default;

const VoiceWave::Config& VoiceWave::config() const {
    return impl->config;
}

void VoiceWave::update(Config config) {
    markConfigured();
    const bool changed = config != impl->config;
    impl->config = std::move(config);
    if (changed) {
        invalidate(Dirty::Paint);
    }
}

Result VoiceWave::build(Context& ctx) {
    return ctx.attach(*this, impl.get());
}

Result VoiceWave::paint() {
    impl->translation = translation();
    impl->inherited = inheritedClip();
    return impl->paint();
}

void VoiceWave::conceal() {
    impl->conceal();
}

}  // namespace Jetstream::Sakura
