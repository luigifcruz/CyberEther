#include <jetstream/render/sakura/components/curves.hh>

#include <jetstream/render/base.hh>

#include "../detail/drawable.hh"
#include "../detail/helpers.hh"

#include <algorithm>
#include <utility>

namespace Jetstream::Sakura {

struct Curves::Impl : public Drawable {
    Config config;
    U64 drawn = 0;
    U64 uploadedInstance = 0;
    bool instancesDirty = true;
    bool active = false;
    Extent2D<F32> uploadedOrigin = {0.0f, 0.0f};
    F32 uploadedScale = 0.0f;
    F32 uploadedFringe = 0.0f;
    std::optional<Render::ScissorRect> clipRect;
    Painter::Slot slot;

    U64 count() const {
        return std::min<U64>(config.instances.size(), std::max<U64>(1, config.capacity));
    }

    U64 segments() const {
        return std::clamp<U64>(config.maxSegments, 1, Painter::CurveSegments);
    }

    Result attach(Context* context, Render::Surface::Config&) override {
        JST_CHECK(join(context));
        instancesDirty = true;
        painter->demand(Painter::Kind::Curves, {.instances = count()});
        return Result::SUCCESS;
    }

    void conceal() override {
        active = false;
    }

    void abandon() override {
        slot.reset();
        instancesDirty = true;
    }

    void steps(std::vector<Render::Surface::Step>& out) const override {
        if (!active || !slot.held() || drawn == 0) {
            return;
        }
        out.push_back({
            .program = painter->program(*slot),
            .range = Render::Draw::Range{
                .count = segments() * 18,
                .firstInstance = slot->instance,
                .instanceCount = drawn,
            },
            .scissor = clipRect,
        });
    }

    Result upload() override {
        if (!painter || !context) {
            return Result::SUCCESS;
        }

        const U64 instances = count();
        active = instances > 0;
        if (!active) {
            return Result::SUCCESS;
        }
        if (!slot.reserve(*painter, Painter::Kind::Curves, {.instances = instances})) {
            active = false;
            return Result::SUCCESS;
        }
        clipRect = scissor(config.clip);

        const Extent2D<F32> origin = {config.origin.x + translation.x, config.origin.y + translation.y};
        if (!instancesDirty && uploadedInstance == slot->instance && uploadedOrigin == origin &&
            uploadedScale == config.scale && uploadedFringe == config.fringe) {
            return Result::SUCCESS;
        }
        instancesDirty = false;
        uploadedInstance = slot->instance;
        uploadedOrigin = origin;
        uploadedScale = config.scale;
        uploadedFringe = config.fringe;

        const F32 maxSegments = static_cast<F32>(segments());
        auto data = painter->instances(*slot);
        for (U64 i = 0; i < instances; ++i) {
            const auto& source = config.instances[i];
            const F32 next[Painter::CurveStride] = {
                source.p0.x, source.p0.y, source.p1.x, source.p1.y,
                source.p2.x, source.p2.y, source.p3.x, source.p3.y,
                source.color.r, source.color.g, source.color.b, source.color.a,
                source.thickness, std::clamp(static_cast<F32>(source.segments), 1.0f, maxSegments), 0.0f, 0.0f,
                origin.x, origin.y, config.scale, config.fringe,
            };
            std::copy(next, next + Painter::CurveStride, data.begin() + static_cast<I64>(i * Painter::CurveStride));
        }
        drawn = instances;
        painter->touch(*slot);

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
    const bool instancesChanged = impl->config.instances != config.instances ||
                                  impl->config.capacity != config.capacity ||
                                  impl->config.maxSegments != config.maxSegments;
    const bool changed = instancesChanged ||
                         impl->config.origin != config.origin ||
                         impl->config.scale != config.scale ||
                         impl->config.clip != config.clip ||
                         impl->config.fringe != config.fringe;
    impl->config = std::move(config);
    if (instancesChanged) {
        impl->instancesDirty = true;
    }
    if (changed) {
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
