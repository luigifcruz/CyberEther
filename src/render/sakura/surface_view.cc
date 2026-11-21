#include "surface_view.hh"

#include "metrics.hh"
#include "surface_presenter.hh"
#include "detail/helpers.hh"

#include <cmath>
#include <optional>

namespace Jetstream::Sakura {

namespace {

constexpr F32 kFringe = 1.0f;

Extent2D<F32> ResolveSurfaceLogicalDrawSize(const SurfaceView::Config& config,
                                            const Extent2D<F32>& available) {
    Extent2D<F32> size = config.size;
    if (size.x <= 0.0f) {
        size.x = available.x;
    }
    if (config.height.has_value()) {
        size.y = std::isfinite(*config.height) ? std::max(0.0f, *config.height) : 0.0f;
    } else if (size.y <= 0.0f) {
        size.y = available.y;
    }
    if (size.x <= 0.0f || size.y <= 0.0f) {
        return {0.0f, 0.0f};
    }

    return size;
}

}  // namespace

struct SurfaceView::Impl {
    Config config;
    Private::SurfacePresenter presenter;
};

SurfaceView::SurfaceView() {
    this->impl = std::make_unique<Impl>();
}

SurfaceView::~SurfaceView() = default;
SurfaceView::SurfaceView(SurfaceView&&) noexcept = default;
SurfaceView& SurfaceView::operator=(SurfaceView&&) noexcept = default;

void SurfaceView::update(Config config) {
    if (this->impl->config.id != config.id ||
        this->impl->config.textureSource != config.textureSource) {
        this->impl->presenter.resetResize();
    }
    if (this->impl->config.id != config.id) {
        this->impl->presenter.resetSettle();
    }
    this->impl->config = std::move(config);
}

SurfaceView::Frame SurfaceView::render(const Context& ctx, const Rect& region, const Rect& clip) const {
    const auto& config = impl->config;
    const U64 frame = ctx.render->frameIndex();
    impl->presenter.beginFrame(frame);

    const Extent2D<F32> origin = DisplayMetrics::Snap(Metrics(ctx), {region.x, region.y});
    Frame result{.rect = {origin.x, origin.y, 0.0f, 0.0f}};

    const Extent2D<F32> available = DisplayMetrics::LogicalSize(Metrics(ctx), {region.width, region.height});
    const Extent2D<F32> logicalDrawSize = ResolveSurfaceLogicalDrawSize(config, available);
    if (logicalDrawSize.x <= 0.0f || logicalDrawSize.y <= 0.0f) {
        return result;
    }
    const auto resolvedResize = ResolveSurfaceResize(ctx, logicalDrawSize);
    impl->presenter.emitResize(resolvedResize, config.onSize);
    Extent2D<F32> displaySize = Scale(ctx, logicalDrawSize);
    if (resolvedResize.has_value()) {
        displaySize = DisplayMetrics::FramebufferToDisplay(Metrics(ctx), resolvedResize->framebufferSize);
    }
    result.rect.width = displaySize.x;
    result.rect.height = displaySize.y;

    const std::shared_ptr<const Render::Texture> texture = config.resolveTexture
        ? config.resolveTexture()
        : config.textureSource;
    if (!texture || texture->raw() == 0) {
        return result;
    }

    const bool visible = Overlaps(Inflate(result.rect, kFringe), clip);
    if (config.textureSource && config.textureSource == texture) {
        config.textureSource->presentationHint(frame, visible);
    }
    const bool settled = impl->presenter.settle(config.textureSource.get(), resolvedResize, frame);
    if (visible && settled) {
        result.layer = Render::PresentList::Layer{
            .texture = texture,
            .rect = result.rect,
            .clip = clip,
        };
        result.blend = config.premultiplied ? Render::PresentList::Blend::Premultiplied
                                     : Render::PresentList::Blend::Straight;
    }

    return result;
}

}  // namespace Jetstream::Sakura
