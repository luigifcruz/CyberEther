#include <jetstream/render/sakura/components/chat.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/chat_composer.hh>

#include "../context.hh"
#include "../metrics.hh"
#include "../detail/chat_transcript.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>

namespace Jetstream::Sakura {

namespace {

constexpr F32 kSurfacePadding = 8.0f;
constexpr F32 kUnbounded = std::numeric_limits<F32>::infinity();

}  // namespace

struct Chat::Impl {
    Config config;
    Box background;
    ChatTranscript transcript;
    Box cover;
    ChatComposer composer;

    F32 padding(const Context& ctx) const {
        return Px(ctx, kSurfacePadding * config.fontSize / Typography::FontSize);
    }

    F32 column(const Context& ctx, F32 width) const {
        F32 column = std::max(0.0f, width - 2.0f * padding(ctx));
        if (config.maxContentWidth > 0.0f) {
            column = std::min(column, Px(ctx, config.maxContentWidth));
        }
        return column;
    }

    Context surface(const Context& ctx) const {
        Context inner = ctx;
        inner.backdropColorKey = config.surfaceColorKey;
        inner.backdropTopColorKey.clear();
        return inner;
    }
};

Chat::Chat() {
    impl = std::make_unique<Impl>();
    setClipsChildren(true);
    addChild(impl->background);
    addChild(impl->transcript);
    addChild(impl->cover);
    addChild(impl->composer);
}

Chat::~Chat() = default;

void Chat::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
    const auto& current = impl->config;
    impl->transcript.update({
        .id = current.id + ":transcript",
        .snapshot = current.snapshot,
        .fontSize = current.fontSize,
        .surfaceColorKey = current.surfaceColorKey,
        .tailRequest = current.clearRequest,
    });
    impl->composer.update({
        .id = current.id + ":composer",
        .style = current.composer,
        .busy = current.snapshot.streaming,
        .selector = current.selector,
        .usage = current.usage,
        .onSubmit = current.onSubmit,
        .onCancel = current.onCancel,
        .onClear = current.onClear,
        .onVoice = current.onVoice,
        .onFocusChange = current.onFocusChange,
        .focusRequest = current.focusRequest,
        .clearRequest = current.clearRequest,
    });
}

Extent2D<F32> Chat::measure(const Context& ctx, Extent2D<F32> available) {
    const F32 pad = impl->padding(ctx);
    const auto composer = measureChild(impl->composer, impl->surface(ctx), {impl->column(ctx, available.x), kUnbounded});
    return {FillLength(available.x, composer.x + 2.0f * pad), FitLength(composer.y + 2.0f * pad, available.y)};
}

void Chat::layout(const Context& ctx) {
    auto& state = *impl;
    const auto& config = state.config;
    const Rect bounds = frame();
    const Context inner = state.surface(ctx);
    const F32 pad = state.padding(ctx);

    const auto background = ctx.color(config.backgroundColorKey);
    state.background.update({
        .instances = {{.rect = bounds, .visible = !bounds.empty() && background.a > 0.0f,
                       .backgroundColor = background}},
        .clip = bounds,
    });
    layoutChild(ctx, state.background, bounds);

    const F32 column = state.column(ctx, bounds.width);
    const F32 columnX = bounds.x + std::round((bounds.width - column) * 0.5f);
    const F32 composerHeight = measureChild(state.composer, inner, {column, kUnbounded}).y;
    const Rect composerRect = {columnX, bounds.bottom() - pad - composerHeight, column, composerHeight};
    const F32 top = bounds.y + pad;
    const Rect transcriptRect = {columnX, top, column, std::max(0.0f, composerRect.y - pad - top)};
    const F32 opacity = std::clamp(config.transcriptOpacity, 0.0f, 1.0f);
    const bool shown = opacity > 0.0f;
    if (state.transcript.visible() != shown) {
        if (!shown) {
            cancelMouse(state.transcript);
            releaseFocusWithin(state.transcript);
        }
        state.transcript.setVisible(shown);
    }
    layoutChild(inner, state.transcript, transcriptRect);

    state.cover.update({
        .instances = {{.rect = transcriptRect, .visible = shown && opacity < 1.0f && !transcriptRect.empty(),
                       .backgroundColor = WithAlpha(ctx.color(config.surfaceColorKey), 1.0f - opacity)}},
        .clip = transcriptRect,
    });
    layoutChild(ctx, state.cover, transcriptRect);
    layoutChild(inner, state.composer, composerRect);
}

}  // namespace Jetstream::Sakura
