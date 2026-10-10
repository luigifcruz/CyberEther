#include <jetstream/render/sakura/components/retained/chat.hh>

#include <jetstream/logger.hh>
#include <jetstream/render/sakura/components/retained/box.hh>
#include <jetstream/render/sakura/components/retained/chat_composer.hh>

#include "../../context.hh"
#include "../../state.hh"
#include "chat_transcript.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>

namespace Jetstream::Sakura::Retained {

namespace {

constexpr F32 kSurfacePadding = 8.0f;

struct ChatBody : public Component {
    Chat::Config config;
    U64 clearRequest = 0;
    Box background;
    ChatMessageList transcript;
    ChatComposer composer;

    ChatBody() : transcript(config) {
        setClipsChildren(true);
        add(background);
        add(transcript);
        add(composer);
    }

    void apply(Chat::Config next) {
        if (next.id != config.id ||
            next.snapshot.generation != config.snapshot.generation) {
            transcript.reset();
        }
        if (config.snapshot != next.snapshot || config.id != next.id ||
            config.maxContentWidth != next.maxContentWidth ||
            config.transcriptOpacity != next.transcriptOpacity ||
            config.backgroundColorKey != next.backgroundColorKey ||
            config.surfaceColorKey != next.surfaceColorKey ||
            config.composer != next.composer) {
            invalidate(Dirty::Paint);
        }
        if (Private::ConsumeRequest(clearRequest, next.clearRequest)) {
            transcript.stickToTail();
        }
        config = std::move(next);
        syncComposer();
    }

    void syncComposer() {
        composer.update({
            .id = jst::fmt::format("{}:composer", config.id),
            .style = config.composer,
            .busy = config.snapshot.streaming,
            .selector = config.selector,
            .usage = config.usage,
            .onSubmit = config.onSubmit,
            .onCancel = config.onCancel,
            .onClear = config.onClear,
            .onVoice = config.onVoice,
            .focusRequest = config.focusRequest,
            .clearRequest = config.clearRequest,
        });
    }

    F32 unit() const {
        return config.fontSize / Typography::FontSize;
    }

    F32 columnWidth(F32 width) const {
        const F32 pad = kSurfacePadding * unit();
        F32 column = std::max(0.0f, width - 2.0f * pad);
        if (config.maxContentWidth > 0.0f) {
            column = std::min(column, config.maxContentWidth);
        }
        return column;
    }

    F32 composerHeightAt(const Context& ctx, F32 column) {
        return measureChild(composer, ctx, {column, std::numeric_limits<F32>::infinity()}).y;
    }

    Extent2D<F32> measureComposer(const Context& ctx, Extent2D<F32> available) {
        const F32 pad = kSurfacePadding * unit();
        return {available.x, composerHeightAt(ctx, columnWidth(available.x)) + 2.0f * pad};
    }

 protected:
    void layout(const Context& ctx) override {
        const Rect bounds = frame();
        const F32 pad = kSurfacePadding * unit();
        const bool visible = !bounds.empty();

        background.update({
            .id = jst::fmt::format("{}:bg", config.id),
            .instances = {{.rect = bounds, .visible = visible, .backgroundColor = ctx.color(config.backgroundColorKey)}},
        });
        layoutChild(ctx, background, bounds);

        const F32 column = columnWidth(bounds.width);
        const F32 columnX = bounds.x + std::round((bounds.width - column) * 0.5f);
        const F32 composerHeight = composerHeightAt(ctx, column);

        const Rect composerRect = {
            columnX,
            bounds.bottom() - pad - composerHeight,
            column,
            std::max(0.0f, composerHeight),
        };
        const Rect transcriptRect = {
            columnX,
            bounds.y + pad,
            column,
            std::max(0.0f, composerRect.y - pad - (bounds.y + pad)),
        };
        Context faded = ctx;
        faded.opacity *= config.transcriptOpacity;
        layoutChild(faded, transcript, transcriptRect);
        layoutChild(ctx, composer, composerRect);
    }
};

}  // namespace

struct Chat::Impl {
    ChatBody body;
};

Chat::Chat() {
    impl = std::make_unique<Impl>();
    setClipsChildren(true);
    add(impl->body);
}

Chat::~Chat() = default;

bool Chat::update(Config config) {
    impl->body.apply(std::move(config));
    return true;
}

Extent2D<F32> Chat::measure(const Context& ctx, Extent2D<F32> available) {
    return impl->body.measureComposer(ctx, available);
}

void Chat::layout(const Context& ctx) {
    layoutChild(ctx, impl->body, frame());
}

}  // namespace Jetstream::Sakura::Retained
