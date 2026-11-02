#include <jetstream/render/sakura/components/log_view.hh>

#include <jetstream/render/sakura/components/text_view.hh>

#include "../context.hh"
#include "../metrics.hh"

#include <limits>
#include <utility>

namespace Jetstream::Sakura {

struct LogView::Impl {
    Config config;
    TextView view;
    F32 pixelRatio = 0.0f;
    bool stale = true;

    void sync(const Context& ctx) {
        if (!stale && ctx.pixelRatio == pixelRatio) {
            return;
        }
        stale = false;
        pixelRatio = ctx.pixelRatio;
        const F32 padding = config.padding * pixelRatio;
        view.update({
            .id = jst::fmt::format("log:{}", static_cast<const void*>(this)),
            .value = config.value,
            .fontSize = config.fontSize * pixelRatio,
            .fontName = Typography::MonoFont,
            .monospace = true,
            .lineNumbers = false,
            .stickToBottom = config.stickToBottom,
            .scrollbar = true,
            .wrap = TextGrid::Wrap::Word,
            .padding = Padding{padding, padding, padding, padding},
            .backgroundColorKey = config.backgroundColorKey,
            .textColorKey = config.textColorKey,
        });
    }
};

LogView::LogView() {
    impl = std::make_unique<Impl>();
    setClipsChildren(true);
    addChild(impl->view);
}

LogView::~LogView() = default;

void LogView::update(Config config) {
    if (config == impl->config) {
        impl->config = std::move(config);
        return;
    }
    const auto& current = impl->config;
    impl->stale = impl->stale || config.value != current.value || config.fontSize != current.fontSize ||
                  config.padding != current.padding || config.stickToBottom != current.stickToBottom ||
                  config.backgroundColorKey != current.backgroundColorKey ||
                  config.textColorKey != current.textColorKey;
    impl->config = std::move(config);
    invalidate(Dirty::Layout);
}

Extent2D<F32> LogView::measure(const Context& ctx, Extent2D<F32> available) {
    impl->sync(ctx);
    const F32 width = FillLength(available.x);
    const F32 natural = impl->config.height == 0.0f
        ? measureChild(impl->view, ctx, {width, std::numeric_limits<F32>::infinity()}).y
        : 0.0f;
    return {width, ResolveLength(ctx, impl->config.height, natural, available.y)};
}

void LogView::layout(const Context& ctx) {
    impl->sync(ctx);
    layoutChild(ctx, impl->view, frame());
}

}  // namespace Jetstream::Sakura
