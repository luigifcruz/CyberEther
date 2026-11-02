#include <jetstream/render/sakura/components/popup.hh>

#include "../context.hh"
#include "../detail/popup_layer.hh"

#include <memory>
#include <optional>
#include <utility>

namespace Jetstream::Sakura {

struct Popup::Impl {
    Config config;
    bool wasOpen = false;
    Extent2D<F32> cursor = {0.0f, 0.0f};
};

Popup::Popup() {
    impl = std::make_unique<Impl>();
}

Popup::~Popup() = default;

void Popup::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
}

Extent2D<F32> Popup::measure(const Context&, Extent2D<F32>) {
    return {0.0f, 0.0f};
}

void Popup::layout(const Context& ctx) {
    const auto& config = impl->config;
    if (config.open && !impl->wasOpen) {
        impl->cursor = ctx.mouse.value_or(Extent2D<F32>{frame().x, frame().y});
    }
    impl->wasOpen = config.open;
    if (!config.open || !config.content) {
        return;
    }
    const Rect anchor = config.atCursor ? Rect{impl->cursor.x, impl->cursor.y, 1.0f, 1.0f} : frame();
    Callback<void()> dismiss;
    if (config.onDismiss) {
        dismiss = [this] {
            if (impl->config.onDismiss) {
                impl->config.onDismiss();
            }
        };
    }
    RequestPopup(ctx, {
        .content = config.content,
        .anchor = anchor,
        .placement = config.placement,
        .panel = config.panel,
        .padding = config.padding,
        .maxWidth = config.maxWidth,
        .size = config.size,
        .onDismiss = std::move(dismiss),
        .passthrough = config.passthroughAnchor ? std::optional<Rect>(frame()) : std::nullopt,
    });
}

}  // namespace Jetstream::Sakura
