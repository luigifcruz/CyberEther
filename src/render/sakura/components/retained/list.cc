#include <jetstream/render/sakura/components/retained/list.hh>
#include <jetstream/render/sakura/components/retained/scroll_view.hh>

#include <algorithm>
#include <cmath>
#include <utility>

namespace Jetstream::Sakura::Retained {

struct List::Impl {
    Config config;
    ScrollView scroll;
    F32 currentScrollY = 0.0f;
    Rect scrollContentRect;
    std::optional<Rect> scrollClipRect;

    F32 contentHeight() const {
        return static_cast<F32>(config.rowCount) * config.rowHeight;
    }

    void bindRows() {
        if (!config.onBindRow) {
            return;
        }

        const auto content = scrollContentRect;
        const auto clip = scrollClipRect;
        const F32 scrollY = currentScrollY;
        const F32 rowHeight = std::max(1.0f, config.rowHeight);
        const U64 firstRow = static_cast<U64>(std::max(0.0f, std::floor(scrollY / rowHeight)));

        for (U64 slot = 0; slot < config.slotCapacity; ++slot) {
            const U64 row = firstRow + slot;
            const F32 top = content.y + static_cast<F32>(row) * rowHeight - scrollY;
            const bool visible = row < config.rowCount && top < content.bottom();
            const Rect rowRect = {content.x, top, content.width, rowHeight};
            config.onBindRow(slot, visible ? std::optional<U64>(row) : std::nullopt, rowRect, clip);
        }
    }
};

List::List() {
    this->impl = std::make_unique<Impl>();
    add(this->impl->scroll);
}

List::~List() = default;

bool List::update(Config config) {
    this->impl->config = std::move(config);
    invalidate(Dirty::Paint);
    return true;
}

void List::layout(const Context& ctx) {
    impl->currentScrollY = std::clamp(impl->config.scrollY, 0.0f,
                                      std::max(0.0f, impl->contentHeight() - frame().height));

    impl->scroll.update({
        .id = impl->config.id + ":scroll",
        .contentHeight = impl->contentHeight(),
        .scrollY = impl->currentScrollY,
        .onLayout = [impl = this->impl.get()](Rect contentRect, std::optional<Rect> clipRect) {
            impl->scrollContentRect = contentRect;
            impl->scrollClipRect = std::move(clipRect);
            impl->bindRows();
        },
        .onScrollY = [impl = this->impl.get()](F32 scrollY) {
            impl->currentScrollY = scrollY;
            impl->bindRows();
            if (impl->config.onScrollY) {
                impl->config.onScrollY(scrollY);
            }
        },
    });

    layoutChild(ctx, impl->scroll, frame());
}

}  // namespace Jetstream::Sakura::Retained
