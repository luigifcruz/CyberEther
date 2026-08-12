#include <jetstream/render/sakura/components/retained/scroll_view.hh>

#include <jetstream/render/sakura/components/retained/box.hh>

#include "../../helpers.hh"

#include <algorithm>
#include <optional>
#include <utility>

namespace Jetstream::Sakura::Retained {

namespace {

constexpr F32 kScrollbarMinThumbThicknessRatio = 4.0f;
constexpr F32 kScrollbarPillRadius = 9999.0f;

}  // namespace

struct ScrollView::Impl {
    enum class DragAxis {
        None,
        Vertical,
        Horizontal,
    };

    Config config;
    Box vTrack;
    Box vThumb;
    Box hTrack;
    Box hThumb;

    Rect viewport;
    F32 scrollX = 0.0f;
    F32 scrollY = 0.0f;
    DragAxis dragAxis = DragAxis::None;
    F32 dragGrabOffset = 0.0f;
    ColorRGBA<F32> trackColor = {1.0f, 1.0f, 1.0f, 0.08f};
    ColorRGBA<F32> thumbColor = {1.0f, 1.0f, 1.0f, 0.25f};

    F32 maxScrollX() const {
        return std::max(0.0f, config.contentWidth - viewport.width);
    }

    F32 maxScrollY() const {
        return std::max(0.0f, config.contentHeight - viewport.height);
    }

    bool verticalVisible() const {
        return config.scrollbar && maxScrollY() > 0.0f;
    }

    bool horizontalVisible() const {
        return config.scrollbar && maxScrollX() > 0.0f;
    }

    F32 reserved() const {
        return config.thickness + 2.0f * config.margin;
    }

    Rect verticalTrackRect() const {
        const F32 bottomInset = horizontalVisible() ? reserved() : 0.0f;
        return {
            viewport.right() - config.thickness - config.margin,
            viewport.y + config.margin,
            config.thickness,
            std::max(0.0f, viewport.height - 2.0f * config.margin - bottomInset),
        };
    }

    Rect horizontalTrackRect() const {
        const F32 rightInset = verticalVisible() ? reserved() : 0.0f;
        return {
            viewport.x + config.margin,
            viewport.bottom() - config.thickness - config.margin,
            std::max(0.0f, viewport.width - 2.0f * config.margin - rightInset),
            config.thickness,
        };
    }

    Rect verticalThumbRect() const {
        const auto track = verticalTrackRect();
        const F32 viewRatio = config.contentHeight > 0.0f
            ? std::clamp(viewport.height / config.contentHeight, 0.0f, 1.0f)
            : 1.0f;
        const F32 minThumb = std::min(kScrollbarMinThumbThicknessRatio * config.thickness, track.height);
        const F32 thumbHeight = std::clamp(track.height * viewRatio, minThumb, track.height);
        const F32 range = std::max(0.0f, track.height - thumbHeight);
        const F32 progress = maxScrollY() > 0.0f ? scrollY / maxScrollY() : 0.0f;
        return {track.x, track.y + range * progress, track.width, thumbHeight};
    }

    Rect horizontalThumbRect() const {
        const auto track = horizontalTrackRect();
        const F32 viewRatio = config.contentWidth > 0.0f
            ? std::clamp(viewport.width / config.contentWidth, 0.0f, 1.0f)
            : 1.0f;
        const F32 minThumb = std::min(kScrollbarMinThumbThicknessRatio * config.thickness, track.width);
        const F32 thumbWidth = std::clamp(track.width * viewRatio, minThumb, track.width);
        const F32 range = std::max(0.0f, track.width - thumbWidth);
        const F32 progress = maxScrollX() > 0.0f ? scrollX / maxScrollX() : 0.0f;
        return {track.x + range * progress, track.y, thumbWidth, track.height};
    }

    Rect contentRect() const {
        auto rect = viewport;
        if (verticalVisible()) {
            rect.width = std::max(0.0f, rect.width - reserved());
        }
        if (horizontalVisible()) {
            rect.height = std::max(0.0f, rect.height - reserved());
        }
        return rect;
    }

    std::optional<Rect> clipRect() const {
        return viewport;
    }

    void notifyLayout() const {
        if (config.onLayout) {
            config.onLayout(contentRect(), clipRect());
        }
    }

    bool setScrollY(F32 value) {
        const F32 next = std::clamp(value, 0.0f, maxScrollY());
        if (next == scrollY) {
            return false;
        }
        scrollY = next;
        if (config.onScrollY) {
            config.onScrollY(scrollY);
        }
        return true;
    }

    bool setScrollX(F32 value) {
        const F32 next = std::clamp(value, 0.0f, maxScrollX());
        if (next == scrollX) {
            return false;
        }
        scrollX = next;
        if (config.onScrollX) {
            config.onScrollX(scrollX);
        }
        return true;
    }

    F32 scrollYFromThumbTop(F32 thumbTop) const {
        const auto track = verticalTrackRect();
        const F32 range = std::max(1e-3f, track.height - verticalThumbRect().height);
        return std::clamp((thumbTop - track.y) / range, 0.0f, 1.0f) * maxScrollY();
    }

    F32 scrollXFromThumbLeft(F32 thumbLeft) const {
        const auto track = horizontalTrackRect();
        const F32 range = std::max(1e-3f, track.width - horizontalThumbRect().width);
        return std::clamp((thumbLeft - track.x) / range, 0.0f, 1.0f) * maxScrollX();
    }

    void resolveTheme(const Context& ctx) {
        trackColor = ctx.color(config.trackColorKey);
        thumbColor = ctx.color(config.thumbColorKey);
    }
};

ScrollView::ScrollView() {
    this->impl = std::make_unique<Impl>();
    add(this->impl->vTrack);
    add(this->impl->vThumb);
    add(this->impl->hTrack);
    add(this->impl->hThumb);
}

ScrollView::~ScrollView() = default;

bool ScrollView::update(Config config) {
    impl->config = std::move(config);
    impl->scrollY = std::clamp(impl->config.scrollY, 0.0f, impl->maxScrollY());
    impl->scrollX = std::clamp(impl->config.scrollX, 0.0f, impl->maxScrollX());
    return true;
}

void ScrollView::layout(const Context& ctx) {
    impl->viewport = frame();
    impl->resolveTheme(ctx);
    impl->scrollY = std::clamp(impl->scrollY, 0.0f, impl->maxScrollY());
    impl->scrollX = std::clamp(impl->scrollX, 0.0f, impl->maxScrollX());

    const bool vVisible = impl->verticalVisible();
    const bool hVisible = impl->horizontalVisible();

    impl->vTrack.update({
        .id = impl->config.id + ":vtrack",
        .instances = {{.rect = impl->verticalTrackRect(), .visible = vVisible, .backgroundColor = impl->trackColor}},
        .cornerRadius = kScrollbarPillRadius,
    });
    impl->vThumb.update({
        .id = impl->config.id + ":vthumb",
        .instances = {{.rect = impl->verticalThumbRect(), .visible = vVisible, .backgroundColor = impl->thumbColor}},
        .cornerRadius = kScrollbarPillRadius,
    });
    impl->hTrack.update({
        .id = impl->config.id + ":htrack",
        .instances = {{.rect = impl->horizontalTrackRect(), .visible = hVisible, .backgroundColor = impl->trackColor}},
        .cornerRadius = kScrollbarPillRadius,
    });
    impl->hThumb.update({
        .id = impl->config.id + ":hthumb",
        .instances = {{.rect = impl->horizontalThumbRect(), .visible = hVisible, .backgroundColor = impl->thumbColor}},
        .cornerRadius = kScrollbarPillRadius,
    });

    impl->notifyLayout();

    layoutChildren(ctx);
}

bool ScrollView::event(const MouseEvent& event) {
    switch (event.type) {
        case MouseEventType::Scroll: {
            if (!impl->config.wheel || !impl->viewport.contains(event.position.x, event.position.y)) {
                return false;
            }
            F32 dx = event.scroll.x;
            F32 dy = event.scroll.y;
            if ((ImGui::GetIO().KeyMods & ImGuiMod_Shift) != 0 && dx == 0.0f) {
                dx = dy;
                dy = 0.0f;
            }
            const bool consumedY = dy != 0.0f && impl->verticalVisible();
            const bool consumedX = dx != 0.0f && impl->horizontalVisible();
            if (consumedY) {
                (void)impl->setScrollY(impl->scrollY - dy * impl->config.wheelStep);
            }
            if (consumedX) {
                (void)impl->setScrollX(impl->scrollX - dx * impl->config.wheelStep);
            }
            return consumedY || consumedX;
        }
        case MouseEventType::Click: {
            if (event.button != MouseButton::Left) {
                return false;
            }
            if (impl->verticalVisible()) {
                const auto thumb = impl->verticalThumbRect();
                if (thumb.contains(event.position.x, event.position.y)) {
                    impl->dragAxis = Impl::DragAxis::Vertical;
                    impl->dragGrabOffset = event.position.y - thumb.y;
                    return true;
                }
                if (impl->verticalTrackRect().contains(event.position.x, event.position.y)) {
                    impl->dragAxis = Impl::DragAxis::Vertical;
                    impl->dragGrabOffset = thumb.height * 0.5f;
                    (void)impl->setScrollY(impl->scrollYFromThumbTop(event.position.y - impl->dragGrabOffset));
                    return true;
                }
            }
            if (impl->horizontalVisible()) {
                const auto thumb = impl->horizontalThumbRect();
                if (thumb.contains(event.position.x, event.position.y)) {
                    impl->dragAxis = Impl::DragAxis::Horizontal;
                    impl->dragGrabOffset = event.position.x - thumb.x;
                    return true;
                }
                if (impl->horizontalTrackRect().contains(event.position.x, event.position.y)) {
                    impl->dragAxis = Impl::DragAxis::Horizontal;
                    impl->dragGrabOffset = thumb.width * 0.5f;
                    (void)impl->setScrollX(impl->scrollXFromThumbLeft(event.position.x - impl->dragGrabOffset));
                    return true;
                }
            }
            return false;
        }
        case MouseEventType::Move: {
            if (impl->dragAxis == Impl::DragAxis::Vertical) {
                (void)impl->setScrollY(impl->scrollYFromThumbTop(event.position.y - impl->dragGrabOffset));
                return true;
            }
            if (impl->dragAxis == Impl::DragAxis::Horizontal) {
                (void)impl->setScrollX(impl->scrollXFromThumbLeft(event.position.x - impl->dragGrabOffset));
                return true;
            }
            return false;
        }
        case MouseEventType::Release: {
            if (event.button != MouseButton::Left || impl->dragAxis == Impl::DragAxis::None) {
                return false;
            }
            impl->dragAxis = Impl::DragAxis::None;
            return true;
        }
        default:
            return false;
    }
}

}  // namespace Jetstream::Sakura::Retained
