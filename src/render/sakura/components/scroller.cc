#include <jetstream/render/sakura/components/scroller.hh>

#include <jetstream/render/sakura/components/box.hh>

#include "../context.hh"
#include "../detail/helpers.hh"
#include "../detail/wheel_motion.hh"
#include "../metrics.hh"

#include <algorithm>
#include <cmath>
#include <optional>
#include <utility>

namespace Jetstream::Sakura {

namespace {

constexpr F32 kScrollbarMinThumbThicknessRatio = 4.0f;
constexpr F32 kScrollbarPillRadius = 9999.0f;
constexpr F32 kRubberBand = 0.55f;
constexpr F32 kSpringRate = 12.5f;
constexpr F32 kBounceAmplitude = 0.31f;
constexpr F32 kFallbackEventRate = 60.0f;
constexpr F64 kSpeedWindow = 0.1;
constexpr F32 kSettleDistance = 0.25f;
constexpr F32 kSettleSpeed = 10.0f;
constexpr F32 kMinShrunkThumbRatio = 0.5f;
constexpr F64 kMaxStep = 1.0 / 30.0;
constexpr F64 kGestureTimeout = 0.3;

F32 RubberBand(F32 stretch, F32 extent) {
    if (extent <= 0.0f) {
        return 0.0f;
    }
    const F32 amount = std::abs(stretch) * kRubberBand / extent;
    return std::copysign((1.0f - 1.0f / (amount + 1.0f)) * extent, stretch);
}

F32 Unstretch(F32 offset, F32 extent) {
    if (extent <= 0.0f || offset == 0.0f) {
        return 0.0f;
    }
    const F32 ratio = std::min(std::abs(offset) / extent, 0.99f);
    return std::copysign((1.0f / (1.0f - ratio) - 1.0f) * extent / kRubberBand, offset);
}

struct Elastic {
    F32 offset = 0.0f;
    F32 stretch = 0.0f;
    F32 velocity = 0.0f;
    bool touching = false;
    bool bounced = false;
    bool restart = false;

    bool resting() const {
        return offset == 0.0f && velocity == 0.0f;
    }

    void grab(F32 extent) {
        touching = true;
        bounced = false;
        stretch = Unstretch(offset, extent);
        velocity = 0.0f;
    }

    void release() {
        touching = false;
        stretch = 0.0f;
        bounced = bounced || offset != 0.0f;
        restart = offset != 0.0f;
    }

    bool springing() const {
        return !touching && !resting();
    }

    F32 drag(F32 scroll, F32 max, F32 delta, F32 extent) {
        if (stretch == 0.0f) {
            const F32 next = scroll + delta;
            if (next < 0.0f) {
                stretch = next;
                scroll = 0.0f;
            } else if (next > max) {
                stretch = next - max;
                scroll = max;
            } else {
                scroll = next;
            }
        } else {
            const F32 next = stretch + delta;
            if (next == 0.0f || (next < 0.0f) != (stretch < 0.0f)) {
                stretch = 0.0f;
                scroll = std::clamp(scroll + next, 0.0f, max);
            } else {
                stretch = next;
            }
        }
        offset = RubberBand(stretch, extent);
        velocity = 0.0f;
        return scroll;
    }

    F32 coast(F32 scroll, F32 max, F32 delta, F32 speed) {
        const F32 next = scroll + delta;
        if (next >= 0.0f && next <= max) {
            return next;
        }
        if (bounced || offset != 0.0f) {
            bounced = true;
            return std::clamp(next, 0.0f, max);
        }
        bounced = true;
        restart = true;
        velocity = speed * kBounceAmplitude;
        return std::clamp(next, 0.0f, max);
    }

    bool spring(F32 dt) {
        if (touching || resting()) {
            return false;
        }
        const F32 start = offset;
        const F32 slope = velocity + kSpringRate * start;
        const F32 decay = std::exp(-kSpringRate * dt);
        offset = (start + slope * dt) * decay;
        velocity = (slope - kSpringRate * (start + slope * dt)) * decay;
        if (std::abs(offset) < kSettleDistance && std::abs(velocity) < kSettleSpeed) {
            offset = 0.0f;
            velocity = 0.0f;
            return false;
        }
        return true;
    }
};

}  // namespace

struct Scroller::Impl {
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
    F32 targetX = 0.0f;
    F32 targetY = 0.0f;
    DragAxis dragAxis = DragAxis::None;
    F32 dragGrabOffset = 0.0f;
    ColorRGBA<F32> trackColor = {1.0f, 1.0f, 1.0f, 0.08f};
    ColorRGBA<F32> thumbColor = {1.0f, 1.0f, 1.0f, 0.25f};
    Elastic elasticX;
    Elastic elasticY;
    F32 pixelRatio = 1.0f;
    F64 steppedAt = -1.0;
    F64 lastGesture = 0.0;
    bool gesturePending = false;
    Extent2D<F32> speed = {0.0f, 0.0f};
    Extent2D<F32> travel = {0.0f, 0.0f};
    F64 sampledAt = -1.0;
    WheelMotion wheelX;
    WheelMotion wheelY;
    F64 wheelSteppedAt = -1.0;

    void sample(Extent2D<F32> delta) {
        if (steppedAt != sampledAt) {
            const F64 span = steppedAt - sampledAt;
            const bool recent = sampledAt >= 0.0 && span > 0.0 && span < kSpeedWindow;
            speed = recent ? Extent2D<F32>{static_cast<F32>(travel.x / span), static_cast<F32>(travel.y / span)}
                           : Extent2D<F32>{0.0f, 0.0f};
            travel = {0.0f, 0.0f};
            sampledAt = steppedAt;
        }
        travel.x += delta.x;
        travel.y += delta.y;
    }

    static F32 Speed(F32 measured, F32 delta) {
        return measured != 0.0f && (measured < 0.0f) == (delta < 0.0f) ? measured : delta * kFallbackEventRate;
    }

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
        return config.thickness + 2.0f * config.margin + config.trackInset.x;
    }

    Rect verticalTrackRect() const {
        const F32 bottomInset = horizontalVisible() ? reserved() : 0.0f;
        return {
            viewport.right() - config.thickness - config.margin - config.trackInset.x,
            viewport.y + config.margin + config.trackInset.y,
            config.thickness,
            std::max(0.0f, viewport.height - 2.0f * (config.margin + config.trackInset.y) - bottomInset),
        };
    }

    Rect horizontalTrackRect() const {
        const F32 rightInset = verticalVisible() ? reserved() : 0.0f;
        return {
            viewport.x + config.margin + config.trackInset.y,
            viewport.bottom() - config.thickness - config.margin - config.trackInset.x,
            std::max(0.0f, viewport.width - 2.0f * (config.margin + config.trackInset.y) - rightInset),
            config.thickness,
        };
    }

    Rect thumbTrack(Rect track) const {
        const F32 inset = std::min({config.thumbInset, track.width * 0.5f, track.height * 0.5f});
        return {track.x + inset, track.y + inset, track.width - 2.0f * inset, track.height - 2.0f * inset};
    }

    static F32 ShrinkThumb(F32 length, F32 minimum, F32 track, F32 content, F32 offset) {
        if (offset == 0.0f || content <= 0.0f) {
            return length;
        }
        const F32 shrunk = length - std::abs(offset) * track / content;
        return std::clamp(shrunk, std::min(minimum * kMinShrunkThumbRatio, length), length);
    }

    Rect verticalThumbRect() const {
        const auto track = thumbTrack(verticalTrackRect());
        const F32 viewRatio = config.contentHeight > 0.0f
            ? std::clamp(viewport.height / config.contentHeight, 0.0f, 1.0f)
            : 1.0f;
        const F32 minThumb = std::min(kScrollbarMinThumbThicknessRatio * config.thickness, track.height);
        const F32 thumbHeight = std::clamp(track.height * viewRatio, minThumb, track.height);
        const F32 range = std::max(0.0f, track.height - thumbHeight);
        const F32 progress = maxScrollY() > 0.0f ? scrollY / maxScrollY() : 0.0f;
        const F32 offset = elasticY.offset;
        const F32 height = ShrinkThumb(thumbHeight, minThumb, track.height, config.contentHeight, offset);
        if (offset < 0.0f) {
            return {track.x, track.y, track.width, height};
        }
        if (offset > 0.0f) {
            return {track.x, track.bottom() - height, track.width, height};
        }
        return {track.x, track.y + range * progress, track.width, thumbHeight};
    }

    Rect horizontalThumbRect() const {
        const auto track = thumbTrack(horizontalTrackRect());
        const F32 viewRatio = config.contentWidth > 0.0f
            ? std::clamp(viewport.width / config.contentWidth, 0.0f, 1.0f)
            : 1.0f;
        const F32 minThumb = std::min(kScrollbarMinThumbThicknessRatio * config.thickness, track.width);
        const F32 thumbWidth = std::clamp(track.width * viewRatio, minThumb, track.width);
        const F32 range = std::max(0.0f, track.width - thumbWidth);
        const F32 progress = maxScrollX() > 0.0f ? scrollX / maxScrollX() : 0.0f;
        const F32 offset = elasticX.offset;
        const F32 width = ShrinkThumb(thumbWidth, minThumb, track.width, config.contentWidth, offset);
        if (offset < 0.0f) {
            return {track.x, track.y, width, track.height};
        }
        if (offset > 0.0f) {
            return {track.right() - width, track.y, width, track.height};
        }
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

    void notifyOverscroll(Scroller& self) const {
        self.invalidate(Dirty::Layout);
        if (config.onOverscroll) {
            config.onOverscroll();
        }
    }

    void settle(Scroller& self, Elastic& elastic) const {
        if (elastic.resting() && !elastic.touching) {
            return;
        }
        elastic = {};
        notifyOverscroll(self);
    }

    bool setScrollY(Scroller& self, F32 value) {
        const F32 next = std::clamp(value, 0.0f, maxScrollY());
        if (next == scrollY) {
            return false;
        }
        scrollY = next;
        targetY = next;
        self.invalidate(Dirty::Layout);
        if (config.onScrollY) {
            config.onScrollY(scrollY);
        }
        return true;
    }

    bool setScrollX(Scroller& self, F32 value) {
        const F32 next = std::clamp(value, 0.0f, maxScrollX());
        if (next == scrollX) {
            return false;
        }
        scrollX = next;
        targetX = next;
        self.invalidate(Dirty::Layout);
        if (config.onScrollX) {
            config.onScrollX(scrollX);
        }
        return true;
    }

    F32 scrollYFromThumbTop(F32 thumbTop) const {
        const auto track = thumbTrack(verticalTrackRect());
        const F32 range = std::max(1e-3f, track.height - verticalThumbRect().height);
        return std::clamp((thumbTop - track.y) / range, 0.0f, 1.0f) * maxScrollY();
    }

    F32 scrollXFromThumbLeft(F32 thumbLeft) const {
        const auto track = thumbTrack(horizontalTrackRect());
        const F32 range = std::max(1e-3f, track.width - horizontalThumbRect().width);
        return std::clamp((thumbLeft - track.x) / range, 0.0f, 1.0f) * maxScrollX();
    }

    void resolveTheme(const Context& ctx) {
        trackColor = ctx.color(config.trackColorKey);
        thumbColor = ctx.color(config.thumbColorKey);
    }

    bool touching() const {
        return elasticX.touching || elasticY.touching;
    }

    void grab() {
        if (maxScrollX() > 0.0f) {
            elasticX.grab(viewport.width);
        }
        if (maxScrollY() > 0.0f) {
            elasticY.grab(viewport.height);
        }
        gesturePending = true;
    }

    void release() {
        elasticX.release();
        elasticY.release();
    }

    void step(Scroller& self, F64 time) {
        if (gesturePending) {
            gesturePending = false;
            lastGesture = time;
        }
        if (time != steppedAt) {
            const bool restart = elasticX.restart || elasticY.restart;
            elasticX.restart = false;
            elasticY.restart = false;
            const F64 elapsed = steppedAt < 0.0 || restart ? 0.0 : time - steppedAt;
            const F32 dt = static_cast<F32>(std::clamp(elapsed, 0.0, kMaxStep));
            const F64 wheelElapsed = wheelSteppedAt < 0.0 ? 0.0 : time - wheelSteppedAt;
            (void)setScrollX(self, wheelX.advance(scrollX, wheelElapsed, maxScrollX()));
            (void)setScrollY(self, wheelY.advance(scrollY, wheelElapsed, maxScrollY()));
            wheelSteppedAt = wheelX.active() || wheelY.active() ? time : -1.0;
            steppedAt = time;
            if (touching() && time - lastGesture > kGestureTimeout) {
                release();
            }
            const Extent2D<F32> before = {elasticX.offset, elasticY.offset};
            elasticX.spring(dt);
            elasticY.spring(dt);
            if (before.x != elasticX.offset || before.y != elasticY.offset) {
                notifyOverscroll(self);
            }
        }
        schedule(self);
    }

    void schedule(Scroller& self) const {
        if (wheelX.active() || wheelY.active()) {
            self.requestFrame();
        }
        if (elasticX.springing() || elasticY.springing()) {
            self.requestFrame();
            if (config.onOverscroll) {
                config.onOverscroll();
            }
        } else if (touching() && (elasticX.offset != 0.0f || elasticY.offset != 0.0f)) {
            self.requestFrame(lastGesture + kGestureTimeout);
        }
    }
};

Scroller::Scroller() {
    this->impl = std::make_unique<Impl>();
    addChild(this->impl->vTrack);
    addChild(this->impl->vThumb);
    addChild(this->impl->hTrack);
    addChild(this->impl->hThumb);
}

Scroller::~Scroller() = default;

void Scroller::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    const bool identity = config.id != impl->config.id;
    impl->config = std::move(config);
    if (identity) {
        reset();
    }
    if (impl->config.scrollY != impl->targetY) {
        impl->wheelY.cancel();
        impl->targetY = impl->config.scrollY;
        invalidate(Dirty::Layout);
    }
    if (impl->config.scrollX != impl->targetX) {
        impl->wheelX.cancel();
        impl->targetX = impl->config.scrollX;
        invalidate(Dirty::Layout);
    }
}

void Scroller::reset() {
    auto& s = *impl;
    s.wheelX.cancel();
    s.wheelY.cancel();
    s.elasticX = {};
    s.elasticY = {};
    s.speed = {0.0f, 0.0f};
    s.travel = {0.0f, 0.0f};
    s.sampledAt = -1.0;
    s.gesturePending = false;
    s.dragAxis = Impl::DragAxis::None;
    invalidate(Dirty::Layout);
}

Extent2D<F32> Scroller::overscroll(const Context& ctx) {
    impl->step(*this, ctx.time);
    return {impl->elasticX.offset, impl->elasticY.offset};
}

void Scroller::layout(const Context& ctx) {
    impl->viewport = frame();
    impl->pixelRatio = ctx.pixelRatio;
    impl->resolveTheme(ctx);
    impl->scrollY = std::clamp(impl->targetY, 0.0f, impl->maxScrollY());
    impl->scrollX = std::clamp(impl->targetX, 0.0f, impl->maxScrollX());
    impl->step(*this, ctx.time);

    const bool vVisible = impl->verticalVisible();
    const bool hVisible = impl->horizontalVisible();

    impl->vTrack.update({
        .instances = {{.rect = impl->verticalTrackRect(), .visible = vVisible, .backgroundColor = impl->trackColor}},
        .cornerRadius = kScrollbarPillRadius,
    });
    impl->vThumb.update({
        .instances = {{.rect = vVisible ? Snap(impl->verticalThumbRect()) : Rect{}, .visible = vVisible,
                       .backgroundColor = impl->thumbColor}},
        .cornerRadius = kScrollbarPillRadius,
    });
    impl->hTrack.update({
        .instances = {{.rect = impl->horizontalTrackRect(), .visible = hVisible, .backgroundColor = impl->trackColor}},
        .cornerRadius = kScrollbarPillRadius,
    });
    impl->hThumb.update({
        .instances = {{.rect = hVisible ? Snap(impl->horizontalThumbRect()) : Rect{}, .visible = hVisible,
                       .backgroundColor = impl->thumbColor}},
        .cornerRadius = kScrollbarPillRadius,
    });

    impl->notifyLayout();

    layoutChildren(ctx);
}

bool Scroller::wheel(const MouseEvent& event) {
    auto& s = *impl;
    if (!s.viewport.contains(event.position.x, event.position.y)) {
        return false;
    }
    F32 dx = event.scroll.x;
    F32 dy = event.scroll.y;
    if (event.modifiers.shift && dx == 0.0f) {
        dx = dy;
        dy = 0.0f;
    }
    const bool scrollsX = s.maxScrollX() > 0.0f;
    const bool scrollsY = s.maxScrollY() > 0.0f;
    const bool held = s.touching();

    if (!event.smooth) {
        s.wheelX.cancel();
        s.wheelY.cancel();
    }

    switch (event.phase) {
        case ScrollPhase::Began:
            s.grab();
            break;
        case ScrollPhase::Changed:
            if (!held) {
                s.grab();
            }
            break;
        case ScrollPhase::Ended:
            s.release();
            s.notifyOverscroll(*this);
            if (dx == 0.0f && dy == 0.0f) {
                return false;
            }
            break;
        case ScrollPhase::Momentum:
            if (held) {
                s.release();
            }
            break;
        case ScrollPhase::MomentumEnded:
            if (held) {
                s.release();
            }
            s.elasticX.bounced = false;
            s.elasticY.bounced = false;
            break;
        case ScrollPhase::None:
            break;
    }
    if (dx == 0.0f && dy == 0.0f) {
        return event.phase == ScrollPhase::Began && (scrollsX || scrollsY);
    }

    const bool sampled = event.phase != ScrollPhase::None || event.precise;
    const F32 unit = WheelStep(event, s.config.wheelStep, s.pixelRatio);
    const bool dragging = event.phase == ScrollPhase::Began || event.phase == ScrollPhase::Changed;
    const bool coasting = event.phase == ScrollPhase::Momentum;
    const Extent2D<F32> before = {s.elasticX.offset, s.elasticY.offset};
    const Extent2D<F32> delta = {-dx * unit, -dy * unit};
    if (event.smooth && !sampled) {
        const bool consumedX = dx != 0.0f && scrollsX;
        const bool consumedY = dy != 0.0f && scrollsY;
        if (!s.wheelX.active() && !s.wheelY.active()) {
            s.wheelSteppedAt = -1.0;
        }
        if (consumedX) {
            s.settle(*this, s.elasticX);
            s.wheelX.add(s.scrollX, delta.x, s.maxScrollX());
        }
        if (consumedY) {
            s.settle(*this, s.elasticY);
            s.wheelY.add(s.scrollY, delta.y, s.maxScrollY());
        }
        if (consumedX || consumedY) {
            invalidate(Dirty::Layout);
            s.schedule(*this);
        }
        return consumedX || consumedY;
    }
    if (sampled) {
        s.sample(delta);
    }
    const auto move = [&](Elastic& elastic, F32 scroll, F32 max, F32 step, F32 measured, F32 extent) {
        if (dragging) {
            return elastic.drag(scroll, max, step, extent);
        }
        if (coasting) {
            return elastic.coast(scroll, max, step, Impl::Speed(measured, step));
        }
        return std::clamp(scroll + step, 0.0f, max);
    };
    const bool consumedY = dy != 0.0f && scrollsY;
    const bool consumedX = dx != 0.0f && scrollsX;
    if (consumedY) {
        (void)s.setScrollY(*this, move(s.elasticY, s.scrollY, s.maxScrollY(), delta.y, s.speed.y, s.viewport.height));
    }
    if (consumedX) {
        (void)s.setScrollX(*this, move(s.elasticX, s.scrollX, s.maxScrollX(), delta.x, s.speed.x, s.viewport.width));
    }
    if (consumedX || consumedY) {
        s.gesturePending = true;
        invalidate(Dirty::Layout);
    }
    if (before.x != s.elasticX.offset || before.y != s.elasticY.offset) {
        s.notifyOverscroll(*this);
    }
    return consumedY || consumedX;
}

bool Scroller::mouseEvent(const MouseEvent& event) {
    switch (event.type) {
        case MouseEventType::Scroll: {
            if (!impl->config.wheel) {
                return false;
            }
            return wheel(event);
        }
        case MouseEventType::Click: {
            if (event.button != MouseButton::Left) {
                return false;
            }
            impl->wheelX.cancel();
            impl->wheelY.cancel();
            if (impl->verticalVisible()) {
                const auto thumb = impl->verticalThumbRect();
                if (thumb.contains(event.position.x, event.position.y)) {
                    impl->settle(*this, impl->elasticY);
                    impl->dragAxis = Impl::DragAxis::Vertical;
                    impl->dragGrabOffset = event.position.y - impl->verticalThumbRect().y;
                    return true;
                }
                if (impl->verticalTrackRect().contains(event.position.x, event.position.y)) {
                    impl->settle(*this, impl->elasticY);
                    impl->dragAxis = Impl::DragAxis::Vertical;
                    impl->dragGrabOffset = thumb.height * 0.5f;
                    (void)impl->setScrollY(*this, impl->scrollYFromThumbTop(event.position.y - impl->dragGrabOffset));
                    return true;
                }
            }
            if (impl->horizontalVisible()) {
                const auto thumb = impl->horizontalThumbRect();
                if (thumb.contains(event.position.x, event.position.y)) {
                    impl->settle(*this, impl->elasticX);
                    impl->dragAxis = Impl::DragAxis::Horizontal;
                    impl->dragGrabOffset = event.position.x - impl->horizontalThumbRect().x;
                    return true;
                }
                if (impl->horizontalTrackRect().contains(event.position.x, event.position.y)) {
                    impl->settle(*this, impl->elasticX);
                    impl->dragAxis = Impl::DragAxis::Horizontal;
                    impl->dragGrabOffset = thumb.width * 0.5f;
                    (void)impl->setScrollX(*this, impl->scrollXFromThumbLeft(event.position.x - impl->dragGrabOffset));
                    return true;
                }
            }
            return false;
        }
        case MouseEventType::Move: {
            if (impl->dragAxis == Impl::DragAxis::Vertical) {
                (void)impl->setScrollY(*this, impl->scrollYFromThumbTop(event.position.y - impl->dragGrabOffset));
                return true;
            }
            if (impl->dragAxis == Impl::DragAxis::Horizontal) {
                (void)impl->setScrollX(*this, impl->scrollXFromThumbLeft(event.position.x - impl->dragGrabOffset));
                return true;
            }
            return false;
        }
        case MouseEventType::Cancel: {
            impl->wheelX.cancel();
            impl->wheelY.cancel();
            impl->dragAxis = Impl::DragAxis::None;
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

}  // namespace Jetstream::Sakura
