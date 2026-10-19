#include <jetstream/render/sakura/components/root.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/canvas.hh>
#include <jetstream/render/sakura/components/text.hh>
#include <jetstream/render/sakura/typography.hh>

#include "../context.hh"
#include "../helpers.hh"
#include "../metrics.hh"
#include "../detail/helpers.hh"
#include "../detail/dock_drag.hh"
#include "../detail/popup_layer.hh"
#include "../detail/root_component.hh"
#include "../detail/access.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace Jetstream::Sakura {

namespace {

constexpr F32 kSubmenuOffset = 8.0f;
constexpr F32 kPopupGap = 6.0f;
constexpr F32 kTooltipBelow = 24.0f;
constexpr F32 kTooltipAbove = 8.0f;

Rect Fit(Rect rect, const Rect& bounds) {
    rect.x = std::max(bounds.x, std::min(rect.x, bounds.right() - rect.width));
    rect.y = std::max(bounds.y, std::min(rect.y, bounds.bottom() - rect.height));
    return rect;
}

Rect Place(const Context& ctx, const PopupRequest& request, F32 width, F32 height, const Rect& bounds) {
    const Rect& anchor = request.anchor;
    Rect rect = {anchor.x, anchor.y, width, height};
    switch (request.placement) {
        case PopupPlacement::Below:
            rect.width = std::max(width, anchor.width);
            rect.y = anchor.bottom();
            if (rect.bottom() > bounds.bottom() && anchor.y - height >= bounds.y) {
                rect.y = anchor.y - height;
            }
            break;
        case PopupPlacement::BelowEnd:
            rect.x = anchor.right() - width;
            rect.y = anchor.bottom() + Px(ctx, kPopupGap);
            if (rect.bottom() > bounds.bottom() && anchor.y - height - Px(ctx, kPopupGap) >= bounds.y) {
                rect.y = anchor.y - height - Px(ctx, kPopupGap);
            }
            break;
        case PopupPlacement::Point:
            if (rect.right() > bounds.right()) {
                rect.x = anchor.x - width;
            }
            if (rect.bottom() > bounds.bottom()) {
                rect.y = anchor.y - height;
            }
            break;
        case PopupPlacement::Side:
            rect.x = anchor.right();
            rect.y = anchor.y - Px(ctx, kSubmenuOffset);
            if (rect.right() > bounds.right()) {
                rect.x = anchor.x - width;
            }
            break;
        case PopupPlacement::Tooltip:
            rect.y = anchor.y + Px(ctx, kTooltipBelow);
            if (rect.bottom() > bounds.bottom()) {
                rect.y = anchor.y - Px(ctx, kTooltipAbove) - height;
            }
            break;
        case PopupPlacement::Center:
            rect.x = bounds.x + std::floor((bounds.width - width) * 0.5f);
            rect.y = bounds.y + std::floor((bounds.height - height) * 0.5f);
            break;
    }
    rect.x = std::round(rect.x);
    rect.y = std::round(rect.y);
    return Fit(rect, bounds);
}

}  // namespace

Layer::Layer() = default;

Layer::~Layer() = default;

void Layer::update(Config config) {
    auto* next = config.content;
    if (next == content && (!next || hasChild(*next))) {
        return;
    }
    content = next;
    invalidate(Dirty::Layout);
    std::vector<Component*> children;
    if (next) {
        children.push_back(next);
    }
    mountChildren(children);
}

bool Layer::covers(const Extent2D<F32>&) const {
    return content != nullptr;
}

bool Layer::modal() const {
    return false;
}

bool Layer::occludes(const Extent2D<F32>& point) const {
    return covers(point);
}

Extent2D<F32> Layer::measure(const Context&, Extent2D<F32> available) {
    return {FillLength(available.x), FillLength(available.y)};
}

void Layer::layout(const Context& ctx) {
    if (content) {
        layoutChild(ctx, *content, frame());
    }
}

PopupLayer::PopupLayer() = default;

PopupLayer::~PopupLayer() = default;

void PopupLayer::replace(Component& owner, std::weak_ptr<bool> alive, std::vector<PopupRequest> requests) {
    std::vector<std::weak_ptr<bool>> contents;
    for (const auto& request : requests) {
        contents.push_back(Access::alive(*request.content));
    }
    const auto it = std::find_if(entries.begin(), entries.end(), [&](const Entry& entry) {
        return entry.owner == &owner;
    });
    if (it == entries.end()) {
        if (requests.empty()) {
            return;
        }
        entries.push_back({.owner = &owner, .alive = std::move(alive), .requests = std::move(requests),
                           .contents = std::move(contents)});
        changed();
        return;
    }
    if (requests.empty()) {
        entries.erase(it);
        changed();
        return;
    }
    const bool differs = it->requests != requests;
    it->alive = std::move(alive);
    it->requests = std::move(requests);
    it->contents = std::move(contents);
    if (differs) {
        changed();
    }
}

void PopupLayer::changed() {
    ++revision;
    if (!arranging) {
        invalidate(Dirty::Layout);
        withdraw();
    }
}

void PopupLayer::withdraw() {
    std::erase_if(placed, [this](const Placed& item) {
        if (!item.alive.expired() && claim(item.content).request) {
            return false;
        }
        if (const auto backdrop = frames.find(item.content); backdrop != frames.end()) {
            backdrop->second->setVisible(false);
        }
        if (!item.alive.expired()) {
            cancelMouse(*item.content);
            Access::releaseFocusWithin(*item.content);
            item.content->setVisible(false);
        }
        return true;
    });
}

void PopupLayer::expire() {
    bool dropped = false;
    std::erase_if(entries, [&](Entry& entry) {
        if (entry.alive.expired()) {
            dropped = true;
            return true;
        }
        for (U64 index = entry.requests.size(); index-- > 0;) {
            if (entry.contents[index].expired()) {
                entry.drop(index);
                dropped = true;
            }
        }
        return entry.requests.empty();
    });
    if (dropped) {
        changed();
    }
}

void PopupLayer::prune() {
    expire();
    const auto before = entries.size();
    std::erase_if(entries, [this](const Entry& entry) {
        if (Access::attached(*entry.owner, *this) && Access::reachable(*entry.owner)) {
            return false;
        }
        Access::invalidate(*entry.owner);
        return true;
    });
    if (entries.size() != before) {
        changed();
    }
    pruneFrames();
}

void PopupLayer::pruneFrames() {
    for (auto it = frames.begin(); it != frames.end();) {
        const auto lifetime = frameContents.find(it->first);
        const bool requested = std::any_of(entries.begin(), entries.end(), [&](const auto& entry) {
            for (U64 index = 0; index < entry.requests.size(); ++index) {
                if (entry.requests[index].content == it->first && !entry.contents[index].expired()) {
                    return true;
                }
            }
            return false;
        });
        if (requested || (lifetime != frameContents.end() && !lifetime->second.expired())) {
            ++it;
            continue;
        }
        removeChild(*it->second);
        frameContents.erase(it->first);
        it = frames.erase(it);
    }
}

void PopupLayer::wentOnscreen(U64) {
    prune();
}

PopupLayer::Claim PopupLayer::claim(const Component* content) const {
    for (const auto& entry : entries) {
        if (entry.alive.expired() || !Access::attached(*entry.owner, *this) || !Access::reachable(*entry.owner)) {
            continue;
        }
        for (U64 index = 0; index < entry.requests.size(); ++index) {
            if (entry.requests[index].content == content && !entry.contents[index].expired()) {
                return {.entry = &entry, .request = &entry.requests[index]};
            }
        }
    }
    return {};
}

bool PopupLayer::interactiveAt(const Extent2D<F32>& point) const {
    for (const auto& item : placed) {
        if (!item.rect.contains(point.x, point.y)) {
            continue;
        }
        const auto found = claim(item.content);
        if (found.request && found.request->interactive && Access::acceptsInput(*item.content)) {
            return true;
        }
    }
    return false;
}

bool PopupLayer::ownedBy(const Component& content, const Component& owner) const {
    std::vector<const Component*> pending = {&content};
    std::vector<const Component*> seen;
    while (!pending.empty()) {
        const auto* next = pending.back();
        pending.pop_back();
        if (std::find(seen.begin(), seen.end(), next) != seen.end()) {
            continue;
        }
        seen.push_back(next);
        if (!Access::attached(*next, *this)) {
            const auto* unmounted = &Access::root(*next);
            const bool requested = std::any_of(entries.begin(), entries.end(), [&](const auto& entry) {
                if (entry.alive.expired()) return false;
                for (U64 index = 0; index < entry.requests.size(); ++index) {
                    if (!entry.contents[index].expired() && entry.requests[index].content == unmounted) return true;
                }
                return false;
            });
            if (!requested) continue;
        }
        if (Access::within(*next, owner)) {
            return true;
        }
        for (const auto& entry : entries) {
            if (entry.alive.expired() || !Access::reachable(*entry.owner)) {
                continue;
            }
            for (U64 index = 0; index < entry.requests.size(); ++index) {
                if (!entry.contents[index].expired() && Access::within(*next, *entry.requests[index].content)) {
                    pending.push_back(entry.owner);
                }
            }
        }
    }
    return false;
}

bool PopupLayer::covers(const Extent2D<F32>& point) const {
    if (interactiveAt(point)) {
        return true;
    }
    bool dismissable = false;
    for (const auto& item : placed) {
        const auto found = claim(item.content);
        if (!found.request || !Access::acceptsInput(*item.content)) {
            continue;
        }
        const auto& passthrough = found.request->passthrough;
        if (passthrough.has_value() && passthrough->contains(point.x, point.y)) {
            return false;
        }
        dismissable = dismissable || static_cast<bool>(found.request->onDismiss);
    }
    return dismissable;
}

bool PopupLayer::capturesInput() const {
    return captured && !capturedAlive.expired() && claim(captured).request &&
           Access::reachable(*captured) && Access::acceptsInput(*captured);
}

void PopupLayer::prepareInput() {
    if (!captured || capturesInput()) {
        return;
    }
    auto* previous = captured;
    const auto alive = capturedAlive;
    captured = nullptr;
    capturedButtons.fill(false);
    if (!alive.expired()) {
        Access::mouseEvent(*previous, MouseEvent{.type = MouseEventType::Cancel});
    }
}

bool PopupLayer::route(const MouseEvent& event) {
    const auto token = Access::alive(*this);
    if (capturesInput()) {
        return deliverChild(*captured, event);
    }
    std::vector<Placed> live;
    for (const auto& item : placed) {
        const auto found = claim(item.content);
        if (found.request && found.request->interactive && Access::acceptsInput(*item.content)) {
            live.push_back(item);
        }
    }
    for (U64 index = live.size(); index-- > 0;) {
        const auto& item = live[index];
        if (item.alive.expired() || !claim(item.content).request || !Access::acceptsInput(*item.content)) {
            continue;
        }
        if ((event.type == MouseEventType::Click || event.type == MouseEventType::Scroll) &&
            !item.rect.contains(event.position.x, event.position.y)) {
            continue;
        }
        if (event.type == MouseEventType::Click) {
            captured = item.content;
            capturedAlive = item.alive;
        }
        if (deliverChild(*item.content, event) || token.expired()) {
            return true;
        }
    }
    return false;
}

Extent2D<F32> PopupLayer::measure(const Context&, Extent2D<F32> available) {
    return {FillLength(available.x), FillLength(available.y)};
}

void PopupLayer::layout(const Context& ctx) {
    for (U64 pass = 0; pass < kArrangePasses; ++pass) {
        const U64 seen = revision;
        arranging = true;
        arrange(ctx);
        arranging = false;
        if (seen == revision) {
            return;
        }
    }
    invalidate(Dirty::Layout);
}

void PopupLayer::arrange(const Context& ctx) {
    const Rect bounds = frame();
    const F32 infinite = std::numeric_limits<F32>::infinity();
    expire();
    pruneFrames();
    const U64 seen = revision;

    std::vector<PopupRequest> shown;
    std::vector<const Component*> claimed;
    for (const auto& entry : entries) {
        if (!ownedBy(*entry.owner, Access::root(*this))) {
            continue;
        }
        for (const auto& request : entry.requests) {
            if (!Access::acceptsInput(*this, *request.content)) {
                continue;
            }
            if (std::find(claimed.begin(), claimed.end(), request.content) != claimed.end()) {
                continue;
            }
            claimed.push_back(request.content);
            const F64 since = requestedSince.try_emplace(request.content, ctx.time).first->second;
            if (request.delayed && ctx.time - since < ctx.hoverDelay) {
                requestFrame(since + ctx.hoverDelay);
                continue;
            }
            shown.push_back(request);
        }
    }
    std::erase_if(requestedSince, [&](const auto& entry) {
        return std::none_of(entries.begin(), entries.end(), [&](const Entry& owner) {
            return std::any_of(owner.requests.begin(), owner.requests.end(), [&](const PopupRequest& request) {
                return request.content == entry.first;
            });
        });
    });

    std::vector<Component*> children;
    for (const auto& request : shown) {
        auto& backdrop = frames[request.content];
        if (!backdrop) {
            backdrop = std::make_unique<Box>();
        }
        frameContents[request.content] = Access::alive(*request.content);
        children.push_back(backdrop.get());
        children.push_back(request.content);
    }
    mountChildren(children);
    prune();
    if (revision != seen) {
        return;
    }

    placed.clear();
    for (const auto& request : shown) {
        Box& backdrop = *frames[request.content];
        const F32 padding = request.panel ? Px(ctx, request.padding) : 0.0f;
        const F32 limit = request.maxWidth > 0.0f ? Px(ctx, request.maxWidth) : bounds.width;
        const F32 available = std::max(0.0f, limit - 2.0f * padding);
        Extent2D<F32> natural = {0.0f, 0.0f};
        if (request.size.x == 0.0f) {
            const auto size = measureChild(*request.content, ctx, {available, infinite});
            natural = {std::ceil(std::min(size.x, available) + 2.0f * padding), std::ceil(size.y + 2.0f * padding)};
        }
        const F32 width = ResolveLength(ctx, request.size.x, natural.x, bounds.width);
        if (request.size.y == 0.0f && (request.size.x != 0.0f || width < natural.x)) {
            const F32 inner = std::max(0.0f, width - 2.0f * padding);
            natural.y = std::ceil(measureChild(*request.content, ctx, {inner, infinite}).y + 2.0f * padding);
        }
        const F32 height = ResolveLength(ctx, request.size.y, natural.y, bounds.height);
        const Rect rect = Place(ctx, request, width, height, bounds);

        backdrop.update({
            .instances = {{.rect = rect, .visible = request.panel && !rect.empty(),
                           .backgroundColor = ctx.color("popup_bg")}},
            .cornerRadius = Px(ctx, kPopupRounding),
            .borderWidth = Px(ctx, 1.0f),
            .borderColor = ctx.color("border"),
            .strokeFringe = true,
        });
        layoutChild(ctx, backdrop, rect);
        layoutChild(ctx, *request.content,
                    {rect.x + padding, rect.y + padding, std::max(0.0f, rect.width - 2.0f * padding),
                     std::max(0.0f, rect.height - 2.0f * padding)});
        placed.push_back({.content = request.content, .alive = Access::alive(*request.content), .rect = rect});
    }
}

bool PopupLayer::mouseEvent(const MouseEvent& event) {
    const auto token = Access::alive(*this);
    prepareInput();
    if (token.expired()) {
        return true;
    }
    const U64 button = static_cast<U64>(event.button);
    if ((event.type == MouseEventType::Click || event.type == MouseEventType::Release) &&
        button >= capturedButtons.size()) {
        return false;
    }
    const bool inside = interactiveAt(event.position);
    switch (event.type) {
        case MouseEventType::Click: {
            if (inside || capturesInput()) {
                capturedButtons[button] = true;
                route(event);
                return true;
            }
            expire();
            std::vector<std::pair<std::weak_ptr<bool>, std::function<void()>>> dismiss;
            for (const auto& item : placed) {
                const auto found = claim(item.content);
                if (found.request && found.request->onDismiss && Access::acceptsInput(*item.content)) {
                    dismiss.emplace_back(found.entry->alive, found.request->onDismiss);
                }
            }
            for (auto& entry : entries) {
                const auto count = entry.requests.size();
                for (U64 index = count; index-- > 0;) {
                    if (entry.requests[index].onDismiss && Access::acceptsInput(*entry.requests[index].content)) {
                        entry.drop(index);
                    }
                }
                if (entry.requests.size() != count) {
                    Access::invalidate(*entry.owner);
                }
            }
            std::erase_if(entries, [](const Entry& entry) {
                return entry.requests.empty();
            });
            if (!dismiss.empty()) {
                invalidate(Dirty::Layout);
            }
            for (const auto& [alive, callback] : dismiss) {
                if (!alive.expired()) {
                    callback();
                    if (token.expired()) {
                        return true;
                    }
                }
            }
            return !dismiss.empty();
        }
        case MouseEventType::Scroll:
            if (inside) {
                route(event);
            }
            return inside;
        case MouseEventType::Move:
            route(event);
            return inside;
        case MouseEventType::Release:
            if (capturedButtons[button]) {
                capturedButtons[button] = false;
                route(event);
                if (!token.expired() && std::none_of(capturedButtons.begin(), capturedButtons.end(), [](bool held) { return held; })) {
                    captured = nullptr;
                }
            }
            return inside;
        default:
            captured = nullptr;
            capturedButtons.fill(false);
            mouseEventChildren(event);
            return false;
    }
}

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

struct Root::Impl {
    Config config;
    Canvas canvas;
    RootComponent root;

    Impl() {
        canvas.mount(root);
    }
};

Root::Root() {
    impl = std::make_unique<Impl>();
}

Root::~Root() = default;

void Root::update(Config config) {
    impl->config = std::move(config);
    impl->root.setLayers(impl->config.layers);
    impl->canvas.update({.id = impl->config.id + ":root"});
}

void Root::render(const Sakura::Context& ctx) {
    impl->canvas.render(ctx, DisplayMetrics::RootRegion(Metrics(ctx), impl->config.topOffset));
}

}  // namespace Jetstream::Sakura
