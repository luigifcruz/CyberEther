#ifndef JETSTREAM_TESTS_CORE_SAKURA_COMPONENT_HARNESS_HH
#define JETSTREAM_TESTS_CORE_SAKURA_COMPONENT_HARNESS_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/curves.hh>
#include <jetstream/render/sakura/components/dock.hh>
#include <jetstream/render/sakura/components/image.hh>
#include <jetstream/render/sakura/components/label.hh>
#include <jetstream/render/sakura/components/polyline.hh>
#include <jetstream/render/sakura/components/root.hh>

#include "render/sakura/context.hh"
#include "render/sakura/detail/access.hh"
#include "render/sakura/detail/dock_drag.hh"
#include "render/sakura/detail/input_router.hh"
#include "render/sakura/detail/popup_layer.hh"
#include "render/sakura/detail/root_component.hh"

#include "render/sakura/themes.hh"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <variant>
#include <vector>

namespace SakuraTest {

using Jetstream::Extent2D;
using Jetstream::F32;
using Jetstream::MouseButton;
using Jetstream::MouseEvent;
using Jetstream::MouseEventType;
using Jetstream::Rect;
using Jetstream::U64;
using Jetstream::U8;

constexpr F32 kInfinite = std::numeric_limits<F32>::infinity();
constexpr Rect kScreen = {0.0f, 0.0f, 800.0f, 600.0f};

inline bool Tap(Jetstream::Sakura::Component& root, Jetstream::KeyCode key, Jetstream::KeyModifiers modifiers = {}) {
    using Jetstream::KeyEventType;
    const bool handled = Jetstream::Sakura::Access::key(root, {KeyEventType::Press, key, modifiers});
    Jetstream::Sakura::Access::key(root, {KeyEventType::Release, key, modifiers});
    return handled;
}

inline void Type(Jetstream::Sakura::Component& root, const std::string& text) {
    for (const char character : text) {
        Jetstream::Sakura::Access::text(root, {static_cast<char32_t>(static_cast<unsigned char>(character))});
    }
}

inline void ClickFocus(Jetstream::Sakura::Component& root, std::optional<Extent2D<F32>> point) {
    using Jetstream::Sakura::Access;
    Access::clickFocus(root, point, point.has_value(), Access::focusGeneration(root));
}

inline const Jetstream::Sakura::Component* FocusOwner(const Jetstream::Sakura::Component& root) {
    return Jetstream::Sakura::Access::focusOwner(root);
}

inline Jetstream::Sakura::Context MakeContext(Extent2D<F32> size = {800.0f, 600.0f}) {
    Jetstream::Sakura::Context ctx;
    ctx.pixelRatio = 1.0f;
    ctx.framebufferSize = {static_cast<U64>(size.x), static_cast<U64>(size.y)};
    ctx.windowFocused = true;
    return ctx;
}

inline MouseEvent Mouse(MouseEventType type, F32 x, F32 y, MouseButton button = MouseButton::Left, U8 clicks = 1) {
    return {.type = type, .button = button, .position = {x, y}, .clicks = clicks};
}

inline MouseEvent Wheel(F32 x, F32 y, F32 dx, F32 dy) {
    MouseEvent event = Mouse(MouseEventType::Scroll, x, y);
    event.scroll = {dx, dy};
    return event;
}

inline MouseEvent Wheel(F32 x, F32 y, F32 dy) {
    return Wheel(x, y, 0.0f, dy);
}

struct Peek : Jetstream::Sakura::Component {
    static const std::vector<Jetstream::Sakura::Component*>& Children(const Jetstream::Sakura::Component& component) {
        return (component.*(&Peek::children))();
    }

    static Rect Frame(const Jetstream::Sakura::Component& component) {
        const Rect frame = (component.*(&Peek::frame))();
        const Extent2D<F32> shift = (component.*(&Peek::translation))();
        return {frame.x + shift.x, frame.y + shift.y, frame.width, frame.height};
    }

    static Rect Clip(const Jetstream::Sakura::Component& component) {
        return (component.*(&Peek::clip))();
    }

    static Extent2D<F32> Translation(const Jetstream::Sakura::Component& component) {
        return (component.*(&Peek::translation))();
    }

    static std::optional<Rect> InheritedClip(const Jetstream::Sakura::Component& component) {
        return (component.*(&Peek::inheritedClip))();
    }

    static std::optional<Jetstream::SurfaceCursor> CursorAt(const Jetstream::Sakura::Component& component,
                                                           Extent2D<F32> point) {
        return (component.*(&Peek::cursorAt))(point);
    }
};

template<typename T>
void Collect(Jetstream::Sakura::Component& node, std::vector<T*>& found) {
    if (!node.visible()) {
        return;
    }
    if (auto* match = dynamic_cast<T*>(&node)) {
        found.push_back(match);
    }
    for (auto* child : Peek::Children(node)) {
        if (child) {
            Collect(*child, found);
        }
    }
}

struct DrawnLeaf {
    std::variant<Jetstream::Sakura::Box::Config,
                 Jetstream::Sakura::Label::Config,
                 Jetstream::Sakura::Polyline::Config,
                 Jetstream::Sakura::Curves::Config,
                 Jetstream::Sakura::Image::Config> config;
    Extent2D<F32> translation = {0.0f, 0.0f};
    std::optional<Rect> clip;

    bool operator==(const DrawnLeaf&) const = default;
};

template<typename... Leaves>
bool CollectLeaf(const Jetstream::Sakura::Component& node, std::vector<DrawnLeaf>& drawn) {
    return ([&] {
        if (const auto* leaf = dynamic_cast<const Leaves*>(&node)) {
            drawn.push_back({
                .config = leaf->config(),
                .translation = Peek::Translation(node),
                .clip = Peek::InheritedClip(node),
            });
            return true;
        }
        return false;
    }() || ...);
}

inline void CollectDrawn(const Jetstream::Sakura::Component& node, std::vector<DrawnLeaf>& drawn) {
    namespace Sakura = Jetstream::Sakura;
    if (!node.visible()) {
        return;
    }
    if (CollectLeaf<Sakura::Box, Sakura::Label, Sakura::Polyline, Sakura::Curves, Sakura::Image>(node, drawn)) {
        return;
    }
    for (auto* child : Peek::Children(node)) {
        if (child) {
            CollectDrawn(*child, drawn);
        }
    }
}

inline std::vector<DrawnLeaf> Drawn(const Jetstream::Sakura::Component& root) {
    std::vector<DrawnLeaf> drawn;
    CollectDrawn(root, drawn);
    return drawn;
}

template<typename T>
std::vector<T*> Find(Jetstream::Sakura::Component& root) {
    std::vector<T*> found;
    Collect(root, found);
    return found;
}

template<typename T>
std::vector<T*> Direct(Jetstream::Sakura::Component& root) {
    std::vector<T*> found;
    for (auto* child : Peek::Children(root)) {
        if (child && child->visible()) {
            if (auto* match = dynamic_cast<T*>(child)) {
                found.push_back(match);
            }
        }
    }
    return found;
}

struct Probe : Jetstream::Sakura::Component {
    Extent2D<F32> desired = {0.0f, 0.0f};
    bool consume = true;
    U64 layouts = 0;
    U64 measures = 0;
    Rect laidFrame;
    Rect laidClip;
    std::optional<Rect> laidInherited;
    std::vector<MouseEvent> events;

    explicit Probe(Extent2D<F32> desired = {0.0f, 0.0f}) : desired(desired) {}

    Rect currentFrame() const {
        return frame();
    }

    Rect currentClip() const {
        return clip();
    }

    std::optional<Rect> inherited() const {
        return inheritedClip();
    }

    void prepare() {
        markConfigured();
    }

    void resize(Extent2D<F32> next) {
        desired = next;
        invalidate(Dirty::Layout);
    }

    void adopt(Jetstream::Sakura::Component& child) {
        addChild(child);
    }

    U64 count(MouseEventType type) const {
        U64 total = 0;
        for (const auto& event : events) {
            total += event.type == type ? 1 : 0;
        }
        return total;
    }

 protected:
    Extent2D<F32> measure(const Jetstream::Sakura::Context&, Extent2D<F32>) override {
        ++measures;
        return desired;
    }

    void layout(const Jetstream::Sakura::Context&) override {
        ++layouts;
        laidFrame = frame();
        laidClip = clip();
        laidInherited = inheritedClip();
    }

    bool mouseEvent(const MouseEvent& event) override {
        events.push_back(event);
        return consume;
    }
};

struct Flow : Probe {
    F32 naturalWidth = 0.0f;
    F32 area = 0.0f;

    Flow(F32 naturalWidth, F32 lineHeight) : naturalWidth(naturalWidth), area(naturalWidth * lineHeight) {}

 protected:
    Extent2D<F32> measure(const Jetstream::Sakura::Context& ctx, Extent2D<F32> available) override {
        Probe::measure(ctx, available);
        const F32 width = std::min(naturalWidth, available.x);
        return {width, std::ceil(area / std::max(1.0f, width))};
    }
};

struct Host : Jetstream::Sakura::Component {
    explicit Host(Jetstream::Sakura::Component& child, bool clipsChildren = false) : child(child) {
        setClipsChildren(clipsChildren);
        addChild(child);
    }

    void place(const Jetstream::Sakura::Context& ctx, Rect bounds) {
        Jetstream::Sakura::Access::advance(ctx.time);
        layoutRoot(ctx, bounds);
        Jetstream::Sakura::Access::settleFocus(*this);
    }

    Extent2D<F32> measureOf(const Jetstream::Sakura::Context& ctx, Extent2D<F32> available) {
        return measureChild(child, ctx, available);
    }

    bool send(const MouseEvent& event) {
        return mouseEventChildren(event);
    }

    bool click(F32 x, F32 y, MouseButton button = MouseButton::Left) {
        const bool pressed = send(Mouse(MouseEventType::Click, x, y, button));
        const bool released = send(Mouse(MouseEventType::Release, x, y, button));
        return pressed || released;
    }

 protected:
    void layout(const Jetstream::Sakura::Context& ctx) override {
        layoutChild(ctx, child, frame());
    }

 private:
    Jetstream::Sakura::Component& child;
};

struct Place : Jetstream::Sakura::Component {
    Rect target;

    Place(Jetstream::Sakura::Component& child, Rect target) : target(target), child(child) {
        addChild(child);
    }

    void place(const Jetstream::Sakura::Context& ctx, Rect bounds) {
        layoutRoot(ctx, bounds);
    }

    void move(Rect next) {
        target = next;
        invalidate(Dirty::Layout);
    }

    bool send(const MouseEvent& event) {
        return mouseEventChildren(event);
    }

 protected:
    void layout(const Jetstream::Sakura::Context& ctx) override {
        layoutChild(ctx, child, target);
    }

 private:
    Jetstream::Sakura::Component& child;
};

struct LayerHost : Jetstream::Sakura::RootComponent {
    LayerHost() = default;

    explicit LayerHost(std::vector<Jetstream::Sakura::Layer*> next) {
        setLayers(std::move(next));
    }

    void place(const Jetstream::Sakura::Context& ctx, Rect bounds) {
        Jetstream::Sakura::Access::advance(ctx.time);
        layoutRoot(ctx, bounds);
        Jetstream::Sakura::Access::settleFocus(*this);
    }

    bool send(const MouseEvent& event) {
        return RootComponent::mouseEvent(event);
    }

    bool click(F32 x, F32 y, MouseButton button = MouseButton::Left) {
        const bool pressed = send(Mouse(MouseEventType::Click, x, y, button));
        const bool released = send(Mouse(MouseEventType::Release, x, y, button));
        return pressed || released;
    }
};

inline Jetstream::Sakura::DockTiles Leaf(const std::string& pane) {
    return Jetstream::Sakura::DockTiles{.pane = pane};
}

inline Jetstream::Sakura::DockTiles Row(std::vector<F32> weights,
                                                  std::vector<Jetstream::Sakura::DockTiles> children) {
    return Jetstream::Sakura::DockTiles{
        .vertical = false,
        .weights = std::move(weights),
        .children = std::move(children),
    };
}

inline Jetstream::Sakura::DockTiles Column(std::vector<F32> weights,
                                                     std::vector<Jetstream::Sakura::DockTiles> children) {
    return Jetstream::Sakura::DockTiles{
        .vertical = true,
        .weights = std::move(weights),
        .children = std::move(children),
    };
}

inline Jetstream::Sakura::Dock::Pane MakePane(const std::string& id,
                                                        Jetstream::Sakura::Component* content = nullptr) {
    return Jetstream::Sakura::Dock::Pane{.id = id, .title = id, .group = "panel", .content = content};
}

struct LayerSlot : Jetstream::Sakura::Layer {
    Jetstream::Sakura::Component* child = nullptr;
    Rect rect;
    std::vector<Jetstream::Sakura::PopupRequest> requests;
    std::function<void(const Jetstream::Sakura::Context&)> onLayout;

    void hold(Jetstream::Sakura::Component& next, Rect at) {
        child = &next;
        rect = at;
        update({.content = &next});
        invalidate(Dirty::Layout);
    }

    void request(std::vector<Jetstream::Sakura::PopupRequest> next) {
        requests = std::move(next);
        invalidate(Dirty::Layout);
    }

    void relayout() {
        invalidate(Dirty::Layout);
    }

    bool covers(const Extent2D<F32>& point) const override {
        return child && rect.contains(point.x, point.y);
    }

 protected:
    void layout(const Jetstream::Sakura::Context& ctx) override {
        if (child) {
            layoutChild(ctx, *child, rect);
        }
        for (const auto& request : requests) {
            RequestPopup(ctx, request);
        }
        if (onLayout) {
            onLayout(ctx);
        }
    }
};

inline bool LiveVerification() {
    return GENERATE(true, false);
}

struct FrameStats {
    bool rebuilt = false;
    bool restructured = false;
    bool repainted = false;
    bool laidOut = false;

    bool clean() const {
        return !rebuilt && !restructured && !repainted && !laidOut;
    }
};

inline bool RepaintOnly(const FrameStats& stats) {
    return stats.repainted && !stats.rebuilt && !stats.restructured;
}

inline bool Unpainted(const FrameStats& stats) {
    return !stats.repainted && !stats.rebuilt && !stats.restructured;
}

inline bool Kept(const FrameStats& stats) {
    return !stats.rebuilt && !stats.restructured;
}

struct Frames {
    using Access = Jetstream::Sakura::Access;

    Jetstream::Sakura::Component& root;
    Rect bounds;
    Jetstream::Sakura::Context ctx;
    bool built = false;
    bool verify = LiveVerification();
    Jetstream::Sakura::Component* reference = nullptr;
    bool referenceBuilt = false;
    Jetstream::F64 timeStep = 0.0;
    U64 rebuilds = 0;
    U64 restructures = 0;
    U64 repaints = 0;
    U64 layouts = 0;
    std::optional<Jetstream::Sakura::Palette> palette;

    explicit Frames(Jetstream::Sakura::Component& root,
                    Rect bounds = kScreen,
                    Jetstream::Sakura::Context context = MakeContext()) :
        root(root),
        bounds(bounds),
        ctx(std::move(context)) {
        stub();
    }

    Frames(Jetstream::Sakura::Component& root,
           Jetstream::Sakura::Component& reference,
           Rect bounds = kScreen,
           Jetstream::Sakura::Context context = MakeContext()) :
        root(root),
        bounds(bounds),
        ctx(std::move(context)),
        verify(false),
        reference(&reference) {
        stub();
    }

    void stub() {
        ctx.attach = [](const Jetstream::Sakura::Component&, Jetstream::Sakura::Drawable*) {
            return Jetstream::Result::SUCCESS;
        };
        ctx.release = [](Jetstream::Sakura::Drawable*,
                         std::shared_ptr<Jetstream::Render::Components::Generic>) {};
    }

    FrameStats run() {
        ++ctx.frame;
        ctx.time += timeStep;
        FrameStats stats;
        prepare(root, built, stats);
        if (!palette.has_value() || *palette != ctx.palette.get()) {
            palette = ctx.palette.get();
            Access::invalidateLayout(root);
        }
        Access::advance(ctx.time);
        const U64 before = Access::layoutRuns();
        Access::layout(root, ctx, bounds);
        stats.laidOut = Access::layoutRuns() != before;
        layouts += stats.laidOut ? 1 : 0;
        Access::present(root, ctx.frame);
        Access::settleFocus(root);
        if (Access::resourceDirty(root) || Access::paintDirty(root)) {
            (void)Access::paint(root);
            stats.repainted = true;
        }
        if (verify) {
            check();
        }
        if (reference) {
            compare();
        }
        rebuilds += stats.rebuilt ? 1 : 0;
        restructures += stats.restructured ? 1 : 0;
        repaints += stats.repainted ? 1 : 0;
        return stats;
    }

    void prepare(Jetstream::Sakura::Component& tree, bool& treeBuilt, FrameStats& stats) {
        if (!treeBuilt || Access::resourceDirty(tree)) {
            treeBuilt = Access::build(tree, ctx) == Jetstream::Result::SUCCESS;
            stats.rebuilt = true;
        } else if (Access::structureDirty(tree)) {
            if (Access::hasUnbuilt(tree)) {
                (void)Access::buildPending(tree, ctx);
            }
            Access::clearStructureDirty(tree);
            stats.restructured = true;
        }
    }

    void compare() {
        FrameStats ignored;
        prepare(*reference, referenceBuilt, ignored);
        auto forced = ctx;
        forced.deltaTime = 0.0f;
        Access::invalidateSubtree(*reference);
        Access::layout(*reference, forced, bounds);
        Access::present(*reference, ctx.frame);
        Access::settleFocus(*reference);
        if (Access::resourceDirty(*reference) || Access::paintDirty(*reference)) {
            (void)Access::paint(*reference);
        }
        const auto live = Drawn(root);
        const auto expected = Drawn(*reference);
        U64 first = 0;
        while (first < live.size() && first < expected.size() && live[first] == expected[first]) {
            ++first;
        }
        INFO("The retained tree drew differently from a fully laid out reference tree");
        INFO("frame " << ctx.frame << ", first differing leaf " << first << " of " << live.size() << " and "
                      << expected.size());
        CHECK(live == expected);
    }

    void check() {
        if (Access::layoutDirty(root) || Access::structureDirty(root) || Access::resourceDirty(root) ||
            Access::paintDirty(root)) {
            return;
        }
        auto forced = ctx;
        forced.deltaTime = 0.0f;
        Access::invalidateLayout(root);
        Access::layout(root, forced, bounds);
        const bool stale = Access::resourceDirty(root) || Access::paintDirty(root) || Access::structureDirty(root);
        INFO("A full layout changed what the skipped layouts left on screen");
        CHECK_FALSE(stale);
        if (stale) {
            (void)Access::paint(root);
            Access::clearStructureDirty(root);
        }
    }

    template<typename Update>
    FrameStats run(Update update) {
        update();
        return run();
    }

    template<typename Update>
    bool settle(Update update, U64 limit = 8) {
        for (U64 frame = 0; frame < limit; ++frame) {
            if (run(update).clean() && frame > 0) {
                return true;
            }
        }
        return false;
    }

    bool settle(U64 limit = 8) {
        return settle([] {}, limit);
    }
};

template<typename Update>
U64 DirtyFrames(Frames& frames, Update update, U64 count = 5) {
    U64 dirty = 0;
    for (U64 frame = 0; frame < count; ++frame) {
        dirty += frames.run(update).clean() ? 0 : 1;
    }
    return dirty;
}

inline U64 DirtyFrames(Frames& frames, U64 count = 5) {
    return DirtyFrames(frames, [] {}, count);
}

template<typename Update>
bool Retains(Frames& frames, Update update, U64 count = 5) {
    return DirtyFrames(frames, update, count) == 0;
}

template<typename Update>
bool QuietMoves(Frames& frames, Update update, const std::vector<Extent2D<F32>>& points,
                const std::function<bool(const MouseEvent&)>& send) {
    for (const auto& point : points) {
        send(Mouse(MouseEventType::Move, point.x, point.y));
        if (!frames.run(update).clean()) {
            return false;
        }
    }
    return true;
}

template<typename Root, typename Update>
void CheckHoverRepaintsOnce(Frames& frames, Root& root, Update update, Extent2D<F32> inside, Extent2D<F32> outside) {
    root.send(Mouse(MouseEventType::Move, outside.x, outside.y));
    REQUIRE(frames.settle(update));

    root.send(Mouse(MouseEventType::Move, inside.x, inside.y));
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(frames.run(update).clean());

    root.send(Mouse(MouseEventType::Move, inside.x + 1.0f, inside.y + 1.0f));
    CHECK(frames.run(update).clean());

    root.send(Mouse(MouseEventType::Move, outside.x, outside.y));
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(frames.run(update).clean());

    root.send(Mouse(MouseEventType::Move, outside.x + 1.0f, outside.y + 1.0f));
    CHECK(frames.run(update).clean());
}

template<typename Root, typename Update>
void CheckLeaveRepaintsOnce(Frames& frames, Root& root, Update update, Extent2D<F32> inside) {
    REQUIRE(frames.settle(update));
    root.send(Mouse(MouseEventType::Leave, inside.x, inside.y));
    CHECK(frames.run(update).clean());
    root.send(Mouse(MouseEventType::Cancel, inside.x, inside.y));
    CHECK(frames.run(update).clean());

    root.send(Mouse(MouseEventType::Move, inside.x, inside.y));
    CHECK(RepaintOnly(frames.run(update)));
    root.send(Mouse(MouseEventType::Leave, inside.x, inside.y));
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(frames.run(update).clean());
}

inline void Hover(LayerHost& host, Frames& frames, F32 x, F32 y) {
    frames.ctx.mouse = Extent2D<F32>{x, y};
    host.send(Mouse(MouseEventType::Move, x, y));
}

inline void Press(LayerHost& host, Frames& frames, F32 x, F32 y) {
    frames.ctx.mouse = Extent2D<F32>{x, y};
    host.send(Mouse(MouseEventType::Click, x, y));
}

inline void Release(LayerHost& host, Frames& frames, F32 x, F32 y) {
    frames.ctx.mouse = Extent2D<F32>{x, y};
    host.send(Mouse(MouseEventType::Release, x, y));
}

inline const Jetstream::Sakura::Palette& DistinctPalette() {
    static const Jetstream::Sakura::Palette palette = [] {
        std::set<std::string> keys;
        for (const auto& [name, theme] : Jetstream::Sakura::Themes) {
            for (const auto& [key, color] : theme) {
                keys.insert(key);
            }
        }
        Jetstream::Sakura::Palette result;
        U64 index = 0;
        for (const auto& key : keys) {
            const F32 shade = static_cast<F32>(++index) / static_cast<F32>(keys.size() + 1);
            result[key] = {shade, 1.0f - shade, 0.5f, 1.0f};
        }
        result["transparent"] = {0.0f, 0.0f, 0.0f, 0.0f};
        return result;
    }();
    return palette;
}

struct InputDriver {
    Jetstream::Render::InputState state;
    Jetstream::Sakura::InputFrame frame;
    Jetstream::Sakura::InputRouter router;
    Jetstream::F64 clock = 1.0;

    Jetstream::U64 run(LayerHost& host,
                       const Jetstream::Sakura::Context& ctx,
                       const Rect& screen,
                       const std::vector<Jetstream::Render::InputQueue::Event>& batch) {
        Jetstream::U64 frames = 0;
        auto queued = batch;
        do {
            state.advance(clock, queued);
            queued.clear();
            clock += 1.0 / 60.0;
            ++frames;
            Jetstream::Sakura::Private::SampleInput(frame, state);
            for (const auto& event : router.route(frame, screen).events) {
                if (const auto* mouse = std::get_if<MouseEvent>(&event)) {
                    MouseEvent absolute = *mouse;
                    absolute.position = {screen.x + mouse->position.x * screen.width,
                                         screen.y + mouse->position.y * screen.height};
                    host.send(absolute);
                } else if (const auto* key = std::get_if<Jetstream::KeyEvent>(&event)) {
                    Jetstream::Sakura::Access::key(host, *key);
                } else if (const auto* text = std::get_if<Jetstream::TextEvent>(&event)) {
                    Jetstream::Sakura::Access::text(host, *text);
                }
            }
            host.place(ctx, screen);
        } while (state.hasPending());
        return frames;
    }
};

}  // namespace SakuraTest

#endif  // JETSTREAM_TESTS_CORE_SAKURA_COMPONENT_HARNESS_HH
