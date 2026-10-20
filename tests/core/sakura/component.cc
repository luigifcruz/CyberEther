#include <catch2/catch_test_macros.hpp>

#include "component_harness.hh"

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/stack.hh>
#include <jetstream/render/sakura/components/text_input.hh>

#include "scene.hh"

#include <optional>
#include <utility>
#include <vector>

using namespace Jetstream;
using namespace SakuraTest;

namespace {

struct Board : Sakura::Component {
    using Component::children;
    using Component::clip;
    using Component::mouseEventChildren;
    using Component::frame;
    using Component::inheritedClip;
    using Component::mountChildren;
    using Component::setChildren;

    std::vector<std::pair<Sakura::Component*, Rect>> slots;
    std::optional<Rect> bound;
    Extent2D<F32> shift = {0.0f, 0.0f};
    bool defaultLayout = false;

    explicit Board(bool clipsChildren = false) {
        setClipsChildren(clipsChildren);
    }

    void put(Sakura::Component& child, Rect rect) {
        addChild(child);
        slots.push_back({&child, rect});
    }

    void attach(Sakura::Component& child) {
        addChild(child);
    }

    void place(const Sakura::Context& ctx, Rect bounds) {
        invalidate(Dirty::Layout);
        layoutRoot(ctx, bounds);
    }

    void relayout() {
        invalidate(Dirty::Layout);
    }

 protected:
    void layout(const Sakura::Context& ctx) override {
        if (defaultLayout) {
            Component::layout(ctx);
            return;
        }
        const Sakura::Context inner = ctx.translated(shift);
        for (auto& [child, rect] : slots) {
            if (bound.has_value()) {
                layoutChild(inner, *child, rect, *bound);
            } else {
                layoutChild(inner, *child, rect);
            }
        }
    }
};

struct StaleClipProbe : Sakura::Component {
    Sakura::Box box;
    U64 seen = 0;

    StaleClipProbe() {
        addChild(box);
    }

 protected:
    void layout(const Sakura::Context& ctx) override {
        box.update({.instances = {{.rect = {20.0f, 20.0f, 100.0f, 40.0f}}}});
        if (layoutRevision() != seen) {
            seen = layoutRevision();
            layoutChild(ctx, box, frame());
        }
    }
};

struct RootHost : Host {
    using Component::clip;
    using Component::frame;
    using Component::inheritedClip;

    using Host::Host;
};

}  // namespace

TEST_CASE("Root layout sets the frame and clip to the given rect", "[core][sakura][component]") {
    const auto ctx = MakeContext();
    Probe child({10.0f, 10.0f});
    RootHost root(child);

    root.place(ctx, {10.0f, 20.0f, 300.0f, 200.0f});

    CHECK(root.frame() == Rect{10.0f, 20.0f, 300.0f, 200.0f});
    CHECK(root.clip() == Rect{10.0f, 20.0f, 300.0f, 200.0f});
    CHECK(root.inheritedClip() == Rect{10.0f, 20.0f, 300.0f, 200.0f});
    CHECK(child.layouts == 1);
    CHECK(child.laidFrame == Rect{10.0f, 20.0f, 300.0f, 200.0f});
    CHECK(child.laidClip == Rect{10.0f, 20.0f, 300.0f, 200.0f});
}

TEST_CASE("Placed children inherit the parent clip", "[core][sakura][component]") {
    const auto ctx = MakeContext();
    Board outer;
    Board inner(false);
    Probe probe;
    outer.put(inner, {50.0f, 50.0f, 100.0f, 100.0f});
    inner.put(probe, {60.0f, 60.0f, 500.0f, 500.0f});

    outer.place(ctx, {0.0f, 0.0f, 400.0f, 400.0f});

    CHECK(inner.clip() == Rect{0.0f, 0.0f, 400.0f, 400.0f});
    CHECK(probe.laidFrame == Rect{60.0f, 60.0f, 500.0f, 500.0f});
    CHECK(probe.laidClip == Rect{0.0f, 0.0f, 400.0f, 400.0f});
}

TEST_CASE("Clipping parents intersect the child clip with their frame", "[core][sakura][component]") {
    const auto ctx = MakeContext();
    Board outer;
    Board inner(true);
    Probe probe;
    outer.put(inner, {50.0f, 50.0f, 100.0f, 100.0f});
    inner.put(probe, {60.0f, 60.0f, 500.0f, 500.0f});

    outer.place(ctx, {0.0f, 0.0f, 400.0f, 400.0f});

    CHECK(probe.laidFrame == Rect{60.0f, 60.0f, 500.0f, 500.0f});
    CHECK(probe.laidClip == Rect{50.0f, 50.0f, 100.0f, 100.0f});

    outer.slots[0].second = {300.0f, 300.0f, 200.0f, 200.0f};
    outer.place(ctx, {0.0f, 0.0f, 400.0f, 400.0f});

    CHECK(probe.laidClip == Rect{300.0f, 300.0f, 100.0f, 100.0f});
}

TEST_CASE("An explicit layout clip narrows the inherited clip", "[core][sakura][component]") {
    const auto ctx = MakeContext();
    Board outer;
    Board inner(true);
    Probe probe;
    outer.put(inner, {50.0f, 50.0f, 100.0f, 100.0f});
    inner.put(probe, {60.0f, 60.0f, 80.0f, 80.0f});

    inner.bound = Rect{40.0f, 70.0f, 40.0f, 200.0f};
    outer.place(ctx, {0.0f, 0.0f, 400.0f, 400.0f});

    CHECK(probe.laidFrame == Rect{60.0f, 60.0f, 80.0f, 80.0f});
    CHECK(probe.laidClip == Rect{50.0f, 70.0f, 30.0f, 80.0f});

    inner.bound = Rect{500.0f, 500.0f, 10.0f, 10.0f};
    inner.relayout();
    outer.place(ctx, {0.0f, 0.0f, 400.0f, 400.0f});

    CHECK(probe.laidClip.empty());
}

TEST_CASE("Translated children keep their clip in local space", "[core][sakura][component]") {
    const auto ctx = MakeContext();
    Board board(true);
    Probe probe;
    board.put(probe, {10.0f, 10.0f, 50.0f, 50.0f});
    board.shift = {100.0f, 50.0f};

    board.place(ctx, {0.0f, 0.0f, 400.0f, 300.0f});

    CHECK(probe.laidFrame == Rect{10.0f, 10.0f, 50.0f, 50.0f});
    CHECK(probe.laidClip == Rect{-100.0f, -50.0f, 400.0f, 300.0f});
    CHECK(probe.inherited() == Rect{-100.0f, -50.0f, 400.0f, 300.0f});
}

TEST_CASE("Translated children receive events in local coordinates", "[core][sakura][component]") {
    const auto ctx = MakeContext();
    Board board(true);
    Probe probe;
    board.put(probe, {10.0f, 10.0f, 50.0f, 50.0f});
    board.shift = {100.0f, 50.0f};
    board.place(ctx, {0.0f, 0.0f, 400.0f, 300.0f});

    CHECK(board.mouseEventChildren(Mouse(MouseEventType::Click, 120.0f, 70.0f)));
    REQUIRE(probe.events.size() == 1);
    CHECK(probe.events[0].position.x == 20.0f);
    CHECK(probe.events[0].position.y == 20.0f);

    CHECK_FALSE(board.mouseEventChildren(Mouse(MouseEventType::Click, 420.0f, 70.0f)));
    CHECK(probe.events.size() == 1);
}

TEST_CASE("Children laid out with the parent translation are not shifted", "[core][sakura][component]") {
    auto ctx = MakeContext().translated({30.0f, 40.0f});
    Board board(true);
    Probe probe;
    board.put(probe, {10.0f, 10.0f, 50.0f, 50.0f});

    board.place(ctx, {0.0f, 0.0f, 200.0f, 100.0f});

    CHECK(probe.laidClip == Rect{0.0f, 0.0f, 200.0f, 100.0f});
    CHECK(board.mouseEventChildren(Mouse(MouseEventType::Click, 15.0f, 15.0f)));
    REQUIRE(probe.events.size() == 1);
    CHECK(probe.events[0].position.x == 15.0f);
    CHECK(probe.events[0].position.y == 15.0f);
}

TEST_CASE("Placed components report their own clip as inherited", "[core][sakura][component]") {
    const auto ctx = MakeContext();
    Board board(true);
    Probe probe;
    board.put(probe, {10.0f, 10.0f, 50.0f, 50.0f});
    board.bound = Rect{0.0f, 0.0f, 30.0f, 30.0f};

    board.place(ctx, {0.0f, 0.0f, 400.0f, 300.0f});

    CHECK(probe.laidInherited == Rect{0.0f, 0.0f, 30.0f, 30.0f});
    CHECK(probe.inherited() == probe.currentClip());
}

TEST_CASE("Unplaced components inherit the nearest placed ancestor child clip", "[core][sakura][component]") {
    const auto ctx = MakeContext();
    Board outer;
    Board holder(true);
    Probe probe;
    Probe grandchild;
    outer.put(holder, {50.0f, 50.0f, 100.0f, 100.0f});
    holder.attach(probe);
    probe.adopt(grandchild);

    outer.place(ctx, {0.0f, 0.0f, 400.0f, 400.0f});

    CHECK(holder.clip() == Rect{0.0f, 0.0f, 400.0f, 400.0f});
    CHECK(probe.layouts == 0);
    CHECK(probe.inherited() == Rect{50.0f, 50.0f, 100.0f, 100.0f});
    CHECK(grandchild.inherited() == Rect{50.0f, 50.0f, 100.0f, 100.0f});
}

TEST_CASE("Configured but unplaced components inherit from their ancestor", "[core][sakura][component]") {
    const auto ctx = MakeContext();
    Board outer;
    Board holder(true);
    Probe probe;
    outer.put(holder, {50.0f, 50.0f, 100.0f, 100.0f});
    holder.attach(probe);
    probe.prepare();

    outer.place(ctx, {0.0f, 0.0f, 400.0f, 400.0f});

    CHECK(probe.inherited() == Rect{50.0f, 50.0f, 100.0f, 100.0f});
    CHECK(probe.inherited() != probe.currentClip());
}

TEST_CASE("Detached components have no inherited clip", "[core][sakura][component]") {
    Probe lone;
    CHECK_FALSE(lone.inherited().has_value());

    lone.prepare();
    CHECK_FALSE(lone.inherited().has_value());

    Probe parent;
    Probe child;
    parent.adopt(child);
    parent.prepare();
    child.prepare();
    CHECK_FALSE(child.inherited().has_value());
}

TEST_CASE("Default layout re-lays visible children at their current frame", "[core][sakura][component]") {
    const auto ctx = MakeContext();
    Board board;
    Probe shown;
    Probe hidden;
    board.put(shown, {10.0f, 10.0f, 50.0f, 50.0f});
    board.put(hidden, {70.0f, 10.0f, 50.0f, 50.0f});
    board.place(ctx, {0.0f, 0.0f, 400.0f, 300.0f});
    REQUIRE(shown.layouts == 1);
    REQUIRE(hidden.layouts == 1);

    hidden.setVisible(false);
    board.defaultLayout = true;
    shown.resize(shown.desired);
    board.place(ctx, {0.0f, 0.0f, 400.0f, 300.0f});

    CHECK(shown.layouts == 2);
    CHECK(shown.laidFrame == Rect{10.0f, 10.0f, 50.0f, 50.0f});
    CHECK(hidden.layouts == 1);
}

TEST_CASE("Hidden children do not receive events", "[core][sakura][component]") {
    const auto ctx = MakeContext();
    Board board;
    Probe below;
    Probe above;
    board.put(below, {0.0f, 0.0f, 100.0f, 100.0f});
    board.put(above, {0.0f, 0.0f, 100.0f, 100.0f});
    board.place(ctx, {0.0f, 0.0f, 100.0f, 100.0f});

    above.setVisible(false);

    CHECK(board.mouseEventChildren(Mouse(MouseEventType::Click, 50.0f, 50.0f)));
    CHECK(above.events.empty());
    CHECK(below.count(MouseEventType::Click) == 1);
}

TEST_CASE("Mounting children orders the active ones and hides the rest where they were", "[core][sakura][component]") {
    Board board;
    Probe a;
    Probe b;
    Probe c;
    board.attach(a);
    board.attach(b);
    board.attach(c);

    board.mountChildren({&c, &a});

    CHECK(board.children() == std::vector<Sakura::Component*>{&c, &a, &b});
    CHECK(c.visible());
    CHECK(a.visible());
    CHECK_FALSE(b.visible());

    board.mountChildren({&b, &b});

    CHECK(board.children() == std::vector<Sakura::Component*>{&c, &a, &b});
    CHECK(b.visible());
    CHECK_FALSE(c.visible());
    CHECK_FALSE(a.visible());
}

TEST_CASE("Mounting an unknown child adopts it alongside the existing ones", "[core][sakura][component]") {
    Board board;
    Probe a;
    Probe b;
    Probe fresh;
    board.attach(a);
    board.attach(b);

    board.mountChildren({&fresh, &b});

    CHECK(board.children() == std::vector<Sakura::Component*>{&a, &fresh, &b});
    CHECK(fresh.visible());
    CHECK(b.visible());
    CHECK_FALSE(a.visible());
}

TEST_CASE("Setting children with a permutation reorders without detaching", "[core][sakura][component]") {
    Board board;
    Probe a;
    Probe b;
    board.attach(a);
    board.attach(b);

    board.setChildren({&b, &a});

    CHECK(board.children() == std::vector<Sakura::Component*>{&b, &a});
    CHECK(a.events.empty());
    CHECK(b.events.empty());

    Probe c;
    board.setChildren({&c});

    CHECK(board.children() == std::vector<Sakura::Component*>{&c});
    CHECK(a.count(MouseEventType::Cancel) == 1);
    CHECK(b.count(MouseEventType::Cancel) == 1);
}

TEST_CASE("Adding and removing siblings leaves the surviving children untouched", "[core][sakura][component]") {
    Board board;
    Probe a;
    Probe b;
    board.attach(a);
    board.attach(b);

    Probe c;
    board.setChildren({&a, &c, &b});

    CHECK(board.children() == std::vector<Sakura::Component*>{&a, &c, &b});
    CHECK(a.events.empty());
    CHECK(b.events.empty());

    board.setChildren({&c, &b, &c});

    CHECK(board.children() == std::vector<Sakura::Component*>{&c, &b});
    CHECK(a.count(MouseEventType::Cancel) == 1);
    CHECK(b.events.empty());
    CHECK(c.events.empty());
}

TEST_CASE("Events reach the topmost child first and stop when consumed", "[core][sakura][component]") {
    const auto ctx = MakeContext();
    Board board;
    Probe below;
    Probe above;
    board.put(below, {0.0f, 0.0f, 100.0f, 100.0f});
    board.put(above, {0.0f, 0.0f, 100.0f, 100.0f});
    board.place(ctx, {0.0f, 0.0f, 100.0f, 100.0f});

    CHECK(board.mouseEventChildren(Mouse(MouseEventType::Click, 50.0f, 50.0f)));
    CHECK(above.count(MouseEventType::Click) == 1);
    CHECK(below.events.empty());

    above.consume = false;
    CHECK(board.mouseEventChildren(Mouse(MouseEventType::Click, 50.0f, 50.0f)));
    CHECK(above.count(MouseEventType::Click) == 2);
    CHECK(below.count(MouseEventType::Click) == 1);

    below.consume = false;
    CHECK_FALSE(board.mouseEventChildren(Mouse(MouseEventType::Click, 50.0f, 50.0f)));
    CHECK(above.count(MouseEventType::Click) == 3);
    CHECK(below.count(MouseEventType::Click) == 2);
}

TEST_CASE("Click and scroll events outside a child clip are skipped", "[core][sakura][component]") {
    const auto ctx = MakeContext();
    Board board;
    Probe probe;
    probe.consume = false;
    board.put(probe, {0.0f, 0.0f, 100.0f, 100.0f});
    board.bound = Rect{0.0f, 0.0f, 50.0f, 50.0f};
    board.place(ctx, {0.0f, 0.0f, 200.0f, 200.0f});

    board.mouseEventChildren(Mouse(MouseEventType::Click, 75.0f, 25.0f));
    board.mouseEventChildren(Mouse(MouseEventType::Scroll, 75.0f, 25.0f));
    CHECK(probe.events.empty());

    board.mouseEventChildren(Mouse(MouseEventType::Move, 75.0f, 25.0f));
    board.mouseEventChildren(Mouse(MouseEventType::Leave, 75.0f, 25.0f));
    board.mouseEventChildren(Mouse(MouseEventType::Release, 75.0f, 25.0f));
    CHECK(probe.count(MouseEventType::Move) == 1);
    CHECK(probe.count(MouseEventType::Leave) == 1);
    CHECK(probe.count(MouseEventType::Release) == 1);

    board.mouseEventChildren(Mouse(MouseEventType::Click, 25.0f, 25.0f));
    board.mouseEventChildren(Mouse(MouseEventType::Scroll, 25.0f, 25.0f));
    CHECK(probe.count(MouseEventType::Click) == 1);
    CHECK(probe.count(MouseEventType::Scroll) == 1);
}

TEST_CASE("Cancel reaches every child including hidden ones", "[core][sakura][component]") {
    const auto ctx = MakeContext();
    Board board;
    Probe below;
    Probe above;
    Probe hidden;
    board.put(below, {0.0f, 0.0f, 100.0f, 100.0f});
    board.put(above, {0.0f, 0.0f, 100.0f, 100.0f});
    board.put(hidden, {0.0f, 0.0f, 100.0f, 100.0f});
    board.place(ctx, {0.0f, 0.0f, 100.0f, 100.0f});
    hidden.setVisible(false);

    CHECK_FALSE(board.mouseEventChildren(Mouse(MouseEventType::Cancel, 500.0f, 500.0f)));
    CHECK(below.count(MouseEventType::Cancel) == 1);
    CHECK(above.count(MouseEventType::Cancel) == 1);
    CHECK(hidden.count(MouseEventType::Cancel) == 1);
}

TEST_CASE("Reference snapshots catch a leaf that keeps a stale clip after the viewport shrinks",
          "[core][sakura][component]") {
    StaleClipProbe live;
    StaleClipProbe mirror;
    Place liveRoot(live, {10.0f, 10.0f, 300.0f, 200.0f});
    Place mirrorRoot(mirror, {10.0f, 10.0f, 300.0f, 200.0f});
    Frames liveFrames(liveRoot);
    Frames mirrorFrames(mirrorRoot);
    liveFrames.verify = false;
    mirrorFrames.verify = false;
    const auto step = [&] {
        liveFrames.run();
        Sakura::Access::invalidateSubtree(mirrorRoot);
        mirrorFrames.run();
    };
    step();
    REQUIRE(Drawn(liveRoot) == Drawn(mirrorRoot));

    liveFrames.bounds = {0.0f, 0.0f, 150.0f, 100.0f};
    mirrorFrames.bounds = liveFrames.bounds;
    step();
    const auto retained = Drawn(liveRoot);
    const auto expected = Drawn(mirrorRoot);
    REQUIRE(retained.size() == 1);
    REQUIRE(expected.size() == 1);
    CHECK(retained[0].config == expected[0].config);
    CHECK(retained[0].translation == expected[0].translation);
    CHECK(retained[0].clip != expected[0].clip);
    CHECK(retained != expected);
}

TEST_CASE("Parking a child through mountChildren cancels its press and releases focus inside it",
          "[core][sakura][component]") {
    Scene scene;
    Probe probe({100.0f, 20.0f});
    Sakura::TextInput input;
    Sakura::Stack stack;
    Host host(stack);
    bool parked = false;
    const auto update = [&] {
        input.update({.value = "abc", .focus = true});
        std::vector<Sakura::Stack::Item> items;
        if (!parked) {
            items = {{.child = &probe}, {.child = &input}};
        }
        stack.update({.items = std::move(items)});
    };
    update();
    host.place(scene.ctx, {0.0f, 0.0f, 300.0f, 200.0f});
    update();
    host.place(scene.ctx, {0.0f, 0.0f, 300.0f, 200.0f});
    REQUIRE(input.focused());
    host.send(Mouse(MouseEventType::Click, 10.0f, 10.0f));
    const U64 cancels = probe.count(MouseEventType::Cancel);

    parked = true;
    update();
    CHECK(probe.count(MouseEventType::Cancel) == cancels + 1);
    CHECK_FALSE(input.focused());
    CHECK_FALSE(probe.visible());

    update();
    CHECK(probe.count(MouseEventType::Cancel) == cancels + 1);
}
