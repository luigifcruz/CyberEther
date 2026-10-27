#include <catch2/catch_test_macros.hpp>

#include "harness.hh"
#include "fonts.hh"
#include "component_harness.hh"
#include "scene.hh"

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/button.hh>
#include <jetstream/render/sakura/components/chat.hh>
#include <jetstream/render/sakura/components/chat_dock.hh>
#include <jetstream/render/sakura/components/label.hh>
#include <jetstream/render/sakura/components/text_grid.hh>
#include <jetstream/render/sakura/typography.hh>

#include <cmath>
#include <functional>
#include <optional>
#include <string>
#include <vector>

using namespace Jetstream;
using namespace SakuraTest;

namespace {

constexpr const char* kTags = "[core][sakura][chat-dock]";
constexpr F32 kStep = 0.05f;
constexpr U64 kSettleFrames = 40;

template<typename T>
T& Part(Sakura::Component& component, U64 index) {
    return *dynamic_cast<T*>(Peek::Children(component).at(index));
}

bool Near(F32 value, F32 expected) {
    return std::abs(value - expected) < 1e-3f;
}

Extent2D<F32> Center(const Rect& rect) {
    return {rect.x + rect.width * 0.5f, rect.y + rect.height * 0.5f};
}

Sakura::Chat::Snapshot Conversation() {
    Sakura::Chat::Snapshot snapshot;
    snapshot.messages.push_back({.role = Sakura::Chat::Role::User, .text = "Tune to the FM band."});
    snapshot.messages.push_back({.role = Sakura::Chat::Role::Assistant, .text = "Tuned to 96.9 MHz."});
    return snapshot;
}

struct DockRig {
    Scene scene;
    Sakura::ChatDock dock;
    LayerHost host{std::vector<Sakura::Layer*>{&dock}};
    Frames frames{host, kScreen, scene.context()};
    Sakura::ChatDock::Config config;
    U64 opened = 0;
    U64 minimized = 0;
    U64 cleared = 0;

    DockRig() {
        frames.ctx.deltaTime = kStep;
        frames.timeStep = kStep;
        config.title = "Assistant";
        config.size = {440.0f, 400.0f};
        config.backgroundColorKey = "agent_panel";
        config.outlineColorKey = "agent_outline";
        config.alertColorKey = "agent_activity";
        config.onOpen = [this] { ++opened; };
        config.onMinimize = [this] { ++minimized; };
        config.chat.onClear = [this] { ++cleared; };
    }

    DockRig(const DockRig&) = delete;
    DockRig& operator=(const DockRig&) = delete;

    std::function<void()> updater() {
        return [this] { dock.update(config); };
    }

    bool settle() {
        return frames.settle(updater(), kSettleFrames);
    }

    const Sakura::Box& background() {
        return Part<Sakura::Box>(dock, 0);
    }

    Sakura::Chat& chat() {
        return Part<Sakura::Chat>(dock, 1);
    }

    const Sakura::Label& title() {
        return Part<Sakura::Label>(dock, 2);
    }

    Sakura::Button& clear() {
        return Part<Sakura::Button>(dock, 3);
    }

    Sakura::Button& minimize() {
        return Part<Sakura::Button>(dock, 4);
    }

    Rect card() {
        return background().config().instances.front().rect;
    }

    std::optional<SurfaceCursor> cursor(Extent2D<F32> point) {
        return Sakura::Access::cursor(host, point);
    }

    void drag(Extent2D<F32> from, Extent2D<F32> to) {
        Press(host, frames, from.x, from.y);
        Hover(host, frames, to.x, to.y);
        Release(host, frames, to.x, to.y);
        frames.run(updater());
    }

    void density(F32 ratio) {
        frames.ctx.pixelRatio = ratio;
        frames.ctx.framebufferSize = {static_cast<U64>(kScreen.width * ratio),
                                      static_cast<U64>(kScreen.height * ratio)};
        frames.bounds = {0.0f, 0.0f, kScreen.width * ratio, kScreen.height * ratio};
        Sakura::Access::invalidateLayout(host);
    }

    bool alerted() {
        const auto alert = DistinctPalette().at("agent_activity");
        for (auto* box : Find<Sakura::Box>(dock)) {
            const auto& color = box->config().borderColor;
            if (box->config().borderWidth > 0.0f && color.r == alert.r && color.g == alert.g && color.b == alert.b) {
                return true;
            }
        }
        return false;
    }
};

}  // namespace

TEST_CASE("Chat docks rest as a compact bar in the bottom right corner and settle there", kTags) {
    DockRig rig;
    const auto update = rig.updater();
    REQUIRE(rig.settle());
    CHECK(Retains(rig.frames, update));

    const Rect bar = rig.card();
    CHECK(bar.right() == 788.0f);
    CHECK(Near(bar.bottom(), 588.0f));
    CHECK(bar.width == 240.0f);
    CHECK(bar.height > 0.0f);
    CHECK(bar.height < 400.0f);
    CHECK(Peek::Frame(rig.chat()) == bar);
    CHECK_FALSE(rig.background().config().instances.front().visible);
    CHECK_FALSE(rig.title().config().instances.front().visible);
    CHECK_FALSE(rig.clear().visible());
    CHECK_FALSE(rig.minimize().visible());

    CHECK(rig.dock.covers({bar.x + 10.0f, bar.y + 10.0f}));
    CHECK_FALSE(rig.dock.covers({20.0f, 20.0f}));
    CHECK_FALSE(rig.cursor({bar.x + 1.0f, bar.y + 1.0f}) == SurfaceCursor::ResizeNWSE);
    CHECK(QuietMoves(rig.frames, update, {{20.0f, 20.0f}, {30.0f, 40.0f}}, [&](const MouseEvent& event) {
        return rig.host.send(event);
    }));
}

TEST_CASE("Opened chat docks expand into a card with a header and settle there", kTags) {
    DockRig rig;
    const auto update = rig.updater();
    rig.config.open = true;
    REQUIRE(rig.settle());
    CHECK(Retains(rig.frames, update));

    CHECK(rig.card() == Rect{348.0f, 188.0f, 440.0f, 400.0f});
    CHECK(Peek::Frame(rig.chat()) == Rect{348.0f, 218.0f, 440.0f, 370.0f});
    const auto& background = rig.background().config();
    CHECK(background.instances.front().visible);
    CHECK(background.instances.front().backgroundColor == DistinctPalette().at("agent_panel"));
    CHECK(background.cornerRadius == 20.0f);
    CHECK(background.borderWidth == 1.0f);

    const auto& title = rig.title().config();
    CHECK(title.fontName == "default_body_bold");
    CHECK(title.clip == rig.card());
    REQUIRE(title.instances.size() == 1);
    CHECK(title.instances.front().visible);
    CHECK(title.instances.front().str == "Assistant");
    CHECK(title.instances.front().fontSize == Sakura::Typography::FontSize * 1.1f);
    CHECK(title.instances.front().color == DistinctPalette().at("text_secondary"));
    CHECK(title.instances.front().rect == Rect{364.0f, 188.0f, 408.0f, 40.0f});
    CHECK(title.instances.front().alignment == Extent2D<I32>{0, 1});

    CHECK(rig.minimize().visible());
    CHECK(rig.clear().visible());
    CHECK(Peek::Frame(rig.minimize()) == Rect{756.0f, 196.0f, 24.0f, 24.0f});
    CHECK(Peek::Frame(rig.clear()) == Rect{730.0f, 196.0f, 24.0f, 24.0f});
    CHECK(QuietMoves(rig.frames, update, {{20.0f, 20.0f}, {500.0f, 200.0f}, {520.0f, 205.0f}},
                     [&](const MouseEvent& event) {
                         return rig.host.send(event);
                     }));
}

TEST_CASE("Chat docks repaint only while they animate and hold still while the clock is frozen", kTags) {
    DockRig rig;
    const auto update = rig.updater();
    REQUIRE(rig.settle());
    rig.config.open = true;
    REQUIRE(rig.settle());
    rig.config.open = false;
    REQUIRE(rig.settle());
    CHECK(rig.frames.rebuilds == 1);
    const U64 restructures = rig.frames.restructures;

    for (const bool open : {true, false, true}) {
        rig.frames.timeStep = 0.0;
        rig.config.open = open;
        CHECK(RepaintOnly(rig.frames.run(update)));
        CHECK(Retains(rig.frames, update));

        rig.frames.timeStep = kStep;
        U64 animated = 0;
        while (animated < kSettleFrames) {
            const auto stats = rig.frames.run(update);
            if (stats.clean()) {
                break;
            }
            CHECK(Kept(stats));
            ++animated;
        }
        CHECK(animated > 2);
        CHECK(animated < kSettleFrames);
        CHECK(Retains(rig.frames, update));
    }
    CHECK(rig.frames.rebuilds == 1);
    CHECK(rig.frames.restructures == restructures);
    CHECK(rig.card() == Rect{348.0f, 188.0f, 440.0f, 400.0f});
    CHECK(rig.minimize().visible());
}

TEST_CASE("Closed chat docks widen while their composer has focus and shrink when it leaves", kTags) {
    DockRig rig;
    const auto update = rig.updater();
    REQUIRE(rig.settle());
    CHECK(rig.card().width == 240.0f);

    const auto inside = Center(rig.card());
    rig.host.click(inside.x, inside.y);
    ClickFocus(rig.host, inside);
    REQUIRE(dynamic_cast<const Sakura::TextGrid*>(FocusOwner(rig.host)) != nullptr);
    rig.frames.settle(update, kSettleFrames);
    CHECK(rig.card().width == 440.0f);
    CHECK(rig.card().right() == 788.0f);
    CHECK(Near(rig.card().bottom(), 588.0f));

    Sakura::Access::releaseFocusWithin(rig.dock);
    CHECK(FocusOwner(rig.host) == nullptr);
    REQUIRE(rig.settle());
    CHECK(Retains(rig.frames, update));
    CHECK(rig.card().width == 240.0f);

    rig.config.open = true;
    REQUIRE(rig.settle());
    CHECK(rig.card() == Rect{348.0f, 188.0f, 440.0f, 400.0f});
}

TEST_CASE("Chat docks stay below the top offset and within the margins", kTags) {
    DockRig rig;
    const auto update = rig.updater();
    rig.config.open = true;
    rig.config.size = {440.0f, 680.0f};
    rig.config.topOffset = 40.0f;
    REQUIRE(rig.settle());
    CHECK(Retains(rig.frames, update));
    CHECK(rig.card() == Rect{348.0f, 52.0f, 440.0f, 536.0f});

    rig.config.size = {900.0f, 200.0f};
    REQUIRE(rig.settle());
    CHECK(rig.card() == Rect{12.0f, 268.0f, 776.0f, 320.0f});
}

TEST_CASE("Chat docks resize from the left edge, the top edge, and the top left corner", kTags) {
    DockRig rig;
    const auto update = rig.updater();
    rig.config.open = true;
    REQUIRE(rig.settle());
    REQUIRE(rig.card() == Rect{348.0f, 188.0f, 440.0f, 400.0f});

    CHECK(rig.cursor({350.0f, 300.0f}) == SurfaceCursor::ResizeEW);
    CHECK(rig.cursor({500.0f, 190.0f}) == SurfaceCursor::ResizeNS);
    CHECK(rig.cursor({350.0f, 190.0f}) == SurfaceCursor::ResizeNWSE);
    CHECK_FALSE(rig.cursor({20.0f, 20.0f}).has_value());

    rig.drag({350.0f, 300.0f}, {300.0f, 300.0f});
    CHECK(rig.card() == Rect{298.0f, 188.0f, 490.0f, 400.0f});
    rig.drag({500.0f, 190.0f}, {500.0f, 140.0f});
    CHECK(rig.card() == Rect{298.0f, 138.0f, 490.0f, 450.0f});
    rig.drag({300.0f, 140.0f}, {250.0f, 100.0f});
    CHECK(rig.card() == Rect{248.0f, 98.0f, 540.0f, 490.0f});
    CHECK(Peek::Frame(rig.chat()) == Rect{248.0f, 128.0f, 540.0f, 460.0f});
    CHECK(rig.frames.rebuilds == 1);
    CHECK(Retains(rig.frames, update));

    rig.drag({250.0f, 300.0f}, {700.0f, 300.0f});
    CHECK(rig.card().width == 360.0f);
    rig.drag({430.0f, 300.0f}, {-200.0f, 300.0f});
    CHECK(rig.card().width == 776.0f);
    CHECK(rig.card().x == 12.0f);
}

TEST_CASE("Chat docks cover the whole layer during a resize drag and Cancel ends it", kTags) {
    DockRig rig;
    const auto update = rig.updater();
    rig.config.open = true;
    REQUIRE(rig.settle());

    Press(rig.host, rig.frames, 350.0f, 300.0f);
    CHECK(rig.dock.covers({20.0f, 20.0f}));
    CHECK(rig.cursor({20.0f, 20.0f}) == SurfaceCursor::ResizeEW);
    Hover(rig.host, rig.frames, 330.0f, 300.0f);
    CHECK(RepaintOnly(rig.frames.run(update)));
    CHECK(rig.card().width == 460.0f);

    rig.host.send(Mouse(MouseEventType::Cancel, 330.0f, 300.0f));
    CHECK_FALSE(rig.dock.covers({20.0f, 20.0f}));
    Hover(rig.host, rig.frames, 200.0f, 300.0f);
    Release(rig.host, rig.frames, 200.0f, 300.0f);
    CHECK(rig.frames.run(update).clean());
    CHECK(rig.card().width == 460.0f);
}

TEST_CASE("Chat docks only resize while open and fully expanded", kTags) {
    DockRig rig;
    const auto update = rig.updater();
    REQUIRE(rig.settle());
    const Rect bar = rig.card();
    CHECK_FALSE(rig.cursor({bar.x + 1.0f, bar.y + 1.0f}) == SurfaceCursor::ResizeNWSE);
    rig.drag({bar.x + 1.0f, bar.y + 1.0f}, {bar.x - 100.0f, bar.y - 100.0f});
    CHECK(rig.card() == bar);

    rig.config.open = true;
    rig.frames.run(update);
    CHECK_FALSE(rig.cursor({rig.card().x + 1.0f, rig.card().y + 1.0f}) == SurfaceCursor::ResizeNWSE);
    rig.drag({rig.card().x + 1.0f, rig.card().y + 1.0f}, {100.0f, 100.0f});
    REQUIRE(rig.settle());
    CHECK(rig.card() == Rect{348.0f, 188.0f, 440.0f, 400.0f});
}

TEST_CASE("Chat dock sizes are logical and follow the pixel ratio", kTags) {
    DockRig rig;
    const auto update = rig.updater();
    rig.config.open = true;
    REQUIRE(rig.settle());
    rig.drag({350.0f, 300.0f}, {290.0f, 300.0f});
    REQUIRE(rig.card() == Rect{288.0f, 188.0f, 500.0f, 400.0f});

    rig.density(2.0f);
    REQUIRE(rig.settle());
    CHECK(Retains(rig.frames, update));
    CHECK(rig.card() == Rect{576.0f, 376.0f, 1000.0f, 800.0f});
    CHECK(Peek::Frame(rig.chat()) == Rect{576.0f, 436.0f, 1000.0f, 740.0f});
    CHECK(rig.background().config().cornerRadius == 40.0f);
    CHECK(rig.background().config().borderWidth == 2.0f);
    CHECK(rig.title().config().instances.front().fontSize == 2.0f * (Sakura::Typography::FontSize * 1.1f));
    CHECK(rig.title().config().instances.front().rect == Rect{608.0f, 376.0f, 936.0f, 80.0f});
    CHECK(Peek::Frame(rig.minimize()) == Rect{1512.0f, 392.0f, 48.0f, 48.0f});
    CHECK(Peek::Frame(rig.clear()) == Rect{1460.0f, 392.0f, 48.0f, 48.0f});
    CHECK(rig.cursor({586.0f, 600.0f}) == SurfaceCursor::ResizeEW);
    CHECK_FALSE(rig.cursor({590.0f, 600.0f}) == SurfaceCursor::ResizeEW);

    rig.drag({578.0f, 600.0f}, {478.0f, 600.0f});
    CHECK(rig.card() == Rect{476.0f, 376.0f, 1100.0f, 800.0f});

    rig.density(1.0f);
    REQUIRE(rig.settle());
    CHECK(rig.card() == Rect{238.0f, 188.0f, 550.0f, 400.0f});
}

TEST_CASE("Clicking the closed bar opens a conversation but not an empty dock", kTags) {
    DockRig rig;
    REQUIRE(rig.settle());
    const Rect bar = rig.card();
    rig.host.click(bar.x + 12.0f, bar.y + bar.height * 0.5f);
    CHECK(rig.opened == 0);

    rig.config.chat.snapshot = Conversation();
    REQUIRE(rig.settle());
    const Rect talking = rig.card();
    rig.host.click(talking.x + 12.0f, talking.y + talking.height * 0.5f);
    CHECK(rig.opened == 1);
    rig.host.click(20.0f, 20.0f);
    CHECK(rig.opened == 1);

    rig.config.open = true;
    REQUIRE(rig.settle());
    rig.host.click(Center(rig.card()).x, Center(rig.card()).y);
    CHECK(rig.opened == 1);
}

TEST_CASE("Chat dock header buttons minimize and clear on release", kTags) {
    DockRig rig;
    const auto update = rig.updater();
    rig.config.open = true;
    rig.config.chat.snapshot = Conversation();
    REQUIRE(rig.settle());

    const auto minimize = Center(Peek::Frame(rig.minimize()));
    const auto clear = Center(Peek::Frame(rig.clear()));
    CHECK(rig.cursor(minimize) == SurfaceCursor::Hand);
    CHECK(rig.cursor(clear) == SurfaceCursor::Hand);

    Press(rig.host, rig.frames, minimize.x, minimize.y);
    CHECK(rig.minimized == 0);
    Release(rig.host, rig.frames, 20.0f, 20.0f);
    CHECK(rig.minimized == 0);

    Press(rig.host, rig.frames, minimize.x, minimize.y);
    rig.host.send(Mouse(MouseEventType::Cancel, minimize.x, minimize.y));
    Release(rig.host, rig.frames, minimize.x, minimize.y);
    CHECK(rig.minimized == 0);

    rig.host.click(clear.x, clear.y);
    CHECK(rig.cleared == 1);
    CHECK(rig.minimized == 0);
    rig.host.click(minimize.x, minimize.y);
    CHECK(rig.minimized == 1);
    CHECK(rig.cleared == 1);

    rig.config.open = false;
    REQUIRE(rig.frames.settle(update, kSettleFrames));
    rig.host.click(minimize.x, minimize.y);
    rig.host.click(clear.x, clear.y);
    CHECK(rig.minimized == 1);
    CHECK(rig.cleared == 1);
}

TEST_CASE("Chat dock header buttons repaint once per hover transition", kTags) {
    DockRig rig;
    const auto update = rig.updater();
    rig.config.open = true;
    REQUIRE(rig.settle());

    const auto minimize = Center(Peek::Frame(rig.minimize()));
    CheckHoverRepaintsOnce(rig.frames, rig.host, update, minimize, {20.0f, 20.0f});
    CheckLeaveRepaintsOnce(rig.frames, rig.host, update, minimize);
    const auto clear = Center(Peek::Frame(rig.clear()));
    CheckHoverRepaintsOnce(rig.frames, rig.host, update, clear, {500.0f, 200.0f});
}

TEST_CASE("Hiding the header while a button is pressed cancels the press", kTags) {
    DockRig rig;
    const auto update = rig.updater();
    rig.config.open = true;
    REQUIRE(rig.settle());
    const auto minimize = Center(Peek::Frame(rig.minimize()));

    Press(rig.host, rig.frames, minimize.x, minimize.y);
    rig.config.open = false;
    REQUIRE(rig.frames.settle(update, kSettleFrames));
    rig.config.open = true;
    REQUIRE(rig.frames.settle(update, kSettleFrames));
    Release(rig.host, rig.frames, minimize.x, minimize.y);
    CHECK(rig.minimized == 0);
}

TEST_CASE("A reply that finishes while the dock is closed outlines the bar until the dock opens", kTags) {
    DockRig rig;
    const auto update = rig.updater();
    rig.config.chat.snapshot = Conversation();
    REQUIRE(rig.settle());
    CHECK_FALSE(rig.alerted());

    rig.config.chat.snapshot.streaming = true;
    rig.frames.run(update);
    rig.config.chat.snapshot.streaming = false;
    CHECK(RepaintOnly(rig.frames.run(update)));
    REQUIRE(rig.frames.settle(update, kSettleFrames));
    CHECK(rig.alerted());
    CHECK(Retains(rig.frames, update));

    rig.config.open = true;
    REQUIRE(rig.settle());
    CHECK_FALSE(rig.alerted());
    rig.config.open = false;
    REQUIRE(rig.settle());
    CHECK_FALSE(rig.alerted());

    rig.config.open = true;
    rig.config.chat.snapshot.streaming = true;
    rig.frames.run(update);
    rig.config.chat.snapshot.streaming = false;
    REQUIRE(rig.settle());
    rig.config.open = false;
    REQUIRE(rig.settle());
    CHECK_FALSE(rig.alerted());
}

TEST_CASE("Minimizing the dock releases keyboard focus held inside it", kTags) {
    DockRig rig;
    rig.config.open = true;
    REQUIRE(rig.settle());

    REQUIRE(Sakura::Access::focus(rig.clear()));
    REQUIRE(FocusOwner(rig.host) == &rig.clear());
    rig.dock.update(rig.config);
    CHECK(FocusOwner(rig.host) == &rig.clear());

    const auto minimize = Center(Peek::Frame(rig.minimize()));
    rig.host.click(minimize.x, minimize.y);
    CHECK(rig.minimized == 1);
    CHECK(FocusOwner(rig.host) == nullptr);
}
