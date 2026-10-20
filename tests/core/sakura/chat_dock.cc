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
#include <jetstream/render/sakura/components/text.hh>
#include <jetstream/render/sakura/components/text_grid.hh>
#include <jetstream/render/sakura/components/voice.hh>
#include <jetstream/render/sakura/typography.hh>
#include <jetstream/render/tools/icons_fa.hh>

#include "render/sakura/detail/orb.hh"

#include <cmath>
#include <functional>
#include <optional>
#include <string>
#include <vector>

using namespace Jetstream;
using namespace SakuraTest;

namespace {

constexpr const char* kTags = "[core][sakura][chat-dock]";
constexpr const char* kVoiceTags = "[core][sakura][voice]";
constexpr const char* kOrbTags = "[core][sakura][orb]";
constexpr F32 kStep = 0.05f;
constexpr U64 kSettleFrames = 40;
constexpr U64 kBlendFrames = 120;

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

struct VoiceRig {
    Scene scene;
    Sakura::Voice voice;
    Place place;
    Frames frames;
    Sakura::Voice::Config config;
    U64 started = 0;
    U64 dismissed = 0;

    explicit VoiceRig(Rect target = {10.0f, 10.0f, 175.0f, 175.0f}) :
        place(voice, target),
        frames(place, kScreen, scene.context()) {
        frames.ctx.deltaTime = kStep;
        frames.timeStep = kStep;
        config.onStart = [this] { ++started; };
        config.onDismiss = [this] { ++dismissed; };
    }

    VoiceRig(const VoiceRig&) = delete;
    VoiceRig& operator=(const VoiceRig&) = delete;

    std::function<void()> updater() {
        return [this] { voice.update(config); };
    }

    void advance(U64 count) {
        for (U64 frame = 0; frame < count; ++frame) {
            frames.run(updater());
        }
    }

    const Sakura::Orb& orb() {
        return Part<Sakura::Orb>(voice, 0);
    }

    const Sakura::Box& badge() {
        return Part<Sakura::Box>(voice, 1);
    }

    const Sakura::Text& glyph() {
        return Part<Sakura::Text>(voice, 2);
    }

    Rect badgeRect() {
        return badge().config().instances.front().rect;
    }

    bool send(MouseEventType type, Extent2D<F32> point) {
        return place.send(Mouse(type, point.x, point.y));
    }

    bool click(Extent2D<F32> point) {
        const bool pressed = send(MouseEventType::Click, point);
        const bool released = send(MouseEventType::Release, point);
        return pressed || released;
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

TEST_CASE("Voice orbs measure their fixed, natural, and filled sizes", kVoiceTags) {
    Sakura::Voice voice;
    Host host(voice);
    auto ctx = MakeContext();

    voice.update({});
    CHECK(host.measureOf(ctx, {kInfinite, kInfinite}) == Extent2D<F32>{175.0f, 175.0f});
    CHECK(host.measureOf(ctx, {100.0f, 80.0f}) == Extent2D<F32>{100.0f, 80.0f});

    voice.update({.size = {0.0f, 0.0f}});
    CHECK(host.measureOf(ctx, {kInfinite, kInfinite}) == Extent2D<F32>{175.0f, 175.0f});
    CHECK(host.measureOf(ctx, {120.0f, 300.0f}) == Extent2D<F32>{120.0f, 175.0f});

    voice.update({.size = {120.0f, 90.0f}});
    CHECK(host.measureOf(ctx, {kInfinite, kInfinite}) == Extent2D<F32>{120.0f, 90.0f});
    CHECK(host.measureOf(ctx, {100.0f, 50.0f}) == Extent2D<F32>{100.0f, 50.0f});

    voice.update({.size = {Sakura::Fill, Sakura::Fill}});
    CHECK(host.measureOf(ctx, {300.0f, 200.0f}) == Extent2D<F32>{300.0f, 200.0f});
    CHECK(host.measureOf(ctx, {kInfinite, kInfinite}) == Extent2D<F32>{175.0f, 175.0f});

    auto dense = ctx;
    dense.pixelRatio = 2.0f;
    voice.update({.size = {120.0f, 90.0f}});
    CHECK(host.measureOf(dense, {kInfinite, kInfinite}) == Extent2D<F32>{240.0f, 180.0f});
    voice.update({.size = {0.0f, 0.0f}});
    CHECK(host.measureOf(dense, {kInfinite, kInfinite}) == Extent2D<F32>{350.0f, 350.0f});
}

TEST_CASE("Voice orbs center the orb and park the dismiss badge on its top right diagonal", kVoiceTags) {
    VoiceRig rig;
    rig.frames.run(rig.updater());

    const auto& orb = rig.orb().config();
    CHECK(orb.rect == Rect{10.0f, 10.0f, 175.0f, 175.0f});
    CHECK(orb.clip == Rect{10.0f, 10.0f, 175.0f, 175.0f});
    CHECK(orb.center == Extent2D<F32>{97.5f, 97.5f});
    CHECK(orb.radius == 175.0f * 0.34f);
    CHECK(orb.listeningColor == DistinctPalette().at("voice_orb_listening"));
    CHECK(orb.busyColor == DistinctPalette().at("voice_orb_busy"));
    CHECK(orb.listeningAccent == DistinctPalette().at("voice_orb_listening_accent"));
    CHECK(orb.busyAccent == DistinctPalette().at("voice_orb_busy_accent"));

    CHECK(rig.badgeRect() == Rect{129.0f, 44.0f, 22.0f, 22.0f});
    CHECK(rig.badge().config().cornerRadius == 11.0f);
    CHECK(rig.badge().config().borderWidth == 1.0f);
    CHECK_FALSE(rig.badge().config().instances.front().visible);
    CHECK(rig.glyph().config().str == ICON_FA_XMARK);
    CHECK(rig.glyph().config().opacity == 0.0f);
}

TEST_CASE("Voice orbs animate on every frame while visible and stop when hidden or frozen", kVoiceTags) {
    VoiceRig rig;
    const auto update = rig.updater();
    REQUIRE(rig.frames.run(update).rebuilt);
    rig.frames.run(update);
    const U64 restructures = rig.frames.restructures;

    F32 time = rig.orb().config().time;
    for (U64 frame = 0; frame < 5; ++frame) {
        const auto stats = rig.frames.run(update);
        CHECK(RepaintOnly(stats));
        CHECK(rig.orb().config().time > time);
        time = rig.orb().config().time;
    }

    rig.frames.timeStep = 0.0;
    rig.frames.run(update);
    CHECK(Retains(rig.frames, update));
    CHECK(QuietMoves(rig.frames, update, {{300.0f, 300.0f}, {310.0f, 300.0f}}, [&](const MouseEvent& event) {
        return rig.place.send(event);
    }));

    rig.frames.timeStep = kStep;
    rig.voice.setVisible(false);
    REQUIRE(rig.frames.settle(update));
    CHECK(Retains(rig.frames, update));
    CHECK(rig.frames.rebuilds == 1);
    CHECK(rig.frames.restructures == restructures);

    rig.voice.setVisible(true);
    CHECK(RepaintOnly(rig.frames.run(update)));
    CHECK(RepaintOnly(rig.frames.run(update)));
    CHECK(rig.orb().config().time > time);
}

TEST_CASE("Voice orbs ease toward the busy, activity, and level of their phase", kVoiceTags) {
    VoiceRig rig;
    rig.advance(2);
    CHECK(rig.orb().config().busy == 0.0f);
    CHECK(rig.orb().config().activity == 0.0f);

    rig.config.phase = Sakura::Voice::Phase::Busy;
    rig.config.level = 0.8f;
    rig.frames.run(rig.updater());
    CHECK(rig.orb().config().busy > 0.0f);
    CHECK(rig.orb().config().busy < 1.0f);
    rig.advance(kBlendFrames);
    CHECK(rig.orb().config().busy == 1.0f);
    CHECK(rig.orb().config().activity == 1.0f);
    CHECK(rig.orb().config().level == 0.8f);

    rig.config.phase = Sakura::Voice::Phase::Connecting;
    rig.advance(kBlendFrames);
    CHECK(rig.orb().config().busy == 0.0f);
    CHECK(rig.orb().config().activity == 0.35f);
    CHECK(rig.orb().config().level == 0.8f);

    rig.config.phase = Sakura::Voice::Phase::Listening;
    rig.config.level = 0.0f;
    rig.advance(kBlendFrames);
    CHECK(rig.orb().config().activity == 1.0f);
    CHECK(rig.orb().config().level == 0.0f);

    rig.config.level = 1.0f;
    rig.frames.run(rig.updater());
    const F32 rise = rig.orb().config().level;
    rig.advance(kBlendFrames);
    rig.config.level = 0.0f;
    rig.frames.run(rig.updater());
    const F32 fall = 1.0f - rig.orb().config().level;
    CHECK(rise > fall);
    CHECK(fall > 0.0f);

    rig.config.phase = Sakura::Voice::Phase::Idle;
    rig.config.level = 1.0f;
    rig.advance(kBlendFrames);
    CHECK(rig.orb().config().activity == 0.0f);
    CHECK(rig.orb().config().level == 0.0f);
}

TEST_CASE("Clicking an idle voice orb starts a session on release over the orb", kVoiceTags) {
    VoiceRig rig;
    rig.advance(2);
    const Extent2D<F32> center = {97.5f, 97.5f};

    CHECK(Peek::CursorAt(rig.voice, center) == SurfaceCursor::Hand);
    CHECK_FALSE(Peek::CursorAt(rig.voice, {15.0f, 15.0f}).has_value());
    CHECK_FALSE(Peek::CursorAt(rig.voice, {300.0f, 300.0f}).has_value());

    CHECK(rig.send(MouseEventType::Click, center));
    CHECK_FALSE(rig.send(MouseEventType::Release, {15.0f, 15.0f}));
    CHECK(rig.started == 0);

    CHECK(rig.send(MouseEventType::Click, center));
    rig.send(MouseEventType::Cancel, center);
    rig.send(MouseEventType::Release, center);
    CHECK(rig.started == 0);

    CHECK_FALSE(rig.send(MouseEventType::Click, {15.0f, 15.0f}));
    rig.send(MouseEventType::Release, center);
    CHECK(rig.started == 0);

    CHECK(rig.click(center));
    CHECK(rig.started == 1);

    rig.config.phase = Sakura::Voice::Phase::Listening;
    rig.advance(1);
    CHECK_FALSE(Peek::CursorAt(rig.voice, center).has_value());
    CHECK_FALSE(rig.click(center));
    CHECK(rig.started == 1);

    rig.config.phase = Sakura::Voice::Phase::Idle;
    rig.config.onStart = {};
    rig.advance(1);
    CHECK_FALSE(Peek::CursorAt(rig.voice, center).has_value());
    CHECK_FALSE(rig.click(center));
}

TEST_CASE("Hovering a voice orb fades in a dismiss badge that dismisses on release", kVoiceTags) {
    VoiceRig rig;
    rig.advance(2);
    const Extent2D<F32> badge = Center(rig.badgeRect());
    CHECK(badge == Extent2D<F32>{140.0f, 55.0f});

    CHECK_FALSE(Peek::CursorAt(rig.voice, badge).has_value());
    CHECK_FALSE(rig.click(badge));
    CHECK(rig.dismissed == 0);

    rig.send(MouseEventType::Move, {97.5f, 97.5f});
    rig.frames.run(rig.updater());
    CHECK(rig.badge().config().instances.front().visible);
    CHECK(rig.glyph().config().opacity > 0.0f);
    CHECK(rig.glyph().config().opacity < 1.0f);
    rig.advance(kSettleFrames);
    CHECK(rig.glyph().config().opacity == 1.0f);
    CHECK(rig.badge().config().instances.front().backgroundColor == DistinctPalette().at("card"));
    CHECK(rig.badge().config().borderColor == DistinctPalette().at("border"));

    CHECK(Peek::CursorAt(rig.voice, badge) == SurfaceCursor::Hand);
    rig.send(MouseEventType::Move, badge);
    rig.frames.run(rig.updater());
    CHECK(rig.badge().config().instances.front().backgroundColor == DistinctPalette().at("button_hovered"));

    CHECK(rig.send(MouseEventType::Click, badge));
    rig.frames.run(rig.updater());
    CHECK(rig.badge().config().instances.front().backgroundColor == DistinctPalette().at("button_active"));
    CHECK(rig.send(MouseEventType::Release, {300.0f, 300.0f}));
    CHECK(rig.dismissed == 0);

    CHECK(rig.send(MouseEventType::Click, badge));
    rig.send(MouseEventType::Cancel, badge);
    rig.send(MouseEventType::Release, badge);
    CHECK(rig.dismissed == 0);

    rig.send(MouseEventType::Move, badge);
    CHECK(rig.click(badge));
    CHECK(rig.dismissed == 1);
    CHECK(rig.started == 0);

    rig.send(MouseEventType::Leave, badge);
    rig.advance(kSettleFrames);
    CHECK_FALSE(rig.badge().config().instances.front().visible);
    CHECK(rig.glyph().config().opacity == 0.0f);
    CHECK_FALSE(rig.click(badge));
    CHECK(rig.dismissed == 1);
}

TEST_CASE("Voice hover, leave, and the dismiss badge each repaint once while the clock is frozen", kVoiceTags) {
    VoiceRig rig;
    const auto update = rig.updater();
    const Extent2D<F32> center = {97.5f, 97.5f};
    rig.advance(2);
    rig.frames.timeStep = 0.0;
    REQUIRE(rig.frames.settle(update));

    CheckHoverRepaintsOnce(rig.frames, rig.place, update, center, {300.0f, 300.0f});
    CheckLeaveRepaintsOnce(rig.frames, rig.place, update, center);

    rig.frames.timeStep = kStep;
    rig.send(MouseEventType::Move, center);
    rig.advance(kSettleFrames);
    REQUIRE(rig.badge().config().instances.front().visible);
    rig.frames.timeStep = 0.0;
    REQUIRE(rig.frames.settle(update));
    CheckHoverRepaintsOnce(rig.frames, rig.place, update, Center(rig.badgeRect()), center);
}

TEST_CASE("Voice orbs scale their badge and orb with the pixel ratio", kVoiceTags) {
    VoiceRig rig({20.0f, 20.0f, 350.0f, 350.0f});
    rig.frames.ctx.pixelRatio = 2.0f;
    rig.frames.ctx.framebufferSize = {1600, 1200};
    rig.frames.bounds = {0.0f, 0.0f, 1600.0f, 1200.0f};
    rig.advance(2);

    CHECK(rig.orb().config().radius == 350.0f * 0.34f);
    CHECK(rig.orb().config().center == Extent2D<F32>{195.0f, 195.0f});
    const Rect badge = rig.badgeRect();
    CHECK(badge.width == 44.0f);
    CHECK(badge.height == 44.0f);
    CHECK(rig.badge().config().cornerRadius == 22.0f);
    CHECK(rig.badge().config().borderWidth == 2.0f);

    rig.send(MouseEventType::Move, {195.0f, 195.0f});
    rig.advance(kSettleFrames);
    const Rect glyph = Peek::Frame(rig.glyph());
    CHECK(glyph.x >= badge.x);
    CHECK(glyph.right() <= badge.right());
    CHECK(std::abs(Center(glyph).x - Center(badge).x) <= 1.0f);
    CHECK(std::abs(Center(glyph).y - Center(badge).y) <= 1.0f);
    CHECK(rig.click(Center(badge)));
    CHECK(rig.dismissed == 1);
}

TEST_CASE("Orb leaves settle and repaint for every field without rebuilding", kOrbTags) {
    Sakura::Orb orb;
    Host host(orb);
    Frames frames(host);
    Sakura::Orb::Config config = {
        .rect = {10.0f, 10.0f, 100.0f, 100.0f},
        .center = {60.0f, 60.0f},
        .radius = 34.0f,
        .listeningColor = {0.1f, 0.4f, 1.0f, 1.0f},
    };
    const auto update = [&] {
        orb.update(config);
    };
    REQUIRE(frames.run(update).rebuilt);
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));

    const std::vector<std::function<void()>> edits = {
        [&] { config.rect = {20.0f, 10.0f, 100.0f, 100.0f}; },
        [&] { config.center = {70.0f, 60.0f}; },
        [&] { config.radius = 30.0f; },
        [&] { config.time = 1.5f; },
        [&] { config.busy = 0.5f; },
        [&] { config.activity = 0.25f; },
        [&] { config.level = 0.75f; },
        [&] { config.listeningColor = {0.2f, 0.4f, 1.0f, 1.0f}; },
        [&] { config.busyColor = {0.9f, 0.3f, 0.3f, 1.0f}; },
        [&] { config.listeningAccent = {0.1f, 1.0f, 1.0f, 1.0f}; },
        [&] { config.busyAccent = {1.0f, 0.8f, 0.4f, 1.0f}; },
        [&] { config.clip = Rect{0.0f, 0.0f, 50.0f, 50.0f}; },
    };
    for (const auto& edit : edits) {
        edit();
        CHECK(RepaintOnly(frames.run(update)));
        CHECK(Retains(frames, update));
    }
    CHECK(frames.rebuilds == 1);
    CHECK(frames.restructures == 0);
}

TEST_CASE("Hiding and showing orb leaves repaints without rebuilding", kOrbTags) {
    Sakura::Orb orb;
    Host host(orb);
    Frames frames(host);
    const auto update = [&] {
        orb.update({.rect = {10.0f, 10.0f, 100.0f, 100.0f}, .center = {60.0f, 60.0f}, .radius = 34.0f});
    };
    REQUIRE(frames.settle(update));

    orb.setVisible(false);
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    orb.setVisible(true);
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    CHECK(frames.rebuilds == 1);
    CHECK(frames.restructures == 0);
}
