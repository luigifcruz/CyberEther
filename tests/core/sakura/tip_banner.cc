#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "fonts.hh"
#include "component_harness.hh"

#include <jetstream/render/sakura/components/text.hh>
#include <jetstream/render/sakura/components/tip_banner.hh>
#include <jetstream/render/tools/icons_lucide.hh>

#include "harness.hh"
#include "scene.hh"

#include <string>

using namespace Jetstream;
using namespace SakuraTest;

namespace {

constexpr Rect kSlot = {100.0f, 20.0f, 360.0f, 40.0f};

struct Rig {
    Scene scene;
    Sakura::TipBanner banner;
    Place place{banner, kSlot};
    Frames frames{place, kScreen, scene.context()};
    Sakura::TipBanner::Config config = {
        .str = "Drag blocks to move them.",
        .icon = ICON_LC_LIGHTBULB,
    };

    void update() {
        banner.update(config);
    }

    FrameStats at(F64 time) {
        frames.ctx.time = time;
        return frames.run([this] { update(); });
    }

    const Sakura::Text* text(const std::string& str) {
        for (auto* found : Find<Sakura::Text>(banner)) {
            if (found->config().str == str) {
                return found;
            }
        }
        return nullptr;
    }

    const Sakura::Text* close() {
        return text(ICON_LC_X);
    }

    F32 opacity() {
        const auto* label = text(config.str);
        return label ? label->config().opacity : 0.0f;
    }

    bool shown() {
        return opacity() > 0.0f;
    }

    bool click(Extent2D<F32> point) {
        const bool pressed = place.send(Mouse(MouseEventType::Click, point.x, point.y));
        const bool released = place.send(Mouse(MouseEventType::Release, point.x, point.y));
        return pressed && released;
    }

    Extent2D<F32> closePoint() {
        const auto* glyph = close();
        REQUIRE(glyph != nullptr);
        const Rect frame = Peek::Frame(*glyph);
        return {frame.x + frame.width * 0.5f, frame.y + frame.height * 0.5f};
    }
};

}  // namespace

TEST_CASE("Tip banners wait for their delay, fade in, hold, fade out and settle", "[core][sakura][tip-banner]") {
    Rig rig;
    CHECK(rig.at(10.0).laidOut);
    CHECK_FALSE(rig.shown());
    CHECK(rig.at(10.5).clean());
    CHECK_FALSE(rig.shown());

    CHECK(RepaintOnly(rig.at(11.1)));
    CHECK(rig.shown());
    CHECK(RepaintOnly(rig.at(11.3)));
    CHECK(rig.shown());
    for (const F64 time : {12.0, 15.0, 19.0}) {
        CAPTURE(time);
        CHECK(rig.at(time).clean());
    }

    CHECK(RepaintOnly(rig.at(19.3)));
    CHECK(rig.shown());
    rig.at(19.5);
    CHECK_FALSE(rig.shown());
    CHECK(Retains(rig.frames, [&] { rig.update(); }));
}

TEST_CASE("Persistent tip banners hold until released and then run their full duration",
          "[core][sakura][tip-banner]") {
    Rig rig;
    rig.config.persistent = true;
    rig.at(10.0);
    rig.at(11.3);
    REQUIRE(rig.shown());
    for (const F64 time : {20.0, 40.0, 80.0}) {
        CAPTURE(time);
        CHECK(rig.at(time).clean());
        CHECK(rig.shown());
    }

    rig.config.persistent = false;
    rig.at(100.0);
    CHECK(rig.shown());
    rig.at(107.5);
    CHECK(rig.shown());
    rig.at(108.1);
    rig.at(108.4);
    CHECK_FALSE(rig.shown());
}

TEST_CASE("Releasing a tip banner before it fades in keeps its delay and fade", "[core][sakura][tip-banner]") {
    Rig rig;
    rig.config.persistent = true;
    rig.at(10.0);
    CHECK_FALSE(rig.shown());

    rig.config.persistent = false;
    rig.at(10.5);
    CHECK_FALSE(rig.shown());
    rig.at(11.1);
    CHECK(rig.opacity() == Catch::Approx(0.5f).margin(0.01f));
    rig.at(11.3);
    CHECK(rig.opacity() == 1.0f);
    rig.at(19.0);
    CHECK(rig.shown());
    rig.at(19.5);
    CHECK_FALSE(rig.shown());
}

TEST_CASE("Pinning a faded tip banner again waits for its delay and fades back in", "[core][sakura][tip-banner]") {
    Rig rig;
    rig.at(10.0);
    rig.at(11.3);
    rig.at(19.5);
    REQUIRE_FALSE(rig.shown());

    rig.config.persistent = true;
    rig.at(25.0);
    CHECK_FALSE(rig.shown());
    CHECK(rig.at(25.5).clean());
    CHECK_FALSE(rig.shown());
    rig.at(26.1);
    CHECK(rig.opacity() == Catch::Approx(0.5f).margin(0.01f));
    rig.at(26.3);
    CHECK(rig.opacity() == 1.0f);
    CHECK(rig.at(60.0).clean());
    CHECK(rig.opacity() == 1.0f);
}

TEST_CASE("Pinning a tip banner while it fades out fades it back in from where it was",
          "[core][sakura][tip-banner]") {
    Rig rig;
    rig.at(10.0);
    rig.at(11.3);
    rig.at(19.3);
    const F32 fading = rig.opacity();
    REQUIRE(fading > 0.0f);
    REQUIRE(fading < 1.0f);

    rig.config.persistent = true;
    rig.at(19.3);
    CHECK(rig.opacity() == Catch::Approx(fading).margin(0.01f));
    rig.at(19.6);
    CHECK(rig.opacity() == 1.0f);
    CHECK(rig.at(40.0).clean());
    CHECK(rig.opacity() == 1.0f);
}

TEST_CASE("Hovering holds a tip banner and its close button dismisses it for good",
          "[core][sakura][tip-banner]") {
    Rig rig;
    rig.at(10.0);
    rig.at(11.3);
    REQUIRE(rig.shown());

    rig.place.send(Mouse(MouseEventType::Move, kSlot.x + 40.0f, kSlot.y + 20.0f));
    rig.at(12.0);
    CHECK(rig.at(30.0).clean());
    CHECK(rig.shown());
    rig.place.send(Mouse(MouseEventType::Leave, 0.0f, 0.0f));
    rig.at(31.0);
    CHECK(rig.shown());

    REQUIRE(rig.close() != nullptr);
    const F32 idle = rig.close()->config().opacity;
    rig.place.send(Mouse(MouseEventType::Move, rig.closePoint().x, rig.closePoint().y));
    rig.at(31.1);
    CHECK(rig.close()->config().opacity > idle);
    CHECK(rig.click(rig.closePoint()));
    rig.at(31.2);
    CHECK_FALSE(rig.shown());
    CHECK(Retains(rig.frames, [&] { rig.update(); }));

    rig.config.persistent = true;
    rig.at(40.0);
    rig.at(41.3);
    CHECK_FALSE(rig.shown());
}

TEST_CASE("Tip banner close presses only pair with a left release", "[core][sakura][tip-banner]") {
    Rig rig;
    rig.config.persistent = true;
    rig.at(10.0);
    rig.at(11.3);
    REQUIRE(rig.shown());
    const auto point = rig.closePoint();
    rig.place.send(Mouse(MouseEventType::Move, point.x, point.y));
    rig.at(11.4);

    rig.place.send(Mouse(MouseEventType::Click, point.x, point.y));
    rig.place.send(Mouse(MouseEventType::Click, point.x, point.y, MouseButton::Right));
    rig.place.send(Mouse(MouseEventType::Release, point.x, point.y, MouseButton::Right));
    rig.at(11.5);
    CHECK(rig.shown());

    rig.place.send(Mouse(MouseEventType::Release, point.x, point.y));
    rig.at(11.6);
    CHECK_FALSE(rig.shown());
}

TEST_CASE("Hidden tip banners let clicks through and only the close button shows a hand cursor",
          "[core][sakura][tip-banner]") {
    Rig rig;
    rig.at(10.0);
    CHECK_FALSE(rig.shown());
    CHECK_FALSE(rig.place.send(Mouse(MouseEventType::Click, kSlot.x + 40.0f, kSlot.y + 20.0f)));
    CHECK_FALSE(Peek::CursorAt(rig.banner, rig.closePoint()).has_value());

    rig.at(11.3);
    REQUIRE(rig.shown());
    CHECK(rig.place.send(Mouse(MouseEventType::Click, kSlot.x + 40.0f, kSlot.y + 20.0f)));
    rig.place.send(Mouse(MouseEventType::Release, kSlot.x + 40.0f, kSlot.y + 20.0f));
    CHECK(rig.shown());
    CHECK_FALSE(Peek::CursorAt(rig.banner, {kSlot.x + 40.0f, kSlot.y + 20.0f}).has_value());
    CHECK(Peek::CursorAt(rig.banner, rig.closePoint()) == SurfaceCursor::Hand);
}

TEST_CASE("Narrow tip banners keep the close cursor and press inside their frame", "[core][sakura][tip-banner]") {
    Scene scene;
    Sakura::TipBanner banner;
    const Rect slot = {100.0f, 20.0f, 30.0f, 40.0f};
    Place place(banner, slot);
    Frames frames(place, kScreen, scene.context());
    const auto update = [&] {
        banner.update({.str = "Drag blocks to move them.", .icon = ICON_LC_LIGHTBULB, .persistent = true});
    };
    frames.ctx.time = 10.0;
    frames.run(update);
    frames.ctx.time = 11.3;
    frames.run(update);

    const F32 y = slot.y + slot.height * 0.5f;
    CHECK_FALSE(Peek::CursorAt(banner, {slot.x - 4.0f, y}).has_value());
    CHECK(Peek::CursorAt(banner, {slot.x + 4.0f, y}) == SurfaceCursor::Hand);
    CHECK_FALSE(place.send(Mouse(MouseEventType::Click, slot.x - 4.0f, y)));
}

TEST_CASE("Tip banners measure their content, cap to the available space and scale with the pixel ratio",
          "[core][sakura][tip-banner]") {
    Scene scene;
    Sakura::TipBanner banner;
    Host host(banner);
    banner.update({.str = "Drag blocks to move them.", .icon = ICON_LC_LIGHTBULB});
    const auto natural = host.measureOf(scene.ctx, {kInfinite, kInfinite});
    CHECK(natural.y == 40.0f);
    CHECK(natural.x > 100.0f);

    auto doubled = scene.ctx;
    doubled.pixelRatio = 2.0f;
    const auto scaled = host.measureOf(doubled, {kInfinite, kInfinite});
    CHECK(scaled.y == 80.0f);
    CHECK(scaled.x > natural.x);

    CHECK(host.measureOf(scene.ctx, {80.0f, 30.0f}) == Extent2D<F32>{80.0f, 30.0f});
}

TEST_CASE("Cancelling a held close press keeps the tip banner and moves stay quiet", "[core][sakura][tip-banner]") {
    Rig rig;
    rig.config.persistent = true;
    rig.at(10.0);
    rig.at(11.3);
    REQUIRE(rig.shown());
    const auto update = [&] {
        rig.update();
    };
    REQUIRE(rig.frames.settle(update));

    CHECK(QuietMoves(rig.frames, update, {{kSlot.x + 300.0f, kSlot.y + 200.0f}, {kSlot.x + 310.0f, kSlot.y + 210.0f}},
                     [&](const MouseEvent& e) {
                         return rig.place.send(e);
                     }));
    rig.place.send(Mouse(MouseEventType::Move, kSlot.x + 40.0f, kSlot.y + 20.0f));
    CHECK(Unpainted(rig.frames.run(update)));
    CHECK(QuietMoves(rig.frames, update, {{kSlot.x + 40.0f, kSlot.y + 20.0f}, {kSlot.x + 60.0f, kSlot.y + 20.0f}},
                     [&](const MouseEvent& e) {
                         return rig.place.send(e);
                     }));

    const auto point = rig.closePoint();
    rig.place.send(Mouse(MouseEventType::Move, point.x, point.y));
    rig.frames.run(update);
    CHECK(rig.place.send(Mouse(MouseEventType::Click, point.x, point.y)));
    rig.place.send(Mouse(MouseEventType::Cancel, point.x, point.y));
    CHECK_FALSE(rig.place.send(Mouse(MouseEventType::Release, point.x, point.y)));
    rig.frames.run(update);
    CHECK(rig.shown());
    REQUIRE(rig.frames.settle(update));
    CHECK(Retains(rig.frames, update));
}
