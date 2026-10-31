#include <catch2/catch_test_macros.hpp>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/text.hh>
#include <jetstream/render/sakura/components/voice.hh>

#include "render/sakura/detail/voice_wave.hh"

#include "component_harness.hh"
#include "harness.hh"
#include "scene.hh"

#include <functional>
#include <vector>

using namespace Jetstream;
using namespace SakuraTest;

namespace {

constexpr Rect kPill = {500.0f, 10.0f, 140.0f, 28.0f};
constexpr Extent2D<F32> kBody = {kPill.x + 70.0f, kPill.y + 14.0f};

struct Rig {
    Scene scene;
    Sakura::Voice voice;
    Place place{voice, kPill};
    Frames frames{place, kScreen, scene.context()};
    U64 starts = 0;
    U64 stops = 0;
    Sakura::Voice::Config config;

    Rig() {
        config = {
            .phase = Sakura::Voice::Phase::Listening,
            .startHint = "Click to reconnect",
            .stopHint = "Click to end",
            .onStart = [this] { ++starts; },
            .onStop = [this] { ++stops; },
        };
        frames.ctx.deltaTime = 0.1f;
        frames.timeStep = 0.1;
    }

    void update() {
        voice.update(config);
    }

    FrameStats run() {
        return frames.run([this] { update(); });
    }

    void settle() {
        for (int frame = 0; frame < 30; ++frame) {
            run();
        }
    }

    void click(Extent2D<F32> point) {
        place.send(Mouse(MouseEventType::Click, point.x, point.y));
        place.send(Mouse(MouseEventType::Release, point.x, point.y));
    }

    const Sakura::VoiceWave::Config& wave() {
        const auto waves = Find<Sakura::VoiceWave>(voice);
        REQUIRE(waves.size() == 1);
        return waves.front()->config();
    }

    const Sakura::Box::Config& cover() {
        const auto boxes = Direct<Sakura::Box>(voice);
        REQUIRE(boxes.size() == 1);
        return boxes.front()->config();
    }

    const Sakura::Text::Config& hint() {
        const auto texts = Find<Sakura::Text>(voice);
        REQUIRE(texts.size() == 1);
        return texts.front()->config();
    }
};

}  // namespace

TEST_CASE("Voice pills measure a fixed width wave and scale with the pixel ratio", "[core][sakura][voice]") {
    Scene scene;
    Sakura::Voice voice;
    Host host(voice);
    voice.update({});
    CHECK(host.measureOf(scene.ctx, {kInfinite, kInfinite}) == Extent2D<F32>{140.0f, 28.0f});

    auto doubled = scene.ctx;
    doubled.pixelRatio = 2.0f;
    CHECK(host.measureOf(doubled, {kInfinite, kInfinite}) == Extent2D<F32>{280.0f, 56.0f});
    CHECK(host.measureOf(scene.ctx, {100.0f, 20.0f}) == Extent2D<F32>{100.0f, 20.0f});
}

TEST_CASE("Voice pills fill with the wave and only show the end hint and a red outline while hovered", "[core][sakura][voice]") {
    Rig rig;
    rig.settle();
    CHECK(rig.wave().rect == kPill);
    CHECK(rig.wave().cornerRadius == 14.0f);
    CHECK(rig.wave().margin > rig.wave().dot);
    CHECK(rig.hint().opacity == 0.0f);

    rig.place.send(Mouse(MouseEventType::Move, kBody.x, kBody.y));
    rig.settle();
    CHECK(rig.hint().opacity == 1.0f);
    CHECK(rig.hint().str == "Click to end");
    CHECK(rig.hint().colorKey == "voice_stop");
    CHECK(rig.cover().borderWidth == 1.5f);
    CHECK(rig.cover().borderColor == rig.frames.ctx.color("voice_stop"));
    CHECK(rig.cover().instances.front().rect == kPill);

    rig.place.send(Mouse(MouseEventType::Leave, 0.0f, 0.0f));
    rig.settle();
    CHECK(rig.hint().opacity == 0.0f);
    CHECK_FALSE(rig.cover().instances.front().visible);
}

TEST_CASE("Voice pills end the session on click and reconnect once the session has dropped",
          "[core][sakura][voice]") {
    Rig rig;
    rig.run();
    rig.click(kBody);
    CHECK(rig.stops == 1);
    CHECK(rig.starts == 0);

    rig.place.send(Mouse(MouseEventType::Click, kBody.x, kBody.y));
    rig.place.send(Mouse(MouseEventType::Release, kPill.x - 10.0f, kBody.y));
    CHECK(rig.stops == 1);

    rig.config.phase = Sakura::Voice::Phase::Idle;
    rig.run();
    rig.place.send(Mouse(MouseEventType::Move, kBody.x, kBody.y));
    rig.settle();
    CHECK(rig.hint().str == "Click to reconnect");
    rig.click(kBody);
    CHECK(rig.starts == 1);
    CHECK(rig.stops == 1);

    CHECK(Peek::CursorAt(rig.voice, kBody) == SurfaceCursor::Hand);
    CHECK_FALSE(Peek::CursorAt(rig.voice, {kPill.x - 5.0f, kPill.y}).has_value());
}

TEST_CASE("Voice pills act on the phase they were pressed in even when it changes before the release",
          "[core][sakura][voice]") {
    Rig rig;
    rig.frames.verify = false;
    rig.run();
    rig.place.send(Mouse(MouseEventType::Click, kBody.x, kBody.y));
    rig.config.phase = Sakura::Voice::Phase::Idle;
    rig.run();
    rig.place.send(Mouse(MouseEventType::Release, kBody.x, kBody.y));
    CHECK(rig.stops == 1);
    CHECK(rig.starts == 0);

    rig.place.send(Mouse(MouseEventType::Click, kBody.x, kBody.y));
    rig.config.phase = Sakura::Voice::Phase::Listening;
    rig.run();
    rig.place.send(Mouse(MouseEventType::Release, kBody.x, kBody.y));
    CHECK(rig.starts == 1);
    CHECK(rig.stops == 1);
}

TEST_CASE("Voice pills animate while live, blend toward the busy color while delegating and settle when idle",
          "[core][sakura][voice]") {
    Rig rig;
    rig.config.level = 1.0f;
    rig.run();
    for (int frame = 0; frame < 3; ++frame) {
        CHECK(RepaintOnly(rig.run()));
    }
    CHECK(rig.wave().level > 0.5f);
    CHECK(rig.wave().activity > 0.0f);
    CHECK(rig.wave().busy == 0.0f);
    const F32 time = rig.wave().time;
    rig.run();
    CHECK(rig.wave().time > time);

    rig.config.phase = Sakura::Voice::Phase::Busy;
    for (int frame = 0; frame < 60; ++frame) {
        rig.run();
    }
    CHECK(rig.wave().busy == 1.0f);
    CHECK(rig.wave().activity == 1.0f);

    rig.config.phase = Sakura::Voice::Phase::Listening;
    for (int frame = 0; frame < 60; ++frame) {
        rig.run();
    }
    CHECK(rig.wave().busy == 0.0f);

    rig.config.phase = Sakura::Voice::Phase::Idle;
    for (int frame = 0; frame < 120; ++frame) {
        rig.run();
    }
    CHECK(rig.wave().activity == 0.0f);
    CHECK(rig.wave().level == 0.0f);
    CHECK(Retains(rig.frames, [&] { rig.update(); }));
}

TEST_CASE("Voice pills ignore cancelled presses and keep moves quiet once idle", "[core][sakura][voice]") {
    Rig rig;
    rig.config.phase = Sakura::Voice::Phase::Idle;
    const auto update = [&] {
        rig.update();
    };
    rig.settle();
    REQUIRE(rig.frames.settle(update));
    CHECK(QuietMoves(rig.frames, update, {{100.0f, 300.0f}, {110.0f, 300.0f}}, [&](const MouseEvent& e) {
        return rig.place.send(e);
    }));

    rig.place.send(Mouse(MouseEventType::Move, kBody.x, kBody.y));
    rig.settle();
    REQUIRE(rig.frames.settle(update));
    CHECK(QuietMoves(rig.frames, update, {{kBody.x + 1.0f, kBody.y}, {kBody.x + 10.0f, kBody.y + 2.0f}},
                     [&](const MouseEvent& e) {
                         return rig.place.send(e);
                     }));

    CHECK(rig.place.send(Mouse(MouseEventType::Click, kBody.x, kBody.y)));
    rig.place.send(Mouse(MouseEventType::Cancel, kBody.x, kBody.y));
    CHECK_FALSE(rig.place.send(Mouse(MouseEventType::Release, kBody.x, kBody.y)));
    CHECK(rig.starts == 0);
    CHECK(rig.stops == 0);
    rig.settle();
    CHECK(rig.hint().opacity == 0.0f);
}

TEST_CASE("Voice pills stop animating while offscreen and resume once shown", "[core][sakura][voice]") {
    Rig rig;
    rig.config.level = 0.5f;
    rig.run();
    CHECK(RepaintOnly(rig.run()));
    CHECK(RepaintOnly(rig.run()));

    const F32 time = rig.wave().time;
    rig.voice.setVisible(false);
    rig.run();
    for (int frame = 0; frame < 5; ++frame) {
        CHECK(rig.run().clean());
    }

    rig.voice.setVisible(true);
    CHECK(RepaintOnly(rig.run()));
    CHECK(RepaintOnly(rig.run()));
    CHECK(rig.wave().time > time);
}

TEST_CASE("Voice waves repaint only when their config changes", "[core][sakura][voice]") {
    Scene scene;
    Sakura::VoiceWave wave;
    Host host(wave);
    Frames frames(host, kScreen, scene.context());
    F32 level = 0.25f;
    std::optional<Rect> clip;
    const auto update = [&] {
        wave.update({
            .rect = kPill,
            .pitch = 2.5f,
            .dot = 0.85f,
            .cornerRadius = 14.0f,
            .margin = 2.35f,
            .level = level,
            .clip = clip,
        });
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));

    level = 0.75f;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    clip = Rect{kPill.x, kPill.y, 70.0f, kPill.height};
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));

    wave.setVisible(false);
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    wave.setVisible(true);
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    CHECK(frames.rebuilds == 1);
    CHECK(frames.restructures == 0);
}

TEST_CASE("Voice waves repaint for every field, a new translation and a new inherited clip", "[core][sakura][voice]") {
    Scene scene;
    Sakura::VoiceWave wave;
    Place place(wave, kScreen);
    Host host(place, true);
    Frames frames(host, kScreen, scene.context());
    Sakura::VoiceWave::Config config = {
        .rect = kPill,
        .pitch = 2.5f,
        .dot = 0.85f,
        .cornerRadius = 14.0f,
        .margin = 2.35f,
        .level = 0.25f,
    };
    const auto update = [&] {
        wave.update(config);
    };
    REQUIRE(frames.settle(update));

    const std::vector<std::function<void()>> edits = {
        [&] { config.rect.x += 4.0f; },
        [&] { config.pitch = 3.0f; },
        [&] { config.dot = 0.5f; },
        [&] { config.cornerRadius = 10.0f; },
        [&] { config.margin = 3.0f; },
        [&] { config.time = 1.5f; },
        [&] { config.busy = 0.5f; },
        [&] { config.activity = 0.75f; },
        [&] { config.level = 0.9f; },
        [&] { config.listeningColor.r = 0.5f; },
        [&] { config.busyColor.g = 0.5f; },
        [&] { config.dimColor.b = 0.5f; },
        [&] { config.clip = Rect{kPill.x, kPill.y, 70.0f, kPill.height}; },
    };
    for (U64 index = 0; index < edits.size(); ++index) {
        CAPTURE(index);
        edits[index]();
        CHECK(RepaintOnly(frames.run(update)));
        CHECK(Retains(frames, update));
    }

    place.pan({30.0f, 40.0f});
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Peek::Translation(wave) == Extent2D<F32>{30.0f, 40.0f});
    CHECK(Retains(frames, update));

    const auto before = Peek::InheritedClip(wave);
    frames.bounds = {0.0f, 0.0f, 600.0f, 300.0f};
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Peek::InheritedClip(wave) != before);
    CHECK(Retains(frames, update));
    CHECK(frames.rebuilds == 1);
    CHECK(frames.restructures == 0);
}
