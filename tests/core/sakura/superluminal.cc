#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "scene.hh"

#include "render/sakura/titlebar.hh"
#include "superluminal/view.hh"

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

using namespace Jetstream;
using namespace SakuraTest;

namespace {

using View = SuperluminalView;

constexpr Jetstream::Rect kWide = {0.0f, 0.0f, 1200.0f, 600.0f};
constexpr F32 kGap = Sakura::Dock::ContentPadding;
constexpr F32 kHeader = Sakura::Dock::HeaderHeight;
constexpr F32 kBar = Sakura::Titlebar::MinimumHeight;

class PlotTexture final : public Render::Texture {
 public:
    PlotTexture() : Texture(Config{.size = {64, 64}}) {}

    Result create() override {
        return Result::SUCCESS;
    }

    Result destroy() override {
        return Result::SUCCESS;
    }

    uint64_t raw() const override {
        return 1;
    }
};

struct ScaledFontWindow : FontWindow {
    explicit ScaledFontWindow(F32 scale) {
        _scalingFactor = scale;
        _previousScalingFactor = scale;
        load("default_mono", jbmm_compressed_data);
    }
};

Jetstream::Rect GlyphRect(Sakura::Text& text) {
    for (auto* label : Find<Sakura::Label>(text)) {
        for (const auto& instance : label->config().instances) {
            if (instance.visible && instance.str == text.config().str) {
                return instance.rect;
            }
        }
    }
    return {};
}

Sakura::DockTiles Id(U64 index) {
    return Leaf(View::PaneId(index));
}

View::Tile Tile(const std::string& name, Extent2D<U8> offset, Extent2D<U8> size) {
    return View::Tile{.name = name, .offset = offset, .size = size};
}

struct Rig {
    Scene scene{kWide};
    View view;
    LayerHost host;
    std::string title = "Superluminal";
    Sakura::Titlebar::Chrome chrome;
    std::vector<U8> presses;
    View::InterfaceCallback global;
    std::vector<View::Tile> tiles;
    Extent2D<U8> mosaic = {0, 0};

    Rig() {
        host.setLayers(view.layers());
    }

    void update() {
        view.update({
            .title = title,
            .chrome = chrome,
            .onBarPress = [this](U8 clicks) {
                presses.push_back(clicks);
            },
            .global = global,
            .tiles = tiles,
            .mosaic = mosaic,
        });
    }

    Sakura::Div& bar() {
        return *Find<Sakura::Div>(host).at(0);
    }

    Sakura::Dock& dock() {
        return *Find<Sakura::Dock>(host).at(0);
    }

    std::vector<std::string> headers() {
        std::vector<std::string> titles;
        for (auto* text : Direct<Sakura::Text>(dock())) {
            titles.push_back(text->config().str);
        }
        return titles;
    }

    std::vector<std::string> labels() {
        std::vector<std::string> strings;
        for (auto* range : Find<Sakura::NodeRangeInput>(host)) {
            for (auto* label : Find<Sakura::Label>(*range)) {
                for (const auto& instance : label->config().instances) {
                    if (instance.visible && !instance.str.empty()) {
                        strings.push_back(instance.str);
                    }
                }
            }
        }
        return strings;
    }
};

View::Tile Plot(const std::string& name,
                Extent2D<U8> offset,
                Extent2D<U8> size,
                const std::shared_ptr<PlotTexture>& texture,
                std::vector<Sakura::SurfaceResize>* sizes = nullptr,
                std::vector<InputEvent>* inputs = nullptr) {
    auto tile = Tile(name, offset, size);
    tile.surfaces.push_back({
        .texture = texture,
        .onSize = [sizes](const Sakura::SurfaceResize& resize) {
            if (sizes) {
                sizes->push_back(resize);
            }
        },
        .onInput = [inputs](InputEvent event) {
            if (inputs) {
                inputs->push_back(event);
            }
        },
        .resolveCursor = [] {
            return SurfaceCursor::ResizeAll;
        },
    });
    return tile;
}

}  // namespace

TEST_CASE("Superluminal registered callbacks retain mutable captures and stateful function objects",
          "[core][sakura][superluminal][callbacks]") {
    struct Counter {
        std::vector<int>* seen;
        int count = 0;

        void operator()() {
            seen->push_back(++count);
        }
    };

    const bool stateful = GENERATE(false, true);
    Rig rig;
    std::vector<int> globalSeen;
    std::vector<int> boxSeen;
    const auto registration = [stateful](std::vector<int>& seen) -> std::function<void()> {
        if (stateful) {
            return Counter{&seen};
        }
        return [count = 0, &seen]() mutable {
            seen.push_back(++count);
        };
    };
    rig.global = registration(globalSeen);
    rig.tiles = {Tile("Controls", {0, 0}, {1, 1})};
    rig.tiles[0].callback = registration(boxSeen);
    for (int frame = 0; frame < 3; ++frame) {
        rig.update();
    }
    CHECK(globalSeen == std::vector<int>{1, 2, 3});
    CHECK(boxSeen == std::vector<int>{1, 2, 3});
}

TEST_CASE("Superluminal callback snapshots survive registration replacement and tile vector mutation",
          "[core][sakura][superluminal][callbacks]") {
    View view;
    View::Config registration;
    std::vector<int> seen;
    auto lifetime = std::make_shared<int>(42);
    std::weak_ptr<int> weak = lifetime;
    registration.global = [lifetime, &registration, &seen] {
        registration.global = {};
        registration.tiles.clear();
        seen.push_back(*lifetime);
    };
    registration.tiles = {Tile("Controls", {0, 0}, {1, 1})};
    registration.tiles[0].callback = [lifetime, &registration, &seen] {
        for (U64 index = 0; index < 64; ++index) {
            registration.tiles.push_back(Tile(std::to_string(index), {0, 0}, {1, 1}));
        }
        seen.push_back(*lifetime + 1);
    };
    lifetime.reset();
    view.update(registration);
    CHECK(seen == std::vector<int>{42, 43});
    CHECK(weak.expired());
    CHECK(registration.tiles.size() == 64);
    CHECK_FALSE(registration.global);
    CHECK_FALSE(View::InterfaceCallback(std::function<void()>{}));

    View::InterfaceCallback callback;
    lifetime = std::make_shared<int>(44);
    weak = lifetime;
    callback = [lifetime, &callback, &seen] {
        callback = {};
        seen.push_back(*lifetime);
    };
    lifetime.reset();
    callback();
    CHECK(seen == std::vector<int>{42, 43, 44});
    CHECK(weak.expired());
    CHECK_FALSE(callback);
}

TEST_CASE("Superluminal slider edits persist in mutable by-value callback captures",
          "[core][sakura][superluminal][callbacks][interaction]") {
    const bool global = GENERATE(false, true);
    Rig rig;
    std::vector<F32> seen;
    const auto callback = [gain = 2.0f, &rig, &seen]() mutable {
        CHECK(rig.view.interface().slider("Gain", 0.0f, 10.0f, gain, false) == Result::SUCCESS);
        seen.push_back(gain);
    };
    if (global) {
        rig.global = callback;
    } else {
        rig.tiles = {Tile("Controls", {0, 0}, {1, 1})};
        rig.tiles[0].callback = callback;
    }
    Frames frames(rig.host, kWide, rig.scene.context());
    frames.verify = false;
    const auto update = [&] { rig.update(); };
    REQUIRE(frames.settle(update));
    const auto rect = Peek::Frame(*Find<Sakura::NodeRangeInput>(rig.host).at(0));
    const F32 x = rect.x + rect.width * 0.75f;
    const F32 y = rect.y + rect.height * 0.5f;
    rig.host.send(Mouse(MouseEventType::Click, x, y));
    rig.host.send(Mouse(MouseEventType::Release, x, y));
    seen.clear();
    for (int frame = 0; frame < 3; ++frame) {
        frames.run(update);
    }
    REQUIRE(seen.size() == 3);
    for (const F32 value : seen) {
        CHECK(value == Catch::Approx(7.5f));
    }
}

TEST_CASE("Superluminal hidden slider edits expire without consuming other sliders or box queues",
          "[core][sakura][superluminal][pending][interaction]") {
    const bool global = GENERATE(false, true);
    const U64 absent = GENERATE(U64{1}, View::ParkedUpdates + 1);
    Rig rig;
    bool shown = true;
    F32 first = 1.0f;
    F32 gain = 2.0f;
    F32 other = 4.0f;
    const auto callback = [&] {
        CHECK(rig.view.interface().slider("Gain", 0.0f, 10.0f, first, false) == Result::SUCCESS);
        if (shown) {
            CHECK(rig.view.interface().slider("Gain", 0.0f, 10.0f, gain, false) == Result::SUCCESS);
        }
    };
    if (global) {
        rig.global = callback;
    } else {
        rig.tiles = {Tile("Controls", {0, 0}, {1, 1})};
        rig.tiles[0].callback = callback;
    }
    rig.tiles.push_back(Tile("Other", {1, 0}, {1, 1}));
    rig.tiles.back().callback = [&] {
        CHECK(rig.view.interface().slider("Gain", 0.0f, 10.0f, other, false) == Result::SUCCESS);
    };
    Frames frames(rig.host, kWide, rig.scene.context());
    frames.verify = false;
    const auto update = [&] { rig.update(); };
    REQUIRE(frames.settle(update));
    const auto ranges = Find<Sakura::NodeRangeInput>(rig.host);
    REQUIRE(ranges.size() == 3);
    for (U64 index = 0; index < ranges.size(); ++index) {
        const auto rect = Peek::Frame(*ranges[index]);
        const F32 fraction = std::array<F32, 3>{0.5f, 0.75f, 0.9f}[index];
        const F32 x = rect.x + rect.width * fraction;
        const F32 y = rect.y + rect.height * 0.5f;
        rig.host.send(Mouse(MouseEventType::Click, x, y));
        rig.host.send(Mouse(MouseEventType::Release, x, y));
    }
    shown = false;
    frames.run(update);
    CHECK(first == Catch::Approx(5.0f));
    CHECK(gain == 2.0f);
    CHECK(other == Catch::Approx(9.0f));
    gain = 3.0f;
    for (U64 frame = 1; frame < absent; ++frame) {
        frames.run(update);
    }
    REQUIRE(frames.settle(update));
    shown = true;
    frames.run(update);
    CHECK(gain == 3.0f);
    CHECK(first == Catch::Approx(5.0f));
    CHECK(other == Catch::Approx(9.0f));
    REQUIRE(frames.settle(update));
    CHECK(Find<Sakura::NodeRangeInput>(rig.host).size() == 3);
}

TEST_CASE("Superluminal recording ends expire only their own unconsumed label occurrences",
          "[core][sakura][superluminal][pending]") {
    View::Recorder recorder;
    for (const std::string box : {std::string(View::GlobalBox), std::string("A"), std::string("B")}) {
        recorder.slide(box, "Gain", 0, 7.0f);
        recorder.slide(box, "Gain", 1, 8.0f);
    }
    F32 first = 1.0f;
    F32 second = 2.0f;
    recorder.begin(View::GlobalBox);
    CHECK(recorder.slider("Gain", 0.0f, 10.0f, first, false) == Result::SUCCESS);
    recorder.end();
    CHECK(first == 7.0f);
    recorder.begin("A");
    recorder.end();
    recorder.begin("B");
    CHECK(recorder.slider("Gain", 0.0f, 10.0f, first, false) == Result::SUCCESS);
    CHECK(recorder.slider("Gain", 0.0f, 10.0f, second, false) == Result::SUCCESS);
    recorder.end();
    CHECK(first == 7.0f);
    CHECK(second == 8.0f);
    for (const std::string box : {std::string(View::GlobalBox), std::string("A")}) {
        first = 3.0f;
        second = 4.0f;
        recorder.begin(box);
        CHECK(recorder.slider("Gain", 0.0f, 10.0f, first, false) == Result::SUCCESS);
        CHECK(recorder.slider("Gain", 0.0f, 10.0f, second, false) == Result::SUCCESS);
        recorder.end();
        CHECK(first == 3.0f);
        CHECK(second == 4.0f);
    }
}

TEST_CASE("Superluminal unsigned text edits write back even at the original float projection",
          "[core][sakura][superluminal][slider-edit]") {
    const bool global = GENERATE(false, true);
    Rig rig;
    U64 value = 16777217;
    const auto callback = [&] {
        REQUIRE(rig.view.interface().slider("Samples", 0.0f, 33554432.0f, value) == Result::SUCCESS);
    };
    if (global) {
        rig.global = callback;
    } else {
        rig.tiles = {Tile("Controls", {0, 0}, {1, 1})};
        rig.tiles[0].callback = callback;
    }
    Frames frames(rig.host, kWide, rig.scene.context());
    frames.verify = false;
    const auto update = [&] { rig.update(); };
    REQUIRE(frames.settle(update));
    REQUIRE(value == 16777217);
    const auto rect = Peek::Frame(*Find<Sakura::NodeRangeInput>(rig.host).at(0));
    const F32 x = rect.x + rect.width * 0.5f;
    const F32 y = rect.y + rect.height * 0.5f;
    REQUIRE(rig.host.send(Mouse(MouseEventType::Click, x, y)));
    rig.host.send(Mouse(MouseEventType::Release, x, y));
    frames.run(update);
    REQUIRE(value == 16777217);
    for (const auto& [text, expected] : std::vector<std::pair<std::string, U64>>{{"16777216", 16777216}, {"42", 42}}) {
        REQUIRE(rig.host.send(Mouse(MouseEventType::Click, x, y, MouseButton::Left, 2)));
        frames.run(update);
        REQUIRE(FocusOwner(rig.host) != nullptr);
        Type(rig.host, text);
        REQUIRE(Tap(rig.host, KeyCode::Enter));
        frames.run(update);
        CHECK(value == expected);
    }
}

TEST_CASE("Superluminal unchanged numeric submissions round trip independently of display precision",
          "[core][sakura][superluminal][slider-edit]") {
    const bool global = GENERATE(false, true);
    const bool integer = GENERATE(false, true);
    const U64 index = GENERATE(U64{0}, U64{1}, U64{2});
    const U64 initialCount = std::array<U64, 3>{1000001, 16777216, 12345679}[index];
    const F32 initialFloat = std::array<F32, 3>{1000001.0f, 0.123456791f, 123456.789f}[index];
    const F32 initial = integer ? static_cast<F32>(initialCount) : initialFloat;
    U64 count = initialCount;
    F32 value = initialFloat;
    Rig rig;
    const auto callback = [&] {
        if (integer) {
            REQUIRE(rig.view.interface().slider("Value", 0.0f, 2.0f * initial, count) == Result::SUCCESS);
        } else {
            REQUIRE(rig.view.interface().slider("Value", 0.0f, 2.0f * initial, value, false) == Result::SUCCESS);
        }
    };
    if (global) {
        rig.global = callback;
    } else {
        rig.tiles = {Tile("Controls", {0, 0}, {1, 1})};
        rig.tiles[0].callback = callback;
    }
    Frames frames(rig.host, kWide, rig.scene.context());
    frames.verify = false;
    const auto update = [&] { rig.update(); };
    REQUIRE(frames.settle(update));
    const auto labels = rig.labels();
    const auto display = jst::fmt::format("{:.{}f}", initial, integer ? 0 : View::SliderPrecision);
    CHECK(std::find(labels.begin(), labels.end(), display) != labels.end());
    const auto rect = Peek::Frame(*Find<Sakura::NodeRangeInput>(rig.host).at(0));
    const F32 x = rect.x + rect.width * 0.5f;
    const F32 y = rect.y + rect.height * 0.5f;
    REQUIRE(rig.host.send(Mouse(MouseEventType::Click, x, y)));
    rig.host.send(Mouse(MouseEventType::Release, x, y));
    frames.run(update);
    REQUIRE(count == initialCount);
    REQUIRE(value == initialFloat);
    REQUIRE(rig.host.send(Mouse(MouseEventType::Click, x, y, MouseButton::Left, 2)));
    frames.run(update);
    REQUIRE(FocusOwner(rig.host) != nullptr);
    REQUIRE(Tap(rig.host, KeyCode::Enter));
    frames.run(update);
    CHECK(count == initialCount);
    CHECK(value == initialFloat);
    REQUIRE(frames.settle(update));
}

TEST_CASE("Superluminal mosaics map onto dock tiles that keep their arrangement",
          "[core][sakura][superluminal][mosaic]") {
    CHECK_FALSE(View::Arrange({}, {1, 1}).has_value());
    CHECK(View::Arrange({Tile("a", {0, 0}, {2, 2})}, {2, 2}) == Id(0));

    CHECK(View::Arrange({Tile("Controls", {0, 0}, {1, 2}), Tile("Sine", {1, 0}, {2, 2})}, {3, 2}) ==
          Row({1.0f, 2.0f}, {Id(0), Id(1)}));

    CHECK(View::Arrange({Tile("Time", {0, 0}, {1, 1}), Tile("Frequency", {0, 1}, {1, 1})}, {1, 2}) ==
          Column({1.0f, 1.0f}, {Id(0), Id(1)}));

    CHECK(View::Arrange({Tile("top", {0, 0}, {2, 1}), Tile("a", {0, 1}, {1, 2}), Tile("b", {1, 1}, {1, 2})},
                        {2, 3}) ==
          Column({1.0f, 2.0f}, {Id(0), Row({1.0f, 1.0f}, {Id(1), Id(2)})}));

    CHECK(View::Arrange({Tile("a", {0, 0}, {1, 1}), Tile("b", {1, 0}, {1, 1}),
                         Tile("c", {0, 1}, {1, 1}), Tile("d", {1, 1}, {1, 1})}, {2, 2}) ==
          Column({1.0f, 1.0f}, {Row({1.0f, 1.0f}, {Id(0), Id(1)}),
                                Row({1.0f, 1.0f}, {Id(2), Id(3)})}));

    CHECK_FALSE(View::Arrange({Tile("a", {0, 0}, {1, 1}), Tile("b", {2, 0}, {1, 1})}, {3, 1}).has_value());
    CHECK_FALSE(View::Arrange({Tile("a", {0, 0}, {1, 1})}, {2, 1}).has_value());
    CHECK_FALSE(View::Arrange({Tile("a", {0, 0}, {2, 1}), Tile("b", {1, 0}, {1, 1})}, {2, 1}).has_value());

    const auto pinwheel = View::Arrange({Tile("a", {0, 0}, {2, 1}), Tile("b", {2, 0}, {1, 2}),
                                         Tile("c", {1, 2}, {2, 1}), Tile("d", {0, 1}, {1, 2}),
                                         Tile("e", {1, 1}, {1, 1})}, {3, 3});
    CHECK_FALSE(pinwheel.has_value());
}

TEST_CASE("Sparse Superluminal mosaics keep their tiles at the outer cells of the real mosaic dims",
          "[core][sakura][superluminal][mosaic]") {
    Rig rig;
    rig.mosaic = {3, 1};
    rig.tiles = {Tile("left", {0, 0}, {1, 1}), Tile("right", {2, 0}, {1, 1})};
    for (auto& tile : rig.tiles) {
        tile.callback = [] {};
    }
    Frames frames(rig.host, kWide, rig.scene.context());
    REQUIRE(frames.settle([&] { rig.update(); }));
    CHECK(rig.view.freeform());
    const auto viewports = Find<View::ScrollView>(rig.host);
    REQUIRE(viewports.size() == 2);
    const F32 unit = (kWide.width - 2.0f * kGap + kGap) / 3.0f;
    const auto left = Peek::Frame(*viewports[0]);
    const auto right = Peek::Frame(*viewports[1]);
    CHECK(left.x == 2.0f * kGap);
    CHECK(left.right() == std::round(kGap + unit - kGap) - kGap);
    CHECK(right.x == std::round(kGap + 2.0f * unit) + kGap);
    CHECK(right.right() == kWide.width - 2.0f * kGap);
}

TEST_CASE("Superluminal mosaics without a guillotine cut keep every tile at its mosaic rectangle",
          "[core][sakura][superluminal][mosaic]") {
    Rig rig;
    rig.tiles = {Tile("a", {0, 0}, {2, 1}), Tile("b", {2, 0}, {1, 2}), Tile("c", {1, 2}, {2, 1}),
                 Tile("d", {0, 1}, {1, 2}), Tile("e", {1, 1}, {1, 1})};
    for (auto& tile : rig.tiles) {
        tile.callback = [] {};
    }
    Frames frames(rig.host, kWide, rig.scene.context());
    const auto update = [&] {
        rig.update();
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
    CHECK(rig.view.freeform());

    const Jetstream::Rect area = {kGap, kBar, kWide.width - 2.0f * kGap, kWide.height - kBar - kGap};
    const F32 unitX = (area.width + kGap) / 3.0f;
    const F32 unitY = (area.height + kGap) / 3.0f;
    const auto expected = [&](const View::Tile& tile) {
        const F32 x0 = std::round(area.x + unitX * tile.offset.x);
        const F32 y0 = std::round(area.y + unitY * tile.offset.y);
        const F32 x1 = std::round(area.x + unitX * (tile.offset.x + tile.size.x) - kGap);
        const F32 y1 = std::round(area.y + unitY * (tile.offset.y + tile.size.y) - kGap);
        return Jetstream::Rect{x0 + kGap, y0 + kHeader + kGap, x1 - x0 - 2.0f * kGap, y1 - y0 - kHeader - 2.0f * kGap};
    };
    const auto docks = Find<Sakura::Dock>(rig.host);
    REQUIRE(docks.size() == 5);
    const auto viewports = Find<View::ScrollView>(rig.host);
    REQUIRE(viewports.size() == 5);
    for (const auto& tile : rig.tiles) {
        const auto want = expected(tile);
        const bool found = std::any_of(viewports.begin(), viewports.end(), [&](auto* viewport) {
            return Peek::Frame(*viewport) == want;
        });
        INFO(tile.name);
        CHECK(found);
        bool titled = false;
        for (auto* dock : docks) {
            for (auto* text : Direct<Sakura::Text>(*dock)) {
                titled = titled || text->config().str == tile.name;
            }
        }
        CHECK(titled);
    }
}

TEST_CASE("Superluminal tiles are placed from the mosaic with the dock chrome and their titles",
          "[core][sakura][superluminal][mosaic]") {
    Rig rig;
    const auto texture = std::make_shared<PlotTexture>();
    rig.tiles = {
        Tile("Controls", {0, 0}, {1, 2}),
        Plot("Sine", {1, 0}, {2, 2}, texture),
    };
    rig.tiles[0].callback = [&] {
        CHECK(rig.view.interface().text("Signal Parameters") == Result::SUCCESS);
    };
    Frames frames(rig.host, kWide, rig.scene.context());
    REQUIRE(frames.settle([&] { rig.update(); }));

    CHECK(rig.view.tiles() == Row({1.0f, 2.0f}, {Id(0), Id(1)}));
    const auto titles = rig.headers();
    CHECK(std::find(titles.begin(), titles.end(), "Controls") != titles.end());
    CHECK(std::find(titles.begin(), titles.end(), "Sine") != titles.end());

    const auto panels = Find<Sakura::NodeFieldPanel>(rig.host);
    const auto images = Find<Sakura::SurfaceImage>(rig.host);
    REQUIRE(panels.size() == 1);
    REQUIRE(images.size() == 1);
    const auto controls = Peek::Frame(*panels[0]);
    const auto plot = Peek::Frame(*images[0]);
    const F32 usable = kWide.width - 3.0f * kGap;
    const F32 split = std::round(kGap + usable / 3.0f);
    CHECK(controls.x == 2.0f * kGap);
    CHECK(controls.y == kBar + kHeader + kGap);
    CHECK(controls.right() == split - kGap);
    CHECK(plot.x == split + 2.0f * kGap);
    CHECK(plot.y == controls.y);
    CHECK(plot.right() == kWide.width - 2.0f * kGap);
    CHECK(plot.bottom() == kWide.height - 2.0f * kGap);
}

TEST_CASE("Superluminal box values repaint without rebuilding while the item list holds",
          "[core][sakura][superluminal][retention]") {
    Rig rig;
    F32 frequency = 1000.0f;
    F32 amplitude = 1.0f;
    bool extra = false;
    rig.tiles = {Tile("Controls", {0, 0}, {1, 1})};
    rig.tiles[0].callback = [&] {
        auto& recorder = rig.view.interface();
        CHECK(recorder.text(jst::fmt::format("Frequency {:.0f} Hz", frequency)) == Result::SUCCESS);
        CHECK(recorder.slider("Frequency (Hz)", 100.0f, 5000.0f, frequency, false) == Result::SUCCESS);
        CHECK(recorder.slider("Amplitude", 0.1f, 2.0f, amplitude, false) == Result::SUCCESS);
        if (extra) {
            F32 noise = 0.1f;
            CHECK(recorder.slider("Noise", 0.0f, 0.5f, noise, false) == Result::SUCCESS);
        }
    };
    Frames frames(rig.host, kWide, rig.scene.context());
    const auto update = [&] {
        rig.update();
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
    CHECK(QuietMoves(frames, update, {{300.0f, 300.0f}, {320.0f, 310.0f}}, [&](const MouseEvent& event) {
        return rig.host.send(event);
    }));
    const U64 restructures = frames.restructures;
    const U64 rebuilds = frames.rebuilds;

    frequency = 2500.0f;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));

    amplitude = 0.25f;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    CHECK(frames.restructures == restructures);
    CHECK(frames.rebuilds == rebuilds);
    const auto labels = rig.labels();
    CHECK(std::find(labels.begin(), labels.end(), "2500.000") != labels.end());
    CHECK(std::find(labels.begin(), labels.end(), "0.250") != labels.end());

    extra = true;
    CHECK(frames.run(update).restructured);
    CHECK(frames.settle(update));
    CHECK(frames.restructures == restructures + 1);

    extra = false;
    CHECK(Kept(frames.run(update)));
    CHECK(frames.settle(update));
    extra = true;
    CHECK(Kept(frames.run(update)));
    CHECK(frames.settle(update));
    CHECK(frames.restructures == restructures + 1);
    CHECK(frames.rebuilds == rebuilds);
}

TEST_CASE("Superluminal slider drags write the value into the caller reference on the next callback",
          "[core][sakura][superluminal][interaction]") {
    Rig rig;
    F32 gain = 2.0f;
    F32 seen = -1.0f;
    U64 calls = 0;
    rig.tiles = {Tile("Controls", {0, 0}, {1, 1})};
    rig.tiles[0].callback = [&] {
        ++calls;
        CHECK(rig.view.interface().slider("Gain", 0.0f, 10.0f, gain, false) == Result::SUCCESS);
        seen = gain;
    };
    Frames frames(rig.host, kWide, rig.scene.context());
    frames.verify = false;
    const auto update = [&] {
        rig.update();
    };
    REQUIRE(frames.settle(update));
    CHECK(seen == 2.0f);

    const auto ranges = Find<Sakura::NodeRangeInput>(rig.host);
    REQUIRE(ranges.size() == 1);
    const auto rect = Peek::Frame(*ranges[0]);
    const F32 y = rect.y + rect.height * 0.5f;
    rig.host.send(Mouse(MouseEventType::Click, rect.x + rect.width * 0.75f, y));
    rig.host.send(Mouse(MouseEventType::Release, rect.x + rect.width * 0.75f, y));
    CHECK(gain == 2.0f);

    const U64 before = calls;
    frames.run(update);
    CHECK(calls == before + 1);
    CHECK(gain == Catch::Approx(7.5f));
    CHECK(seen == Catch::Approx(7.5f));
    const auto labels = rig.labels();
    CHECK(std::find(labels.begin(), labels.end(), "7.500") != labels.end());

    gain = 1.0f;
    frames.run(update);
    CHECK(gain == 1.0f);
    CHECK(seen == 1.0f);
    CHECK(frames.settle(update));

    rig.host.send(Mouse(MouseEventType::Click, rect.x + rect.width * 0.5f, y));
    rig.host.send(Mouse(MouseEventType::Move, rect.x + rect.width * 0.25f, y));
    rig.host.send(Mouse(MouseEventType::Cancel, rect.x + rect.width * 0.25f, y));
    frames.run(update);
    CHECK(gain == Catch::Approx(2.5f));
}

TEST_CASE("Superluminal unsigned sliders write rounded values back and the recorder rejects calls outside callbacks",
          "[core][sakura][superluminal][interaction]") {
    Rig rig;
    U64 taps = 3;
    rig.tiles = {Tile("Controls", {0, 0}, {1, 1})};
    rig.tiles[0].callback = [&] {
        CHECK(rig.view.interface().slider("Taps", 1.0f, 65.0f, taps) == Result::SUCCESS);
    };
    Frames frames(rig.host, kWide, rig.scene.context());
    frames.verify = false;
    const auto update = [&] {
        rig.update();
    };
    REQUIRE(frames.settle(update));
    auto labels = rig.labels();
    CHECK(std::find(labels.begin(), labels.end(), "3") != labels.end());
    CHECK(taps == 3);

    const auto rect = Peek::Frame(*Find<Sakura::NodeRangeInput>(rig.host).at(0));
    rig.host.send(Mouse(MouseEventType::Click, rect.x + rect.width * 0.51f, rect.y + rect.height * 0.5f));
    rig.host.send(Mouse(MouseEventType::Release, rect.x + rect.width * 0.51f, rect.y + rect.height * 0.5f));
    frames.run(update);
    CHECK(taps == 34);
    labels = rig.labels();
    CHECK(std::find(labels.begin(), labels.end(), "34") != labels.end());

    F32 orphan = 1.0f;
    CHECK(rig.view.interface().slider("Orphan", 0.0f, 1.0f, orphan, false) == Result::ERROR);
    CHECK(rig.view.interface().text("Orphan") == Result::ERROR);
}

TEST_CASE("Superluminal unsigned sliders preserve untouched values and bound edited conversions",
          "[core][sakura][superluminal][interaction]") {
    View::Recorder recorder;
    for (const U64 original : {U64{16777217}, std::numeric_limits<U64>::max()}) {
        U64 value = original;
        recorder.begin("Controls");
        CHECK(recorder.slider("Count", 0.0f, 100.0f, value) == Result::SUCCESS);
        recorder.end();
        CHECK(value == original);
        CHECK(recorder.slider("Count", 0.0f, 100.0f, value) == Result::ERROR);
        CHECK(value == original);
    }
    for (const auto [edit, expected] : std::vector<std::pair<F32, U64>>{
             {-1.0f, 0}, {3.6f, 4},
             {static_cast<F32>(std::numeric_limits<U64>::max()), std::numeric_limits<U64>::max()},
             {std::numeric_limits<F32>::quiet_NaN(), 1}, {std::numeric_limits<F32>::infinity(), 1}}) {
        U64 value = 1;
        recorder.slide("Controls", "Count", 0, edit);
        recorder.begin("Controls");
        CHECK(recorder.slider("Count", 0.0f, 100.0f, value) == Result::SUCCESS);
        recorder.end();
        CHECK(value == expected);
    }
}

TEST_CASE("The Superluminal global interface is a full width tile above the mosaic sized to its content",
          "[core][sakura][superluminal][global]") {
    Rig rig;
    const auto texture = std::make_shared<PlotTexture>();
    F32 level = 0.5f;
    F32 boxLevel = 0.5f;
    rig.global = [&] {
        auto& recorder = rig.view.interface();
        CHECK(recorder.text("Signal Parameters") == Result::SUCCESS);
        CHECK(recorder.slider("Level", 0.0f, 1.0f, level, false) == Result::SUCCESS);
    };
    rig.tiles = {Tile("Level", {0, 0}, {1, 1}), Plot("Time", {1, 0}, {1, 1}, texture)};
    rig.tiles[0].callback = [&] {
        CHECK(rig.view.interface().slider("Level", 0.0f, 1.0f, boxLevel, false) == Result::SUCCESS);
    };
    Frames frames(rig.host, kWide, rig.scene.context());
    const auto update = [&] {
        rig.update();
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));

    const auto tiles = rig.view.tiles();
    REQUIRE(tiles.has_value());
    CHECK(tiles->vertical);
    REQUIRE(tiles->children.size() == 2);
    CHECK(tiles->children[0] == Leaf(View::GlobalPane));
    CHECK(tiles->children[1] == Row({1.0f, 1.0f}, {Id(0), Id(1)}));
    CHECK(tiles->weights.front() != View::GlobalWeight);
    const auto titles = rig.headers();
    CHECK(std::find(titles.begin(), titles.end(), View::GlobalTitle) != titles.end());

    const auto fits = Find<View::ScrollView>(rig.host);
    REQUIRE(fits.size() == 2);
    const auto global = Peek::Frame(*fits[0]);
    CHECK(global.x == 2.0f * kGap);
    CHECK(global.y == kBar + kHeader + kGap);
    CHECK(global.right() == kWide.width - 2.0f * kGap);
    CHECK(global.height < kWide.height * 0.5f);

    const auto ranges = Find<Sakura::NodeRangeInput>(rig.host);
    REQUIRE(ranges.size() == 2);
    const auto globalRange = std::find_if(ranges.begin(), ranges.end(), [&](auto* range) {
        return Peek::Frame(*range).bottom() <= global.bottom();
    });
    REQUIRE(globalRange != ranges.end());
    const auto rect = Peek::Frame(**globalRange);
    CHECK(global.bottom() - rect.bottom() <= 2.0f * View::FitSlack);
    const auto image = Peek::Frame(*Find<Sakura::SurfaceImage>(rig.host).at(0));
    CHECK(image.y == Catch::Approx(global.bottom() + 3.0f * kGap + kHeader).margin(1.0f));

    rig.host.send(Mouse(MouseEventType::Click, rect.x + rect.width * 0.9f, rect.y + rect.height * 0.5f));
    rig.host.send(Mouse(MouseEventType::Release, rect.x + rect.width * 0.9f, rect.y + rect.height * 0.5f));
    frames.run(update);
    CHECK(level == Catch::Approx(0.9f));
    CHECK(boxLevel == 0.5f);
    CHECK(frames.settle(update));
}

TEST_CASE("The Superluminal window bar shows the title past the traffic lights and routes bar presses",
          "[core][sakura][superluminal][titlebar]") {
    Rig rig;
    rig.title = "Scope";
    rig.chrome = {.leadingInset = 78.0f, .height = 52.0f, .controlsCenter = 27.0f};
    F32 level = 0.5f;
    rig.global = [&] {
        CHECK(rig.view.interface().text("Signal Parameters") == Result::SUCCESS);
        CHECK(rig.view.interface().slider("Level", 0.0f, 1.0f, level, false) == Result::SUCCESS);
    };
    Frames frames(rig.host, kWide, rig.scene.context());
    const auto update = [&] {
        rig.update();
    };
    REQUIRE(frames.settle(update));
    CHECK(rig.view.tiles() == Leaf(View::GlobalPane));

    const auto bar = Find<Sakura::Text>(rig.bar());
    const auto title = std::find_if(bar.begin(), bar.end(), [](auto* text) {
        return text->config().str == "Scope";
    });
    REQUIRE(title != bar.end());
    const auto caption = Peek::Frame(**title);
    CHECK(caption.x == 78.0f + View::TitleInset);
    CHECK((*title)->config().font == Sakura::Text::Font::DisplaySemiBold);
    CHECK((*title)->config().scale * Sakura::Typography::FontSize == View::TitleSize);
    const auto glyph = GlyphRect(**title);
    CHECK(glyph.y + glyph.height * 0.5f == Catch::Approx(27.0f));
    CHECK(glyph.height == View::TitleSize);
    CHECK(caption.bottom() <= 52.0f - kGap);
    CHECK(Peek::Frame(rig.dock()).y == 52.0f - kGap);

    for (auto* text : Find<Sakura::Text>(rig.dock())) {
        CHECK(Peek::Frame(*text).y >= 52.0f);
    }
    for (auto* range : Find<Sakura::NodeRangeInput>(rig.host)) {
        CHECK(Peek::Frame(*range).y >= 52.0f);
    }

    rig.host.send(Mouse(MouseEventType::Click, 1000.0f, 20.0f));
    rig.host.send(Mouse(MouseEventType::Release, 1000.0f, 20.0f));
    rig.host.send(Mouse(MouseEventType::Click, 1000.0f, 20.0f, MouseButton::Left, 2));
    rig.host.send(Mouse(MouseEventType::Release, 1000.0f, 20.0f));
    CHECK(rig.presses == std::vector<U8>{1, 2});

    rig.chrome = {};
    frames.run(update);
    CHECK(frames.settle(update));
    CHECK(Peek::Frame(rig.dock()).y == kBar - kGap);
    const auto fallback = GlyphRect(*Find<Sakura::Text>(rig.bar()).at(0));
    CHECK(fallback.y + fallback.height * 0.5f == Catch::Approx(kBar * 0.5f));
    CHECK(fallback.x == View::TitleInset);
}

TEST_CASE("Superluminal box content lines up with the tile header title and labels use the UI font",
          "[core][sakura][superluminal][box]") {
    Rig rig;
    F32 gain = 0.5f;
    rig.tiles = {Tile("Controls", {0, 0}, {1, 1})};
    rig.tiles[0].callback = [&] {
        auto& recorder = rig.view.interface();
        CHECK(recorder.text("Signal Parameters") == Result::SUCCESS);
        CHECK(recorder.slider("Gain", 0.0f, 1.0f, gain, false) == Result::SUCCESS);
        CHECK(recorder.markdown("Some **notes**.") == Result::SUCCESS);
    };
    rig.tiles.push_back(Tile("Other", {1, 0}, {1, 1}));
    rig.tiles.back().callback = [] {};
    Frames frames(rig.host, kWide, rig.scene.context());
    REQUIRE(frames.settle([&] { rig.update(); }));

    const auto headers = Direct<Sakura::Text>(rig.dock());
    const auto header = std::find_if(headers.begin(), headers.end(), [](auto* text) {
        return text->config().str == "Controls";
    });
    REQUIRE(header != headers.end());
    const F32 left = Peek::Frame(**header).x;

    const auto texts = Find<Sakura::Text>(rig.dock());
    const auto label = std::find_if(texts.begin(), texts.end(), [](auto* text) {
        return text->config().str == "Signal Parameters";
    });
    REQUIRE(label != texts.end());
    CHECK(Peek::Frame(**label).x == left);
    CHECK((*label)->config().font == Sakura::Text::Font::DisplaySemiBold);
    CHECK((*label)->config().tone == Sakura::Text::Tone::Primary);
    CHECK((*label)->config().colorKey.empty());

    auto* field = Find<Sakura::NodeField>(rig.host).at(0);
    const auto fieldLabels = Direct<Sakura::Label>(*field);
    REQUIRE_FALSE(fieldLabels.empty());
    CHECK(fieldLabels[0]->config().instances.at(0).rect.x == left);

    auto* markdown = Find<Sakura::Markdown>(rig.host).at(0);
    F32 markdownLeft = kWide.width;
    for (auto* glyphs : Find<Sakura::Label>(*markdown)) {
        for (const auto& instance : glyphs->config().instances) {
            if (instance.visible && !instance.str.empty()) {
                markdownLeft = std::min(markdownLeft, instance.rect.x);
            }
        }
    }
    CHECK(markdownLeft == Catch::Approx(left).margin(1.0f));

    const auto range = Peek::Frame(*Find<Sakura::NodeRangeInput>(rig.host).at(0));
    const auto heading = Peek::Frame(**label);
    CHECK(range.y > heading.bottom());
    CHECK(Peek::Frame(*markdown).y > range.bottom());
}

TEST_CASE("The Superluminal window bar is a plain background with only the title and no tab pill",
          "[core][sakura][superluminal][titlebar]") {
    Rig rig;
    rig.title = "Scope";
    rig.tiles = {Tile("Controls", {0, 0}, {1, 1})};
    rig.tiles[0].callback = [&] {
        CHECK(rig.view.interface().text("Signal Parameters") == Result::SUCCESS);
    };
    Frames frames(rig.host, kWide, rig.scene.context());
    const auto update = [&] {
        rig.update();
    };
    REQUIRE(frames.settle(update));

    CHECK(Find<Sakura::Dock>(rig.host).size() == 1);
    const auto texts = Find<Sakura::Text>(rig.bar());
    REQUIRE(texts.size() == 1);
    CHECK(texts[0]->config().str == "Scope");
    const auto background = frames.ctx.color("background");
    U64 shapes = 0;
    for (auto* box : Find<Sakura::Box>(rig.bar())) {
        for (const auto& instance : box->config().instances) {
            if (!instance.visible) {
                continue;
            }
            ++shapes;
            CHECK(instance.backgroundColor == background);
            CHECK(instance.rect == Peek::Frame(rig.bar()));
        }
    }
    CHECK(shapes == 1);
    CHECK(Peek::Frame(rig.bar()) == Jetstream::Rect{0.0f, 0.0f, kWide.width, kBar - kGap});

    rig.host.send(Mouse(MouseEventType::Move, 600.0f, 10.0f));
    CHECK(Unpainted(frames.run(update)));
    CHECK(frames.settle(update));
}

TEST_CASE("Superluminal plot tiles forward input, cursor and resizes when their tile resizes",
          "[core][sakura][superluminal][surface]") {
    Rig rig;
    const auto texture = std::make_shared<PlotTexture>();
    std::vector<Sakura::SurfaceResize> sizes;
    std::vector<InputEvent> inputs;
    rig.tiles = {Plot("Sine", {0, 0}, {1, 1}, texture, &sizes, &inputs),
                 Plot("Cosine", {1, 0}, {1, 1}, texture)};
    Frames frames(rig.host, kWide, rig.scene.context());
    const auto update = [&] {
        rig.update();
    };
    REQUIRE(frames.settle(update));
    REQUIRE(sizes.size() == 1);
    auto* image = Find<Sakura::SurfaceImage>(rig.host).at(0);
    auto frame = Peek::Frame(*image);
    CHECK(sizes[0].framebufferSize == Extent2D<U64>{static_cast<U64>(frame.width), static_cast<U64>(frame.height)});
    CHECK(sizes[0].logicalSize == sizes[0].framebufferSize);
    CHECK(sizes[0].scale == Catch::Approx(0.5f));

    const F32 x = frame.x + frame.width * 0.5f;
    const F32 y = frame.y + frame.height * 0.25f;
    CHECK(Peek::CursorAt(*image, {x, y}) == SurfaceCursor::ResizeAll);
    rig.host.send(Mouse(MouseEventType::Click, x, y));
    rig.host.send(Mouse(MouseEventType::Release, x, y));
    const auto click = std::find_if(inputs.begin(), inputs.end(), [](const InputEvent& event) {
        const auto* mouse = std::get_if<MouseEvent>(&event);
        return mouse && mouse->type == MouseEventType::Click;
    });
    REQUIRE(click != inputs.end());
    CHECK(std::get<MouseEvent>(*click).position.x == Catch::Approx(0.5f));
    CHECK(std::get<MouseEvent>(*click).position.y == Catch::Approx(0.25f));

    frames.bounds = {0.0f, 0.0f, 1000.0f, 500.0f};
    Frames::Access::invalidateLayout(rig.host);
    frames.run(update);
    REQUIRE(sizes.size() == 2);
    frame = Peek::Frame(*image);
    CHECK(sizes[1].framebufferSize == Extent2D<U64>{static_cast<U64>(frame.width), static_cast<U64>(frame.height)});
    CHECK(sizes[1].framebufferSize.x < sizes[0].framebufferSize.x);

    const auto row = rig.view.tiles();
    REQUIRE(row.has_value());
    REQUIRE(row->children.size() == 2);
    const auto second = Peek::Frame(*Find<Sakura::SurfaceImage>(rig.host).at(1));
    const F32 middle = (frame.right() + second.x) * 0.5f;
    rig.host.send(Mouse(MouseEventType::Move, middle, 250.0f));
    rig.host.send(Mouse(MouseEventType::Click, middle, 250.0f));
    rig.host.send(Mouse(MouseEventType::Move, middle - 100.0f, 250.0f));
    rig.host.send(Mouse(MouseEventType::Release, middle - 100.0f, 250.0f));
    frames.run(update);
    frames.run(update);
    REQUIRE(sizes.size() == 3);
    CHECK(static_cast<F64>(sizes[2].framebufferSize.x) ==
          Catch::Approx(static_cast<F64>(sizes[1].framebufferSize.x) - 100.0).margin(4.0));
    CHECK(rig.view.tiles() != row);
    CHECK(frames.settle(update));
}

TEST_CASE("Superluminal tiles and plot surfaces scale with a pixel ratio of two",
          "[core][sakura][superluminal][pixel-ratio]") {
    const auto measure = [](F32 ratio) {
        ImGuiContextGuard imgui;
        ScaledFontWindow window(ratio);
        auto ctx = MakeContext({kWide.width * ratio, kWide.height * ratio});
        ctx.render = &window;
        ctx.pixelRatio = ratio;
        ctx.palette = DistinctPalette();
        View view;
        LayerHost host(view.layers());
        const auto texture = std::make_shared<PlotTexture>();
        std::vector<Sakura::SurfaceResize> sizes;
        F32 value = 0.5f;
        std::vector<View::Tile> tiles = {Tile("Controls", {0, 0}, {1, 1}), Plot("Sine", {1, 0}, {1, 1}, texture, &sizes)};
        tiles[0].callback = [&] {
            CHECK(view.interface().slider("Value", 0.0f, 1.0f, value, false) == Result::SUCCESS);
        };
        Frames frames(host, {0.0f, 0.0f, kWide.width * ratio, kWide.height * ratio}, ctx);
        REQUIRE(frames.settle([&] { view.update({.tiles = tiles}); }));
        REQUIRE(sizes.size() == 1);
        return std::make_tuple(Peek::Frame(*Find<Sakura::NodeFieldPanel>(host).at(0)),
                               Peek::Frame(*Find<Sakura::SurfaceImage>(host).at(0)),
                               Peek::Frame(*Find<Sakura::NodeRangeInput>(host).at(0)),
                               sizes.back());
    };
    const auto [panel, image, range, resize] = measure(1.0f);
    const auto [panel2, image2, range2, resize2] = measure(2.0f);
    CHECK(panel2.x == 2.0f * panel.x);
    CHECK(panel2.y == 2.0f * panel.y);
    CHECK(image2.x == Catch::Approx(2.0f * image.x).margin(1.0f));
    CHECK(image2.width == Catch::Approx(2.0f * image.width).margin(1.0f));
    CHECK(range2.height == 2.0f * range.height);
    CHECK(resize2.framebufferSize.x == Catch::Approx(2.0 * static_cast<F64>(resize.framebufferSize.x)).margin(1.0));
    CHECK(resize2.logicalSize.x == Catch::Approx(static_cast<F64>(resize.logicalSize.x)).margin(1.0));
    CHECK(resize2.scale == Catch::Approx(2.0f * resize.scale));
}

TEST_CASE("Superluminal viewports report their content height in logical units only when it changes",
          "[core][sakura][superluminal][retention]") {
    Probe content({100.0f, 40.0f});
    View::ScrollView viewport;
    std::vector<Extent2D<F32>> reports;
    Place place(viewport, {10.0f, 10.0f, 300.0f, 200.0f});
    auto ctx = MakeContext();
    ctx.pixelRatio = 2.0f;
    Frames frames(place, kScreen, ctx);
    frames.verify = false;
    const auto update = [&] {
        viewport.update({
            .content = &content,
            .onLayout = [&](F32 natural, F32 height) {
                reports.push_back({natural, height});
            },
        });
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
    CHECK(reports == std::vector<Extent2D<F32>>{{20.0f, 100.0f}});
    CHECK(Peek::Frame(content) == Jetstream::Rect{10.0f, 10.0f, 300.0f, 200.0f});

    place.move({10.0f, 10.0f, 300.0f, 120.0f});
    frames.run(update);
    CHECK(reports.back() == Extent2D<F32>{20.0f, 60.0f});
    content.resize({100.0f, 50.0f});
    frames.run(update);
    CHECK(reports.back() == Extent2D<F32>{25.0f, 60.0f});
    CHECK(reports.size() == 3);
    CHECK(frames.settle(update));
}

TEST_CASE("Superluminal viewports scroll overflowing content and stretch it when it fits",
          "[core][sakura][superluminal][scroll]") {
    Probe content({100.0f, 40.0f});
    content.consume = false;
    View::ScrollView viewport;
    Place place(viewport, {10.0f, 10.0f, 300.0f, 200.0f});
    Frames frames(place, kScreen, MakeContext());
    const auto update = [&] {
        viewport.update({.content = &content});
    };
    REQUIRE(frames.settle(update));
    CHECK(Peek::Frame(content) == Jetstream::Rect{10.0f, 10.0f, 300.0f, 200.0f});
    CHECK_FALSE(place.send(Wheel(100.0f, 100.0f, -1.0f)));

    content.resize({100.0f, 500.0f});
    frames.run(update);
    REQUIRE(frames.settle(update));
    auto frame = Peek::Frame(content);
    CHECK(frame.y == 10.0f);
    CHECK(frame.height == 500.0f);
    CHECK(frame.width < 300.0f);
    CHECK(place.send(Wheel(100.0f, 100.0f, -1.0f)));
    frames.run(update);
    frame = Peek::Frame(content);
    CHECK(frame.y < 10.0f);
    for (int step = 0; step < 20; ++step) {
        place.send(Wheel(100.0f, 100.0f, -1.0f));
        frames.run(update);
    }
    CHECK(Peek::Frame(content).bottom() == Catch::Approx(210.0f));
    CHECK(frames.settle(update));
}

TEST_CASE("Superluminal markdown items render with text and sliders and change without rebuilding",
          "[core][sakura][superluminal][markdown]") {
    Rig rig;
    F32 gain = 0.5f;
    std::string notes = "# Notes\nFirst **draft**.";
    bool before = true;
    rig.tiles = {Tile("Controls", {0, 0}, {1, 1})};
    rig.tiles[0].callback = [&] {
        auto& recorder = rig.view.interface();
        if (before) {
            CHECK(recorder.markdown("# Key Features\n- **Signal Generation**") == Result::SUCCESS);
        }
        CHECK(recorder.text("Signal Parameters") == Result::SUCCESS);
        CHECK(recorder.slider("Gain", 0.0f, 1.0f, gain, false) == Result::SUCCESS);
        CHECK(recorder.markdown(notes) == Result::SUCCESS);
    };
    const auto rendered = [](Sakura::Markdown& markdown) {
        std::string text;
        for (auto* label : Find<Sakura::Label>(markdown)) {
            for (const auto& instance : label->config().instances) {
                if (instance.visible) {
                    text += instance.str + "\n";
                }
            }
        }
        return text;
    };
    Frames frames(rig.host, kWide, rig.scene.context());
    const auto update = [&] {
        rig.update();
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));

    auto markdowns = Find<Sakura::Markdown>(rig.host);
    REQUIRE(markdowns.size() == 2);
    CHECK(rendered(*markdowns[0]).find("Key Features") != std::string::npos);
    CHECK(rendered(*markdowns[1]).find("draft") != std::string::npos);
    const auto first = Peek::Frame(*markdowns[0]);
    const auto last = Peek::Frame(*markdowns[1]);
    const auto range = Peek::Frame(*Find<Sakura::NodeRangeInput>(rig.host).at(0));
    const auto panel = Peek::Frame(*Find<Sakura::NodeFieldPanel>(rig.host).at(0));
    CHECK(first.bottom() <= range.y);
    CHECK(range.bottom() <= last.y);
    CHECK(last.bottom() == panel.bottom());
    const U64 restructures = frames.restructures;

    notes = "# Notes\nSecond revision.";
    CHECK(Kept(frames.run(update)));
    CHECK(rendered(*Find<Sakura::Markdown>(rig.host).at(1)).find("Second revision") != std::string::npos);
    CHECK(frames.settle(update));
    CHECK(Retains(frames, update));

    before = false;
    CHECK_FALSE(frames.run(update).rebuilt);
    CHECK(frames.settle(update));
    markdowns = Find<Sakura::Markdown>(rig.host);
    REQUIRE(markdowns.size() == 1);
    CHECK(rendered(*markdowns[0]).find("Second revision") != std::string::npos);
    CHECK(frames.restructures <= restructures + 1);

    F32 orphan = 0.0f;
    CHECK(rig.view.interface().markdown("orphan") == Result::ERROR);
    CHECK(rig.view.interface().slider("Orphan", 0.0f, 1.0f, orphan, false) == Result::ERROR);
}

TEST_CASE("Destroying the Superluminal root during a tile drag leaves the root memory untouched",
          "[core][sakura][superluminal][lifetime]") {
    Scene scene(kWide);
    auto view = std::make_unique<View>();
    std::vector<View::Tile> tiles = {Tile("a", {0, 0}, {1, 1}), Tile("b", {1, 0}, {1, 1})};
    for (auto& tile : tiles) {
        tile.callback = [] {};
    }
    alignas(LayerHost) std::array<std::byte, sizeof(LayerHost)> storage;
    auto* host = new (storage.data()) LayerHost(view->layers());
    const auto ctx = scene.context();
    view->update({.tiles = tiles});
    host->place(ctx, kWide);
    host->place(ctx, kWide);

    const F32 y = kBar + kHeader * 0.5f;
    host->send(Mouse(MouseEventType::Click, 100.0f, y));
    host->send(Mouse(MouseEventType::Move, 160.0f, y + 40.0f));
    REQUIRE(host->drag.active);

    host->~LayerHost();
    std::fill(storage.begin(), storage.end(), std::byte{0xA5});
    view.reset();
    CHECK(std::all_of(storage.begin(), storage.end(), [](std::byte value) {
        return value == std::byte{0xA5};
    }));
}

TEST_CASE("Superluminal boxes scroll so a slider below many text rows stays reachable",
          "[core][sakura][superluminal][scroll]") {
    Rig rig;
    F32 last = 0.0f;
    rig.tiles = {Tile("Controls", {0, 0}, {1, 1})};
    rig.tiles[0].callback = [&] {
        auto& recorder = rig.view.interface();
        for (int index = 0; index < 40; ++index) {
            CHECK(recorder.text(jst::fmt::format("Row {}", index)) == Result::SUCCESS);
        }
        CHECK(recorder.slider("Last", 0.0f, 1.0f, last, false) == Result::SUCCESS);
    };
    Frames frames(rig.host, kWide, rig.scene.context());
    frames.verify = false;
    const auto update = [&] {
        rig.update();
    };
    REQUIRE(frames.settle(update));
    const auto viewport = Peek::Frame(*Find<View::ScrollView>(rig.host).at(0));
    auto* range = Find<Sakura::NodeRangeInput>(rig.host).at(0);
    CHECK(Peek::Frame(*range).y > viewport.bottom());

    for (int step = 0; step < 100 && Peek::Frame(*range).bottom() > viewport.bottom(); ++step) {
        rig.host.send(Wheel(viewport.x + 100.0f, viewport.y + 100.0f, -1.0f));
        frames.run(update);
    }
    const auto rect = Peek::Frame(*range);
    REQUIRE(rect.bottom() <= viewport.bottom());
    REQUIRE(rect.y >= viewport.y);
    rig.host.send(Mouse(MouseEventType::Click, rect.x + rect.width * 0.5f, rect.y + rect.height * 0.5f));
    rig.host.send(Mouse(MouseEventType::Release, rect.x + rect.width * 0.5f, rect.y + rect.height * 0.5f));
    frames.run(update);
    CHECK(last == Catch::Approx(0.5f));
}

TEST_CASE("The Superluminal global tile keeps fitting its content until the user moves its splitter",
          "[core][sakura][superluminal][global]") {
    Rig rig;
    const auto texture = std::make_shared<PlotTexture>();
    U64 count = 1;
    std::vector<F32> values(8, 0.5f);
    rig.global = [&] {
        for (U64 index = 0; index < count; ++index) {
            CHECK(rig.view.interface().slider(jst::fmt::format("Value {}", index), 0.0f, 1.0f, values[index], false) ==
                  Result::SUCCESS);
        }
    };
    rig.tiles = {Plot("Time", {0, 0}, {1, 1}, texture)};
    Frames frames(rig.host, kWide, rig.scene.context());
    frames.verify = false;
    const auto update = [&] {
        rig.update();
    };
    const auto fitted = [&] {
        const auto viewport = Peek::Frame(*Find<View::ScrollView>(rig.host).at(0));
        const auto ranges = Find<Sakura::NodeRangeInput>(rig.host);
        F32 bottom = 0.0f;
        for (auto* range : ranges) {
            bottom = std::max(bottom, Peek::Frame(*range).bottom());
        }
        return viewport.bottom() - bottom >= 0.0f && viewport.bottom() - bottom <= 2.0f * View::FitSlack;
    };
    REQUIRE(frames.settle(update));
    CHECK(fitted());

    count = 6;
    frames.bounds = {0.0f, 0.0f, 500.0f, 600.0f};
    Frames::Access::invalidateLayout(rig.host);
    REQUIRE(frames.settle(update));
    CHECK(fitted());

    frames.bounds = {0.0f, 0.0f, 1200.0f, 800.0f};
    Frames::Access::invalidateLayout(rig.host);
    REQUIRE(frames.settle(update));
    CHECK(fitted());

    const auto viewport = Peek::Frame(*Find<View::ScrollView>(rig.host).at(0));
    const F32 splitter = viewport.bottom() + kGap + kGap * 0.5f;
    rig.host.send(Mouse(MouseEventType::Move, 600.0f, splitter));
    rig.host.send(Mouse(MouseEventType::Click, 600.0f, splitter));
    rig.host.send(Mouse(MouseEventType::Move, 600.0f, splitter + 120.0f));
    frames.run(update);
    frames.run(update);
    rig.host.send(Mouse(MouseEventType::Release, 600.0f, splitter + 120.0f));
    REQUIRE(frames.settle(update));
    const F32 dragged = Peek::Frame(*Find<View::ScrollView>(rig.host).at(0)).height;
    CHECK(dragged > viewport.height + 100.0f);

    count = 2;
    REQUIRE(frames.settle(update));
    CHECK(Peek::Frame(*Find<View::ScrollView>(rig.host).at(0)).height == dragged);
}

TEST_CASE("Space reaches the Superluminal plot whose tile was selected and yields to focused surfaces",
          "[core][sakura][superluminal][keyboard]") {
    Rig rig;
    const auto texture = std::make_shared<PlotTexture>();
    std::vector<InputEvent> left;
    std::vector<InputEvent> right;
    rig.tiles = {Plot("Left", {0, 0}, {1, 1}, texture, nullptr, &left),
                 Plot("Right", {1, 0}, {1, 1}, texture, nullptr, &right)};
    Frames frames(rig.host, kWide, rig.scene.context());
    const auto update = [&] {
        rig.update();
    };
    const auto spaces = [](const std::vector<InputEvent>& events) {
        U64 count = 0;
        for (const auto& event : events) {
            const auto* key = std::get_if<KeyEvent>(&event);
            count += key && key->key == KeyCode::Space && key->type == KeyEventType::Press ? 1 : 0;
        }
        return count;
    };
    REQUIRE(frames.settle(update));
    CHECK_FALSE(Tap(rig.host, KeyCode::Space));
    CHECK(spaces(left) + spaces(right) == 0);

    const F32 header = kBar + kHeader * 0.5f;
    rig.host.send(Mouse(MouseEventType::Click, kWide.width * 0.75f, header));
    rig.host.send(Mouse(MouseEventType::Release, kWide.width * 0.75f, header));
    ClickFocus(rig.host, Extent2D<F32>{kWide.width * 0.75f, header});
    frames.run(update);
    CHECK(Tap(rig.host, KeyCode::Space));
    CHECK(spaces(right) == 1);
    CHECK(spaces(left) == 0);

    rig.host.send(Mouse(MouseEventType::Click, kWide.width * 0.25f, header));
    rig.host.send(Mouse(MouseEventType::Release, kWide.width * 0.25f, header));
    ClickFocus(rig.host, Extent2D<F32>{kWide.width * 0.25f, header});
    frames.run(update);
    CHECK(Tap(rig.host, KeyCode::Space));
    CHECK(spaces(left) == 1);
    CHECK(spaces(right) == 1);

    const auto image = Peek::Frame(*Find<Sakura::SurfaceImage>(rig.host).at(1));
    const F32 x = image.x + image.width * 0.5f;
    const F32 y = image.y + image.height * 0.5f;
    rig.host.send(Mouse(MouseEventType::Click, x, y));
    ClickFocus(rig.host, Extent2D<F32>{x, y});
    rig.host.send(Mouse(MouseEventType::Release, x, y));
    frames.run(update);
    CHECK(Tap(rig.host, KeyCode::Space));
    CHECK(Tap(rig.host, KeyCode::A));
    CHECK(spaces(right) == 2);
    CHECK(spaces(left) == 1);
    const bool letter = std::any_of(right.begin(), right.end(), [](const InputEvent& event) {
        const auto* key = std::get_if<KeyEvent>(&event);
        return key && key->key == KeyCode::A;
    });
    CHECK(letter);
}

TEST_CASE("Freeform Superluminal tiles keep a single selection across header clicks on separate docks",
          "[core][sakura][superluminal][keyboard]") {
    Rig rig;
    const auto texture = std::make_shared<PlotTexture>();
    std::vector<InputEvent> left;
    std::vector<InputEvent> right;
    rig.mosaic = {3, 1};
    rig.tiles = {Plot("Left", {0, 0}, {1, 1}, texture, nullptr, &left),
                 Plot("Right", {2, 0}, {1, 1}, texture, nullptr, &right)};
    Frames frames(rig.host, kWide, rig.scene.context());
    const auto update = [&] {
        rig.update();
    };
    const auto spaces = [](const std::vector<InputEvent>& events) {
        U64 count = 0;
        for (const auto& event : events) {
            const auto* key = std::get_if<KeyEvent>(&event);
            count += key && key->key == KeyCode::Space && key->type == KeyEventType::Press ? 1 : 0;
        }
        return count;
    };
    REQUIRE(frames.settle(update));
    REQUIRE(rig.view.freeform());
    const F32 header = kBar + kHeader * 0.5f;
    const auto press = [&](F32 x) {
        rig.host.send(Mouse(MouseEventType::Click, x, header));
        rig.host.send(Mouse(MouseEventType::Release, x, header));
        ClickFocus(rig.host, Extent2D<F32>{x, header});
        frames.run(update);
        CHECK(Tap(rig.host, KeyCode::Space));
    };
    press(100.0f);
    CHECK(spaces(left) == 1);
    press(kWide.width - 100.0f);
    CHECK(spaces(right) == 1);
    press(100.0f);
    CHECK(spaces(left) == 2);
    CHECK(spaces(right) == 1);
}

TEST_CASE("Freeform Superluminal tiles cannot be dropped into another tile dock",
          "[core][sakura][superluminal][mosaic]") {
    Rig rig;
    const auto texture = std::make_shared<PlotTexture>();
    rig.mosaic = {3, 1};
    rig.global = [&] {
        CHECK(rig.view.interface().text("Signal Parameters") == Result::SUCCESS);
    };
    rig.tiles = {Plot("Left", {0, 0}, {1, 1}, texture), Plot("Right", {2, 0}, {1, 1}, texture)};
    Frames frames(rig.host, kWide, rig.scene.context());
    const auto update = [&] {
        rig.update();
    };
    REQUIRE(frames.settle(update));
    REQUIRE(rig.view.freeform());
    const auto controls = Peek::Frame(*Find<View::ScrollView>(rig.host).at(0));
    const auto plot = Peek::Frame(*Find<View::Keys>(rig.host).at(1));
    const F32 header = plot.y - kGap - kHeader * 0.5f;
    const F32 x = plot.x + 40.0f;

    rig.host.send(Mouse(MouseEventType::Click, x, header));
    rig.host.send(Mouse(MouseEventType::Move, x + 20.0f, header + 20.0f));
    REQUIRE(rig.host.drag.active);
    const Extent2D<F32> edge = {controls.x + 4.0f, controls.y + controls.height * 0.5f};
    Hover(rig.host, frames, edge.x, edge.y);
    frames.run(update);
    Hover(rig.host, frames, edge.x + 1.0f, edge.y);
    frames.run(update);
    CHECK_FALSE(rig.host.drag.target.has_value());
    Release(rig.host, frames, edge.x + 1.0f, edge.y);
    REQUIRE(frames.settle(update));
    for (auto* text : Find<Sakura::Text>(rig.host)) {
        CHECK(text->config().str != "Loading...");
    }
    CHECK(Find<View::Keys>(rig.host).size() == 2);
}

TEST_CASE("The freeform Superluminal global tile is capped and scrolls when its content overflows",
          "[core][sakura][superluminal][global]") {
    Rig rig;
    rig.global = [&] {
        for (int index = 0; index < 40; ++index) {
            CHECK(rig.view.interface().text(jst::fmt::format("Row {}", index)) == Result::SUCCESS);
        }
    };
    rig.tiles = {Tile("a", {0, 0}, {2, 1}), Tile("b", {2, 0}, {1, 2}), Tile("c", {1, 2}, {2, 1}),
                 Tile("d", {0, 1}, {1, 2}), Tile("e", {1, 1}, {1, 1})};
    for (auto& tile : rig.tiles) {
        tile.callback = [] {};
    }
    Frames frames(rig.host, kWide, rig.scene.context());
    frames.verify = false;
    const auto update = [&] {
        rig.update();
    };
    REQUIRE(frames.settle(update));
    REQUIRE(rig.view.freeform());
    const auto viewports = Find<View::ScrollView>(rig.host);
    REQUIRE(viewports.size() == 6);
    const auto global = Peek::Frame(*viewports[0]);
    CHECK(Peek::Frame(rig.dock()).height <= View::GlobalLimit * kWide.height + 1.0f);
    for (U64 index = 1; index < viewports.size(); ++index) {
        CHECK(Peek::Frame(*viewports[index]).height > 20.0f);
    }
    const auto rows = Find<Sakura::Text>(*viewports[0]);
    REQUIRE_FALSE(rows.empty());
    const F32 before = Peek::Frame(*rows.front()).y;
    rig.host.send(Wheel(global.x + 50.0f, global.y + 50.0f, -1.0f));
    frames.run(update);
    CHECK(Peek::Frame(*rows.front()).y < before);
}

TEST_CASE("A trailing global markdown is fitted at its natural height",
          "[core][sakura][superluminal][global]") {
    Rig rig;
    const auto texture = std::make_shared<PlotTexture>();
    rig.global = [&] {
        CHECK(rig.view.interface().markdown("# Notes\nOne\n\nTwo\n\nThree\n\nFour\n\nFive\n\nSix") == Result::SUCCESS);
    };
    rig.tiles = {Plot("Time", {0, 0}, {1, 1}, texture)};
    Frames frames(rig.host, kWide, rig.scene.context());
    frames.verify = false;
    REQUIRE(frames.settle([&] { rig.update(); }));
    const auto viewport = Peek::Frame(*Find<View::ScrollView>(rig.host).at(0));
    const auto markdown = Peek::Frame(*Find<Sakura::Markdown>(rig.host).at(0));
    CHECK(markdown.height > View::MarkdownMinimum + 40.0f);
    CHECK(viewport.bottom() >= markdown.bottom());
    CHECK(viewport.bottom() - markdown.bottom() <= 2.0f * View::FitSlack);
}

TEST_CASE("A Superluminal plot drag released over another plot ends on the pressed plot",
          "[core][sakura][superluminal][surface]") {
    Rig rig;
    const auto texture = std::make_shared<PlotTexture>();
    std::vector<InputEvent> left;
    std::vector<InputEvent> right;
    rig.tiles = {Plot("Left", {0, 0}, {1, 1}, texture, nullptr, &left),
                 Plot("Right", {1, 0}, {1, 1}, texture, nullptr, &right)};
    Frames frames(rig.host, kWide, rig.scene.context());
    REQUIRE(frames.settle([&] { rig.update(); }));
    const auto count = [](const std::vector<InputEvent>& events, MouseEventType type) {
        return std::count_if(events.begin(), events.end(), [type](const InputEvent& event) {
            const auto* mouse = std::get_if<MouseEvent>(&event);
            return mouse && mouse->type == type;
        });
    };
    const auto images = Find<Sakura::SurfaceImage>(rig.host);
    REQUIRE(images.size() == 2);
    const auto a = Peek::Frame(*images[0]);
    const auto b = Peek::Frame(*images[1]);
    rig.host.send(Mouse(MouseEventType::Click, a.x + 40.0f, a.y + 40.0f));
    rig.host.send(Mouse(MouseEventType::Move, b.x + 40.0f, b.y + 40.0f));
    rig.host.send(Mouse(MouseEventType::Release, b.x + 40.0f, b.y + 40.0f));
    CHECK(count(left, MouseEventType::Release) == 1);
    CHECK(count(right, MouseEventType::Release) == 0);

    left.clear();
    rig.host.send(Mouse(MouseEventType::Move, a.right() + 200.0f, a.y + 40.0f));
    CHECK(count(left, MouseEventType::Move) == 0);
}

TEST_CASE("Selecting overlapping freeform Superluminal tiles picks the one painted on top",
          "[core][sakura][superluminal][keyboard]") {
    Rig rig;
    const auto texture = std::make_shared<PlotTexture>();
    std::vector<std::vector<InputEvent>> inputs(5);
    rig.mosaic = {4, 1};
    const std::vector<std::pair<std::string, U8>> columns = {{"a", 0}, {"b", 1}, {"c", 1}, {"d", 2}, {"e", 3}};
    for (U64 index = 0; index < columns.size(); ++index) {
        rig.tiles.push_back(Plot(columns[index].first, {columns[index].second, 0}, {1, 1}, texture, nullptr,
                                 &inputs[index]));
    }
    Frames frames(rig.host, kWide, rig.scene.context());
    const auto update = [&] {
        rig.update();
    };
    REQUIRE(frames.settle(update));
    REQUIRE(rig.view.freeform());
    const F32 x = kGap + (kWide.width - kGap) / 4.0f * 1.5f;
    const F32 header = kBar + kHeader * 0.5f;
    rig.host.send(Mouse(MouseEventType::Click, x, header));
    rig.host.send(Mouse(MouseEventType::Release, x, header));
    ClickFocus(rig.host, Extent2D<F32>{x, header});
    frames.run(update);
    CHECK(Tap(rig.host, KeyCode::Space));
    const auto spaces = [](const std::vector<InputEvent>& events) {
        return std::count_if(events.begin(), events.end(), [](const InputEvent& event) {
            const auto* key = std::get_if<KeyEvent>(&event);
            return key && key->key == KeyCode::Space && key->type == KeyEventType::Press;
        });
    };
    CHECK(spaces(inputs[2]) == 1);
    CHECK(spaces(inputs[1]) == 0);

    const auto images = Find<Sakura::SurfaceImage>(rig.host);
    REQUIRE(images.size() == 5);
    const auto body = Peek::Frame(*images[2]);
    rig.host.send(Mouse(MouseEventType::Click, body.x + 20.0f, body.y + 20.0f));
    rig.host.send(Mouse(MouseEventType::Release, body.x + 20.0f, body.y + 20.0f));
    const auto clicks = [](const std::vector<InputEvent>& events) {
        return std::count_if(events.begin(), events.end(), [](const InputEvent& event) {
            const auto* mouse = std::get_if<MouseEvent>(&event);
            return mouse && mouse->type == MouseEventType::Click;
        });
    };
    CHECK(clicks(inputs[2]) == 1);
    CHECK(clicks(inputs[1]) == 0);
}

TEST_CASE("A user box named like an internal pane id lays out once without a Controls tile",
          "[core][sakura][superluminal][global]") {
    Rig rig;
    const auto texture = std::make_shared<PlotTexture>();
    rig.tiles = {Tile("superluminal:global", {0, 0}, {1, 1}), Tile("global", {1, 0}, {1, 1}),
                 Plot("Time", {2, 0}, {1, 1}, texture)};
    rig.tiles[0].callback = [&] {
        CHECK(rig.view.interface().text("First") == Result::SUCCESS);
    };
    rig.tiles[1].callback = [&] {
        CHECK(rig.view.interface().text("Second") == Result::SUCCESS);
    };
    Frames frames(rig.host, kWide, rig.scene.context());
    REQUIRE(frames.settle([&] { rig.update(); }));
    CHECK(rig.view.tiles() == Row({1.0f, 1.0f, 1.0f}, {Id(0), Id(1), Id(2)}));
    const auto titles = rig.headers();
    CHECK(std::count(titles.begin(), titles.end(), "superluminal:global") == 1);
    CHECK(std::count(titles.begin(), titles.end(), "global") == 1);
    CHECK(std::count(titles.begin(), titles.end(), View::GlobalTitle) == 0);
    CHECK(Find<View::ScrollView>(rig.host).size() == 2);
}

TEST_CASE("Modified Space on a focused Superluminal plot is forwarded once",
          "[core][sakura][superluminal][keyboard]") {
    Rig rig;
    const auto texture = std::make_shared<PlotTexture>();
    std::vector<InputEvent> inputs;
    rig.tiles = {Plot("Time", {0, 0}, {1, 1}, texture, nullptr, &inputs),
                 Plot("Other", {1, 0}, {1, 1}, texture)};
    Frames frames(rig.host, kWide, rig.scene.context());
    const auto update = [&] {
        rig.update();
    };
    REQUIRE(frames.settle(update));
    const auto image = Peek::Frame(*Find<Sakura::SurfaceImage>(rig.host).at(0));
    const F32 x = image.x + image.width * 0.5f;
    const F32 y = image.y + image.height * 0.5f;
    rig.host.send(Mouse(MouseEventType::Click, x, y));
    ClickFocus(rig.host, Extent2D<F32>{x, y});
    rig.host.send(Mouse(MouseEventType::Release, x, y));
    frames.run(update);
    for (const KeyModifiers modifiers : {KeyModifiers{.control = true}, KeyModifiers{.super = true}, KeyModifiers{}}) {
        inputs.clear();
        Tap(rig.host, KeyCode::Space, modifiers);
        const auto presses = std::count_if(inputs.begin(), inputs.end(), [](const InputEvent& event) {
            const auto* key = std::get_if<KeyEvent>(&event);
            return key && key->key == KeyCode::Space && key->type == KeyEventType::Press;
        });
        CHECK(presses == 1);
    }
}

TEST_CASE("An overlapping Superluminal box blocks pointer input to the plot below it",
          "[core][sakura][superluminal][mosaic]") {
    Rig rig;
    const auto texture = std::make_shared<PlotTexture>();
    std::vector<InputEvent> inputs;
    rig.mosaic = {1, 1};
    rig.tiles = {Plot("a", {0, 0}, {1, 1}, texture, nullptr, &inputs), Tile("z", {0, 0}, {1, 1})};
    rig.tiles[1].callback = [&] {
        CHECK(rig.view.interface().text("Covering") == Result::SUCCESS);
    };
    Frames frames(rig.host, kWide, rig.scene.context());
    const auto update = [&] {
        rig.update();
    };
    REQUIRE(frames.settle(update));
    REQUIRE(rig.view.freeform());
    const Extent2D<F32> center = {kWide.width * 0.5f, kWide.height * 0.6f};
    rig.host.send(Mouse(MouseEventType::Move, center.x, center.y));
    rig.host.send(Mouse(MouseEventType::Click, center.x, center.y));
    rig.host.send(Mouse(MouseEventType::Release, center.x, center.y));
    rig.host.send(Wheel(center.x, center.y, -1.0f));
    rig.host.send(Mouse(MouseEventType::Click, center.x, center.y, MouseButton::Right));
    rig.host.send(Mouse(MouseEventType::Release, center.x, center.y, MouseButton::Right));
    frames.run(update);
    const bool reached = std::any_of(inputs.begin(), inputs.end(), [](const InputEvent& event) {
        const auto* mouse = std::get_if<MouseEvent>(&event);
        return mouse && mouse->type != MouseEventType::Leave;
    });
    CHECK_FALSE(reached);
    CHECK(Frames::Access::cursor(rig.host, center) != SurfaceCursor::ResizeAll);
}

TEST_CASE("A chorded press keeps the free layout capture until every button is released",
          "[core][sakura][superluminal][mosaic]") {
    Rig rig;
    const auto texture = std::make_shared<PlotTexture>();
    std::vector<InputEvent> inputs;
    rig.mosaic = {2, 1};
    rig.tiles = {Plot("Time", {0, 0}, {1, 1}, texture, nullptr, &inputs)};
    Frames frames(rig.host, kWide, rig.scene.context());
    REQUIRE(frames.settle([&] { rig.update(); }));
    REQUIRE(rig.view.freeform());
    const auto image = Peek::Frame(*Find<Sakura::SurfaceImage>(rig.host).at(0));
    const F32 x = image.x + image.width * 0.5f;
    const F32 y = image.y + image.height * 0.5f;
    rig.host.send(Mouse(MouseEventType::Click, x, y));
    rig.host.send(Mouse(MouseEventType::Click, x, y, MouseButton::Middle));
    rig.host.send(Mouse(MouseEventType::Release, x, y, MouseButton::Middle));
    inputs.clear();
    const F32 empty = kWide.width * 0.75f;
    rig.host.send(Mouse(MouseEventType::Move, empty, y));
    const auto moved = std::find_if(inputs.begin(), inputs.end(), [](const InputEvent& event) {
        const auto* mouse = std::get_if<MouseEvent>(&event);
        return mouse && mouse->type == MouseEventType::Move;
    });
    REQUIRE(moved != inputs.end());
    CHECK(std::get<MouseEvent>(*moved).position.x > 1.0f);
    CHECK(std::get<MouseEvent>(*moved).position.x < 3.0f);

    inputs.clear();
    rig.host.send(Mouse(MouseEventType::Release, empty, y));
    const bool released = std::any_of(inputs.begin(), inputs.end(), [](const InputEvent& event) {
        const auto* mouse = std::get_if<MouseEvent>(&event);
        return mouse && mouse->type == MouseEventType::Release && mouse->button == MouseButton::Left;
    });
    CHECK(released);
}

TEST_CASE("A slider edit stays with its slider when an earlier slider disappears",
          "[core][sakura][superluminal][interaction]") {
    Rig rig;
    bool showA = true;
    F32 a = 1.0f;
    F32 b = 2.0f;
    rig.tiles = {Tile("Controls", {0, 0}, {1, 1})};
    rig.tiles[0].callback = [&] {
        if (showA) {
            CHECK(rig.view.interface().slider("A", 0.0f, 10.0f, a, false) == Result::SUCCESS);
        }
        CHECK(rig.view.interface().slider("B", 0.0f, 10.0f, b, false) == Result::SUCCESS);
    };
    Frames frames(rig.host, kWide, rig.scene.context());
    frames.verify = false;
    const auto update = [&] {
        rig.update();
    };
    REQUIRE(frames.settle(update));
    const auto rect = Peek::Frame(*Find<Sakura::NodeRangeInput>(rig.host).at(0));
    rig.host.send(Mouse(MouseEventType::Click, rect.x + 10.0f, rect.y + rect.height * 0.5f, MouseButton::Left, 2));
    rig.host.send(Mouse(MouseEventType::Release, rect.x + 10.0f, rect.y + rect.height * 0.5f));
    frames.run(update);
    frames.run(update);
    REQUIRE(FocusOwner(rig.host) != nullptr);
    Type(rig.host, "9");
    frames.run(update);

    showA = false;
    frames.run(update);
    frames.run(update);
    Tap(rig.host, KeyCode::Enter);
    frames.run(update);
    frames.run(update);
    CHECK(b == 2.0f);
}

TEST_CASE("A slider hidden mid press does not keep dragging when it is shown again",
          "[core][sakura][superluminal][interaction]") {
    Rig rig;
    bool shown = true;
    F32 gain = 2.0f;
    rig.tiles = {Tile("Controls", {0, 0}, {1, 1})};
    rig.tiles[0].callback = [&] {
        if (shown) {
            CHECK(rig.view.interface().slider("Gain", 0.0f, 10.0f, gain, false) == Result::SUCCESS);
        }
    };
    Frames frames(rig.host, kWide, rig.scene.context());
    frames.verify = false;
    const auto update = [&] {
        rig.update();
    };
    REQUIRE(frames.settle(update));
    const auto rect = Peek::Frame(*Find<Sakura::NodeRangeInput>(rig.host).at(0));
    const F32 y = rect.y + rect.height * 0.5f;
    rig.host.send(Mouse(MouseEventType::Click, rect.x + rect.width * 0.2f, y));
    frames.run(update);
    CHECK(gain == Catch::Approx(2.0f));

    shown = false;
    frames.run(update);
    rig.host.send(Mouse(MouseEventType::Release, rect.x + rect.width * 0.2f, y));
    frames.run(update);
    shown = true;
    frames.run(update);
    frames.run(update);
    rig.host.send(Mouse(MouseEventType::Move, rect.x + rect.width * 0.8f, y));
    frames.run(update);
    frames.run(update);
    CHECK(gain == Catch::Approx(2.0f));
}

TEST_CASE("Dragging the second of two same label sliders writes only the second variable",
          "[core][sakura][superluminal][interaction]") {
    Rig rig;
    F32 first = 1.0f;
    F32 second = 1.0f;
    rig.tiles = {Tile("Controls", {0, 0}, {1, 1})};
    rig.tiles[0].callback = [&] {
        CHECK(rig.view.interface().slider("Gain", 0.0f, 10.0f, first, false) == Result::SUCCESS);
        CHECK(rig.view.interface().slider("Gain", 0.0f, 10.0f, second, false) == Result::SUCCESS);
    };
    Frames frames(rig.host, kWide, rig.scene.context());
    frames.verify = false;
    const auto update = [&] {
        rig.update();
    };
    REQUIRE(frames.settle(update));
    const auto ranges = Find<Sakura::NodeRangeInput>(rig.host);
    REQUIRE(ranges.size() == 2);
    const auto rect = Peek::Frame(*ranges[1]);
    rig.host.send(Mouse(MouseEventType::Click, rect.x + rect.width * 0.75f, rect.y + rect.height * 0.5f));
    rig.host.send(Mouse(MouseEventType::Release, rect.x + rect.width * 0.75f, rect.y + rect.height * 0.5f));
    frames.run(update);
    CHECK(first == 1.0f);
    CHECK(second == Catch::Approx(7.5f));
}
