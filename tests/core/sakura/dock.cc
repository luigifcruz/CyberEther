#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <jetstream/render/base/texture.hh>
#include <jetstream/render/sakura/components/dock.hh>
#include <jetstream/render/sakura/components/image.hh>
#include <jetstream/render/sakura/components/label.hh>
#include <jetstream/render/sakura/components/layer.hh>
#include <jetstream/render/sakura/components/spinner.hh>
#include <jetstream/render/sakura/components/stack.hh>
#include <jetstream/render/sakura/components/polyline.hh>
#include <jetstream/render/sakura/components/surface_image.hh>
#include <jetstream/render/sakura/components/text.hh>

#include <jetstream/render/tools/icons_lucide.hh>

#include "harness.hh"
#include "fonts.hh"
#include "component_harness.hh"
#include "scene.hh"
#include "render/sakura/metrics.hh"
#include "render/sakura/themes.hh"

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <unordered_map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

using namespace Jetstream;
using namespace SakuraTest;

template <>
struct Catch::StringMaker<Sakura::DockTiles> {
    static std::string convert(const Sakura::DockTiles& tiles) {
        if (tiles.children.empty()) {
            return tiles.pane + (tiles.minimized ? "*" : "");
        }
        std::string text = tiles.vertical ? "column(" : "row(";
        for (U64 index = 0; index < tiles.children.size(); ++index) {
            text += (index > 0 ? ", " : "") + convert(tiles.children[index]) + ":" +
                    (index < tiles.sizes.size() ? std::to_string(tiles.sizes[index]) : std::string("?"));
        }
        return text + ")";
    }
};

template <>
struct Catch::StringMaker<std::optional<Sakura::DockTiles>> {
    static std::string convert(const std::optional<Sakura::DockTiles>& tiles) {
        return tiles ? Catch::StringMaker<Sakura::DockTiles>::convert(*tiles) : std::string("none");
    }
};

namespace {

using Dock = Sakura::Dock;
using DockTiles = Sakura::DockTiles;
using DockDrag = Sakura::DockDrag;
using Layer = Sakura::Layer;

constexpr F32 kBar = 32.0f;
constexpr F32 kHeader = 20.0f;

Rect Body(const Rect& card) {
    return {card.x, card.y + kHeader, card.width, card.height - kHeader};
}

Extent2D<F32> CloseOf(const Rect& card) {
    return {card.right() - 15.0f, card.y + 15.0f};
}

Extent2D<F32> MaximizeOf(const Rect& card) {
    return {card.right() - 37.0f, card.y + 15.0f};
}

Extent2D<F32> MinimizeOf(const Rect& card) {
    return {card.right() - 59.0f, card.y + 15.0f};
}

struct Emissions {
    std::vector<std::optional<DockTiles>> trees;
    std::vector<std::string>* log = nullptr;

    std::function<void(std::optional<DockTiles>)> sink() {
        return [this](std::optional<DockTiles> tiles) {
            trees.push_back(std::move(tiles));
            if (log) {
                log->push_back("tiles");
            }
        };
    }

    const std::optional<DockTiles>& last() const {
        return trees.back();
    }
};

void CheckSizes(const std::vector<F32>& actual, const std::vector<F32>& expected) {
    REQUIRE(actual.size() == expected.size());
    for (U64 index = 0; index < expected.size(); ++index) {
        CHECK(actual[index] == Catch::Approx(expected[index]).margin(0.01f));
    }
}

void CheckApprox(const DockTiles& actual, const DockTiles& expected) {
    CHECK(actual.pane == expected.pane);
    CHECK(actual.vertical == expected.vertical);
    REQUIRE(actual.sizes.size() == expected.sizes.size());
    for (U64 index = 0; index < expected.sizes.size(); ++index) {
        CHECK(actual.sizes[index] == Catch::Approx(expected.sizes[index]));
    }
    REQUIRE(actual.children.size() == expected.children.size());
    for (U64 index = 0; index < expected.children.size(); ++index) {
        CheckApprox(actual.children[index], expected.children[index]);
    }
}

struct CursorDock : Dock {
    using Dock::cursorAt;
};

struct Columns : Sakura::Component {
    Columns(Sakura::Component& left, Sakura::Component& right) : left(left), right(right) {
        addChild(left);
        addChild(right);
    }

 protected:
    void layout(const Sakura::Context& ctx) override {
        const Rect bounds = frame();
        const F32 half = bounds.width * 0.5f;
        layoutChild(ctx, left, {bounds.x, bounds.y, half, bounds.height});
        layoutChild(ctx, right, {bounds.x + half, bounds.y, bounds.width - half, bounds.height});
    }

 private:
    Sakura::Component& left;
    Sakura::Component& right;
};

Sakura::Context MouseAt(F32 x, F32 y) {
    auto ctx = MakeContext();
    ctx.mouse = Extent2D<F32>{x, y};
    return ctx;
}

class SurfaceTexture final : public Render::Texture {
 public:
    SurfaceTexture() : Texture(Config{.size = {64, 64}}) {}

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

bool ButtonGlyph(const Sakura::Text& text) {
    const auto& str = text.config().str;
    return str == ICON_LC_X || str == ICON_LC_MAXIMIZE_2 ||
           str == ICON_LC_MINIMIZE_2 || str == ICON_LC_DOT;
}

const Sakura::Text* GlyphIn(Sakura::Component& dock, const Rect& card, const std::string& icon) {
    for (auto* text : Direct<Sakura::Text>(dock)) {
        const auto frame = Peek::Frame(*text);
        if (text->config().str == icon &&
            card.contains(frame.x + frame.width * 0.5f, frame.y + frame.height * 0.5f)) {
            return text;
        }
    }
    return nullptr;
}

std::vector<Rect> TextFrames(Sakura::Component& dock) {
    std::vector<Rect> frames;
    for (auto* text : Direct<Sakura::Text>(dock)) {
        if (!ButtonGlyph(*text)) {
            frames.push_back(Peek::Frame(*text));
        }
    }
    std::sort(frames.begin(), frames.end(), [](const auto& a, const auto& b) {
        return a.y == b.y ? a.x < b.x : a.y < b.y;
    });
    return frames;
}

const Sakura::Box::Config* BoxWith(Sakura::Component& dock, const Rect& rect) {
    for (auto* box : Find<Sakura::Box>(dock)) {
        for (const auto& instance : box->config().instances) {
            if (instance.rect == rect) {
                return &box->config();
            }
        }
    }
    return nullptr;
}

}

TEST_CASE("Tab mode lays out only the active pane below the tab bar", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    Probe b;
    Dock dock;
    Host host(dock);
    dock.update({.tabs = true, .panes = {MakePane("a", &a), MakePane("b", &b)}});
    host.place(ctx, kScreen);

    CHECK(a.layouts == 0);
    REQUIRE(b.layouts == 1);
    CHECK(b.laidFrame == Rect{0.0f, kBar, 800.0f, 600.0f - kBar});
    CHECK(b.laidClip == Rect{0.0f, kBar, 800.0f, 600.0f - kBar});
}

TEST_CASE("Tab mode can hide its tabs while keeping the bar and the active pane", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    Probe b;
    Dock dock;
    Host host(dock);
    dock.update({.tabs = true, .panes = {MakePane("a", &a), MakePane("b", &b)}});
    host.place(ctx, kScreen);
    const auto shown = TextFrames(dock).size();
    REQUIRE(shown >= 2);
    const Rect first = TextFrames(dock).front();
    const Rect body = b.laidFrame;
    CHECK(BoxWith(dock, body) != nullptr);

    dock.update({.tabs = true, .hideTabs = true, .panes = {MakePane("a", &a), MakePane("b", &b)}});
    host.place(ctx, kScreen);
    CHECK(TextFrames(dock).size() < shown);
    CHECK(b.laidFrame == Rect{0.0f, kBar, 800.0f, 600.0f - kBar});
    CHECK(BoxWith(dock, body) == nullptr);
    host.click(first.x + 2.0f, first.y + first.height * 0.5f);
    host.place(ctx, kScreen);
    CHECK(a.layouts == 0);
}

TEST_CASE("Tab mode marks an unsaved pane with a dot after its title without restructuring",
          "[core][sakura][dock][retention]") {
    Scene scene;
    Probe a;
    Dock dock;
    Host host(dock);
    Frames frames(host, kScreen, scene.context());
    bool unsaved = false;
    const auto update = [&] {
        auto pane = MakePane("a", &a);
        pane.unsaved = unsaved;
        dock.update({.tabs = true, .panes = {pane}});
    };
    const auto find = [&](const std::string& str) -> const Sakura::Text* {
        for (auto* text : Direct<Sakura::Text>(dock)) {
            if (text->config().str == str) {
                return text;
            }
        }
        return nullptr;
    };
    REQUIRE(frames.settle(update));
    const auto* dot = find(ICON_LC_DOT);
    const auto* title = find("a");
    REQUIRE(dot);
    REQUIRE(title);
    CHECK(dot->config().opacity == 0.0f);
    const U64 restructures = frames.restructures;
    const U64 rebuilds = frames.rebuilds;

    unsaved = true;
    CHECK(RepaintOnly(frames.run(update)));
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
    CHECK(dot->config().opacity == 1.0f);
    CHECK(dot->config().colorKey == "cyber_blue");
    const Rect titleFrame = Peek::Frame(*title);
    const Rect dotFrame = Peek::Frame(*dot);
    CHECK(dotFrame.x + dotFrame.width * 0.5f > titleFrame.x + titleFrame.width);
    CHECK(dotFrame.y + dotFrame.height * 0.5f == Catch::Approx(titleFrame.y + titleFrame.height * 0.5f).margin(1.0f));

    unsaved = false;
    CHECK(RepaintOnly(frames.run(update)));
    REQUIRE(frames.settle(update));
    CHECK(dot->config().opacity == 0.0f);
    CHECK(frames.restructures == restructures);
    CHECK(frames.rebuilds == rebuilds);
}

TEST_CASE("Tab mode places an aside on the right that shares one gutter with the content and draws above it",
          "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    const F32 gap = Dock::ContentPadding;
    Probe b;
    Probe aside({150.0f, 10.0f});
    Dock dock;
    Host host(dock);
    const auto update = [&](Sakura::Component* slot, F32 width, bool hideTabs) {
        dock.update({.tabs = true, .hideTabs = hideTabs, .panes = {MakePane("b", &b)}, .aside = slot, .asideWidth = width});
        host.place(ctx, kScreen);
    };

    update(&aside, 200.0f, false);
    const Rect area = {800.0f - 200.0f - gap, kBar, 200.0f + gap, 600.0f - kBar};
    CHECK(aside.laidFrame == area);
    CHECK(aside.laidClip == area);
    CHECK(b.laidFrame == Rect{0.0f, kBar, 600.0f, 600.0f - kBar});
    CHECK(BoxWith(dock, area) != nullptr);
    host.click(area.x + gap * 0.5f, 300.0f);
    CHECK(aside.count(MouseEventType::Click) == 1);
    CHECK(b.count(MouseEventType::Click) == 0);

    update(&aside, 200.0f, true);
    CHECK(aside.laidFrame == area);
    CHECK(BoxWith(dock, area) == nullptr);

    update(&aside, 0.0f, false);
    CHECK(aside.laidFrame == Rect{800.0f - 150.0f - gap, kBar, 150.0f + gap, 600.0f - kBar});
    CHECK(b.laidFrame.width == 650.0f);

    update(&aside, 1000.0f, false);
    CHECK(aside.laidFrame == Rect{400.0f, kBar, 400.0f, 600.0f - kBar});
    CHECK(b.laidFrame.width == 400.0f + gap);
    update(&aside, Sakura::Fill, false);
    CHECK(aside.laidFrame == Rect{400.0f, kBar, 400.0f, 600.0f - kBar});

    update(nullptr, 200.0f, false);
    CHECK(b.laidFrame == Rect{0.0f, kBar, 800.0f, 600.0f - kBar});
    const U64 clicks = aside.count(MouseEventType::Click);
    host.click(700.0f, 300.0f);
    CHECK(aside.count(MouseEventType::Click) == clicks);
    CHECK(b.count(MouseEventType::Click) == 1);
}

TEST_CASE("An aside settles after it is mounted and keeps its structure when its width changes",
          "[core][sakura][dock][retention]") {
    auto ctx = MakeContext();
    ctx.palette = DistinctPalette();
    Probe b;
    Probe aside({150.0f, 10.0f});
    Dock dock;
    Host host(dock);
    Frames frames(host, kScreen, ctx);
    F32 width = 200.0f;
    const auto update = [&] {
        dock.update({.tabs = true, .panes = {MakePane("b", &b)}, .aside = &aside, .asideWidth = width});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));

    const U64 restructures = frames.restructures;
    width = 240.0f;
    REQUIRE(frames.settle(update));
    CHECK(aside.laidFrame.width == 240.0f + Dock::ContentPadding);
    CHECK(frames.restructures == restructures);
    CHECK(Retains(frames, update));
}

TEST_CASE("Toggling an aside parks it without rebuilding or restructuring", "[core][sakura][dock][retention]") {
    auto ctx = MakeContext();
    ctx.palette = DistinctPalette();
    Probe b;
    Probe aside({150.0f, 10.0f});
    Dock dock;
    Host host(dock);
    Frames frames(host, kScreen, ctx);
    bool shown = true;
    const auto update = [&] {
        dock.update({.tabs = true, .panes = {MakePane("b", &b)}, .aside = shown ? &aside : nullptr, .asideWidth = 200.0f});
    };
    REQUIRE(frames.settle(update));
    REQUIRE(aside.visible());
    const U64 restructures = frames.restructures;
    const U64 rebuilds = frames.rebuilds;

    for (int toggle = 0; toggle < 4; ++toggle) {
        shown = !shown;
        const auto stats = frames.run(update);
        CHECK(Kept(stats));
        REQUIRE(frames.settle(update));
        CHECK(Retains(frames, update));
        CHECK(aside.visible() == shown);
        const auto mounted = Peek::Children(dock);
        CHECK(std::find(mounted.begin(), mounted.end(), &aside) != mounted.end());
        CHECK(b.laidFrame.width == (shown ? 600.0f : 800.0f));
    }
    CHECK(frames.restructures == restructures);
    CHECK(frames.rebuilds == rebuilds);

    shown = false;
    REQUIRE(frames.settle(update));
    const U64 clicks = aside.count(MouseEventType::Click);
    host.click(700.0f, 300.0f);
    CHECK(aside.count(MouseEventType::Click) == clicks);
}

TEST_CASE("Docks can drop tile headers and give the whole tile to the content", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    Probe b;
    Dock dock;
    Host host(dock);
    dock.update({.panes = {MakePane("a", &a), MakePane("b", &b)}, .columns = true, .flexPane = "b"});
    host.place(ctx, kScreen);
    const auto titled = a.laidFrame;
    const auto labels = TextFrames(dock).size();
    REQUIRE(labels > 0);

    dock.update({.panes = {MakePane("a", &a), MakePane("b", &b)}, .columns = true, .flexPane = "b", .headers = false});
    host.place(ctx, kScreen);
    CHECK(a.laidFrame.y < titled.y);
    CHECK(a.laidFrame.height > titled.height);
    CHECK(TextFrames(dock).size() < labels);
}

TEST_CASE("Docks without headers keep a pane that starts minimized open because nothing could restore it",
          "[core][sakura][dock][columns]") {
    const auto ctx = MakeContext();
    Probe a;
    Probe notes;
    Dock dock;
    Host host(dock);
    const auto update = [&](bool headers) {
        auto pane = MakePane("notes", &notes);
        pane.leftmost = true;
        pane.startMinimized = true;
        dock.update({.panes = {MakePane("a", &a), pane}, .columns = true, .flexPane = "a", .headers = headers});
        host.place(ctx, kScreen);
    };

    update(false);
    CHECK(notes.visible());
    CHECK(notes.laidFrame.width > 32.0f);
    CHECK(notes.laidFrame.x < a.laidFrame.x);
    const Extent2D<F32> center = {notes.laidFrame.x + notes.laidFrame.width * 0.5f,
                                  notes.laidFrame.y + notes.laidFrame.height * 0.5f};
    CHECK_FALSE(Peek::CursorAt(dock, center) == SurfaceCursor::Hand);

    update(true);
    CHECK(notes.visible());
}

TEST_CASE("Dock panes without a card keep their slot and bleed through the gaps around it", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    Probe b;
    Dock dock;
    Host host(dock);
    auto bare = MakePane("b", &b);
    dock.update({.panes = {MakePane("a", &a), bare}, .columns = true, .flexPane = "b", .headers = false});
    host.place(ctx, kScreen);
    const auto carded = b.laidFrame;

    bare.card = false;
    dock.update({.panes = {MakePane("a", &a), bare}, .columns = true, .flexPane = "b", .headers = false});
    host.place(ctx, kScreen);
    const F32 gap = Dock::ContentPadding;
    CHECK(b.laidFrame.x == carded.x - gap);
    CHECK(b.laidFrame.y == carded.y - gap);
    CHECK(b.laidFrame.right() == carded.right() + gap);
    CHECK(b.laidFrame.bottom() == carded.bottom() + gap);
    CHECK(b.laidFrame.right() == kScreen.right());
    CHECK(b.laidFrame.bottom() == kScreen.bottom());
    CHECK(a.laidFrame.x > kScreen.x);
    CHECK(b.laidFrame.x == a.laidFrame.right());
}

TEST_CASE("Dock panes see the tile backdrop only when they draw a card", "[core][sakura][dock]") {
    auto ctx = MakeContext();
    ctx.backdropColorKey = "background";
    BackdropProbe a;
    BackdropProbe b;
    Dock dock;
    Host host(dock);
    auto bare = MakePane("b", &b);
    bare.card = false;
    dock.update({.panes = {MakePane("a", &a), bare}, .columns = true, .flexPane = "b", .headers = false});
    host.place(ctx, kScreen);
    CHECK(a.backdrop == "tile_bg");
    CHECK(b.backdrop == "background");
}

TEST_CASE("Dock panes with a fill tint their whole tile under the header and the frame",
          "[core][sakura][dock]") {
    constexpr ColorRGBA<F32> kTint = {0.1f, 0.2f, 0.9f, 1.0f};
    Probe a({0.0f, 0.0f});
    Probe b({0.0f, 0.0f});
    Dock dock;
    Host host(dock);
    auto tinted = MakePane("b", &b);
    tinted.fill = kTint;
    const auto tinting = [&](const Dock::Pane& second) {
        dock.update({.panes = {MakePane("a", &a), second}, .columns = true});
        host.place(MakeContext(), kScreen);
        std::vector<Sakura::Box::Instance> found;
        for (const auto* box : Find<Sakura::Box>(dock)) {
            for (const auto& instance : box->config().instances) {
                if (instance.backgroundColor == kTint) {
                    found.push_back(instance);
                }
            }
        }
        return found;
    };

    CHECK(tinting(MakePane("b", &b)).empty());
    const auto found = tinting(tinted);
    REQUIRE(found.size() == 1);
    CHECK(found[0].rect.y < Peek::Frame(b).y);
    CHECK(found[0].rect.x <= Peek::Frame(b).x);
    CHECK(found[0].rect.right() >= Peek::Frame(b).right());
    CHECK(found[0].rect.bottom() >= Peek::Frame(b).bottom());
    CHECK(found[0].rect.x > Peek::Frame(a).right());
}

TEST_CASE("Turning dock headers off restores a maximized tile", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    Probe b;
    Probe c;
    Dock dock;
    Host host(dock);
    const auto tiles = Row({1000.0f, 1000.0f, 1000.0f}, {Leaf("a"), Leaf("b"), Leaf("c")});
    const auto update = [&](bool headers) {
        dock.update({.panes = {MakePane("a", &a), MakePane("b", &b), MakePane("c", &c)}, .tiles = tiles,
                     .headers = headers});
        host.place(ctx, kScreen);
    };
    update(true);
    const Rect cardB = {271.0f, 6.0f, 258.0f, 588.0f};
    REQUIRE(b.laidFrame == Body(cardB));
    const auto maximize = MaximizeOf(cardB);
    host.click(maximize.x, maximize.y);
    update(true);
    REQUIRE_FALSE(a.visible());

    update(false);
    CHECK(a.visible());
    CHECK(b.visible());
    CHECK(c.visible());
    CHECK(a.laidFrame.right() < b.laidFrame.x);
    CHECK(b.laidFrame.right() < c.laidFrame.x);

    update(true);
    CHECK(a.visible());
    CHECK(b.laidFrame == Body(cardB));
}

TEST_CASE("Turning resizing off during a splitter drag ends the drag once", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Emissions emissions;
    CursorDock dock;
    Host host(dock);
    const auto update = [&](bool resizable) {
        dock.update({.panes = {MakePane("a"), MakePane("b"), MakePane("c")},
                     .tiles = Row({500.0f, 250.0f, 250.0f}, {Leaf("a"), Leaf("b"), Leaf("c")}),
                     .onTiles = emissions.sink(),
                     .resizable = resizable});
        host.place(ctx, kScreen);
    };
    update(true);
    CHECK(host.send(Mouse(MouseEventType::Click, 597.0f, 300.0f)));
    CHECK(host.send(Mouse(MouseEventType::Move, 498.5f, 300.0f)));
    REQUIRE(dock.resizing());

    update(false);
    CHECK_FALSE(dock.resizing());
    CHECK_FALSE(dock.cursorAt({498.5f, 300.0f}) == SurfaceCursor::ResizeEW);
    REQUIRE(emissions.trees.size() == 1);
    REQUIRE(emissions.last().has_value());
    CheckApprox(*emissions.last(), Row({388.0f, 95.5f, 292.5f}, {Leaf("a"), Leaf("b"), Leaf("c")}));

    host.send(Mouse(MouseEventType::Move, 300.0f, 300.0f));
    host.send(Mouse(MouseEventType::Release, 300.0f, 300.0f));
    update(false);
    CHECK_FALSE(dock.resizing());
    CHECK(emissions.trees.size() == 1);
}

TEST_CASE("Tab mode presents only the active pane's surface and swaps hints when the tab changes",
          "[core][sakura][dock]") {
    SakuraTest::FakeWindow window;
    auto ctx = MakeContext();
    ctx.render = &window;
    const auto first = std::make_shared<SurfaceTexture>();
    const auto second = std::make_shared<SurfaceTexture>();
    Sakura::SurfaceImage a;
    Sakura::SurfaceImage b;
    Dock dock;
    Host host(dock);
    Frames frames(host, kScreen, ctx);
    std::string active = "b";
    const auto update = [&] {
        a.update({.texture = first});
        b.update({.texture = second});
        dock.update({.tabs = true, .panes = {MakePane("a", &a), MakePane("b", &b)}, .activate = active});
    };
    const auto check = [&](bool firstShown) {
        for (int frame = 0; frame < 3; ++frame) {
            frames.run(update);
            const U64 count = frames.ctx.frame;
            CHECK(first->visibleForPresentation(count) == firstShown);
            CHECK(second->visibleForPresentation(count) == !firstShown);
        }
    };

    CHECK(frames.run(update).rebuilt);
    const U64 count = frames.ctx.frame;
    CHECK_FALSE(first->visibleForPresentation(count));
    CHECK(second->visibleForPresentation(count));
    check(false);

    active = "a";
    check(true);
    active = "b";
    check(false);
    active = "a";
    check(true);
}

TEST_CASE("Tab mode adopts inactive panes hidden without extra restructures or rebuilds",
          "[core][sakura][dock]") {
    Probe a;
    Probe b;
    Probe c;
    Dock dock;
    Host host(dock);
    Frames frames(host);
    std::string active = "c";
    const auto update = [&] {
        dock.update({.tabs = true, .panes = {MakePane("a", &a), MakePane("b", &b), MakePane("c", &c)},
                     .activate = active});
    };

    CHECK(frames.run(update).rebuilt);
    CHECK(Peek::Children(dock).size() >= 3);
    CHECK_FALSE(a.visible());
    CHECK_FALSE(b.visible());
    CHECK(c.visible());
    REQUIRE(frames.settle(update));
    CHECK(frames.rebuilds == 1);
    CHECK(frames.restructures == 0);
    for (int frame = 0; frame < 4; ++frame) {
        CHECK(frames.run(update).clean());
    }
    CHECK(frames.restructures == 0);

    active = "a";
    CHECK_FALSE(frames.run(update).rebuilt);
    REQUIRE(frames.settle(update));
    CHECK(frames.rebuilds == 1);
    CHECK(a.visible());
    CHECK(a.layouts > 0);
    CHECK_FALSE(c.visible());
    for (int frame = 0; frame < 4; ++frame) {
        CHECK(frames.run(update).clean());
    }
}

TEST_CASE("Tab mode appends new panes and makes the newest one active", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    Probe b;
    Probe c;
    Dock dock;
    Host host(dock);

    dock.update({.tabs = true, .panes = {MakePane("a", &a)}});
    host.place(ctx, kScreen);
    CHECK(a.layouts == 1);

    dock.update({.tabs = true, .panes = {MakePane("a", &a), MakePane("b", &b)}});
    host.place(ctx, kScreen);
    CHECK(a.layouts == 1);
    CHECK(b.layouts == 1);

    dock.update({.tabs = true, .panes = {MakePane("c", &c), MakePane("a", &a), MakePane("b", &b)}});
    host.place(ctx, kScreen);
    CHECK(b.layouts == 1);
    CHECK(c.layouts == 1);
}

TEST_CASE("Tab mode keeps the active pane when the same panes are sent again", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    Probe b;
    Dock dock;
    Host host(dock);
    const Dock::Config config = {.tabs = true, .panes = {MakePane("a", &a), MakePane("b", &b)}};

    dock.update(config);
    host.place(ctx, kScreen);
    dock.update(config);
    host.place(ctx, kScreen);
    dock.update(config);
    host.place(ctx, kScreen);

    CHECK_FALSE(a.visible());
    CHECK(b.visible());
}

TEST_CASE("Tab mode activate selects an existing pane and ignores a missing one", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    Probe b;
    Dock dock;
    Host host(dock);

    dock.update({.tabs = true, .panes = {MakePane("a", &a), MakePane("b", &b)}, .activate = "a"});
    host.place(ctx, kScreen);
    CHECK(a.visible());
    CHECK_FALSE(b.visible());

    dock.update({.tabs = true, .panes = {MakePane("a", &a), MakePane("b", &b)}, .activate = "missing"});
    host.place(ctx, kScreen);
    CHECK(a.visible());
    CHECK_FALSE(b.visible());

    dock.update({.tabs = true, .panes = {MakePane("a", &a), MakePane("b", &b)}});
    host.place(ctx, kScreen);
    CHECK(a.visible());
    CHECK_FALSE(b.visible());
}

TEST_CASE("Tab mode activate wins over a newly appended pane", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    Probe b;
    Dock dock;
    Host host(dock);

    dock.update({.tabs = true, .panes = {MakePane("a", &a)}});
    dock.update({.tabs = true, .panes = {MakePane("a", &a), MakePane("b", &b)}, .activate = "a"});
    host.place(ctx, kScreen);

    CHECK(a.layouts == 1);
    CHECK(b.layouts == 0);
}

TEST_CASE("Tab mode drops removed panes and keeps the active pane selected", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    Probe b;
    Probe c;
    Dock dock;
    Host host(dock);

    dock.update({.tabs = true, .panes = {MakePane("a", &a), MakePane("b", &b), MakePane("c", &c)}});
    dock.update({.tabs = true, .panes = {MakePane("b", &b), MakePane("c", &c)}});
    host.place(ctx, kScreen);
    CHECK(c.layouts == 1);
    CHECK(b.layouts == 0);

    dock.update({.tabs = true, .panes = {MakePane("b", &b)}});
    host.place(ctx, kScreen);
    CHECK(c.layouts == 1);
    CHECK(b.layouts == 1);

    dock.update({.tabs = true});
    host.place(ctx, kScreen);
    CHECK(a.layouts == 0);
    CHECK(b.layouts == 1);
    CHECK(c.layouts == 1);
}

TEST_CASE("Tab mode activates a clicked tab on release and focuses its pane", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    Probe b;
    U64 focusA = 0;
    U64 focusB = 0;
    Dock dock;
    Host host(dock);
    auto paneA = MakePane("a", &a);
    paneA.onFocus = [&] { ++focusA; };
    auto paneB = MakePane("b", &b);
    paneB.onFocus = [&] { ++focusB; };
    dock.update({.tabs = true, .panes = {paneA, paneB}});
    host.place(ctx, kScreen);
    REQUIRE(b.visible());

    CHECK(host.send(Mouse(MouseEventType::Click, 10.0f, 12.0f)));
    host.place(ctx, kScreen);
    CHECK_FALSE(a.visible());
    CHECK(focusA == 0);

    CHECK(host.send(Mouse(MouseEventType::Release, 10.0f, 12.0f)));
    host.place(ctx, kScreen);
    CHECK(a.visible());
    CHECK_FALSE(b.visible());
    CHECK(a.laidFrame == Rect{0.0f, kBar, 800.0f, 600.0f - kBar});
    CHECK(focusA == 1);
    CHECK(focusB == 0);

    host.click(10.0f, 12.0f);
    CHECK(focusA == 1);
}

TEST_CASE("Tab mode close button calls onClose only when released over it", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    U64 closed = 0;
    Dock dock;
    Host host(dock);
    auto pane = MakePane("a", &a);
    pane.onClose = [&] { ++closed; };
    dock.update({.tabs = true, .panes = {pane}});
    host.place(ctx, kScreen);

    host.send(Mouse(MouseEventType::Click, 233.0f, 20.0f));
    host.send(Mouse(MouseEventType::Release, 200.0f, 300.0f));
    CHECK(closed == 0);

    host.click(233.0f, 20.0f);
    CHECK(closed == 1);

    host.place(ctx, kScreen);
    CHECK(a.visible());
}

TEST_CASE("Tab mode non-closable tabs have no close button", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    Probe b;
    U64 closed = 0;
    U64 focused = 0;
    Dock dock;
    Host host(dock);
    auto paneA = MakePane("a", &a);
    paneA.closable = false;
    paneA.onClose = [&] { ++closed; };
    paneA.onFocus = [&] { ++focused; };
    dock.update({.tabs = true, .panes = {paneA, MakePane("b", &b)}});
    host.place(ctx, kScreen);

    host.click(12.0f, 12.0f);
    CHECK(closed == 0);
    CHECK(focused == 1);

    host.click(372.0f, 20.0f);
    host.place(ctx, kScreen);
    CHECK(b.visible());
}

TEST_CASE("Tab mode insets the active content by its padding and clips it to the body",
          "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    Dock dock;
    Host host(dock);
    auto pane = MakePane("a", &a);
    pane.padding = 8.0f;
    dock.update({.tabs = true, .panes = {pane}});
    host.place(ctx, kScreen);

    CHECK(a.laidFrame == Rect{8.0f, kBar + 8.0f, 784.0f, 600.0f - kBar - 16.0f});
    CHECK(a.laidClip == Rect{0.0f, kBar, 800.0f, 600.0f - kBar});
    CHECK(a.laidInherited == a.laidClip);
}

TEST_CASE("Tab mode never shows headers even for a single closable pane", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    U64 closed = 0;
    Dock dock;
    Host host(dock);
    auto pane = MakePane("a", &a);
    pane.onClose = [&] { ++closed; };
    dock.update({.tabs = true, .panes = {pane}});
    host.place(ctx, kScreen);

    CHECK(a.laidFrame.y == kBar);
    host.click(29.0f, kBar + 12.0f);
    CHECK(closed == 0);
    CHECK(a.count(MouseEventType::Click) == 1);
}

TEST_CASE("Tab mode never emits onTiles", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Emissions emissions;
    Dock dock;
    Host host(dock);
    dock.update({.tabs = true, .panes = {MakePane("a"), MakePane("b")}, .onTiles = emissions.sink()});
    host.place(ctx, kScreen);
    host.click(10.0f, 12.0f);
    host.click(29.0f, 12.0f);
    dock.update({.tabs = true, .panes = {MakePane("b")}, .onTiles = emissions.sink()});
    host.place(ctx, kScreen);

    CHECK(emissions.trees.empty());
}

TEST_CASE("Tab mode scales the tab bar and padding with the pixel ratio", "[core][sakura][dock]") {
    auto ctx = MakeContext();
    ctx.pixelRatio = 2.0f;
    Probe a;
    Dock dock;
    Host host(dock);
    auto pane = MakePane("a", &a);
    pane.padding = 4.0f;
    dock.update({.tabs = true, .panes = {pane}});
    host.place(ctx, kScreen);

    CHECK(a.laidFrame == Rect{8.0f, 72.0f, 784.0f, 520.0f});
    CHECK(a.laidClip == Rect{0.0f, 64.0f, 800.0f, 536.0f});
}

TEST_CASE("Dragging a tab inside the tab bar reorders it", "[core][sakura][dock]") {
    Probe a;
    Probe b;
    Probe c;
    U64 focusB = 0;
    Dock dock;
    Layer layer;
    layer.update({.content = &dock});
    LayerHost host({&layer});
    auto paneB = MakePane("b", &b);
    paneB.onFocus = [&] { ++focusB; };
    const Dock::Config config = {
        .tabs = true,
        .accepts = "panel",
        .panes = {MakePane("a", &a), paneB, MakePane("c", &c)},
    };
    dock.update(config);
    host.place(MakeContext(), kScreen);

    host.send(Mouse(MouseEventType::Click, 10.0f, 12.0f));
    host.send(Mouse(MouseEventType::Move, 700.0f, 12.0f));
    REQUIRE(host.drag.active);
    CHECK(host.drag.pane == "a");
    CHECK(host.drag.group == "panel");
    CHECK(host.drag.title == "a");

    host.place(MouseAt(700.0f, 12.0f), kScreen);
    REQUIRE(host.drag.target.has_value());
    CHECK(host.drag.target->preview == Rect{6.0f, kBar + 6.0f, 788.0f, 600.0f - kBar - 12.0f});

    host.send(Mouse(MouseEventType::Release, 700.0f, 12.0f));
    CHECK_FALSE(host.drag.active);

    const U64 before = a.layouts;
    dock.update(config);
    host.place(MakeContext(), kScreen);
    CHECK(a.layouts == before + 1);

    host.click(10.0f, 12.0f);
    CHECK(focusB == 1);
    host.place(MakeContext(), kScreen);
    CHECK(b.layouts == 1);
}

TEST_CASE("Tab drops outside the tab bar are not offered", "[core][sakura][dock]") {
    Probe a;
    Probe b;
    Dock dock;
    Layer layer;
    layer.update({.content = &dock});
    LayerHost host({&layer});
    dock.update({.tabs = true, .accepts = "panel", .panes = {MakePane("a", &a), MakePane("b", &b)}});
    host.place(MakeContext(), kScreen);

    host.send(Mouse(MouseEventType::Click, 10.0f, 12.0f));
    host.send(Mouse(MouseEventType::Move, 300.0f, 300.0f));
    REQUIRE(host.drag.active);
    host.place(MouseAt(300.0f, 300.0f), kScreen);
    CHECK_FALSE(host.drag.target.has_value());

    host.send(Mouse(MouseEventType::Release, 300.0f, 300.0f));
    host.place(MakeContext(), kScreen);
    CHECK(a.layouts == 0);
    CHECK(b.layouts == 3);
}

TEST_CASE("Tab presses that move less than the threshold do not start a drag", "[core][sakura][dock]") {
    Probe a;
    Probe b;
    Dock dock;
    Layer layer;
    layer.update({.content = &dock});
    LayerHost host({&layer});
    dock.update({.tabs = true, .accepts = "panel", .panes = {MakePane("a", &a), MakePane("b", &b)}});
    host.place(MakeContext(), kScreen);

    host.send(Mouse(MouseEventType::Click, 10.0f, 12.0f));
    host.send(Mouse(MouseEventType::Move, 14.0f, 15.0f));
    CHECK_FALSE(host.drag.active);
    host.send(Mouse(MouseEventType::Release, 14.0f, 15.0f));
    host.place(MakeContext(), kScreen);
    CHECK(a.layouts == 1);
}

TEST_CASE("Tile mode appends panes into a horizontal row of unsized equal shares and reports each tree",
          "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Emissions emissions;
    Dock dock;
    Host host(dock);

    dock.update({.panes = {MakePane("a")}, .onTiles = emissions.sink()});
    REQUIRE(emissions.trees.size() == 1);
    CHECK(emissions.last() == Leaf("a"));

    dock.update({.panes = {MakePane("a"), MakePane("b")}, .tiles = emissions.last(), .onTiles = emissions.sink()});
    REQUIRE(emissions.trees.size() == 2);
    CHECK(emissions.last() == Row({0.0f, 0.0f}, {Leaf("a"), Leaf("b")}));

    dock.update({.panes = {MakePane("a"), MakePane("b"), MakePane("c")},
                 .tiles = emissions.last(),
                 .onTiles = emissions.sink()});
    REQUIRE(emissions.trees.size() == 3);
    CHECK(emissions.last() == Row({0.0f, 0.0f, 0.0f}, {Leaf("a"), Leaf("b"), Leaf("c")}));
}

TEST_CASE("Tile mode appends to a vertical tree by wrapping it in a new row", "[core][sakura][dock]") {
    Emissions emissions;
    Dock dock;
    const auto column = Column({500.0f, 500.0f}, {Leaf("a"), Leaf("b")});
    dock.update({.panes = {MakePane("a"), MakePane("b"), MakePane("c")}, .tiles = column, .onTiles = emissions.sink()});

    REQUIRE(emissions.trees.size() == 1);
    CHECK(emissions.last() == Row({0.0f, 0.0f}, {column, Leaf("c")}));
}

TEST_CASE("Tile mode adopts owner tiles without emitting them back", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    Probe b;
    Emissions emissions;
    Dock dock;
    Host host(dock);
    const auto tiles = Row({300.0f, 700.0f}, {Leaf("a"), Leaf("b")});
    dock.update({.panes = {MakePane("a", &a), MakePane("b", &b)}, .tiles = tiles, .onTiles = emissions.sink()});
    host.place(ctx, kScreen);

    CHECK(emissions.trees.empty());
    CHECK(a.laidFrame == Body({6.0f, 6.0f, 235.0f, 588.0f}));
    CHECK(b.laidFrame == Body({247.0f, 6.0f, 547.0f, 588.0f}));
}

TEST_CASE("Re-sending the emitted tiles does not emit again", "[core][sakura][dock]") {
    Emissions emissions;
    Dock dock;
    const std::vector<Dock::Pane> panes = {MakePane("a"), MakePane("b")};
    dock.update({.panes = panes, .onTiles = emissions.sink()});
    REQUIRE(emissions.trees.size() == 1);
    const auto tiles = emissions.last();

    dock.update({.panes = panes, .tiles = tiles, .onTiles = emissions.sink()});
    dock.update({.panes = panes, .tiles = tiles, .onTiles = emissions.sink()});
    CHECK(emissions.trees.size() == 1);
}

TEST_CASE("Tile mode replaces its tree when the owner sends different tiles", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    Probe b;
    Emissions emissions;
    Dock dock;
    Host host(dock);
    const std::vector<Dock::Pane> panes = {MakePane("a", &a), MakePane("b", &b)};
    dock.update({.panes = panes, .tiles = Row({500.0f, 500.0f}, {Leaf("a"), Leaf("b")}), .onTiles = emissions.sink()});
    host.place(ctx, kScreen);
    CHECK(a.laidFrame.x == 6.0f);

    dock.update({.panes = panes, .tiles = Column({500.0f, 500.0f}, {Leaf("b"), Leaf("a")}), .onTiles = emissions.sink()});
    host.place(ctx, kScreen);
    CHECK(emissions.trees.empty());
    CHECK(b.laidFrame == Body({6.0f, 6.0f, 788.0f, 291.0f}));
    CHECK(a.laidFrame == Body({6.0f, 303.0f, 788.0f, 291.0f}));
}

TEST_CASE("Tile mode flattens nested rows along the same axis and keeps the nested row's extent",
          "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    Probe b;
    Probe c;
    Emissions emissions;
    Dock dock;
    Host host(dock);
    const auto nested = Row({500.0f, 500.0f}, {Leaf("a"), Row({500.0f, 500.0f}, {Leaf("b"), Leaf("c")})});
    dock.update({.panes = {MakePane("a", &a), MakePane("b", &b), MakePane("c", &c)},
                 .tiles = nested,
                 .onTiles = emissions.sink()});
    host.place(ctx, kScreen);

    CHECK(emissions.trees.empty());
    CHECK(a.laidFrame == Body({6.0f, 6.0f, 390.0f, 588.0f}));
    CHECK(b.laidFrame == Body({402.0f, 6.0f, 193.0f, 588.0f}));
    CHECK(c.laidFrame == Body({601.0f, 6.0f, 193.0f, 588.0f}));

    host.click(399.0f, 300.0f);
    REQUIRE(emissions.trees.size() == 1);
    CHECK(emissions.last() == Row({500.0f, 247.0f, 247.0f}, {Leaf("a"), Leaf("b"), Leaf("c")}));
}

TEST_CASE("Tile mode keeps nested splits along the other axis", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    Probe b;
    Probe c;
    Emissions emissions;
    Dock dock;
    Host host(dock);
    const auto tiles = Row({500.0f, 500.0f}, {Leaf("a"), Column({500.0f, 500.0f}, {Leaf("b"), Leaf("c")})});
    dock.update({.panes = {MakePane("a", &a), MakePane("b", &b), MakePane("c", &c)},
                 .tiles = tiles,
                 .onTiles = emissions.sink()});
    host.place(ctx, kScreen);

    CHECK(a.laidFrame == Body({6.0f, 6.0f, 391.0f, 588.0f}));
    CHECK(b.laidFrame == Body({403.0f, 6.0f, 391.0f, 291.0f}));
    CHECK(c.laidFrame == Body({403.0f, 303.0f, 391.0f, 291.0f}));

    host.click(400.0f, 100.0f);
    REQUIRE(emissions.trees.size() == 1);
    CHECK(emissions.last() == tiles);
}

TEST_CASE("Tile mode lays out owner sizes proportionally and reports them unchanged", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    Emissions emissions;
    Dock dock;
    Host host(dock);
    dock.update({.panes = {MakePane("a", &a), MakePane("b")},
                 .tiles = Row({3.0f, 1.0f}, {Leaf("a"), Leaf("b")}),
                 .onTiles = emissions.sink()});
    host.place(ctx, kScreen);
    CHECK(a.laidFrame.width == Catch::Approx((788.0f - 6.0f) * 0.75f).margin(1.0f));
    host.click(596.0f, 300.0f);

    REQUIRE(emissions.trees.size() == 1);
    CHECK(emissions.last() == Row({3.0f, 1.0f}, {Leaf("a"), Leaf("b")}));
}

TEST_CASE("Tile mode drops owner tiles whose sizes do not match their children, or leaves with sizes, and appends its panes instead",
          "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Emissions emissions;
    Dock dock;
    Host host(dock);
    dock.update({.panes = {MakePane("a"), MakePane("b")},
                 .tiles = Row({}, {Leaf("b"), Leaf("a")}),
                 .onTiles = emissions.sink()});
    REQUIRE(emissions.trees.size() == 1);
    CHECK(emissions.last() == Row({0.0f, 0.0f}, {Leaf("a"), Leaf("b")}));

    dock.update({.panes = {MakePane("a"), MakePane("b")},
                 .tiles = Row({300.0f, -1.0f}, {Leaf("b"), Leaf("a")}),
                 .onTiles = emissions.sink()});
    REQUIRE(emissions.trees.size() == 2);
    CHECK(emissions.last() == Row({0.0f, 0.0f}, {Leaf("a"), Leaf("b")}));

    dock.update({.panes = {MakePane("a"), MakePane("b")},
                 .tiles = DockTiles{.pane = "b", .sizes = {300.0f}},
                 .onTiles = emissions.sink()});
    REQUIRE(emissions.trees.size() == 3);
    CHECK(emissions.last() == Row({0.0f, 0.0f}, {Leaf("a"), Leaf("b")}));

    auto sized = Leaf("b");
    sized.sizes = {300.0f};
    dock.update({.panes = {MakePane("a"), MakePane("b")},
                 .tiles = Row({300.0f, 300.0f}, {sized, Leaf("a")}),
                 .onTiles = emissions.sink()});
    REQUIRE(emissions.trees.size() == 4);
    CHECK(emissions.last() == Row({0.0f, 0.0f}, {Leaf("a"), Leaf("b")}));
}

TEST_CASE("Unknown leaves keep their place and are described for their header", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    Probe ghost;
    Emissions emissions;
    std::vector<std::string> described;
    Dock dock;
    Host host(dock);
    const auto tiles = Row({500.0f, 500.0f}, {Leaf("a"), Leaf("ghost")});
    const auto describe = [&](const std::string& id) {
        described.push_back(id);
        return "Ghost";
    };
    dock.update({.panes = {MakePane("a", &a)}, .tiles = tiles, .describe = describe, .onTiles = emissions.sink()});
    host.place(ctx, kScreen);

    CHECK(emissions.trees.empty());
    CHECK(described == std::vector<std::string>{"ghost"});
    CHECK(a.laidFrame == Body({6.0f, 6.0f, 391.0f, 588.0f}));

    dock.update({.panes = {MakePane("a", &a), MakePane("ghost", &ghost)},
                 .tiles = tiles,
                 .describe = describe,
                 .onTiles = emissions.sink()});
    host.place(ctx, kScreen);
    CHECK(emissions.trees.empty());
    CHECK(ghost.laidFrame == Body({403.0f, 6.0f, 391.0f, 588.0f}));
}

TEST_CASE("A single non-closable pane has no header", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    Dock dock;
    Host host(dock);
    auto pane = MakePane("a", &a);
    pane.closable = false;
    dock.update({.panes = {pane}});
    host.place(ctx, kScreen);

    CHECK(a.laidFrame == Rect{6.0f, 6.0f, 788.0f, 588.0f});
    CHECK(a.laidClip == Rect{6.0f, 6.0f, 788.0f, 588.0f});
}

TEST_CASE("A single non-closable pane keeps its header when lone headers are requested", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    Dock dock;
    Host host(dock);
    auto pane = MakePane("a", &a);
    pane.closable = false;
    dock.update({.panes = {pane}, .loneHeader = true});
    host.place(ctx, kScreen);

    CHECK(a.laidFrame == Rect{6.0f, 26.0f, 788.0f, 568.0f});
    bool titled = false;
    for (auto* text : Direct<Sakura::Text>(dock)) {
        titled = titled || text->config().str == "a";
    }
    CHECK(titled);
}

TEST_CASE("A single closable pane keeps its header", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    Dock dock;
    Host host(dock);
    dock.update({.panes = {MakePane("a", &a)}});
    host.place(ctx, kScreen);

    CHECK(a.laidFrame == Body({6.0f, 6.0f, 788.0f, 588.0f}));
}

TEST_CASE("Several non-closable panes all get headers", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    Probe b;
    Dock dock;
    Host host(dock);
    auto paneA = MakePane("a", &a);
    paneA.closable = false;
    auto paneB = MakePane("b", &b);
    paneB.closable = false;
    dock.update({.panes = {paneA, paneB}});
    host.place(ctx, kScreen);

    CHECK(a.laidFrame == Body({6.0f, 6.0f, 391.0f, 588.0f}));
    CHECK(b.laidFrame == Body({403.0f, 6.0f, 391.0f, 588.0f}));
}

TEST_CASE("Leaf content is inset by padding and clipped to its own leaf body", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    Probe b;
    Dock dock;
    Host host(dock);
    auto paneA = MakePane("a", &a);
    paneA.padding = 6.0f;
    auto paneB = MakePane("b", &b);
    paneB.padding = 6.0f;
    dock.update({.panes = {paneA, paneB}});
    host.place(ctx, kScreen);

    CHECK(a.laidFrame == Rect{12.0f, 32.0f, 379.0f, 556.0f});
    CHECK(a.laidClip == Body({6.0f, 6.0f, 391.0f, 588.0f}));
    CHECK(b.laidFrame == Rect{409.0f, 32.0f, 379.0f, 556.0f});
    CHECK(b.laidClip == Body({403.0f, 6.0f, 391.0f, 588.0f}));
    CHECK(b.laidInherited == b.laidClip);
}

TEST_CASE("Leaf geometry follows the dock origin", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    Probe b;
    Dock dock;
    Host host(dock);
    dock.update({.panes = {MakePane("a", &a), MakePane("b", &b)}});
    host.place(ctx, {100.0f, 50.0f, 800.0f, 600.0f});

    CHECK(a.laidFrame == Body({106.0f, 56.0f, 391.0f, 588.0f}));
    CHECK(b.laidFrame == Body({503.0f, 56.0f, 391.0f, 588.0f}));
    CHECK(b.laidClip == b.laidFrame);
}

TEST_CASE("Closing a header removes the leaf, emits the tree, then calls onClose", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    Probe b;
    std::vector<std::string> log;
    Emissions emissions;
    emissions.log = &log;
    Dock dock;
    Host host(dock);
    auto paneB = MakePane("b", &b);
    paneB.onClose = [&] { log.push_back("close:b"); };
    dock.update({.panes = {MakePane("a", &a), paneB},
                 .tiles = Row({500.0f, 500.0f}, {Leaf("a"), Leaf("b")}),
                 .onTiles = emissions.sink()});
    host.place(ctx, kScreen);

    const auto closeB = CloseOf({403.0f, 6.0f, 391.0f, 588.0f});
    host.click(closeB.x, closeB.y);
    REQUIRE(emissions.trees.size() == 1);
    CHECK(emissions.last() == Leaf("a"));
    CHECK(log == std::vector<std::string>{"tiles", "close:b"});

    dock.update({.panes = {MakePane("a", &a)}, .tiles = emissions.last(), .onTiles = emissions.sink()});
    host.place(ctx, kScreen);
    CHECK(emissions.trees.size() == 1);
    CHECK(a.laidFrame == Body({6.0f, 6.0f, 788.0f, 588.0f}));
}

TEST_CASE("Closing a header keeps the stored sizes of the remaining tiles", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Emissions emissions;
    Dock dock;
    Host host(dock);
    dock.update({.panes = {MakePane("a"), MakePane("b"), MakePane("c")},
                 .tiles = Row({500.0f, 250.0f, 250.0f}, {Leaf("a"), Leaf("b"), Leaf("c")}),
                 .onTiles = emissions.sink()});
    host.place(ctx, kScreen);

    const auto closeA = CloseOf({6.0f, 6.0f, 388.0f, 588.0f});
    host.click(closeA.x, closeA.y);
    REQUIRE(emissions.trees.size() == 1);
    CHECK(emissions.last() == Row({250.0f, 250.0f}, {Leaf("b"), Leaf("c")}));
}

TEST_CASE("Closing a tile inside a nested split collapses its parent", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Emissions emissions;
    Dock dock;
    Host host(dock);
    dock.update({.panes = {MakePane("a"), MakePane("b"), MakePane("c")},
                 .tiles = Row({400.0f, 600.0f}, {Leaf("a"), Column({500.0f, 500.0f}, {Leaf("b"), Leaf("c")})}),
                 .onTiles = emissions.sink()});
    host.place(ctx, kScreen);

    const auto closeB = CloseOf({325.0f, 6.0f, 469.0f, 291.0f});
    host.click(closeB.x, closeB.y);
    REQUIRE(emissions.trees.size() == 1);
    REQUIRE(emissions.last().has_value());
    CheckApprox(*emissions.last(), Row({400.0f, 600.0f}, {Leaf("a"), Leaf("c")}));
}

TEST_CASE("Closing the last tile emits an empty tree", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    U64 closed = 0;
    Emissions emissions;
    Dock dock;
    Host host(dock);
    auto pane = MakePane("a", &a);
    pane.onClose = [&] { ++closed; };
    dock.update({.panes = {pane}, .tiles = Leaf("a"), .onTiles = emissions.sink()});
    host.place(ctx, kScreen);

    const auto close = CloseOf({6.0f, 6.0f, 788.0f, 588.0f});
    host.click(close.x, close.y);
    REQUIRE(emissions.trees.size() == 1);
    CHECK_FALSE(emissions.last().has_value());
    CHECK(closed == 1);

    dock.update({.tiles = emissions.last(), .onTiles = emissions.sink()});
    host.place(ctx, kScreen);
    CHECK(emissions.trees.size() == 1);
    CHECK(a.layouts == 1);
}

TEST_CASE("Releasing a header close away from the button keeps the tile", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    U64 closed = 0;
    Emissions emissions;
    Dock dock;
    Host host(dock);
    auto paneB = MakePane("b");
    paneB.onClose = [&] { ++closed; };
    dock.update({.panes = {MakePane("a"), paneB},
                 .tiles = Row({500.0f, 500.0f}, {Leaf("a"), Leaf("b")}),
                 .onTiles = emissions.sink()});
    host.place(ctx, kScreen);

    const auto close = CloseOf({403.0f, 6.0f, 391.0f, 588.0f});
    host.send(Mouse(MouseEventType::Click, close.x, close.y));
    host.send(Mouse(MouseEventType::Release, 600.0f, close.y));
    CHECK(emissions.trees.empty());
    CHECK(closed == 0);
}

TEST_CASE("Non-closable tile headers have no close button", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    U64 closed = 0;
    Emissions emissions;
    Dock dock;
    Host host(dock);
    auto paneB = MakePane("b");
    paneB.closable = false;
    paneB.maximizable = false;
    paneB.onClose = [&] { ++closed; };
    dock.update({.panes = {MakePane("a"), paneB},
                 .tiles = Row({500.0f, 500.0f}, {Leaf("a"), Leaf("b")}),
                 .onTiles = emissions.sink()});
    host.place(ctx, kScreen);

    const auto close = CloseOf({403.0f, 6.0f, 391.0f, 588.0f});
    host.click(close.x, close.y);
    CHECK(emissions.trees.empty());
    CHECK(closed == 0);
}

TEST_CASE("Dragging a splitter changes only its neighbours, bakes the laid sizes and emits on release",
          "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Emissions emissions;
    Dock dock;
    Host host(dock);
    dock.update({.panes = {MakePane("a"), MakePane("b"), MakePane("c")},
                 .tiles = Row({500.0f, 250.0f, 250.0f}, {Leaf("a"), Leaf("b"), Leaf("c")}),
                 .onTiles = emissions.sink()});
    host.place(ctx, kScreen);

    CHECK(dock.paneAt({100.0f, 300.0f}) == "a");
    CHECK(dock.paneAt({700.0f, 300.0f}) == "c");
    CHECK(dock.paneAt({597.0f, 300.0f}).empty());
    CHECK_FALSE(dock.resizing());
    CHECK(host.send(Mouse(MouseEventType::Click, 597.0f, 300.0f)));
    CHECK(dock.resizing());
    CHECK(host.send(Mouse(MouseEventType::Move, 498.5f, 300.0f)));
    CHECK(emissions.trees.empty());

    CHECK(host.send(Mouse(MouseEventType::Release, 498.5f, 300.0f)));
    CHECK_FALSE(dock.resizing());
    REQUIRE(emissions.trees.size() == 1);
    REQUIRE(emissions.last().has_value());
    CheckApprox(*emissions.last(), Row({388.0f, 95.5f, 292.5f}, {Leaf("a"), Leaf("b"), Leaf("c")}));
}

TEST_CASE("Double clicking a splitter evens out its two neighbours and leaves the rest alone",
          "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Emissions emissions;
    Dock dock;
    Host host(dock);
    dock.update({.panes = {MakePane("a"), MakePane("b"), MakePane("c")},
                 .tiles = Row({500.0f, 125.0f, 375.0f}, {Leaf("a"), Leaf("b"), Leaf("c")}),
                 .onTiles = emissions.sink()});
    host.place(ctx, kScreen);

    CHECK(host.send(Mouse(MouseEventType::Click, 498.5f, 300.0f, MouseButton::Left, 2)));
    CHECK(host.send(Mouse(MouseEventType::Release, 498.5f, 300.0f)));
    REQUIRE(emissions.trees.size() == 1);
    REQUIRE(emissions.last().has_value());
    CheckApprox(*emissions.last(), Row({388.0f, 194.0f, 194.0f}, {Leaf("a"), Leaf("b"), Leaf("c")}));
}

TEST_CASE("Splitter drags keep a minimum share for each neighbour", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Emissions emissions;
    Dock dock;
    Host host(dock);
    dock.update({.panes = {MakePane("a"), MakePane("b")},
                 .tiles = Row({500.0f, 500.0f}, {Leaf("a"), Leaf("b")}),
                 .onTiles = emissions.sink()});
    host.place(ctx, kScreen);

    host.send(Mouse(MouseEventType::Click, 400.0f, 300.0f));
    host.send(Mouse(MouseEventType::Move, -50.0f, 300.0f));
    host.send(Mouse(MouseEventType::Release, -50.0f, 300.0f));
    REQUIRE(emissions.trees.size() == 1);
    REQUIRE(emissions.last().has_value());
    CheckApprox(*emissions.last(), Row({48.0f, 734.0f}, {Leaf("a"), Leaf("b")}));

    dock.update({.panes = {MakePane("a"), MakePane("b")}, .tiles = emissions.last(), .onTiles = emissions.sink()});
    host.place(ctx, kScreen);
    const F32 splitter = 6.0f + 48.0f + 3.0f;
    host.send(Mouse(MouseEventType::Click, splitter, 300.0f));
    host.send(Mouse(MouseEventType::Move, 900.0f, 300.0f));
    host.send(Mouse(MouseEventType::Release, 900.0f, 300.0f));
    REQUIRE(emissions.trees.size() == 2);
    REQUIRE(emissions.last().has_value());
    CheckApprox(*emissions.last(), Row({734.0f, 48.0f}, {Leaf("a"), Leaf("b")}));
}

TEST_CASE("Vertical splitters resize along the y axis", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    Probe b;
    Emissions emissions;
    Dock dock;
    Host host(dock);
    dock.update({.panes = {MakePane("a", &a), MakePane("b", &b)},
                 .tiles = Column({500.0f, 500.0f}, {Leaf("a"), Leaf("b")}),
                 .onTiles = emissions.sink()});
    host.place(ctx, kScreen);

    host.send(Mouse(MouseEventType::Click, 400.0f, 300.0f));
    host.send(Mouse(MouseEventType::Move, 400.0f, 153.0f));
    host.place(ctx, kScreen);
    CHECK(a.laidFrame == Body({6.0f, 6.0f, 788.0f, 144.0f}));
    CHECK(b.laidFrame == Body({6.0f, 156.0f, 788.0f, 438.0f}));

    host.send(Mouse(MouseEventType::Release, 400.0f, 153.0f));
    REQUIRE(emissions.trees.size() == 1);
    REQUIRE(emissions.last().has_value());
    CheckApprox(*emissions.last(), Column({144.0f, 438.0f}, {Leaf("a"), Leaf("b")}));
}

TEST_CASE("Splitter cursors follow the split axis", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    CursorDock dock;
    Host host(dock);
    dock.update({.panes = {MakePane("a"), MakePane("b"), MakePane("c")},
                 .tiles = Row({500.0f, 500.0f}, {Leaf("a"), Column({500.0f, 500.0f}, {Leaf("b"), Leaf("c")})})});
    host.place(ctx, kScreen);

    CHECK(dock.cursorAt({400.0f, 100.0f}) == SurfaceCursor::ResizeEW);
    CHECK(dock.cursorAt({600.0f, 300.0f}) == SurfaceCursor::ResizeNS);
    CHECK_FALSE(dock.cursorAt({200.0f, 300.0f}).has_value());
}

TEST_CASE("Docks that are not resizable have no splitter to hit or drag", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    Probe b;
    Emissions emissions;
    CursorDock dock;
    Host host(dock);
    const auto update = [&](bool resizable) {
        dock.update({.panes = {MakePane("a", &a), MakePane("b", &b)},
                     .tiles = Row({500.0f, 500.0f}, {Leaf("a"), Leaf("b")}),
                     .onTiles = emissions.sink(),
                     .resizable = resizable});
        host.place(ctx, kScreen);
    };
    update(false);
    const auto left = a.laidFrame;
    const auto right = b.laidFrame;
    CHECK_FALSE(dock.cursorAt({400.0f, 300.0f}).has_value());
    host.send(Mouse(MouseEventType::Click, 400.0f, 300.0f));
    CHECK_FALSE(dock.resizing());
    host.send(Mouse(MouseEventType::Move, 200.0f, 300.0f));
    host.send(Mouse(MouseEventType::Release, 200.0f, 300.0f));
    host.place(ctx, kScreen);
    CHECK_FALSE(dock.resizing());
    CHECK(emissions.trees.empty());
    CHECK(a.laidFrame == left);
    CHECK(b.laidFrame == right);

    update(true);
    CHECK(dock.cursorAt({400.0f, 300.0f}) == SurfaceCursor::ResizeEW);
    CHECK(a.laidFrame == left);
    CHECK(b.laidFrame == right);
}

TEST_CASE("Clicking a header or body focuses its pane once per change", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    Probe b;
    U64 focusA = 0;
    U64 focusB = 0;
    Dock dock;
    Host host(dock);
    auto paneA = MakePane("a", &a);
    paneA.onFocus = [&] { ++focusA; };
    auto paneB = MakePane("b", &b);
    paneB.onFocus = [&] { ++focusB; };
    dock.update({.panes = {paneA, paneB}});
    host.place(ctx, kScreen);

    host.click(200.0f, 300.0f);
    CHECK(focusA == 1);
    CHECK(a.count(MouseEventType::Click) == 1);
    CHECK(b.count(MouseEventType::Click) == 0);

    host.click(200.0f, 400.0f);
    CHECK(focusA == 1);

    host.click(600.0f, 12.0f);
    CHECK(focusB == 1);
    CHECK(b.count(MouseEventType::Click) == 0);

    host.click(600.0f, 300.0f);
    CHECK(focusB == 1);
    CHECK(b.count(MouseEventType::Click) == 1);

    host.click(200.0f, 12.0f);
    CHECK(focusA == 2);
    CHECK(focusB == 1);
}

TEST_CASE("Dragging a header past the threshold starts a dock drag with the pane group",
          "[core][sakura][dock]") {
    Dock dock;
    Layer layer;
    layer.update({.content = &dock});
    LayerHost host({&layer});
    auto paneA = MakePane("a");
    paneA.title = "Alpha";
    paneA.group = "scope";
    dock.update({.accepts = "scope", .panes = {paneA, MakePane("b")}});
    host.place(MakeContext(), kScreen);

    host.send(Mouse(MouseEventType::Click, 200.0f, 12.0f));
    host.send(Mouse(MouseEventType::Move, 203.0f, 14.0f));
    CHECK_FALSE(host.drag.active);

    host.send(Mouse(MouseEventType::Move, 210.0f, 14.0f));
    REQUIRE(host.drag.active);
    CHECK(host.drag.pane == "a");
    CHECK(host.drag.group == "scope");
    CHECK(host.drag.title == "Alpha");
    CHECK(host.drag.source == static_cast<const void*>(&dock));
}

TEST_CASE("Header drags offer a split on the nearest edge of another leaf sized from the room left after the drag",
          "[core][sakura][dock]") {
    struct Case {
        Extent2D<F32> point;
        Rect preview;
        DockTiles tiles;
    };
    const auto test = GENERATE(
        Case{{420.0f, 300.0f}, {403.0f, 6.0f, 391.0f, 588.0f}, Row({391.0f, 391.0f}, {Leaf("a"), Leaf("b")})},
        Case{{790.0f, 300.0f}, {403.0f, 6.0f, 391.0f, 588.0f}, Row({391.0f, 391.0f}, {Leaf("b"), Leaf("a")})},
        Case{{600.0f, 10.0f}, {403.0f, 6.0f, 391.0f, 291.0f}, Column({291.0f, 291.0f}, {Leaf("a"), Leaf("b")})},
        Case{{600.0f, 590.0f}, {403.0f, 303.0f, 391.0f, 291.0f}, Column({291.0f, 291.0f}, {Leaf("b"), Leaf("a")})});
    CAPTURE(test.point.x, test.point.y);

    Emissions emissions;
    Dock dock;
    Layer layer;
    layer.update({.content = &dock});
    LayerHost host({&layer});
    dock.update({.accepts = "panel",
                 .panes = {MakePane("a"), MakePane("b")},
                 .tiles = Row({500.0f, 500.0f}, {Leaf("a"), Leaf("b")}),
                 .onTiles = emissions.sink()});
    host.place(MakeContext(), kScreen);

    host.send(Mouse(MouseEventType::Click, 200.0f, 12.0f));
    host.send(Mouse(MouseEventType::Move, test.point.x, test.point.y));
    REQUIRE(host.drag.active);
    host.place(MouseAt(test.point.x, test.point.y), kScreen);
    REQUIRE(host.drag.target.has_value());
    CHECK(host.drag.target->preview == test.preview);
    CHECK(emissions.trees.empty());

    host.send(Mouse(MouseEventType::Release, test.point.x, test.point.y));
    REQUIRE(emissions.trees.size() == 1);
    CHECK(emissions.last() == test.tiles);
    CHECK_FALSE(host.drag.active);
}

TEST_CASE("Splitting along the row axis inserts beside the target and only the target pays",
          "[core][sakura][dock]") {
    Emissions emissions;
    Dock dock;
    Layer layer;
    layer.update({.content = &dock});
    LayerHost host({&layer});
    dock.update({.accepts = "panel",
                 .panes = {MakePane("a"), MakePane("b"), MakePane("c")},
                 .tiles = Row({500.0f, 250.0f, 250.0f}, {Leaf("a"), Leaf("b"), Leaf("c")}),
                 .onTiles = emissions.sink()});
    host.place(MakeContext(), kScreen);

    host.send(Mouse(MouseEventType::Click, 200.0f, 12.0f));
    host.send(Mouse(MouseEventType::Move, 590.0f, 300.0f));
    host.place(MouseAt(590.0f, 300.0f), kScreen);
    REQUIRE(host.drag.target.has_value());
    CHECK(host.drag.target->preview == Rect{402.0f, 6.0f, 192.0f, 588.0f});

    host.send(Mouse(MouseEventType::Release, 590.0f, 300.0f));
    REQUIRE(emissions.trees.size() == 1);
    REQUIRE(emissions.last().has_value());
    CheckApprox(*emissions.last(), Row({192.5f, 192.5f, 391.0f}, {Leaf("b"), Leaf("a"), Leaf("c")}));
}

TEST_CASE("Dragging a header over its own leaf offers no target", "[core][sakura][dock]") {
    Emissions emissions;
    Dock dock;
    Layer layer;
    layer.update({.content = &dock});
    LayerHost host({&layer});
    dock.update({.accepts = "panel",
                 .panes = {MakePane("a"), MakePane("b")},
                 .tiles = Row({500.0f, 500.0f}, {Leaf("a"), Leaf("b")}),
                 .onTiles = emissions.sink()});
    host.place(MakeContext(), kScreen);

    host.send(Mouse(MouseEventType::Click, 200.0f, 12.0f));
    host.send(Mouse(MouseEventType::Move, 20.0f, 300.0f));
    REQUIRE(host.drag.active);
    host.place(MouseAt(20.0f, 300.0f), kScreen);
    CHECK_FALSE(host.drag.target.has_value());

    host.send(Mouse(MouseEventType::Release, 20.0f, 300.0f));
    CHECK(emissions.trees.empty());
}

TEST_CASE("A drag from a group the dock does not accept gets no offer", "[core][sakura][dock]") {
    Emissions emissions;
    Dock dock;
    Layer layer;
    layer.update({.content = &dock});
    LayerHost host({&layer});
    auto paneA = MakePane("a");
    paneA.group = "scope";
    dock.update({.accepts = "panel",
                 .panes = {paneA, MakePane("b")},
                 .tiles = Row({500.0f, 500.0f}, {Leaf("a"), Leaf("b")}),
                 .onTiles = emissions.sink()});
    host.place(MakeContext(), kScreen);

    host.send(Mouse(MouseEventType::Click, 200.0f, 12.0f));
    host.send(Mouse(MouseEventType::Move, 600.0f, 10.0f));
    REQUIRE(host.drag.active);
    CHECK(host.drag.group == "scope");
    host.place(MouseAt(600.0f, 10.0f), kScreen);
    CHECK_FALSE(host.drag.target.has_value());

    host.send(Mouse(MouseEventType::Release, 600.0f, 10.0f));
    CHECK(emissions.trees.empty());
}

TEST_CASE("A drag with the mouse outside the dock gets no offer", "[core][sakura][dock]") {
    Emissions emissions;
    Dock dock;
    Layer layer;
    layer.update({.content = &dock});
    LayerHost host({&layer});
    dock.update({.accepts = "panel",
                 .panes = {MakePane("a"), MakePane("b")},
                 .tiles = Row({500.0f, 500.0f}, {Leaf("a"), Leaf("b")}),
                 .onTiles = emissions.sink()});
    host.place(MakeContext(), kScreen);

    host.send(Mouse(MouseEventType::Click, 200.0f, 12.0f));
    host.send(Mouse(MouseEventType::Move, 600.0f, 10.0f));
    REQUIRE(host.drag.active);
    host.place(MakeContext(), kScreen);
    CHECK_FALSE(host.drag.target.has_value());
    host.place(MouseAt(900.0f, 300.0f), kScreen);
    CHECK_FALSE(host.drag.target.has_value());

    host.send(Mouse(MouseEventType::Release, 900.0f, 300.0f));
    CHECK(emissions.trees.empty());
}

TEST_CASE("Header drags need a dock drag context", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Emissions emissions;
    Dock dock;
    Host host(dock);
    dock.update({.panes = {MakePane("a"), MakePane("b")},
                 .tiles = Row({500.0f, 500.0f}, {Leaf("a"), Leaf("b")}),
                 .onTiles = emissions.sink()});
    host.place(ctx, kScreen);

    CHECK(host.send(Mouse(MouseEventType::Click, 200.0f, 12.0f)));
    CHECK(host.send(Mouse(MouseEventType::Move, 600.0f, 10.0f)));
    CHECK(host.send(Mouse(MouseEventType::Release, 600.0f, 10.0f)));
    CHECK(emissions.trees.empty());
}

TEST_CASE("Leaving the surface cancels an active drag", "[core][sakura][dock]") {
    Emissions emissions;
    Dock dock;
    Layer layer;
    layer.update({.content = &dock});
    LayerHost host({&layer});
    dock.update({.accepts = "panel",
                 .panes = {MakePane("a"), MakePane("b")},
                 .tiles = Row({500.0f, 500.0f}, {Leaf("a"), Leaf("b")}),
                 .onTiles = emissions.sink()});
    host.place(MakeContext(), kScreen);

    host.send(Mouse(MouseEventType::Click, 200.0f, 12.0f));
    host.send(Mouse(MouseEventType::Move, 600.0f, 10.0f));
    host.place(MouseAt(600.0f, 10.0f), kScreen);
    REQUIRE(host.drag.target.has_value());

    host.send(Mouse(MouseEventType::Leave, 600.0f, 10.0f));
    CHECK_FALSE(host.drag.active);
    CHECK_FALSE(host.drag.target.has_value());

    host.send(Mouse(MouseEventType::Move, 610.0f, 10.0f));
    host.send(Mouse(MouseEventType::Release, 610.0f, 10.0f));
    CHECK(emissions.trees.empty());
}

TEST_CASE("Dragging a tile into an empty dock moves it between docks", "[core][sakura][dock]") {
    Emissions source;
    Emissions target;
    Dock left;
    Dock right;
    Columns columns(left, right);
    Layer layer;
    layer.update({.content = &columns});
    LayerHost host({&layer});
    left.update({.accepts = "panel",
                 .panes = {MakePane("a"), MakePane("b")},
                 .tiles = Row({500.0f, 500.0f}, {Leaf("a"), Leaf("b")}),
                 .onTiles = source.sink()});
    right.update({.accepts = "panel", .onTiles = target.sink()});
    host.place(MakeContext(), kScreen);

    host.send(Mouse(MouseEventType::Click, 100.0f, 12.0f));
    host.send(Mouse(MouseEventType::Move, 600.0f, 300.0f));
    REQUIRE(host.drag.active);
    host.place(MouseAt(600.0f, 300.0f), kScreen);
    REQUIRE(host.drag.target.has_value());
    CHECK(host.drag.target->preview == Rect{406.0f, 6.0f, 388.0f, 588.0f});

    host.send(Mouse(MouseEventType::Release, 600.0f, 300.0f));
    REQUIRE(source.trees.size() == 1);
    CHECK(source.last() == Leaf("b"));
    REQUIRE(target.trees.size() == 1);
    CHECK(target.last() == Leaf("a"));
}

TEST_CASE("Dragging a tile into a dock of another group is refused", "[core][sakura][dock]") {
    Emissions source;
    Emissions target;
    Dock left;
    Dock right;
    Columns columns(left, right);
    Layer layer;
    layer.update({.content = &columns});
    LayerHost host({&layer});
    left.update({.accepts = "panel",
                 .panes = {MakePane("a"), MakePane("b")},
                 .tiles = Row({500.0f, 500.0f}, {Leaf("a"), Leaf("b")}),
                 .onTiles = source.sink()});
    right.update({.accepts = "scope", .onTiles = target.sink()});
    host.place(MakeContext(), kScreen);

    host.send(Mouse(MouseEventType::Click, 100.0f, 12.0f));
    host.send(Mouse(MouseEventType::Move, 600.0f, 300.0f));
    host.place(MouseAt(600.0f, 300.0f), kScreen);
    CHECK_FALSE(host.drag.target.has_value());

    host.send(Mouse(MouseEventType::Release, 600.0f, 300.0f));
    CHECK(source.trees.empty());
    CHECK(target.trees.empty());
}

TEST_CASE("A dock adopts drags whose group it lists in adopts and still refuses other groups",
          "[core][sakura][dock]") {
    Emissions source;
    Emissions target;
    Dock left;
    Dock right;
    Columns columns(left, right);
    Layer layer;
    layer.update({.content = &columns});
    LayerHost host({&layer});
    auto paneA = MakePane("a");
    paneA.group = "terminal";
    left.update({.accepts = "panel",
                 .panes = {paneA, MakePane("b")},
                 .tiles = Row({500.0f, 500.0f}, {Leaf("a"), Leaf("b")}),
                 .onTiles = source.sink()});
    right.update({.accepts = "scope", .adopts = {"terminal"}, .onTiles = target.sink()});
    host.place(MakeContext(), kScreen);

    host.send(Mouse(MouseEventType::Click, 300.0f, 12.0f));
    host.send(Mouse(MouseEventType::Move, 600.0f, 300.0f));
    REQUIRE(host.drag.active);
    CHECK(host.drag.pane == "b");
    CHECK(host.drag.group == "panel");
    host.place(MouseAt(600.0f, 300.0f), kScreen);
    CHECK_FALSE(host.drag.target.has_value());
    host.send(Mouse(MouseEventType::Release, 600.0f, 300.0f));
    CHECK(source.trees.empty());
    CHECK(target.trees.empty());

    host.place(MakeContext(), kScreen);
    host.send(Mouse(MouseEventType::Click, 100.0f, 12.0f));
    host.send(Mouse(MouseEventType::Move, 600.0f, 300.0f));
    REQUIRE(host.drag.active);
    CHECK(host.drag.pane == "a");
    CHECK(host.drag.group == "terminal");
    host.place(MouseAt(600.0f, 300.0f), kScreen);
    REQUIRE(host.drag.target.has_value());
    CHECK(host.drag.target->owner == static_cast<const void*>(&right));
    CHECK(host.drag.target->preview == Rect{406.0f, 6.0f, 388.0f, 588.0f});

    host.send(Mouse(MouseEventType::Release, 600.0f, 300.0f));
    REQUIRE(source.trees.size() == 1);
    CHECK(source.last() == Leaf("b"));
    REQUIRE(target.trees.size() == 1);
    CHECK(target.last() == Leaf("a"));
    CHECK_FALSE(host.drag.active);
}

TEST_CASE("A tab bar adopts a tile of a listed group and shows it as a new tab", "[core][sakura][dock]") {
    Probe a;
    Probe b;
    Probe c;
    Emissions source;
    Dock left;
    Dock right;
    Columns columns(left, right);
    Layer layer;
    layer.update({.content = &columns});
    LayerHost host({&layer});
    c.consume = false;
    auto paneA = MakePane("a", &a);
    paneA.group = "terminal";
    left.update({.accepts = "panel",
                 .panes = {paneA, MakePane("b", &b)},
                 .tiles = Row({500.0f, 500.0f}, {Leaf("a"), Leaf("b")}),
                 .onTiles = source.sink()});
    right.update({.tabs = true, .accepts = "main", .adopts = {"terminal"}, .panes = {MakePane("c", &c)}});
    host.place(MakeContext(), kScreen);

    host.send(Mouse(MouseEventType::Click, 100.0f, 12.0f));
    host.send(Mouse(MouseEventType::Move, 700.0f, 12.0f));
    REQUIRE(host.drag.active);
    host.place(MouseAt(700.0f, 12.0f), kScreen);
    REQUIRE(host.drag.target.has_value());
    CHECK(host.drag.target->owner == static_cast<const void*>(&right));

    host.send(Mouse(MouseEventType::Release, 700.0f, 12.0f));
    REQUIRE(source.trees.size() == 1);
    CHECK(source.last() == Leaf("b"));

    auto moved = MakePane("a", &a);
    moved.group = "terminal";
    left.update({.accepts = "panel", .panes = {MakePane("b", &b)}, .tiles = Leaf("b"), .onTiles = source.sink()});
    right.update({.tabs = true, .accepts = "main", .adopts = {"terminal"}, .panes = {MakePane("c", &c), moved}});
    host.place(MakeContext(), kScreen);
    CHECK(a.laidFrame == Rect{400.0f, kBar, 400.0f, 600.0f - kBar});
}

TEST_CASE("Dropping a tab into another tab bar inserts it at the hovered slot and focuses the source's next tab",
          "[core][sakura][dock]") {
    Probe a;
    Probe b;
    Probe c;
    U64 focusA = 0;
    U64 focusB = 0;
    Dock left;
    Dock right;
    Columns columns(left, right);
    Layer layer;
    layer.update({.content = &columns});
    LayerHost host({&layer});
    c.consume = false;
    auto paneB = MakePane("b", &b);
    paneB.onFocus = [&] { ++focusB; };
    left.update({.tabs = true, .accepts = "panel", .panes = {MakePane("a"), paneB}, .activate = "a"});
    right.update({.tabs = true, .accepts = "panel", .panes = {MakePane("c", &c)}});
    host.place(MakeContext(), kScreen);
    CHECK(b.layouts == 0);

    host.send(Mouse(MouseEventType::Click, 10.0f, 12.0f));
    host.send(Mouse(MouseEventType::Move, 405.0f, 12.0f));
    REQUIRE(host.drag.active);
    host.place(MouseAt(405.0f, 12.0f), kScreen);
    REQUIRE(host.drag.target.has_value());
    CHECK(host.drag.target->preview == Rect{406.0f, kBar + 6.0f, 388.0f, 600.0f - kBar - 12.0f});

    REQUIRE(focusB == 0);
    host.send(Mouse(MouseEventType::Release, 405.0f, 12.0f));
    host.place(MakeContext(), kScreen);
    CHECK(b.layouts == 1);
    CHECK(focusB == 1);

    auto paneA = MakePane("a", &a);
    paneA.onFocus = [&] { ++focusA; };
    left.update({.tabs = true, .accepts = "panel", .panes = {MakePane("b", &b)}});
    right.update({.tabs = true, .accepts = "panel", .panes = {MakePane("c", &c), paneA}});
    const U64 before = c.layouts;
    host.place(MakeContext(), kScreen);
    CHECK(a.laidFrame == Rect{400.0f, kBar, 400.0f, 600.0f - kBar});
    CHECK(c.layouts == before);
    CHECK(focusA == 1);
    host.click(503.0f, 20.0f);
    CHECK(focusA == 1);

    host.click(697.0f, 20.0f);
    host.place(MakeContext(), kScreen);
    CHECK(c.layouts == before + 1);
    host.click(503.0f, 20.0f);
    CHECK(focusA == 2);
}

TEST_CASE("Leading chrome sits after the inset and trailing chrome right after the last tab",
          "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    Probe b;
    Probe leading({40.0f, 20.0f});
    Probe trailing({24.0f, 30.0f});
    Dock dock;
    Host host(dock);
    dock.update({.tabs = true, .panes = {MakePane("a", &a), MakePane("b", &b)}, .leading = &leading,
                 .trailing = &trailing, .leadingInset = 70.0f});
    host.place(ctx, kScreen);

    CHECK(leading.laidFrame == Rect{76.0f, 10.0f, 40.0f, 20.0f});
    CHECK(trailing.laidFrame == Rect{616.0f, 5.0f, 24.0f, 30.0f});
    CHECK(b.laidFrame == Rect{0.0f, 32.0f, 800.0f, 568.0f});

    host.click(125.0f, 19.0f);
    host.place(ctx, kScreen);
    CHECK(a.laidFrame == Rect{0.0f, 32.0f, 800.0f, 568.0f});
}

TEST_CASE("The tab bar grows to the requested height and to its tallest chrome", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    Probe leading({40.0f, 20.0f});
    Dock dock;
    Host host(dock);

    dock.update({.tabs = true, .panes = {MakePane("a", &a)}, .leading = &leading, .tabBarHeight = 52.0f});
    host.place(ctx, kScreen);
    CHECK(a.laidFrame == Rect{0.0f, 46.0f, 800.0f, 554.0f});
    CHECK(leading.laidFrame == Rect{6.0f, 17.0f, 40.0f, 20.0f});

    leading.resize({40.0f, 60.0f});
    host.place(ctx, kScreen);
    CHECK(a.laidFrame == Rect{0.0f, 54.0f, 800.0f, 546.0f});

    dock.update({.tabs = true, .panes = {MakePane("a", &a)}, .leading = &leading});
    host.place(ctx, kScreen);
    CHECK(a.laidFrame == Rect{0.0f, 54.0f, 800.0f, 546.0f});
    CHECK(leading.laidFrame == Rect{6.0f, 1.0f, 40.0f, 60.0f});

    leading.resize({40.0f, 10.0f});
    host.place(ctx, kScreen);
    CHECK(a.laidFrame == Rect{0.0f, kBar, 800.0f, 600.0f - kBar});
}

TEST_CASE("A tab bar with only chrome is still shown and hides trailing chrome until a tab exists",
          "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe leading({40.0f, 20.0f});
    Probe trailing({24.0f, 24.0f});
    Dock dock;
    Host host(dock);
    dock.update({.tabs = true, .leading = &leading, .trailing = &trailing});
    host.place(ctx, kScreen);

    CHECK(leading.visible());
    CHECK_FALSE(trailing.visible());
    CHECK(leading.laidFrame == Rect{6.0f, 10.0f, 40.0f, 20.0f});

    Probe a;
    dock.update({.tabs = true, .panes = {MakePane("a", &a)}, .leading = &leading, .trailing = &trailing});
    host.place(MakeContext(), kScreen);
    CHECK(trailing.visible());
    CHECK(trailing.laidFrame == Rect{302.0f, 8.0f, 24.0f, 24.0f});

    dock.update({.tabs = true, .leading = &leading, .trailing = &trailing});
    host.place(MakeContext(), kScreen);
    CHECK_FALSE(trailing.visible());
}

TEST_CASE("Strip tabs place their close buttons inside the tab and close only through them",
          "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    Probe b;
    std::vector<std::string> closed;
    std::vector<U8> presses;
    Dock dock;
    Host host(dock);
    auto paneA = MakePane("a", &a);
    paneA.onClose = [&] { closed.push_back("a"); };
    auto paneB = MakePane("b", &b);
    paneB.onClose = [&] { closed.push_back("b"); };
    dock.update({.tabs = true, .panes = {paneA, paneB},
                 .onBarPress = [&](U8 clicks) { presses.push_back(clicks); }});
    host.place(ctx, kScreen);

    host.click(235.0f, 19.0f);
    host.place(ctx, kScreen);
    CHECK(closed.empty());
    CHECK(a.visible());
    host.click(235.0f, 19.0f);
    host.place(ctx, kScreen);
    CHECK(closed == std::vector<std::string>{"a"});
    host.click(475.0f, 19.0f);
    host.place(ctx, kScreen);
    CHECK(closed == std::vector<std::string>{"a"});
    CHECK(b.visible());
    host.click(475.0f, 19.0f);
    host.place(ctx, kScreen);
    CHECK(closed == std::vector<std::string>{"a", "b"});

    host.click(100.0f, 19.0f);
    host.place(ctx, kScreen);
    CHECK(a.visible());
    CHECK(closed.size() == 2);
    CHECK(presses.empty());

    host.click(600.0f, 19.0f);
    CHECK(presses == std::vector<U8>{1});
    CHECK(closed.size() == 2);
}

TEST_CASE("Strip tab icons sit right before the title without a badge", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    Probe b;
    std::vector<U8> presses;
    Dock dock;
    Host host(dock);
    auto paneA = MakePane("a", &a);
    paneA.icon = ICON_LC_TERMINAL;
    dock.update({.tabs = true, .panes = {paneA, MakePane("b", &b)}, .activate = "a",
                 .onBarPress = [&](U8 clicks) { presses.push_back(clicks); }});
    host.place(ctx, kScreen);

    const auto texts = TextFrames(dock);
    REQUIRE(texts.size() == 3);
    CHECK(texts[0].x == 18.0f);
    CHECK(texts[0].width == 15.0f);
    CHECK(texts[1].x == 39.0f);
    CHECK(texts[2].x == 262.0f);

    host.click(600.0f, 19.0f);
    CHECK(presses == std::vector<U8>{1});
    host.click(300.0f, 19.0f);
    host.place(ctx, kScreen);
    CHECK(presses.size() == 1);
    CHECK(b.layouts == 1);
}

TEST_CASE("Strip tabs share one track with a bordered active capsule, a tonal rim, dividers and bare icons",
          "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Dock dock;
    Host host(dock);
    auto paneB = MakePane("b");
    paneB.icon = ICON_LC_TERMINAL;
    dock.update({.tabs = true, .panes = {MakePane("a"), paneB, MakePane("c"), MakePane("d")}, .activate = "a"});
    host.place(ctx, kScreen);

    const auto* track = BoxWith(dock, {6.0f, 4.0f, 788.0f, 32.0f});
    REQUIRE(track);
    CHECK(track->instances.size() == 1);
    CHECK(track->cornerRadius == 16.0f);

    const auto* active = BoxWith(dock, {8.0f, 6.0f, 193.0f, 28.0f});
    REQUIRE(active);
    CHECK(active->instances.size() == 1);
    CHECK(active->cornerRadius == 14.0f);
    CHECK(active->borderWidth == 1.0f);

    for (auto* box : Direct<Sakura::Box>(dock)) {
        CHECK((box->config().instances.empty() || box->config().softness == 0.0f));
    }
    U64 litEdges = 0;
    for (auto* box : Direct<Sakura::Box>(dock)) {
        const auto& config = box->config();
        if (config.topLit && !config.instances.empty()) {
            REQUIRE(config.instances.size() == 1);
            CHECK(config.instances[0].rect == Rect{8.0f, 6.0f, 193.0f, 28.0f});
            CHECK(config.cornerRadius == 14.0f);
            CHECK(config.borderColor == Sakura::RimColor(ctx.color("pill_active_bg")));
            ++litEdges;
        }
    }
    CHECK(litEdges == 1);

    const auto* dividers = BoxWith(dock, {597.0f, 12.0f, 1.0f, 15.0f});
    REQUIRE(dividers);
    REQUIRE(dividers->instances.size() == 2);
    CHECK(dividers->instances[0].rect == Rect{400.0f, 12.0f, 1.0f, 15.0f});

    CHECK(BoxWith(dock, {215.0f, 9.0f, 22.0f, 22.0f}) == nullptr);

    const auto texts = TextFrames(dock);
    REQUIRE(texts.size() == 5);
    CHECK(texts[0].x == 18.0f);
    CHECK(texts[1].x == 215.0f);
    CHECK(texts[2].x == 236.0f);
    CHECK(texts[3].x == 412.0f);

    dock.update({.tabs = true, .panes = {MakePane("a"), paneB, MakePane("c"), MakePane("d")}, .activate = "c"});
    host.place(ctx, kScreen);
    CHECK(BoxWith(dock, {400.0f, 12.0f, 1.0f, 15.0f}) == nullptr);
    CHECK(BoxWith(dock, {597.0f, 12.0f, 1.0f, 15.0f}) == nullptr);
    const auto* left = BoxWith(dock, {203.0f, 12.0f, 1.0f, 15.0f});
    REQUIRE(left);
    CHECK(left->instances.size() == 1);
    CHECK(BoxWith(dock, {402.0f, 6.0f, 193.0f, 28.0f}) == active);
}

TEST_CASE("Strip tab titles are ellipsized to the tab and tabs shrink evenly to fit the bar",
          "[core][sakura][dock]") {
    MonoFontWindow window;
    auto ctx = MakeContext();
    ctx.render = &window;
    const std::string title(60, 'w');
    std::vector<U8> presses;
    Dock dock;
    Host host(dock);
    auto paneA = MakePane("a");
    paneA.title = title;
    dock.update({.tabs = true, .panes = {paneA, MakePane("b")},
                 .onBarPress = [&](U8 clicks) { presses.push_back(clicks); }});
    host.place(ctx, kScreen);

    const auto texts = TextFrames(dock);
    REQUIRE(texts.size() == 2);
    CHECK(texts[0].width > 100.0f);
    CHECK(texts[0].width <= 203.0f);
    host.click(600.0f, 19.0f);
    CHECK(presses == std::vector<U8>{1});
    host.click(100.0f, 19.0f);
    CHECK(presses.size() == 1);

    Probe trailing({24.0f, 24.0f});
    std::vector<Dock::Pane> panes;
    for (const auto* id : {"a", "b", "c", "d"}) {
        auto pane = MakePane(id);
        pane.title = title;
        panes.push_back(pane);
    }
    dock.update({.tabs = true, .panes = panes, .trailing = &trailing});
    host.place(ctx, {0.0f, 0.0f, 400.0f, 300.0f});
    const auto squeezed = TextFrames(dock);
    REQUIRE(squeezed.size() == 4);
    for (const auto& frame : squeezed) {
        CHECK(frame.right() <= 364.0f);
    }
    CHECK(trailing.laidFrame.x == 368.0f);
    CHECK(trailing.laidFrame.right() <= 394.0f);
}

TEST_CASE("Right clicking a tab reports the pane and the point in dock coordinates",
          "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    Probe b;
    std::vector<std::pair<std::string, Extent2D<F32>>> menus;
    std::vector<U8> presses;
    Dock dock;
    Host host(dock);
    const auto update = [&] {
        dock.update({.tabs = true, .panes = {MakePane("a", &a), MakePane("b", &b)},
                     .onTabMenu = [&](const std::string& pane, Extent2D<F32> point) {
                         menus.emplace_back(pane, point);
                     },
                     .onBarPress = [&](U8 clicks) { presses.push_back(clicks); }});
    };
    update();
    host.place(ctx, {0.0f, 40.0f, 800.0f, 560.0f});

    host.click(300.0f, 59.0f, MouseButton::Right);
    REQUIRE(menus.size() == 1);
    CHECK(menus[0].first == "b");
    CHECK(menus[0].second == Extent2D<F32>{300.0f, 59.0f});

    host.click(600.0f, 59.0f, MouseButton::Right);
    CHECK(menus.size() == 1);
    CHECK(presses.empty());
    host.place(ctx, {0.0f, 40.0f, 800.0f, 560.0f});
    CHECK(b.visible());
    CHECK_FALSE(a.visible());

    update();
    host.place(ctx, kScreen);
    host.click(10.0f, 19.0f, MouseButton::Right);
    REQUIRE(menus.size() == 2);
    CHECK(menus[1].first == "a");
}

TEST_CASE("Pressing empty tab bar space reports the click count, not on pills or chrome",
          "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    Probe leading({40.0f, 20.0f});
    Probe trailing({24.0f, 24.0f});
    std::vector<U8> presses;
    Dock dock;
    Host host(dock);
    dock.update({.tabs = true, .panes = {MakePane("a", &a)}, .leading = &leading, .trailing = &trailing, .onBarPress = [&](U8 clicks) { presses.push_back(clicks); }});
    host.place(ctx, kScreen);

    host.send(Mouse(MouseEventType::Click, 400.0f, 19.0f, MouseButton::Left, 1));
    host.send(Mouse(MouseEventType::Release, 400.0f, 19.0f));
    host.send(Mouse(MouseEventType::Click, 400.0f, 19.0f, MouseButton::Left, 2));
    host.send(Mouse(MouseEventType::Release, 400.0f, 19.0f));
    CHECK(presses == std::vector<U8>{1, 2});

    host.click(60.0f, 19.0f);
    host.click(20.0f, 19.0f);
    host.click(310.0f, 19.0f);
    host.click(400.0f, 19.0f, MouseButton::Right);
    host.click(400.0f, 300.0f);
    CHECK(presses == std::vector<U8>{1, 2});
    CHECK(leading.count(MouseEventType::Click) == 1);
    CHECK(trailing.count(MouseEventType::Click) == 1);
    CHECK(a.count(MouseEventType::Click) == 1);
}

TEST_CASE("Tab drags in a pill bar preview a card inset into the content", "[core][sakura][dock]") {
    Probe a;
    Probe b;
    U64 focusB = 0;
    Dock dock;
    Layer layer;
    layer.update({.content = &dock});
    LayerHost host({&layer});
    auto paneB = MakePane("b", &b);
    paneB.onFocus = [&] { ++focusB; };
    const Dock::Config config = {.tabs = true, .accepts = "panel", .panes = {MakePane("a", &a), paneB}};
    dock.update(config);
    host.place(MakeContext(), kScreen);

    host.send(Mouse(MouseEventType::Click, 10.0f, 19.0f));
    host.send(Mouse(MouseEventType::Move, 400.0f, 19.0f));
    REQUIRE(host.drag.active);
    CHECK(host.drag.pane == "a");
    host.place(MouseAt(400.0f, 19.0f), kScreen);
    REQUIRE(host.drag.target.has_value());
    CHECK(host.drag.target->preview == Rect{6.0f, 38.0f, 788.0f, 556.0f});
    host.send(Mouse(MouseEventType::Release, 400.0f, 19.0f));
    CHECK_FALSE(host.drag.active);

    dock.update(config);
    host.place(MakeContext(), kScreen);
    host.click(10.0f, 19.0f);
    CHECK(focusB == 1);
}

TEST_CASE("Tile cards keep a gap at the edges and between cards with the splitter inside it",
          "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    CursorDock dock;
    Probe a;
    Probe b;
    Host host(dock);
    dock.update({.panes = {MakePane("a", &a), MakePane("b", &b)}});
    host.place(ctx, kScreen);

    CHECK(a.laidClip == Body({6.0f, 6.0f, 391.0f, 588.0f}));
    CHECK(b.laidClip == Body({403.0f, 6.0f, 391.0f, 588.0f}));
    CHECK_FALSE(dock.cursorAt({3.0f, 300.0f}).has_value());
    CHECK(dock.cursorAt({396.0f, 300.0f}) == SurfaceCursor::ResizeEW);
    CHECK(dock.cursorAt({400.0f, 300.0f}) == SurfaceCursor::ResizeEW);
    CHECK(dock.cursorAt({404.0f, 300.0f}) == SurfaceCursor::ResizeEW);
    CHECK_FALSE(dock.cursorAt({410.0f, 300.0f}).has_value());

    host.click(3.0f, 300.0f);
    CHECK(a.count(MouseEventType::Click) == 0);
}

TEST_CASE("Tile cards cast a soft dropped shadow and carry a top lit tonal rim without painting corner masks",
          "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Dock dock;
    Host host(dock);
    auto paneB = MakePane("b");
    paneB.padding = Dock::ContentPadding;
    dock.update({.panes = {MakePane("a"), paneB}});
    host.place(ctx, kScreen);

    const Rect cardA = {6.0f, 6.0f, 391.0f, 588.0f};
    const Rect cardB = {403.0f, 6.0f, 391.0f, 588.0f};
    const auto* shadows = BoxWith(dock, {0.0f, 2.0f, 403.0f, 600.0f});
    REQUIRE(shadows);
    CHECK(shadows->softness == 6.0f);
    CHECK(shadows->cornerRadius == 12.0f);
    REQUIRE(shadows->instances.size() == 2);
    CHECK(shadows->instances[1].rect == Rect{397.0f, 2.0f, 403.0f, 600.0f});

    std::vector<const Sakura::Box::Config*> lit;
    for (auto* box : Find<Sakura::Box>(dock)) {
        if (box->config().topLit && !box->config().instances.empty()) {
            lit.push_back(&box->config());
        }
    }
    REQUIRE(lit.size() == 1);
    CHECK(lit[0]->cornerRadius == 12.0f);
    CHECK(lit[0]->borderWidth == 1.0f);
    CHECK(lit[0]->borderColor == Sakura::RimColor(ctx.color("tile_bg")));
    REQUIRE(lit[0]->instances.size() == 2);
    CHECK(lit[0]->instances[0].rect == cardA);
    CHECK(lit[0]->instances[1].rect == cardB);

    for (auto* polyline : Direct<Sakura::Polyline>(dock)) {
        for (const auto& instance : polyline->config().instances) {
            CHECK_FALSE(instance.filled);
        }
    }
}

TEST_CASE("Tile headers show only the title, aligned to the content edge, with buttons at the right edge",
          "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    Probe b;
    U64 closedA = 0;
    Dock dock;
    Host host(dock);
    auto paneA = MakePane("a", &a);
    paneA.icon = ICON_LC_CHART_LINE;
    paneA.onClose = [&] { ++closedA; };
    auto paneB = MakePane("b", &b);
    paneB.padding = 12.0f;
    dock.update({.panes = {paneA, paneB}});
    host.place(ctx, kScreen);

    const auto texts = TextFrames(dock);
    REQUIRE(texts.size() == 2);
    CHECK(texts[0].x == 18.0f);
    CHECK(texts[0].y == 12.0f);
    CHECK(texts[0].height == 16.0f);
    CHECK(texts[1].x == 415.0f);
    CHECK(texts[1].y == 12.0f);
    for (auto* text : Direct<Sakura::Text>(dock)) {
        if (!ButtonGlyph(*text)) {
            const Rect clip = Peek::Clip(*text);
            CHECK(clip.y <= Peek::Frame(*text).y);
            CHECK(clip.bottom() >= Peek::Frame(*text).bottom() + 4.0f);
        }
    }

    const Rect card = {6.0f, 6.0f, 391.0f, 588.0f};
    const auto* cross = GlyphIn(dock, card, ICON_LC_X);
    REQUIRE(cross);
    const Rect glyph = Peek::Frame(*cross);
    CHECK(glyph.right() == Catch::Approx(card.right() - (texts[0].x - card.x)));
    CHECK(glyph.y + glyph.height * 0.5f == Catch::Approx(texts[0].y + texts[0].height * 0.5f));
    const auto* crossB = GlyphIn(dock, {403.0f, 6.0f, 391.0f, 588.0f}, ICON_LC_X);
    REQUIRE(crossB);
    CHECK(Peek::Frame(*crossB).right() == Catch::Approx(794.0f - 12.0f));

    const F32 centerX = glyph.x + glyph.width * 0.5f;
    const F32 closeX = std::round(centerX - 10.0f);
    CHECK(Peek::CursorAt(dock, {centerX, 20.0f}) == SurfaceCursor::Hand);
    CHECK(Peek::CursorAt(dock, MaximizeOf(card)) == SurfaceCursor::Hand);
    CHECK_FALSE(Peek::CursorAt(dock, {100.0f, 20.0f}).has_value());
    host.click(closeX - 1.0f, 20.0f);
    CHECK(closedA == 0);
    host.click(closeX + 21.0f, 20.0f);
    CHECK(closedA == 0);
    const auto close = CloseOf(card);
    host.click(close.x, close.y);
    CHECK(closedA == 1);

    const F32 below = card.y + 20.0f + 2.0f;
    REQUIRE(a.laidFrame.y < below);
    const U64 clicks = a.count(MouseEventType::Click);
    host.click(centerX, below);
    CHECK(closedA == 2);
    CHECK(a.count(MouseEventType::Click) == clicks);
}

TEST_CASE("Header buttons are same size icon glyphs that recolor on hover and flip and turn blue when maximized",
          "[core][sakura][dock]") {
    const Sakura::Palette palette = Sakura::Themes.at(Sakura::ResolveThemeKey(""));
    auto ctx = MakeContext();
    ctx.palette = palette;
    Dock dock;
    Host host(dock);
    auto paneA = MakePane("a");
    paneA.onClose = [] {};
    dock.update({.panes = {paneA, MakePane("b")}});
    host.place(ctx, kScreen);

    const auto glyph = [&](const Rect& card, const std::string& icon) {
        const auto* text = GlyphIn(dock, card, icon);
        REQUIRE(text);
        return text->config();
    };
    const Rect card = {6.0f, 6.0f, 391.0f, 588.0f};
    const auto expand = std::string(ICON_LC_MAXIMIZE_2);
    const auto compress = std::string(ICON_LC_MINIMIZE_2);
    CHECK(glyph(card, ICON_LC_X).colorKey == "tile_header_text");
    CHECK(glyph(card, expand).colorKey == "tile_header_text");
    CHECK(glyph(card, expand).scale == glyph(card, ICON_LC_X).scale);
    const Rect expandFrame = Peek::Frame(*GlyphIn(dock, card, expand));
    const Rect crossFrame = Peek::Frame(*GlyphIn(dock, card, ICON_LC_X));
    CHECK(expandFrame.y + expandFrame.height * 0.5f == Catch::Approx(crossFrame.y + crossFrame.height * 0.5f));
    CHECK(expandFrame.right() < crossFrame.x);

    const auto point = MaximizeOf(card);
    host.send(Mouse(MouseEventType::Move, point.x, point.y));
    host.place(ctx, kScreen);
    CHECK(glyph(card, expand).colorKey == "tile_button_hover");
    CHECK(glyph(card, ICON_LC_X).colorKey == "tile_header_text");
    const auto circles = [&] {
        U64 count = 0;
        for (auto* box : Direct<Sakura::Box>(dock)) {
            for (const auto& instance : box->config().instances) {
                if (instance.visible && instance.rect.width == 20.0f && instance.rect.height == 20.0f &&
                    box->config().cornerRadius == 10.0f &&
                    instance.rect.contains(point.x, point.y)) {
                    ++count;
                }
            }
        }
        return count;
    };
    CHECK(circles() == 1);
    host.send(Mouse(MouseEventType::Move, card.x + 40.0f, card.y + 200.0f));
    host.place(ctx, kScreen);
    CHECK(circles() == 0);
    host.send(Mouse(MouseEventType::Move, point.x, point.y));
    host.place(ctx, kScreen);

    host.click(point.x, point.y);
    host.place(ctx, kScreen);
    const Rect full = {6.0f, 6.0f, 788.0f, 588.0f};
    CHECK(glyph(full, compress).str == compress);
    CHECK(glyph(full, compress).colorKey == "tile_button_hover");
    CHECK_FALSE(GlyphIn(dock, full, expand));

    const auto closePoint = CloseOf(full);
    host.send(Mouse(MouseEventType::Move, closePoint.x, closePoint.y));
    host.place(ctx, kScreen);
    CHECK(glyph(full, compress).colorKey == "cyber_blue");
    CHECK(glyph(full, ICON_LC_X).colorKey == "tile_button_hover");

    const auto tinted = [&](const Extent2D<F32>& at, const char* key) {
        U64 count = 0;
        for (auto* box : Direct<Sakura::Box>(dock)) {
            for (const auto& instance : box->config().instances) {
                if (instance.visible && instance.rect.contains(at.x, at.y) && instance.rect.width == 20.0f &&
                    instance.backgroundColor == ctx.color(key)) {
                    ++count;
                }
            }
        }
        return count;
    };
    const auto compressPoint = MaximizeOf(full);
    CHECK(tinted(compressPoint, "tile_button_active_bg") == 1);
    CHECK(tinted(closePoint, "chrome_button_hover") == 1);
    host.send(Mouse(MouseEventType::Move, compressPoint.x, compressPoint.y));
    host.place(ctx, kScreen);
    CHECK(tinted(compressPoint, "tile_button_active_bg") == 1);
    CHECK(tinted(compressPoint, "chrome_button_hover") == 0);

    host.click(compressPoint.x, compressPoint.y);
    host.place(ctx, kScreen);
    CHECK(tinted(MaximizeOf(card), "tile_button_active_bg") == 0);
}

TEST_CASE("Header buttons are not drag handles while the rest of the header is", "[core][sakura][dock]") {
    Dock dock;
    Layer layer;
    layer.update({.content = &dock});
    LayerHost host({&layer});
    dock.update({.accepts = "panel", .panes = {MakePane("a"), MakePane("b")}});
    host.place(MakeContext(), kScreen);
    const Rect card = {6.0f, 6.0f, 391.0f, 588.0f};

    for (const auto point : {CloseOf(card), MaximizeOf(card)}) {
        host.send(Mouse(MouseEventType::Click, point.x, point.y));
        host.send(Mouse(MouseEventType::Move, point.x - 100.0f, point.y + 100.0f));
        CHECK_FALSE(host.drag.active);
        host.send(Mouse(MouseEventType::Release, point.x - 100.0f, point.y + 100.0f));
    }

    host.send(Mouse(MouseEventType::Click, card.right() - 60.0f, 21.0f));
    host.send(Mouse(MouseEventType::Move, card.right() - 160.0f, 121.0f));
    CHECK(host.drag.active);
    host.send(Mouse(MouseEventType::Cancel, 0.0f, 0.0f));
}

TEST_CASE("Maximizing a tile fills the dock until restored and never emits tiles", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    Probe b;
    Probe c;
    Emissions emissions;
    Dock dock;
    Host host(dock);
    const auto tiles = Row({1000.0f, 1000.0f, 1000.0f}, {Leaf("a"), Leaf("b"), Leaf("c")});
    const auto update = [&] {
        dock.update({.panes = {MakePane("a", &a), MakePane("b", &b), MakePane("c", &c)}, .tiles = tiles,
                     .onTiles = emissions.sink()});
    };
    update();
    host.place(ctx, kScreen);
    const Rect cardB = {271.0f, 6.0f, 258.0f, 588.0f};
    REQUIRE(b.laidFrame == Body(cardB));

    const auto maximize = MaximizeOf(cardB);
    host.click(maximize.x, maximize.y);
    update();
    host.place(ctx, kScreen);
    const Rect full = {6.0f, 6.0f, 788.0f, 588.0f};
    CHECK(b.laidFrame == Body(full));
    CHECK(b.visible());
    CHECK_FALSE(a.visible());
    CHECK_FALSE(c.visible());
    CHECK(emissions.trees.empty());

    const auto restore = MaximizeOf(full);
    host.click(restore.x, restore.y);
    update();
    host.place(ctx, kScreen);
    CHECK(b.laidFrame == Body(cardB));
    CHECK(a.visible());
    CHECK(c.visible());
    CHECK(emissions.trees.empty());
}

TEST_CASE("A maximized tile is restored when it goes away or when a pane is added", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    Probe b;
    Probe c;
    Emissions emissions;
    Dock dock;
    Host host(dock);
    std::optional<DockTiles> tiles = Row({500.0f, 500.0f}, {Leaf("a"), Leaf("b")});
    const auto sink = [&](std::optional<DockTiles> next) {
        tiles = std::move(next);
    };
    dock.update({.panes = {MakePane("a", &a), MakePane("b", &b)}, .tiles = tiles, .onTiles = sink});
    host.place(ctx, kScreen);
    const auto maximizeB = MaximizeOf({403.0f, 6.0f, 391.0f, 588.0f});
    host.click(maximizeB.x, maximizeB.y);
    host.place(ctx, kScreen);
    REQUIRE_FALSE(a.visible());

    dock.update({.panes = {MakePane("a", &a), MakePane("b", &b), MakePane("c", &c)}, .tiles = tiles, .onTiles = sink});
    host.place(ctx, kScreen);
    CHECK(a.visible());
    CHECK(c.visible());

    const auto maximizeA = MaximizeOf({6.0f, 6.0f, 259.0f, 588.0f});
    host.click(maximizeA.x, maximizeA.y);
    host.place(ctx, kScreen);
    REQUIRE_FALSE(b.visible());
    tiles = Row({500.0f, 500.0f}, {Leaf("b"), Leaf("c")});
    dock.update({.panes = {MakePane("b", &b), MakePane("c", &c)}, .tiles = tiles, .onTiles = sink});
    host.place(ctx, kScreen);
    CHECK(b.visible());
    CHECK(c.visible());
    CHECK(b.laidFrame == Body({6.0f, 6.0f, 391.0f, 588.0f}));
}

TEST_CASE("A maximized tile is restored when a new pane arrives already placed in the tiles",
          "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    Probe b;
    Probe c;
    Dock dock;
    Host host(dock);
    dock.update({.panes = {MakePane("a", &a), MakePane("b", &b)}, .tiles = Row({500.0f, 500.0f}, {Leaf("a"), Leaf("b")})});
    host.place(ctx, kScreen);
    const auto maximizeB = MaximizeOf({403.0f, 6.0f, 391.0f, 588.0f});
    host.click(maximizeB.x, maximizeB.y);
    host.place(ctx, kScreen);
    REQUIRE_FALSE(a.visible());

    dock.update({.panes = {MakePane("a", &a), MakePane("b", &b), MakePane("c", &c)},
                 .tiles = Row({1000.0f, 1000.0f}, {Leaf("a"), Column({1000.0f, 1000.0f}, {Leaf("b"), Leaf("c")})})});
    host.place(ctx, kScreen);
    CHECK(a.visible());
    CHECK(b.visible());
    CHECK(c.visible());
    CHECK(c.laidFrame.width > 0.0f);
    CHECK(b.laidFrame.width < 700.0f);
}

TEST_CASE("Tiles with different fills draw outlines derived from their own fill", "[core][sakura][dock]") {
    constexpr ColorRGBA<F32> kRed = {0.3f, 0.0f, 0.0f, 1.0f};
    constexpr ColorRGBA<F32> kBlue = {0.0f, 0.0f, 0.3f, 1.0f};
    Probe a;
    Probe b;
    Dock dock;
    Host host(dock);
    auto red = MakePane("a", &a);
    red.fill = kRed;
    auto blue = MakePane("b", &b);
    blue.fill = kBlue;
    dock.update({.panes = {red, blue}, .tiles = Row({500.0f, 500.0f}, {Leaf("a"), Leaf("b")})});
    host.place(MakeContext(), kScreen);

    U64 reds = 0;
    U64 blues = 0;
    for (auto* box : Direct<Sakura::Box>(dock)) {
        const auto& config = box->config();
        if (config.borderWidth <= 0.0f || config.instances.size() != 1) {
            continue;
        }
        const auto& color = config.borderColor;
        const Rect& rect = config.instances.front().rect;
        if (color.r > color.b && color.r > kRed.r) {
            CHECK(rect.contains(Peek::Frame(a).x + 1.0f, Peek::Frame(a).y + 1.0f));
            ++reds;
        } else if (color.b > color.r && color.b > kBlue.b) {
            CHECK(rect.contains(Peek::Frame(b).x + 1.0f, Peek::Frame(b).y + 1.0f));
            ++blues;
        }
    }
    CHECK(reds == 2);
    CHECK(blues == 2);
}

TEST_CASE("The header button hover circle draws above tile contents and below the header glyphs",
          "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    Probe b;
    Dock dock;
    Host host(dock);
    dock.update({.panes = {MakePane("a", &a), MakePane("b", &b)}, .tiles = Row({500.0f, 500.0f}, {Leaf("a"), Leaf("b")})});
    host.place(ctx, kScreen);
    const auto close = CloseOf({403.0f, 6.0f, 391.0f, 588.0f});
    host.send(Mouse(MouseEventType::Move, close.x, close.y));
    host.place(ctx, kScreen);

    const auto children = Peek::Children(dock);
    const auto index = [&](const Sakura::Component* child) {
        return static_cast<U64>(std::find(children.begin(), children.end(), child) - children.begin());
    };
    std::optional<U64> hover;
    for (auto* box : Direct<Sakura::Box>(dock)) {
        for (const auto& instance : box->config().instances) {
            if (instance.visible && instance.rect.contains(close.x, close.y) &&
                instance.backgroundColor == ctx.color("chrome_button_hover")) {
                hover = index(box);
            }
        }
    }
    REQUIRE(hover.has_value());
    CHECK(*hover > index(&a));
    CHECK(*hover > index(&b));
    const auto glyph = GlyphIn(dock, {403.0f, 6.0f, 391.0f, 588.0f}, ICON_LC_X);
    REQUIRE(glyph != nullptr);
    CHECK(*hover < index(glyph));
}

TEST_CASE("A tab dragged over its own content dock is not offered as a drop there", "[core][sakura][dock]") {
    Probe x;
    Dock inner;
    Dock outer;
    Layer layer;
    layer.update({.content = &outer});
    LayerHost host({&layer});
    inner.update({.accepts = "terminal", .panes = {MakePane("x", &x)}});
    auto tab = MakePane("t", &inner);
    tab.group = "terminal";
    outer.update({.tabs = true, .accepts = "main", .adopts = {"terminal"}, .panes = {tab}});
    host.place(MakeContext(), kScreen);
    const auto rect = outer.tabRect("t");
    REQUIRE(rect.has_value());

    host.send(Mouse(MouseEventType::Click, rect->x + rect->width * 0.5f, rect->y + rect->height * 0.5f));
    host.send(Mouse(MouseEventType::Move, 400.0f, 300.0f));
    REQUIRE(host.drag.active);
    CHECK(host.drag.pane == "t");
    CHECK(host.drag.content == static_cast<const void*>(&inner));
    host.place(MouseAt(400.0f, 300.0f), kScreen);
    CHECK_FALSE(host.drag.target.has_value());
    host.send(Mouse(MouseEventType::Release, 400.0f, 300.0f));
    host.place(MakeContext(), kScreen);
    CHECK(x.visible());
    CHECK(outer.tabRect("t").has_value());
}

TEST_CASE("A tab dragged over a dock nested deeper inside its content is not offered as a drop there",
          "[core][sakura][dock]") {
    Probe x;
    Dock inner;
    Dock middle;
    Dock outer;
    Layer layer;
    layer.update({.content = &outer});
    LayerHost host({&layer});
    inner.update({.accepts = "terminal", .panes = {MakePane("x", &x)}});
    middle.update({.accepts = "other", .panes = {MakePane("i", &inner)}});
    auto tab = MakePane("t", &middle);
    tab.group = "terminal";
    outer.update({.tabs = true, .accepts = "main", .adopts = {"terminal"}, .panes = {tab}});
    host.place(MakeContext(), kScreen);
    const auto rect = outer.tabRect("t");
    REQUIRE(rect.has_value());

    host.send(Mouse(MouseEventType::Click, rect->x + rect->width * 0.5f, rect->y + rect->height * 0.5f));
    host.send(Mouse(MouseEventType::Move, 400.0f, 300.0f));
    REQUIRE(host.drag.active);
    REQUIRE(x.laidFrame.contains(400.0f, 300.0f));
    host.place(MouseAt(400.0f, 300.0f), kScreen);
    CHECK_FALSE(host.drag.target.has_value());
    host.send(Mouse(MouseEventType::Release, 400.0f, 300.0f));
    host.place(MakeContext(), kScreen);
    CHECK(x.visible());
    CHECK(outer.tabRect("t").has_value());
}

namespace {

struct BackdropColorProbe : Probe {
    ColorRGBA<F32> color = {0.0f, 0.0f, 0.0f, 0.0f};

 protected:
    void layout(const Jetstream::Sakura::Context& ctx) override {
        Probe::layout(ctx);
        color = ctx.color(ctx.backdropColorKey);
    }
};

}  // namespace

TEST_CASE("Content of a tinted tile sees the tint as its backdrop and follows tint changes", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    BackdropColorProbe a;
    BackdropColorProbe b;
    Dock dock;
    Host host(dock);
    const auto configure = [&](ColorRGBA<F32> tint) {
        auto tinted = MakePane("b", &b);
        tinted.fill = tint;
        dock.update({.panes = {MakePane("a", &a), tinted}});
    };

    const auto same = [](ColorRGBA<F32> actual, ColorRGBA<F32> expected) {
        return actual.r == Catch::Approx(expected.r) && actual.g == Catch::Approx(expected.g) &&
               actual.b == Catch::Approx(expected.b) && actual.a == Catch::Approx(expected.a);
    };

    configure({0.3f, 0.0f, 0.0f, 1.0f});
    host.place(ctx, kScreen);
    CHECK(a.color == ctx.color("tile_bg"));
    CHECK(same(b.color, {0.3f, 0.0f, 0.0f, ctx.color("tile_bg").a}));

    configure({0.0f, 0.4f, 0.0f, 1.0f});
    host.place(ctx, kScreen);
    CHECK(same(b.color, {0.0f, 0.4f, 0.0f, ctx.color("tile_bg").a}));

    const auto base = ctx.color("tile_bg");
    configure({1.0f, 0.0f, 0.0f, 0.5f});
    host.place(ctx, kScreen);
    CHECK(b.color.r == Catch::Approx(base.r + (1.0f - base.r) * 0.5f));
    CHECK(b.color.g == Catch::Approx(base.g * 0.5f));
    CHECK(b.color.a == Catch::Approx(base.a));

    configure({0.0f, 0.0f, 0.0f, 0.0f});
    host.place(ctx, kScreen);
    CHECK(b.color == base);
}

TEST_CASE("Maximize buttons appear only for maximizable panes with company", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    Probe b;
    Dock dock;
    Host host(dock);
    const Rect full = {6.0f, 6.0f, 788.0f, 588.0f};

    dock.update({.panes = {MakePane("a", &a)}});
    host.place(ctx, kScreen);
    const auto lone = MaximizeOf(full);
    host.click(lone.x, lone.y);
    host.place(ctx, kScreen);
    CHECK(a.laidFrame == Body(full));

    auto paneB = MakePane("b", &b);
    paneB.maximizable = false;
    dock.update({.panes = {MakePane("a", &a), paneB}});
    host.place(ctx, kScreen);
    const auto fixed = MaximizeOf({403.0f, 6.0f, 391.0f, 588.0f});
    host.click(fixed.x, fixed.y);
    host.place(ctx, kScreen);
    CHECK(a.visible());
    CHECK(b.laidFrame == Body({403.0f, 6.0f, 391.0f, 588.0f}));
}

TEST_CASE("A maximized tile hides the other surfaces until restored", "[core][sakura][dock]") {
    SakuraTest::FakeWindow window;
    auto ctx = MakeContext();
    ctx.render = &window;
    const auto first = std::make_shared<SurfaceTexture>();
    const auto second = std::make_shared<SurfaceTexture>();
    Sakura::SurfaceImage a;
    Sakura::SurfaceImage b;
    Dock dock;
    Host host(dock);
    Frames frames(host, kScreen, ctx);
    std::optional<DockTiles> tiles = Row({500.0f, 500.0f}, {Leaf("a"), Leaf("b")});
    const auto update = [&] {
        a.update({.texture = first});
        b.update({.texture = second});
        dock.update({.panes = {MakePane("a", &a), MakePane("b", &b)}, .tiles = tiles,
                     .onTiles = [&](std::optional<DockTiles> next) { tiles = std::move(next); }});
    };
    const auto check = [&](bool firstShown) {
        for (int frame = 0; frame < 3; ++frame) {
            frames.run(update);
            const U64 count = frames.ctx.frame;
            CHECK(first->visibleForPresentation(count) == firstShown);
            CHECK(second->visibleForPresentation(count));
        }
    };

    frames.run(update);
    check(true);
    const U64 rebuilds = frames.rebuilds;

    const auto maximize = MaximizeOf({403.0f, 6.0f, 391.0f, 588.0f});
    host.click(maximize.x, maximize.y);
    check(false);

    const auto restore = MaximizeOf({6.0f, 6.0f, 788.0f, 588.0f});
    host.click(restore.x, restore.y);
    check(true);
    CHECK(frames.rebuilds == rebuilds);
}

TEST_CASE("Tile drags over the gap between cards still target the nearest card", "[core][sakura][dock]") {
    Emissions emissions;
    Dock dock;
    Layer layer;
    layer.update({.content = &dock});
    LayerHost host({&layer});
    dock.update({.accepts = "panel",
                 .panes = {MakePane("a"), MakePane("b"), MakePane("c")},
                 .tiles = Row({500.0f, 500.0f}, {Leaf("a"), Column({500.0f, 500.0f}, {Leaf("b"), Leaf("c")})}),
                 .onTiles = emissions.sink()});
    host.place(MakeContext(), kScreen);

    host.send(Mouse(MouseEventType::Click, 200.0f, 21.0f));
    host.send(Mouse(MouseEventType::Move, 600.0f, 299.0f));
    REQUIRE(host.drag.active);
    host.place(MouseAt(600.0f, 299.0f), kScreen);
    REQUIRE(host.drag.target.has_value());
    CHECK(host.drag.target->preview == Rect{403.0f, 155.0f, 391.0f, 142.0f});
    CHECK(host.drag.icon.empty());
    host.send(Mouse(MouseEventType::Cancel, 600.0f, 299.0f));
}

namespace {

constexpr Rect kWideDock = {0.0f, 0.0f, 1600.0f, 600.0f};

void CollectLeaves(const DockTiles& tiles, std::vector<std::string>& leaves) {
    if (tiles.children.empty()) {
        leaves.push_back(tiles.pane);
        return;
    }
    for (const auto& child : tiles.children) {
        CollectLeaves(child, leaves);
    }
}

std::vector<std::string> Leaves(const DockTiles& tiles) {
    std::vector<std::string> leaves;
    CollectLeaves(tiles, leaves);
    return leaves;
}

DockTiles ColumnRow(std::vector<DockTiles> children, std::vector<F32> widths = {}) {
    if (widths.empty()) {
        widths.assign(children.size(), 0.0f);
    }
    return DockTiles{.sizes = std::move(widths), .children = std::move(children)};
}

struct ColumnRig {
    Probe a;
    Probe b;
    Probe c;
    std::string flex = "a";
    std::optional<DockTiles> tiles;
    std::vector<std::optional<DockTiles>> emitted;
    std::vector<std::string> closed;
    std::string closeOnlyMinimized;
    ColorRGBA<F32> fillC = {0.0f, 0.0f, 0.0f, 0.0f};
    bool withA = true;
    bool withC = true;
    bool persistView = false;
    Dock dock;

    void update() {
        std::vector<Dock::Pane> panes;
        if (withA) {
            panes.push_back(MakePane("a", &a));
        }
        panes.push_back(MakePane("b", &b));
        if (withC) {
            panes.push_back(MakePane("c", &c));
        }
        for (auto& pane : panes) {
            pane.onClose = [this, id = pane.id] { closed.push_back(id); };
            pane.closeOnlyMinimized = pane.id == closeOnlyMinimized;
            if (pane.id == "c") {
                pane.fill = fillC;
            }
        }
        dock.update({
            .accepts = "panel",
            .panes = std::move(panes),
            .tiles = tiles,
            .onTiles = [this](std::optional<DockTiles> next) {
                emitted.push_back(next);
                tiles = std::move(next);
            },
            .columns = true,
            .flexPane = flex,
            .persistView = persistView,
        });
    }
};

Sakura::Context WideAt(F32 x, F32 y) {
    auto ctx = MakeContext({1600.0f, 600.0f});
    ctx.mouse = Extent2D<F32>{x, y};
    return ctx;
}

}  // namespace

TEST_CASE("Column mode gives each pane a column with default widths and the flex pane takes spare width",
          "[core][sakura][dock][columns]") {
    const auto ctx = MakeContext({1600.0f, 600.0f});
    ColumnRig rig;
    Host host(rig.dock);
    rig.update();
    host.place(ctx, kWideDock);

    CHECK(rig.a.laidFrame == Rect{6.0f, 26.0f, 736.0f, 568.0f});
    CHECK(rig.b.laidFrame == Rect{748.0f, 26.0f, 420.0f, 568.0f});
    CHECK(rig.c.laidFrame == Rect{1174.0f, 26.0f, 420.0f, 568.0f});
    REQUIRE(rig.emitted.size() == 1);
    REQUIRE(rig.emitted[0].has_value());
    CHECK_FALSE(rig.emitted[0]->vertical);
    CHECK(rig.emitted[0]->sizes == std::vector<F32>{0.0f, 0.0f, 0.0f});
    CHECK(Leaves(*rig.emitted[0]) == std::vector<std::string>{"a", "b", "c"});

    rig.flex.clear();
    rig.update();
    host.place(ctx, kWideDock);
    CHECK(rig.a.laidFrame == Rect{6.0f, 26.0f, 525.0f, 568.0f});
    CHECK(rig.b.laidFrame == Rect{537.0f, 26.0f, 526.0f, 568.0f});
    CHECK(rig.c.laidFrame == Rect{1069.0f, 26.0f, 525.0f, 568.0f});
    CHECK(rig.emitted.size() == 1);
}

TEST_CASE("Column mode keeps stored widths, uses defaults for unsized columns and raises widths below the floor",
          "[core][sakura][dock][columns]") {
    ColumnRig rig;
    rig.tiles = ColumnRow({Leaf("a"), Leaf("b"), Leaf("c")}, {0.0f, 300.0f, 20.0f});
    Host host(rig.dock);
    rig.update();
    host.place(MakeContext({1600.0f, 600.0f}), kWideDock);

    CHECK(rig.a.laidFrame == Rect{6.0f, 26.0f, 1212.0f, 568.0f});
    CHECK(rig.b.laidFrame == Rect{1224.0f, 26.0f, 300.0f, 568.0f});
    CHECK(rig.c.laidFrame == Rect{1530.0f, 26.0f, 64.0f, 568.0f});
    CHECK(rig.emitted.empty());
}

TEST_CASE("Column dividers trade width with the neighbour and stop at the minimum width",
          "[core][sakura][dock][columns]") {
    const auto ctx = MakeContext({1600.0f, 600.0f});
    ColumnRig rig;
    rig.tiles = ColumnRow({Leaf("a"), Leaf("b"), Leaf("c")});
    Host host(rig.dock);
    rig.update();
    host.place(ctx, kWideDock);
    CHECK(Peek::CursorAt(rig.dock, {745.0f, 300.0f}) == SurfaceCursor::ResizeEW);
    CHECK(Peek::CursorAt(rig.dock, {740.0f, 300.0f}) == SurfaceCursor::ResizeEW);

    host.send(Mouse(MouseEventType::Click, 745.0f, 300.0f));
    host.send(Mouse(MouseEventType::Move, 645.0f, 300.0f));
    host.place(ctx, kWideDock);
    CHECK(rig.b.laidFrame == Rect{648.0f, 26.0f, 520.0f, 568.0f});
    CHECK(rig.c.laidFrame.x == 1174.0f);
    CHECK(rig.emitted.empty());
    host.send(Mouse(MouseEventType::Release, 645.0f, 300.0f));
    REQUIRE(rig.emitted.size() == 1);
    CHECK(rig.emitted[0]->sizes == std::vector<F32>{0.0f, 520.0f, 0.0f});

    rig.update();
    host.place(ctx, kWideDock);
    host.send(Mouse(MouseEventType::Click, 1171.0f, 300.0f));
    host.send(Mouse(MouseEventType::Move, 1221.0f, 300.0f));
    host.send(Mouse(MouseEventType::Release, 1221.0f, 300.0f));
    rig.update();
    host.place(ctx, kWideDock);
    REQUIRE(rig.emitted.size() == 2);
    CHECK(rig.emitted[1]->sizes == std::vector<F32>{0.0f, 570.0f, 370.0f});
    CHECK(rig.a.laidFrame.width == 636.0f);
    CHECK(rig.b.laidFrame == Rect{648.0f, 26.0f, 570.0f, 568.0f});
    CHECK(rig.c.laidFrame == Rect{1224.0f, 26.0f, 370.0f, 568.0f});

    host.send(Mouse(MouseEventType::Click, 1221.0f, 300.0f));
    host.send(Mouse(MouseEventType::Move, 1600.0f, 300.0f));
    host.place(ctx, kWideDock);
    CHECK(rig.b.laidFrame == Rect{648.0f, 26.0f, 700.0f, 568.0f});
    CHECK(rig.c.laidFrame == Rect{1354.0f, 26.0f, 240.0f, 568.0f});

    host.send(Mouse(MouseEventType::Move, 100.0f, 300.0f));
    host.place(ctx, kWideDock);
    CHECK(rig.a.laidFrame.width == 636.0f);
    CHECK(rig.b.laidFrame == Rect{648.0f, 26.0f, 240.0f, 568.0f});
    CHECK(rig.c.laidFrame == Rect{894.0f, 26.0f, 700.0f, 568.0f});
    host.send(Mouse(MouseEventType::Release, 100.0f, 300.0f));
    REQUIRE(rig.emitted.size() == 3);
    CHECK(rig.emitted[2]->sizes == std::vector<F32>{0.0f, 240.0f, 700.0f});
}

TEST_CASE("Column dividers next to the flex column shrink it down to the minimum width",
          "[core][sakura][dock][columns]") {
    const auto ctx = MakeContext({1600.0f, 600.0f});
    ColumnRig rig;
    rig.tiles = ColumnRow({Leaf("a"), Leaf("b"), Leaf("c")});
    Host host(rig.dock);
    rig.update();
    host.place(ctx, kWideDock);

    host.send(Mouse(MouseEventType::Click, 745.0f, 300.0f));
    host.send(Mouse(MouseEventType::Move, 489.0f, 300.0f));
    host.place(ctx, kWideDock);
    CHECK(rig.a.laidFrame.width == 480.0f);
    CHECK(rig.b.laidFrame.width == 676.0f);
    CHECK(rig.c.laidFrame.right() == 1594.0f);

    host.send(Mouse(MouseEventType::Move, 100.0f, 300.0f));
    host.place(ctx, kWideDock);
    CHECK(rig.a.laidFrame == Rect{6.0f, 26.0f, 240.0f, 568.0f});
    CHECK(rig.b.laidFrame == Rect{252.0f, 26.0f, 916.0f, 568.0f});
    CHECK(rig.c.laidFrame.right() == 1594.0f);
    host.send(Mouse(MouseEventType::Release, 100.0f, 300.0f));
    REQUIRE(rig.emitted.size() == 1);
    CHECK(rig.emitted[0]->sizes == std::vector<F32>{240.0f, 916.0f, 0.0f});
}

TEST_CASE("Columns wider than the window shrink to fit and keep a soft minimum width",
          "[core][sakura][dock][columns]") {
    ColumnRig rig;
    rig.flex.clear();
    rig.tiles = ColumnRow({Leaf("a"), Leaf("b"), Leaf("c")}, {1200.0f, 250.0f, 600.0f});
    Host host(rig.dock);
    rig.update();
    host.place(MakeContext({1600.0f, 600.0f}), kWideDock);
    CHECK(rig.a.laidFrame == Rect{6.0f, 26.0f, 891.0f, 568.0f});
    CHECK(rig.b.laidFrame == Rect{903.0f, 26.0f, 240.0f, 568.0f});
    CHECK(rig.c.laidFrame == Rect{1149.0f, 26.0f, 445.0f, 568.0f});
    CHECK(rig.emitted.empty());

    host.place(MakeContext(), kScreen);
    CHECK(rig.a.laidFrame == Rect{6.0f, 26.0f, 296.0f, 568.0f});
    CHECK(rig.b.laidFrame == Rect{308.0f, 26.0f, 240.0f, 568.0f});
    CHECK(rig.c.laidFrame == Rect{554.0f, 26.0f, 240.0f, 568.0f});

    rig.tiles = ColumnRow({Leaf("a"), Leaf("b"), Leaf("c")}, {800.0f, 400.0f, 400.0f});
    rig.update();
    host.place(MakeContext({600.0f, 600.0f}), {0.0f, 0.0f, 600.0f, 600.0f});
    CHECK(rig.a.laidFrame == Rect{6.0f, 26.0f, 288.0f, 568.0f});
    CHECK(rig.b.laidFrame == Rect{300.0f, 26.0f, 144.0f, 568.0f});
    CHECK(rig.c.laidFrame == Rect{450.0f, 26.0f, 144.0f, 568.0f});
}

TEST_CASE("Double clicking a divider evens out its two neighbours",
          "[core][sakura][dock][columns]") {
    const auto ctx = MakeContext({1600.0f, 600.0f});
    ColumnRig rig;
    rig.tiles = ColumnRow({Leaf("a"), Leaf("b"), Leaf("c")});
    Host host(rig.dock);
    rig.update();
    host.place(ctx, kWideDock);

    host.send(Mouse(MouseEventType::Click, 745.0f, 300.0f, MouseButton::Left, 2));
    host.send(Mouse(MouseEventType::Release, 745.0f, 300.0f));
    REQUIRE(rig.emitted.size() == 1);
    CHECK(rig.emitted[0]->sizes == std::vector<F32>{0.0f, 578.0f, 0.0f});
    rig.update();
    host.place(ctx, kWideDock);
    CHECK(rig.a.laidFrame.width == 578.0f);
    CHECK(rig.b.laidFrame.width == 578.0f);
    CHECK(rig.c.laidFrame.width == 420.0f);
}

TEST_CASE("Double clicking a tile header maximizes it and double clicking again restores the columns",
          "[core][sakura][dock][columns]") {
    const auto ctx = MakeContext();
    ColumnRig rig;
    rig.tiles = ColumnRow({Leaf("a"), Leaf("b"), Leaf("c")});
    Host host(rig.dock);
    rig.update();
    host.place(ctx, kScreen);

    host.send(Mouse(MouseEventType::Click, 200.0f, 16.0f, MouseButton::Left, 2));
    host.send(Mouse(MouseEventType::Release, 200.0f, 16.0f));
    host.place(ctx, kScreen);
    CHECK(rig.a.laidFrame == Body({6.0f, 6.0f, 788.0f, 588.0f}));
    CHECK_FALSE(rig.b.visible());

    host.send(Mouse(MouseEventType::Click, 200.0f, 16.0f, MouseButton::Left, 2));
    host.send(Mouse(MouseEventType::Release, 200.0f, 16.0f));
    host.place(ctx, kScreen);
    CHECK(rig.a.laidFrame == Rect{6.0f, 26.0f, 282.0f, 568.0f});
    CHECK(rig.b.visible());
    CHECK(rig.emitted.empty());
}

TEST_CASE("Dropping on a tile's edges splits that tile in place and a full height tile splits into columns",
          "[core][sakura][dock][columns]") {
    ColumnRig rig;
    rig.tiles = ColumnRow({Leaf("a"), Leaf("b"), Leaf("c")});
    Layer layer;
    layer.update({.content = &rig.dock});
    LayerHost host({&layer});
    rig.update();
    host.place(MakeContext({1600.0f, 600.0f}), kWideDock);

    host.send(Mouse(MouseEventType::Click, 1300.0f, 16.0f));
    host.send(Mouse(MouseEventType::Move, 1310.0f, 16.0f));
    host.send(Mouse(MouseEventType::Move, 958.0f, 500.0f));
    REQUIRE(host.drag.active);
    CHECK(host.drag.pane == "c");
    host.place(WideAt(958.0f, 500.0f), kWideDock);
    REQUIRE(host.drag.target.has_value());
    CHECK(host.drag.target->preview == Rect{748.0f, 303.0f, 420.0f, 291.0f});
    host.send(Mouse(MouseEventType::Release, 958.0f, 500.0f));
    REQUIRE(rig.emitted.size() == 1);
    REQUIRE(rig.emitted[0]->children.size() == 2);
    CHECK(rig.emitted[0]->children[1].vertical);
    CHECK(Leaves(rig.emitted[0]->children[1]) == std::vector<std::string>{"b", "c"});

    rig.update();
    host.place(MakeContext({1600.0f, 600.0f}), kWideDock);
    CHECK(rig.b.laidFrame == Rect{1174.0f, 26.0f, 420.0f, 271.0f});
    CHECK(rig.c.laidFrame == Rect{1174.0f, 323.0f, 420.0f, 271.0f});

    host.send(Mouse(MouseEventType::Click, 1300.0f, 313.0f));
    host.send(Mouse(MouseEventType::Move, 1310.0f, 313.0f));
    host.send(Mouse(MouseEventType::Move, 1580.0f, 150.0f));
    host.place(WideAt(1580.0f, 150.0f), kWideDock);
    REQUIRE(host.drag.target.has_value());
    CHECK(host.drag.target->preview == Rect{1387.0f, 6.0f, 207.0f, 291.0f});
    host.send(Mouse(MouseEventType::Release, 1580.0f, 150.0f));
    REQUIRE(rig.emitted.size() == 2);
    CHECK(rig.emitted[1]->children.size() == 3);

    rig.update();
    host.place(MakeContext({1600.0f, 600.0f}), kWideDock);
    CHECK(rig.b.laidFrame == Rect{1174.0f, 26.0f, 207.0f, 568.0f});
    CHECK(rig.c.laidFrame == Rect{1387.0f, 26.0f, 207.0f, 568.0f});
}

TEST_CASE("A tab dropped into columns takes half of the target shown by the preview",
          "[core][sakura][dock][columns][drop-sizing]") {
    const U64 side = GENERATE(0, 1, 2, 3);
    const F32 width = GENERATE(800.0f, 1600.0f);
    const F32 scale = GENERATE(1.0f, 1.1f, 1.3f, 2.0f, 2.3f);
    const bool extra = GENERATE(false, true);
    CAPTURE(side, width, scale, extra);
    auto ctx = MakeContext({width * scale, 600.0f * scale});
    ctx.pixelRatio = scale;
    const Rect bounds = {0.0f, 0.0f, width * scale, 600.0f * scale};
    Probe editor;
    Probe other;
    Probe terminal;
    editor.consume = false;
    other.consume = false;
    terminal.consume = false;
    Dock source;
    Dock target;
    Emissions emissions;
    auto tab = MakePane("terminal", &terminal);
    tab.group = "terminal";
    source.update({.tabs = true, .accepts = "main", .panes = {tab}});
    Dock::Config config = {
        .accepts = "flowgraph",
        .adopts = {"terminal"},
        .panes = {MakePane("editor", &editor)},
        .tiles = Leaf("editor"),
        .onTiles = emissions.sink(),
        .columns = true,
        .flexPane = "editor",
    };
    if (extra) {
        config.panes.push_back(MakePane("other", &other));
        config.tiles = ColumnRow({Leaf("editor"), Leaf("other")});
    }
    target.update(config);
    Columns columns(source, target);
    Layer layer;
    layer.update({.content = &columns});
    LayerHost host({&layer});
    host.place(ctx, bounds);
    const Rect original = extra ? other.laidFrame : editor.laidFrame;
    const Rect tile = {original.x, original.y - kHeader * scale,
                       original.width, original.height + kHeader * scale};
    const Extent2D<F32> point = {
        side == 0 ? tile.x + 20.0f * scale : side == 1 ? tile.right() - 20.0f * scale : tile.x + tile.width * 0.5f,
        side == 2 ? tile.y + 20.0f * scale : side == 3 ? tile.bottom() - 20.0f * scale : tile.y + tile.height * 0.5f,
    };
    const auto tabRect = source.tabRect("terminal");
    REQUIRE(tabRect.has_value());
    host.send(Mouse(MouseEventType::Click, (tabRect->x + 5.0f) * scale, (tabRect->y + 12.0f) * scale));
    host.send(Mouse(MouseEventType::Move, (tabRect->x + 15.0f) * scale, (tabRect->y + 12.0f) * scale));
    host.send(Mouse(MouseEventType::Move, point.x, point.y));
    REQUIRE(host.drag.active);
    ctx.mouse = point;
    host.place(ctx, bounds);
    REQUIRE(host.drag.target.has_value());
    const Rect preview = host.drag.target->preview;
    host.send(Mouse(MouseEventType::Release, point.x, point.y));
    REQUIRE(emissions.trees.size() == 1);
    config.tiles = emissions.last();
    config.panes.push_back(tab);
    source.update({.tabs = true, .accepts = "main"});
    target.update(config);
    host.place(ctx, bounds);

    const Rect kept = extra ? other.laidFrame : editor.laidFrame;
    const F32 gap = Dock::ContentPadding * scale;
    if (side < 2) {
        CHECK(terminal.laidFrame.width == Catch::Approx((tile.width - gap) * 0.5f).margin(1.0f));
        CHECK(kept.width == Catch::Approx(terminal.laidFrame.width).margin(1.0f));
    } else {
        CHECK(terminal.laidFrame.height + kHeader * scale ==
              Catch::Approx((tile.height - gap) * 0.5f).margin(1.0f));
        CHECK(kept.height == Catch::Approx(terminal.laidFrame.height).margin(1.0f));
    }
    CHECK(terminal.laidFrame.x == Catch::Approx(preview.x).margin(1.0f));
    CHECK(terminal.laidFrame.y - kHeader * scale == Catch::Approx(preview.y).margin(1.0f));
    CHECK(terminal.laidFrame.width == Catch::Approx(preview.width).margin(1.0f));
    CHECK(terminal.laidFrame.height + kHeader * scale == Catch::Approx(preview.height).margin(1.0f));

    if (side < 2) {
        auto& left = side == 0 ? terminal : extra ? other : editor;
        auto& right = side == 1 ? terminal : extra ? other : editor;
        const F32 leftWidth = left.laidFrame.width;
        const F32 rightWidth = right.laidFrame.width;
        const auto resize = [&](F32 delta) {
            const F32 divider = left.laidFrame.right() + gap * 0.5f;
            host.send(Mouse(MouseEventType::Click, divider, 300.0f * scale));
            host.send(Mouse(MouseEventType::Move, divider + delta, 300.0f * scale));
            host.send(Mouse(MouseEventType::Release, divider + delta, 300.0f * scale));
            config.tiles = emissions.last();
            target.update(config);
            host.place(ctx, bounds);
        };
        resize(20.0f * scale);
        CHECK(left.laidFrame.width == Catch::Approx(leftWidth + 20.0f * scale).margin(1.0f));
        CHECK(right.laidFrame.width == Catch::Approx(rightWidth - 20.0f * scale).margin(1.0f));
        if (std::min(leftWidth, rightWidth) < 240.0f * scale) {
            for (const bool shrinkRight : {true, false}) {
                for (const F32 delta : {bounds.width, 20.0f * scale}) {
                    CAPTURE(shrinkRight, delta);
                    resize(shrinkRight ? delta : -delta);
                    CHECK((shrinkRight ? right : left).laidFrame.width ==
                          Catch::Approx(64.0f * scale).margin(1.0f));
                    REQUIRE(config.tiles.has_value());
                    for (const F32 stored : config.tiles->sizes) {
                        CHECK(stored >= 64.0f);
                    }
                }
            }
        }
    }

    const Rect restored = terminal.laidFrame;
    Dock reopened;
    Layer reopenedLayer;
    reopenedLayer.update({.content = &reopened});
    LayerHost reopenedHost({&reopenedLayer});
    reopened.update(config);
    reopenedHost.place(ctx, {bounds.width * 0.5f, 0.0f, bounds.width * 0.5f, bounds.height});
    CHECK(terminal.laidFrame == restored);
    reopenedHost.place(ctx, {bounds.width * 0.5f, 0.0f, bounds.width * 0.4f, bounds.height});
    CHECK(terminal.laidFrame.width <= restored.width + 1.0f);
}

TEST_CASE("Column divider actions preserve floor widths at fractional scales without a flex pane",
          "[core][sakura][dock][columns][drop-sizing]") {
    const F32 scale = GENERATE(1.0f, 1.1f, 1.3f, 2.3f);
    CAPTURE(scale);
    auto ctx = MakeContext({572.0f * scale, 600.0f * scale});
    ctx.pixelRatio = scale;
    const Rect bounds = {0.0f, 0.0f, 572.0f * scale, 600.0f * scale};
    ColumnRig rig;
    rig.flex.clear();
    rig.tiles = ColumnRow({Leaf("a"), Leaf("b"), Leaf("c")}, {420.0f, 64.0f, 64.0f});
    Host host(rig.dock);
    rig.update();
    host.place(ctx, bounds);
    REQUIRE(rig.b.laidFrame.width == Catch::Approx(64.0f * scale).margin(1.0f));
    REQUIRE(rig.c.laidFrame.width == Catch::Approx(64.0f * scale).margin(1.0f));

    SECTION("Dragging also preserves the untouched floor-width column") {
        for (U64 attempt = 0; attempt < 2; ++attempt) {
            const F32 seam = (rig.a.laidFrame.right() + rig.b.laidFrame.x) * 0.5f;
            host.send(Mouse(MouseEventType::Click, seam, 300.0f * scale));
            host.send(Mouse(MouseEventType::Move, seam + 20.0f * scale, 300.0f * scale));
            host.send(Mouse(MouseEventType::Release, seam + 20.0f * scale, 300.0f * scale));
            rig.update();
            host.place(ctx, bounds);
            CHECK(rig.b.laidFrame.width == Catch::Approx(64.0f * scale).margin(1.0f));
            CHECK(rig.c.laidFrame.width == Catch::Approx(64.0f * scale).margin(1.0f));
        }
    }
    SECTION("Equalizing two floor-width columns keeps their saved widths valid") {
        const F32 seam = (rig.b.laidFrame.right() + rig.c.laidFrame.x) * 0.5f;
        host.send(Mouse(MouseEventType::Click, seam, 300.0f * scale, MouseButton::Left, 2));
        host.send(Mouse(MouseEventType::Release, seam, 300.0f * scale));
        rig.update();
        host.place(ctx, bounds);
    }

    REQUIRE(rig.tiles.has_value());
    for (const F32 stored : rig.tiles->sizes) {
        CHECK(stored >= 64.0f);
    }
    ColumnRig restored;
    restored.flex.clear();
    restored.tiles = rig.tiles;
    restored.update();
    Host restoredHost(restored.dock);
    restoredHost.place(ctx, bounds);
    CHECK(restored.b.laidFrame.width == Catch::Approx(64.0f * scale).margin(1.0f));
    CHECK(restored.c.laidFrame.width == Catch::Approx(64.0f * scale).margin(1.0f));
}

TEST_CASE("Side drops wait until every visible column has a storable width",
          "[core][sakura][dock][columns][drop-sizing]") {
    const bool before = GENERATE(false, true);
    const F32 scale = GENERATE(1.0f, 1.25f, 2.0f);
    CAPTURE(before, scale);
    ColumnRig rig;
    rig.tiles = ColumnRow({Leaf("a"), Leaf("b"), Leaf("c")}, {3000.0f, 480.0f, 420.0f});
    const auto original = rig.tiles;
    Layer layer;
    layer.update({.content = &rig.dock});
    LayerHost host({&layer});
    auto ctx = MakeContext({600.0f * scale, 600.0f * scale});
    ctx.pixelRatio = scale;
    Rect bounds = {0.0f, 0.0f, 600.0f * scale, 600.0f * scale};
    rig.update();
    host.place(ctx, bounds);
    REQUIRE(rig.a.laidFrame.width > 134.0f * scale);
    REQUIRE(rig.c.laidFrame.width < 64.0f * scale);
    const Rect narrow = rig.c.laidFrame;
    const auto offer = [&] {
        host.drag.active = true;
        host.drag.pane = "terminal";
        host.drag.group = "panel";
        ctx.mouse = Extent2D<F32>{before ? rig.a.laidFrame.x + 20.0f * scale : rig.a.laidFrame.right() - 20.0f * scale,
                                 300.0f * scale};
        Sakura::Access::invalidateLayout(rig.dock);
        host.place(ctx, bounds);
    };
    offer();
    CHECK_FALSE(host.drag.target.has_value());
    CHECK(rig.emitted.empty());
    CHECK(rig.tiles == original);
    host.drag.reset();

    ColumnRig restored;
    restored.tiles = rig.tiles;
    restored.update();
    Host restoredHost(restored.dock);
    restoredHost.place(ctx, bounds);
    CHECK(restored.c.laidFrame == narrow);

    bounds.width = 800.0f * scale;
    ctx.framebufferSize.x = static_cast<U64>(bounds.width);
    host.place(ctx, bounds);
    REQUIRE(rig.c.laidFrame.width >= 64.0f * scale);
    offer();
    REQUIRE(host.drag.target.has_value());
    const Rect preview = host.drag.target->preview;
    const auto accept = host.drag.target->accept;
    host.drag.reset();
    accept("terminal");
    REQUIRE(rig.emitted.size() == 1);

    Probe terminal;
    Dock reopened;
    reopened.update({
        .panes = {MakePane("a", &rig.a), MakePane("b", &rig.b), MakePane("c", &rig.c),
                  MakePane("terminal", &terminal)},
        .tiles = rig.tiles,
        .columns = true,
        .flexPane = "a",
    });
    Host reopenedHost(reopened);
    reopenedHost.place(ctx, bounds);
    CHECK(terminal.laidFrame.width == Catch::Approx(preview.width).margin(1.0f));
    CHECK(terminal.laidFrame.x == Catch::Approx(preview.x).margin(1.0f));
    CHECK(rig.c.laidFrame.width >= 64.0f * scale);
}

TEST_CASE("Side drops leave spare editor width available when the window narrows",
          "[core][sakura][dock][columns][drop-sizing]") {
    const bool before = GENERATE(false, true);
    CAPTURE(before);
    auto ctx = MakeContext({2560.0f, 600.0f});
    const Rect bounds = {0.0f, 0.0f, 2560.0f, 600.0f};
    Probe editor;
    Probe description;
    Probe other;
    Probe terminal;
    Emissions emissions;
    Dock dock;
    Dock::Config config = {
        .accepts = "panel",
        .panes = {MakePane("editor", &editor), MakePane("description", &description), MakePane("other", &other)},
        .tiles = ColumnRow({Leaf("editor"), Leaf("description"), Leaf("other")}, {0.0f, 420.0f, 600.0f}),
        .onTiles = emissions.sink(),
        .columns = true,
        .flexPane = "editor",
    };
    dock.update(config);
    Layer layer;
    layer.update({.content = &dock});
    LayerHost host({&layer});
    host.place(ctx, bounds);
    REQUIRE(editor.laidFrame.width == 1516.0f);
    REQUIRE(other.laidFrame.width == 600.0f);

    host.drag.active = true;
    host.drag.pane = "terminal";
    host.drag.group = "panel";
    ctx.mouse = Extent2D<F32>{before ? other.laidFrame.x + 20.0f : other.laidFrame.right() - 20.0f, 300.0f};
    Sakura::Access::invalidateLayout(dock);
    host.place(ctx, bounds);
    REQUIRE(host.drag.target.has_value());
    const auto accept = host.drag.target->accept;
    host.drag.reset();
    accept("terminal");
    REQUIRE(emissions.trees.size() == 1);
    config.tiles = emissions.last();
    config.panes.push_back(MakePane("terminal", &terminal));
    dock.update(config);
    host.place(ctx, bounds);
    CHECK(description.laidFrame.width == 420.0f);
    CHECK(other.laidFrame.width == 297.0f);
    CHECK(terminal.laidFrame.width == 297.0f);

    const auto checkNarrow = [&] {
        CHECK(editor.laidFrame.width == 556.0f);
        CHECK(description.laidFrame.width == 420.0f);
        CHECK(other.laidFrame.width == 297.0f);
        CHECK(terminal.laidFrame.width == 297.0f);
    };
    host.place(MakeContext({1600.0f, 600.0f}), kWideDock);
    checkNarrow();

    Dock reopened;
    Host reopenedHost(reopened);
    reopened.update(config);
    reopenedHost.place(MakeContext({1600.0f, 600.0f}), kWideDock);
    checkNarrow();
}

TEST_CASE("Closing a pane removes its column and the flex column takes the space back",
          "[core][sakura][dock][columns]") {
    const auto ctx = MakeContext({1600.0f, 600.0f});
    ColumnRig rig;
    rig.tiles = ColumnRow({Leaf("a"), Leaf("b"), Leaf("c")});
    Host host(rig.dock);
    rig.update();
    host.place(ctx, kWideDock);

    const auto close = CloseOf({1174.0f, 6.0f, 420.0f, 588.0f});
    host.click(close.x, close.y);
    CHECK(rig.closed == std::vector<std::string>{"c"});
    REQUIRE(rig.emitted.size() == 1);
    CHECK(Leaves(*rig.emitted[0]) == std::vector<std::string>{"a", "b"});

    rig.withC = false;
    rig.update();
    host.place(ctx, kWideDock);
    CHECK(rig.a.laidFrame.width == 1162.0f);
    CHECK(rig.b.laidFrame.x == 1174.0f);
}

TEST_CASE("Maximizing a column pane fills the viewport and hides the other columns",
          "[core][sakura][dock][columns]") {
    const auto ctx = MakeContext();
    ColumnRig rig;
    rig.tiles = ColumnRow({Leaf("a"), Leaf("b"), Leaf("c")});
    Host host(rig.dock);
    rig.update();
    host.place(ctx, kScreen);

    const auto maximize = MaximizeOf({6.0f, 6.0f, 282.0f, 588.0f});
    host.click(maximize.x, maximize.y);
    host.place(ctx, kScreen);
    CHECK(rig.a.laidFrame == Body({6.0f, 6.0f, 788.0f, 588.0f}));
    CHECK_FALSE(rig.b.visible());
    CHECK_FALSE(rig.c.visible());

    const auto restore = MaximizeOf({6.0f, 6.0f, 788.0f, 588.0f});
    host.click(restore.x, restore.y);
    host.place(ctx, kScreen);
    CHECK(rig.b.visible());
    CHECK(rig.a.laidFrame == Rect{6.0f, 26.0f, 282.0f, 568.0f});
    CHECK(rig.b.laidFrame.x == 294.0f);
    CHECK(rig.emitted.empty());
}

TEST_CASE("Pane header actions sit left of maximize, show their state and hover, and run on release",
          "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    U64 clicks = 0;
    bool active = false;
    Dock dock;
    Host host(dock);
    const auto update = [&] {
        auto paneA = MakePane("a");
        paneA.onClose = [] {};
        paneA.actions = {{.icon = ICON_LC_SLIDERS_HORIZONTAL, .active = active, .onClick = [&] { ++clicks; }}};
        dock.update({.panes = {paneA, MakePane("b")}});
        host.place(ctx, kScreen);
    };
    update();

    const Rect card = {6.0f, 6.0f, 391.0f, 588.0f};
    const auto* action = GlyphIn(dock, card, ICON_LC_SLIDERS_HORIZONTAL);
    const auto* expand = GlyphIn(dock, card, ICON_LC_MAXIMIZE_2);
    REQUIRE(action);
    REQUIRE(expand);
    const Rect actionFrame = Peek::Frame(*action);
    const Rect expandFrame = Peek::Frame(*expand);
    CHECK(actionFrame.right() < expandFrame.x);
    CHECK(actionFrame.y + actionFrame.height * 0.5f == Catch::Approx(expandFrame.y + expandFrame.height * 0.5f));
    CHECK(action->config().colorKey == "tile_header_text");
    CHECK_FALSE(GlyphIn(dock, {403.0f, 6.0f, 391.0f, 588.0f}, ICON_LC_SLIDERS_HORIZONTAL));

    active = true;
    update();
    CHECK(GlyphIn(dock, card, ICON_LC_SLIDERS_HORIZONTAL)->config().colorKey == "cyber_blue");

    const Extent2D<F32> center = {actionFrame.x + actionFrame.width * 0.5f, actionFrame.y + actionFrame.height * 0.5f};
    host.send(Mouse(MouseEventType::Move, center.x, center.y));
    update();
    CHECK(GlyphIn(dock, card, ICON_LC_SLIDERS_HORIZONTAL)->config().colorKey == "tile_button_hover");
    CHECK(Peek::CursorAt(dock, center) == SurfaceCursor::Hand);

    host.send(Mouse(MouseEventType::Click, center.x, center.y));
    CHECK(clicks == 0);
    host.send(Mouse(MouseEventType::Release, center.x + 200.0f, center.y));
    CHECK(clicks == 0);
    host.click(center.x, center.y);
    CHECK(clicks == 1);
}

TEST_CASE("Strip tab close button shows on the active or hovered tab, faint until hovered",
          "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Dock dock;
    Host host(dock);
    U64 closed = 0;
    auto first = MakePane("a");
    first.onClose = [&]() {
        ++closed;
    };
    dock.update({.tabs = true, .panes = {first, MakePane("b")}});
    host.place(ctx, kScreen);

    const auto opacities = [&] {
        std::vector<F32> values;
        for (auto* text : Direct<Sakura::Text>(dock)) {
            if (text->config().str == ICON_LC_X && text->config().opacity > 0.0f) {
                values.push_back(text->config().opacity);
            }
        }
        return values;
    };
    CHECK(opacities() == std::vector<F32>{0.35f});
    for (auto* text : Direct<Sakura::Text>(dock)) {
        if (text->config().str == ICON_LC_X && text->config().opacity > 0.0f) {
            const Rect frame = Peek::Frame(*text);
            CHECK(frame.x + frame.width * 0.5f == Catch::Approx(492.0f - 7.0f - 8.0f));
        }
    }

    host.send(Mouse(MouseEventType::Move, 300.0f, 19.0f));
    host.place(ctx, kScreen);
    CHECK(opacities() == std::vector<F32>{1.0f});

    host.send(Mouse(MouseEventType::Move, 100.0f, 19.0f));
    host.place(ctx, kScreen);
    CHECK(opacities() == std::vector<F32>{1.0f, 0.35f});
    std::vector<F32> centers;
    for (auto* text : Direct<Sakura::Text>(dock)) {
        if (text->config().str == ICON_LC_X && text->config().opacity > 0.0f) {
            const Rect frame = Peek::Frame(*text);
            centers.push_back(frame.x + frame.width * 0.5f);
        }
    }
    REQUIRE(centers.size() == 2);
    CHECK(centers[0] == Catch::Approx(248.0f - 7.0f - 8.0f));
    CHECK(centers[1] == Catch::Approx(492.0f - 7.0f - 8.0f));

    host.send(Mouse(MouseEventType::Move, 248.0f - 7.0f - 8.0f, 19.0f));
    host.place(ctx, kScreen);
    host.click(248.0f - 7.0f - 8.0f, 19.0f);
    host.place(ctx, kScreen);
    CHECK(closed == 1);

    host.send(Mouse(MouseEventType::Move, 300.0f, 200.0f));
    host.place(ctx, kScreen);
    CHECK(opacities() == std::vector<F32>{0.35f});
}

TEST_CASE("Column dividers at a pixel ratio of two emit logical widths that lay out again at twice their size",
          "[core][sakura][dock][columns]") {
    auto ctx = MakeContext({3200.0f, 1200.0f});
    ctx.pixelRatio = 2.0f;
    const Rect bounds = {0.0f, 0.0f, 3200.0f, 1200.0f};
    ColumnRig rig;
    rig.tiles = ColumnRow({Leaf("a"), Leaf("b"), Leaf("c")});
    Host host(rig.dock);
    rig.update();
    host.place(ctx, bounds);
    CHECK(rig.a.laidFrame == Rect{12.0f, 52.0f, 1472.0f, 1136.0f});
    CHECK(rig.b.laidFrame == Rect{1496.0f, 52.0f, 840.0f, 1136.0f});
    CHECK(rig.c.laidFrame == Rect{2348.0f, 52.0f, 840.0f, 1136.0f});

    host.send(Mouse(MouseEventType::Click, 1490.0f, 600.0f));
    host.send(Mouse(MouseEventType::Move, 1290.0f, 600.0f));
    host.place(ctx, bounds);
    CHECK(rig.b.laidFrame == Rect{1296.0f, 52.0f, 1040.0f, 1136.0f});
    host.send(Mouse(MouseEventType::Release, 1290.0f, 600.0f));
    REQUIRE(rig.emitted.size() == 1);
    CHECK(rig.emitted[0]->sizes == std::vector<F32>{0.0f, 520.0f, 0.0f});

    rig.update();
    host.place(ctx, bounds);
    CHECK(rig.b.laidFrame == Rect{1296.0f, 52.0f, 1040.0f, 1136.0f});
    host.send(Mouse(MouseEventType::Click, 2342.0f, 600.0f, MouseButton::Left, 2));
    host.send(Mouse(MouseEventType::Release, 2342.0f, 600.0f));
    REQUIRE(rig.emitted.size() == 2);
    CHECK(rig.emitted[1]->sizes == std::vector<F32>{0.0f, 470.0f, 470.0f});

    rig.flex.clear();
    rig.update();
    host.place(ctx, bounds);
    CHECK(rig.a.laidFrame.width == 973.0f);
    CHECK(rig.b.laidFrame.width == 1090.0f);
    CHECK(rig.c.laidFrame.width == 1089.0f);
    host.send(Mouse(MouseEventType::Click, 991.0f, 600.0f));
    host.send(Mouse(MouseEventType::Move, 1091.0f, 600.0f));
    host.send(Mouse(MouseEventType::Release, 1091.0f, 600.0f));
    REQUIRE(rig.emitted.size() == 3);
    CheckSizes(rig.emitted[2]->sizes, {536.70587f, 494.64703f, 544.64703f});
    rig.update();
    host.place(ctx, bounds);
    CHECK(rig.a.laidFrame.width == 1073.0f);
    CHECK(rig.b.laidFrame.width == 990.0f);
    CHECK(rig.c.laidFrame == Rect{2099.0f, 52.0f, 1089.0f, 1136.0f});
}

TEST_CASE("Dropping on a tile's top stacks above it in place and a tile is never offered as its own target",
          "[core][sakura][dock][columns]") {
    ColumnRig rig;
    rig.tiles = ColumnRow({Leaf("a"), Leaf("b"), Leaf("c")});
    Layer layer;
    layer.update({.content = &rig.dock});
    LayerHost host({&layer});
    rig.update();
    host.place(MakeContext({1600.0f, 600.0f}), kWideDock);

    host.send(Mouse(MouseEventType::Click, 1300.0f, 16.0f));
    host.send(Mouse(MouseEventType::Move, 1310.0f, 16.0f));
    host.send(Mouse(MouseEventType::Move, 958.0f, 20.0f));
    host.place(WideAt(958.0f, 20.0f), kWideDock);
    REQUIRE(host.drag.target.has_value());
    CHECK(host.drag.target->preview == Rect{748.0f, 6.0f, 420.0f, 291.0f});
    host.send(Mouse(MouseEventType::Release, 958.0f, 20.0f));
    REQUIRE(rig.emitted.size() == 1);
    REQUIRE(rig.emitted[0].has_value());
    REQUIRE(rig.emitted[0]->children.size() == 2);
    CHECK(rig.emitted[0]->children[1].vertical);
    CHECK(Leaves(rig.emitted[0]->children[1]) == std::vector<std::string>{"c", "b"});

    rig.update();
    host.place(MakeContext({1600.0f, 600.0f}), kWideDock);
    CHECK(rig.c.laidFrame == Body({1174.0f, 6.0f, 420.0f, 291.0f}));
    CHECK(rig.b.laidFrame == Body({1174.0f, 303.0f, 420.0f, 291.0f}));

    host.send(Mouse(MouseEventType::Click, 1300.0f, 16.0f));
    host.send(Mouse(MouseEventType::Move, 1310.0f, 16.0f));
    for (const Extent2D<F32> point : {Extent2D<F32>{1300.0f, 270.0f}, Extent2D<F32>{1570.0f, 150.0f},
                                      Extent2D<F32>{1198.0f, 150.0f}, Extent2D<F32>{1300.0f, 30.0f}}) {
        host.send(Mouse(MouseEventType::Move, point.x, point.y));
        host.place(WideAt(point.x, point.y), kWideDock);
        CHECK_FALSE(host.drag.target.has_value());
    }
}

TEST_CASE("A stacked tile moved onto another column keeps its height and lands where the preview showed",
          "[core][sakura][dock][columns]") {
    ColumnRig rig;
    rig.tiles = ColumnRow({Leaf("a"), Column({700.0f, 300.0f}, {Leaf("b"), Leaf("c")})});
    Layer layer;
    layer.update({.content = &rig.dock});
    LayerHost host({&layer});
    rig.update();
    host.place(MakeContext({1600.0f, 600.0f}), kWideDock);
    const Rect before = rig.c.laidFrame;
    REQUIRE(before.height < 250.0f);

    host.send(Mouse(MouseEventType::Click, before.x + 100.0f, before.y - kHeader * 0.5f));
    host.send(Mouse(MouseEventType::Move, before.x + 110.0f, before.y - kHeader * 0.5f));
    host.send(Mouse(MouseEventType::Move, 300.0f, 590.0f));
    host.place(WideAt(300.0f, 590.0f), kWideDock);
    REQUIRE(host.drag.target.has_value());
    const Rect preview = host.drag.target->preview;
    host.send(Mouse(MouseEventType::Release, 300.0f, 590.0f));
    REQUIRE(rig.emitted.size() == 1);

    rig.update();
    host.place(MakeContext({1600.0f, 600.0f}), kWideDock);
    CHECK(rig.c.laidFrame.x < 600.0f);
    CHECK(rig.c.laidFrame.height == Catch::Approx(before.height).margin(1.0f));
    const Rect landed = Body(preview);
    CHECK(rig.c.laidFrame.x == Catch::Approx(landed.x).margin(1.0f));
    CHECK(rig.c.laidFrame.y == Catch::Approx(landed.y).margin(1.0f));
    CHECK(rig.c.laidFrame.width == Catch::Approx(landed.width).margin(1.0f));
    CHECK(rig.c.laidFrame.height == Catch::Approx(landed.height).margin(1.0f));
}

TEST_CASE("A tile stacked into a column that already holds rows keeps its height",
          "[core][sakura][dock][columns]") {
    Probe a;
    Probe b;
    Probe c;
    Probe d;
    Dock dock;
    Emissions emissions;
    Dock::Config config = {
        .accepts = "panel",
        .panes = {MakePane("a", &a), MakePane("b", &b), MakePane("c", &c), MakePane("d", &d)},
        .tiles = ColumnRow({Column({500.0f, 500.0f}, {Leaf("a"), Leaf("b")}), Column({800.0f, 200.0f}, {Leaf("c"), Leaf("d")})}),
        .onTiles = emissions.sink(),
        .columns = true,
    };
    dock.update(config);
    Layer layer;
    layer.update({.content = &dock});
    LayerHost host({&layer});
    host.place(MakeContext({1600.0f, 600.0f}), kWideDock);
    const Rect before = d.laidFrame;
    const Rect target = a.laidFrame;
    REQUIRE(before.height + kHeader < (target.height + kHeader - 6.0f) * 0.5f);

    host.send(Mouse(MouseEventType::Click, before.x + 100.0f, before.y - kHeader * 0.5f));
    host.send(Mouse(MouseEventType::Move, before.x + 110.0f, before.y - kHeader * 0.5f));
    const Extent2D<F32> point = {target.x + target.width * 0.5f, target.bottom() - 20.0f};
    host.send(Mouse(MouseEventType::Move, point.x, point.y));
    host.place(WideAt(point.x, point.y), kWideDock);
    REQUIRE(host.drag.target.has_value());
    const Rect preview = host.drag.target->preview;
    host.send(Mouse(MouseEventType::Release, point.x, point.y));
    REQUIRE(emissions.trees.size() == 1);

    config.tiles = emissions.last();
    dock.update(config);
    host.place(MakeContext({1600.0f, 600.0f}), kWideDock);
    CHECK(d.laidFrame.x == Catch::Approx(target.x).margin(1.0f));
    CHECK(d.laidFrame.height == Catch::Approx(before.height).margin(1.0f));
    CHECK(preview.bottom() == Catch::Approx(target.bottom()).margin(1.0f));
    CHECK(Body(preview).height == Catch::Approx(d.laidFrame.height).margin(1.0f));
}

TEST_CASE("A tile stacked under a full width editor takes half of the target when moved beside it",
          "[core][sakura][dock][columns]") {
    ColumnRig rig;
    rig.withC = false;
    rig.tiles = Column({600.0f, 400.0f}, {Leaf("a"), Leaf("b")});
    Layer layer;
    layer.update({.content = &rig.dock});
    LayerHost host({&layer});
    rig.update();
    host.place(MakeContext({1600.0f, 600.0f}), kWideDock);
    const Rect editor = rig.a.laidFrame;
    const Rect before = rig.b.laidFrame;
    REQUIRE(before.width > 1500.0f);

    host.send(Mouse(MouseEventType::Click, before.x + 100.0f, before.y - kHeader * 0.5f));
    host.send(Mouse(MouseEventType::Move, before.x + 110.0f, before.y - kHeader * 0.5f));
    const Extent2D<F32> point = {editor.right() - 20.0f, editor.y + editor.height * 0.5f};
    host.send(Mouse(MouseEventType::Move, point.x, point.y));
    host.place(WideAt(point.x, point.y), kWideDock);
    REQUIRE(host.drag.target.has_value());
    const Rect preview = host.drag.target->preview;
    host.send(Mouse(MouseEventType::Release, point.x, point.y));
    REQUIRE(rig.emitted.size() == 1);

    rig.update();
    host.place(MakeContext({1600.0f, 600.0f}), kWideDock);
    CHECK(rig.b.laidFrame.x > rig.a.laidFrame.x);
    CHECK(rig.a.laidFrame.width == Catch::Approx(rig.b.laidFrame.width).margin(1.0f));
    CHECK(rig.a.laidFrame.width > 700.0f);
    CHECK(rig.b.laidFrame.x == Catch::Approx(preview.x).margin(1.0f));
    CHECK(rig.b.laidFrame.width == Catch::Approx(preview.width).margin(1.0f));
}

TEST_CASE("A minimized tile moved to the dock edge stays minimized as a strip column taken from the edge column",
          "[core][sakura][dock][columns]") {
    ColumnRig rig;
    rig.tiles = ColumnRow({Leaf("a"), Column({500.0f, 500.0f}, {Leaf("b"), Leaf("c")})});
    Layer layer;
    layer.update({.content = &rig.dock});
    LayerHost host({&layer});
    rig.update();
    host.place(MakeContext({1600.0f, 600.0f}), kWideDock);

    const auto minimize = MinimizeOf({1174.0f, 303.0f, 420.0f, 291.0f});
    host.send(Mouse(MouseEventType::Click, minimize.x, minimize.y));
    host.send(Mouse(MouseEventType::Release, minimize.x, minimize.y));
    host.place(MakeContext({1600.0f, 600.0f}), kWideDock);
    REQUIRE_FALSE(rig.c.visible());

    host.send(Mouse(MouseEventType::Click, 1250.0f, 578.0f));
    host.send(Mouse(MouseEventType::Move, 1260.0f, 578.0f));
    host.send(Mouse(MouseEventType::Move, 1590.0f, 150.0f));
    host.place(WideAt(1590.0f, 150.0f), kWideDock);
    REQUIRE(host.drag.active);
    REQUIRE(host.drag.target.has_value());
    const Rect preview = host.drag.target->preview;
    CHECK(preview.width == 32.0f);
    CHECK(preview.right() == 1594.0f);
    host.send(Mouse(MouseEventType::Release, 1590.0f, 150.0f));
    REQUIRE_FALSE(rig.emitted.empty());

    rig.update();
    host.place(MakeContext({1600.0f, 600.0f}), kWideDock);
    CHECK_FALSE(rig.c.visible());
    CHECK(rig.b.laidFrame.width == Catch::Approx(420.0f - 6.0f - 32.0f).margin(1.0f));
    REQUIRE(rig.emitted.back().has_value());
    CHECK(rig.emitted.back()->children.back().pane == "c");
}

namespace {

enum class MoveSource : U8 { Lone, Stacked, Row, MinimizedStacked, MinimizedLone, Foreign };
enum class MoveDrop : U8 { Left, Right, Top, Bottom, DockLeft, DockRight };

const DockTiles* FindLeaf(const DockTiles& tiles, const std::string& pane) {
    if (tiles.children.empty()) {
        return tiles.pane == pane ? &tiles : nullptr;
    }
    for (const auto& child : tiles.children) {
        if (const auto* found = FindLeaf(child, pane)) {
            return found;
        }
    }
    return nullptr;
}

std::optional<DockTiles> WithoutPane(const DockTiles& tiles, const std::string& pane) {
    if (tiles.children.empty()) {
        return tiles.pane == pane ? std::nullopt : std::optional<DockTiles>(tiles);
    }
    DockTiles result = tiles;
    result.children.clear();
    result.sizes.clear();
    for (U64 index = 0; index < tiles.children.size(); ++index) {
        if (auto kept = WithoutPane(tiles.children[index], pane)) {
            result.children.push_back(std::move(*kept));
            result.sizes.push_back(index < tiles.sizes.size() ? tiles.sizes[index] : 1.0f);
        }
    }
    if (result.children.empty()) {
        return std::nullopt;
    }
    if (result.children.size() == 1) {
        return std::move(result.children.front());
    }
    return result;
}

struct MoveRig {
    F32 scale = 1.0f;
    MoveSource source = MoveSource::Lone;
    Probe a;
    Probe b;
    Probe c;
    Probe d;
    Probe e;
    Probe f;
    Probe t;
    Dock tabs;
    Dock dock;
    Emissions emissions;
    Dock::Config config;
    Dock::Pane tab;
    Columns columns{tabs, dock};
    Layer layer;
    std::unique_ptr<LayerHost> host;
    Sakura::Context ctx;
    Rect bounds;

    MoveRig(F32 scale, bool flex, MoveSource source, const std::string& skip = "", bool stacked = false)
        : scale(scale), source(source) {
        for (auto* probe : {&a, &b, &c, &d, &e, &f, &t}) {
            probe->consume = false;
        }
        tab = MakePane("t", &t);
        tab.group = "terminal";
        tabs.update({.tabs = true, .accepts = "main", .panes = skip == "t" ? std::vector<Dock::Pane>{}
                                                                            : std::vector<Dock::Pane>{tab}});
        auto upper = Leaf("b");
        upper.minimized = source == MoveSource::MinimizedStacked;
        auto last = Leaf("d");
        last.minimized = source == MoveSource::MinimizedLone;
        std::optional<DockTiles> tiles = ColumnRow(
            {stacked ? Column({500.0f, 500.0f}, {Leaf("a"), Leaf("f")}) : Leaf("a"),
             Column({600.0f, 400.0f}, {upper, Row({500.0f, 500.0f}, {Leaf("c"), Leaf("e")})}), last},
            {0.0f, 0.0f, 360.0f});
        std::vector<Dock::Pane> panes;
        std::vector<std::pair<std::string, Probe*>> named = {{"a", &a}, {"b", &b}, {"c", &c}, {"d", &d}, {"e", &e}};
        if (stacked) {
            named.push_back({"f", &f});
        }
        for (const auto& [name, probe] : named) {
            if (name != skip) {
                panes.push_back(MakePane(name, probe));
            }
        }
        if (!skip.empty() && skip != "t") {
            tiles = WithoutPane(*tiles, skip);
        }
        config = {
            .accepts = "panel",
            .adopts = {"terminal"},
            .panes = std::move(panes),
            .tiles = tiles,
            .onTiles = emissions.sink(),
            .columns = true,
            .flexPane = flex ? "a" : "",
            .persistView = true,
        };
        dock.update(config);
        layer.update({.content = &columns});
        host = std::make_unique<LayerHost>(std::vector<Layer*>{&layer});
        ctx = MakeContext({3200.0f * scale, 600.0f * scale});
        ctx.pixelRatio = scale;
        bounds = {0.0f, 0.0f, 3200.0f * scale, 600.0f * scale};
        host->place(ctx, bounds);
    }

    Probe& probe(const std::string& name) {
        return name == "a"   ? a
               : name == "b" ? b
               : name == "c" ? c
               : name == "d" ? d
               : name == "e" ? e
               : name == "f" ? f
                             : t;
    }

    std::string moved() const {
        switch (source) {
            case MoveSource::Lone:
            case MoveSource::MinimizedLone:
                return "d";
            case MoveSource::Stacked:
            case MoveSource::MinimizedStacked:
                return "b";
            case MoveSource::Row:
                return "c";
            case MoveSource::Foreign:
                return "t";
        }
        return "";
    }

    bool minimizedSource() const {
        return source == MoveSource::MinimizedStacked || source == MoveSource::MinimizedLone;
    }

    Rect tile(const Probe& probe) const {
        const Rect body = probe.laidFrame;
        return {body.x, body.y - kHeader * scale, body.width, body.height + kHeader * scale};
    }

    Rect viewport() const {
        const F32 inset = Dock::ContentPadding * scale;
        return {1600.0f * scale + inset, inset, 1600.0f * scale - 2.0f * inset, 600.0f * scale - 2.0f * inset};
    }

    Rect rect(const std::string& name) {
        Probe& target = probe(name);
        if (target.visible()) {
            return tile(target);
        }
        const F32 strip = 32.0f * scale;
        if (name == "d") {
            return {viewport().right() - strip, viewport().y, strip, viewport().height};
        }
        const Rect left = tile(c);
        const Rect right = tile(e);
        return {left.x, viewport().y, right.right() - left.x, strip};
    }

    Extent2D<F32> grab() {
        switch (source) {
            case MoveSource::Lone:
                return {tile(d).x + 40.0f * scale, tile(d).y + 10.0f * scale};
            case MoveSource::Stacked:
                return {tile(b).x + 40.0f * scale, tile(b).y + 10.0f * scale};
            case MoveSource::Row:
                return {tile(c).x + 40.0f * scale, tile(c).y + 10.0f * scale};
            case MoveSource::MinimizedStacked:
                return {tile(c).x + 40.0f * scale, viewport().y + 10.0f * scale};
            case MoveSource::MinimizedLone:
                return {viewport().right() - 16.0f * scale, 300.0f * scale};
            case MoveSource::Foreign: {
                const auto rect = tabs.tabRect("t");
                REQUIRE(rect.has_value());
                return {(rect->x + 5.0f) * scale, (rect->y + 12.0f) * scale};
            }
        }
        return {};
    }

    Extent2D<F32> point(MoveDrop drop, const Rect& target) const {
        const F32 edge = 20.0f * scale;
        switch (drop) {
            case MoveDrop::Left:
                return {target.x + edge, target.y + target.height * 0.5f};
            case MoveDrop::Right:
                return {target.right() - edge, target.y + target.height * 0.5f};
            case MoveDrop::Top:
                return {target.x + target.width * 0.5f, target.y + edge};
            case MoveDrop::Bottom:
                return {target.x + target.width * 0.5f, target.bottom() - edge};
            case MoveDrop::DockLeft:
                return {1600.0f * scale + 1.0f, 300.0f * scale};
            case MoveDrop::DockRight:
                return {3200.0f * scale - 1.0f, 300.0f * scale};
        }
        return {};
    }

    void start() {
        const auto from = grab();
        host->send(Mouse(MouseEventType::Click, from.x, from.y));
        host->send(Mouse(MouseEventType::Move, from.x + 10.0f * scale, from.y));
    }

    std::optional<Rect> hover(Extent2D<F32> at) {
        host->send(Mouse(MouseEventType::Move, at.x, at.y));
        auto hovered = ctx;
        hovered.mouse = at;
        host->place(hovered, bounds);
        if (!host->drag.target.has_value()) {
            return std::nullopt;
        }
        return host->drag.target->preview;
    }

    void settle() {
        config.tiles = emissions.last();
        if (source == MoveSource::Foreign) {
            config.panes.push_back(tab);
            tabs.update({.tabs = true, .accepts = "main"});
        }
        dock.update(config);
        host->place(ctx, bounds);
    }
};

F32 Along(const Rect& rect, bool vertical) {
    return vertical ? rect.height : rect.width;
}

}  // namespace

TEST_CASE("Dropping on a tile splits it in place and only the target gives up space",
          "[core][sakura][dock][columns][drop-rules]") {
    const F32 scale = GENERATE(1.0f, 1.25f, 1.5f, 2.0f);
    const bool flex = GENERATE(false, true);
    const auto source = GENERATE(MoveSource::Lone, MoveSource::Stacked, MoveSource::Row,
                                 MoveSource::MinimizedStacked, MoveSource::MinimizedLone, MoveSource::Foreign);
    const auto drop = GENERATE(MoveDrop::Left, MoveDrop::Right, MoveDrop::Top, MoveDrop::Bottom);
    const std::string target = GENERATE(std::string("a"), std::string("b"), std::string("e"));
    CAPTURE(scale, flex, static_cast<int>(source), static_cast<int>(drop), target);

    MoveRig rig(scale, flex, source);
    const std::string moved = rig.moved();
    if (target == moved) {
        return;
    }
    const F32 gap = Dock::ContentPadding * scale;
    const F32 strip = 32.0f * scale;
    const F32 margin = 2.0f;
    const bool vertical = drop == MoveDrop::Top || drop == MoveDrop::Bottom;
    Probe& movedProbe = rig.probe(moved);
    Probe& targetProbe = rig.probe(target);
    const bool targetMinimized = source == MoveSource::MinimizedStacked && target == "b";
    const Rect before = rig.tile(targetProbe);
    const F32 own = source == MoveSource::Foreign || rig.minimizedSource() ? 0.0f
                                                                          : Along(rig.tile(movedProbe), vertical);

    rig.start();
    REQUIRE(rig.host->drag.active);
    const auto at = rig.point(drop, before);
    const auto preview = rig.hover(at);
    if (targetMinimized) {
        CHECK_FALSE(preview.has_value());
        return;
    }
    REQUIRE(preview.has_value());
    rig.host->send(Mouse(MouseEventType::Release, at.x, at.y));
    REQUIRE(rig.emissions.trees.size() == 1);
    REQUIRE(rig.emissions.last().has_value());
    rig.settle();

    MoveRig reference(scale, flex, MoveSource::Lone, moved);
    std::unordered_map<std::string, Rect> expected;
    for (const auto name : {"a", "b", "c", "d", "e"}) {
        if (name != moved && reference.probe(name).visible()) {
            expected[name] = reference.tile(reference.probe(name));
        }
    }
    REQUIRE(expected.contains(target));
    const Rect slot = expected.at(target);
    const F32 half = (Along(slot, vertical) - gap) * 0.5f;
    const bool column = !vertical && target == "a";
    F32 size = own > 0.0f ? std::min(own, half) : half;
    if (column) {
        size = std::max(size, 64.0f * scale);
    }
    if (rig.minimizedSource()) {
        size = strip;
    }

    const Rect after = rig.tile(targetProbe);
    CHECK(Along(after, vertical) == Catch::Approx(Along(slot, vertical) - gap - size).margin(margin));
    CHECK(Along(after, !vertical) == Catch::Approx(Along(slot, !vertical)).margin(margin));
    for (const auto& [name, rect] : expected) {
        if (name == target) {
            continue;
        }
        CAPTURE(name);
        REQUIRE(rig.probe(name).visible());
        CHECK(rig.tile(rig.probe(name)).x == Catch::Approx(rect.x).margin(margin));
        CHECK(rig.tile(rig.probe(name)).y == Catch::Approx(rect.y).margin(margin));
        CHECK(rig.tile(rig.probe(name)).width == Catch::Approx(rect.width).margin(margin));
        CHECK(rig.tile(rig.probe(name)).height == Catch::Approx(rect.height).margin(margin));
    }

    const F32 thickness = std::min(size, Along(before, vertical));
    CHECK(Along(*preview, vertical) == Catch::Approx(thickness).margin(margin));
    CHECK(Along(*preview, !vertical) == Catch::Approx(Along(before, !vertical)).margin(margin));
    switch (drop) {
        case MoveDrop::Left:
            CHECK(preview->x == Catch::Approx(before.x).margin(margin));
            CHECK(after.x == Catch::Approx(slot.x + size + gap).margin(margin));
            break;
        case MoveDrop::Right:
            CHECK(preview->right() == Catch::Approx(before.right()).margin(margin));
            CHECK(after.x == Catch::Approx(slot.x).margin(margin));
            break;
        case MoveDrop::Top:
            CHECK(preview->y == Catch::Approx(before.y).margin(margin));
            CHECK(after.y == Catch::Approx(slot.y + size + gap).margin(margin));
            break;
        default:
            CHECK(preview->bottom() == Catch::Approx(before.bottom()).margin(margin));
            CHECK(after.y == Catch::Approx(slot.y).margin(margin));
            break;
    }

    const auto* leaf = FindLeaf(*rig.emissions.last(), moved);
    REQUIRE(leaf != nullptr);
    if (rig.minimizedSource()) {
        CHECK(leaf->minimized);
        CHECK_FALSE(movedProbe.visible());
        return;
    }
    CHECK_FALSE(leaf->minimized);
    REQUIRE(movedProbe.visible());
    const Rect landed = rig.tile(movedProbe);
    CHECK(Along(landed, vertical) == Catch::Approx(size).margin(margin));
    CHECK(Along(landed, !vertical) == Catch::Approx(Along(slot, !vertical)).margin(margin));
    switch (drop) {
        case MoveDrop::Left:
            CHECK(landed.x == Catch::Approx(slot.x).margin(margin));
            break;
        case MoveDrop::Right:
            CHECK(landed.right() == Catch::Approx(slot.right()).margin(margin));
            break;
        case MoveDrop::Top:
            CHECK(landed.y == Catch::Approx(slot.y).margin(margin));
            break;
        default:
            CHECK(landed.bottom() == Catch::Approx(slot.bottom()).margin(margin));
            break;
    }
}

TEST_CASE("Dropping on a gap inserts the tile there and only the two tiles beside the gap give up space",
          "[core][sakura][dock][columns][drop-rules]") {
    struct GapCase {
        const char* name;
        bool vertical;
        std::vector<std::string> before;
        std::vector<std::string> after;
    };
    const std::vector<GapCase> cases = {
        {"dock left", false, {}, {"a", "f"}},
        {"columns a and b", false, {"a", "f"}, {"b", "c", "e"}},
        {"columns b and d", false, {"b", "c", "e"}, {"d"}},
        {"dock right", false, {"d"}, {}},
    };
    const F32 scale = GENERATE(1.0f, 1.25f, 1.5f, 2.0f);
    const bool flex = GENERATE(false, true);
    const auto source = GENERATE(MoveSource::Lone, MoveSource::Stacked, MoveSource::Row,
                                 MoveSource::MinimizedStacked, MoveSource::MinimizedLone, MoveSource::Foreign);
    const U64 which = GENERATE(0, 1, 2, 3);
    const auto& gapCase = cases[which];
    CAPTURE(scale, flex, static_cast<int>(source), gapCase.name);

    MoveRig rig(scale, flex, source, "", true);
    const std::string moved = rig.moved();
    const F32 gap = Dock::ContentPadding * scale;
    const F32 strip = 32.0f * scale;
    const F32 margin = 2.0f;
    const bool vertical = gapCase.vertical;

    const auto span = [&](MoveRig& from, const std::vector<std::string>& names, bool axis) {
        F32 low = std::numeric_limits<F32>::max();
        F32 high = std::numeric_limits<F32>::lowest();
        for (const auto& name : names) {
            if (name == moved || !from.probe(name).visible()) {
                continue;
            }
            const Rect tile = from.tile(from.probe(name));
            low = std::min(low, axis ? tile.y : tile.x);
            high = std::max(high, axis ? tile.bottom() : tile.right());
        }
        return std::make_pair(low, high > low ? high - low : 0.0f);
    };
    const auto open = [&](const std::vector<std::string>& names) {
        return std::any_of(names.begin(), names.end(), [&](const std::string& name) {
            return name != moved && rig.probe(name).visible();
        });
    };
    const auto excluded = [&](const std::vector<std::string>& names) {
        return !names.empty() && !open(names);
    };

    Extent2D<F32> at;
    if (gapCase.before.empty()) {
        at = {1600.0f * scale + 1.0f, 300.0f * scale};
    } else if (gapCase.after.empty()) {
        at = {3200.0f * scale - 1.0f, 300.0f * scale};
    } else {
        const auto edge = [&](const std::vector<std::string>& names, bool end) {
            F32 value = end ? std::numeric_limits<F32>::lowest() : std::numeric_limits<F32>::max();
            for (const auto& name : names) {
                const Rect tile = rig.rect(name);
                value = end ? std::max(value, vertical ? tile.bottom() : tile.right())
                            : std::min(value, vertical ? tile.y : tile.x);
            }
            return value;
        };
        const F32 center = (edge(gapCase.before, true) + edge(gapCase.after, false)) * 0.5f;
        const Rect anchor = rig.rect(gapCase.after.back());
        at = vertical ? Extent2D<F32>{anchor.x + anchor.width * 0.5f, center}
                      : Extent2D<F32>{center, anchor.y + anchor.height * 0.5f};
    }

    const F32 own = source == MoveSource::Foreign || rig.minimizedSource()
                        ? 0.0f
                        : Along(rig.tile(rig.probe(moved)), vertical);
    rig.start();
    REQUIRE(rig.host->drag.active);
    const auto preview = rig.hover(at);
    const bool lonely = excluded(gapCase.before) || excluded(gapCase.after);
    if (lonely) {
        CHECK_FALSE(preview.has_value());
        return;
    }
    MoveRig reference(scale, flex, MoveSource::Lone, moved, true);
    const auto whole = [&](const std::vector<std::string>& names, bool end) -> std::string {
        std::string facing;
        F32 extreme = 0.0f;
        for (const auto& name : names) {
            if (name == moved || !reference.probe(name).visible()) {
                continue;
            }
            const Rect tile = reference.tile(reference.probe(name));
            const F32 edge = end ? tile.right() : -tile.x;
            if (facing.empty() || edge > extreme) {
                facing = name;
                extreme = edge;
            }
        }
        if (facing.empty()) {
            return {};
        }
        const Rect tile = reference.tile(reference.probe(facing));
        const Rect area = reference.viewport();
        return std::abs(tile.y - area.y) <= margin && std::abs(tile.height - area.height) <= margin ? facing
                                                                                                    : std::string();
    };
    const std::string leadWhole = whole(gapCase.before, true);
    const std::string trailWhole = whole(gapCase.after, false);
    if (!leadWhole.empty() || !trailWhole.empty()) {
        const F32 center = gapCase.before.empty() || gapCase.after.empty()
                               ? at.x
                               : (rig.rect(gapCase.before.front()).right() + rig.rect(gapCase.after.front()).x) * 0.5f;
        const bool trailing = !trailWhole.empty() && (leadWhole.empty() || at.x >= center);
        const Rect tile = rig.rect(trailing ? trailWhole : leadWhole);
        const auto split = rig.hover({trailing ? tile.x + 20.0f * scale : tile.right() - 20.0f * scale,
                                      tile.y + tile.height * 0.5f});
        REQUIRE(preview.has_value());
        REQUIRE(split.has_value());
        CHECK(*preview == *split);
        return;
    }
    REQUIRE(preview.has_value());
    rig.host->send(Mouse(MouseEventType::Release, at.x, at.y));
    REQUIRE(rig.emissions.trees.size() == 1);
    REQUIRE(rig.emissions.last().has_value());
    rig.settle();
    const auto facing = [&](const std::vector<std::string>& names, bool end) {
        std::vector<std::string> kept;
        F32 extreme = end ? std::numeric_limits<F32>::lowest() : std::numeric_limits<F32>::max();
        for (const auto& name : names) {
            if (name == moved || !reference.probe(name).visible()) {
                continue;
            }
            const Rect tile = reference.tile(reference.probe(name));
            extreme = end ? std::max(extreme, vertical ? tile.bottom() : tile.right())
                          : std::min(extreme, vertical ? tile.y : tile.x);
        }
        for (const auto& name : names) {
            if (name == moved || !reference.probe(name).visible()) {
                continue;
            }
            const Rect tile = reference.tile(reference.probe(name));
            const F32 edge = end ? (vertical ? tile.bottom() : tile.right()) : (vertical ? tile.y : tile.x);
            if (std::abs(edge - extreme) <= margin) {
                kept.push_back(name);
            }
        }
        return kept;
    };
    const auto leading = facing(gapCase.before, true);
    const auto trailing = facing(gapCase.after, false);
    const F32 lead = span(reference, leading, vertical).second;
    const F32 trail = span(reference, trailing, vertical).second;
    const F32 pool = lead + trail;
    const F32 floor = 64.0f * scale;
    const F32 half = (pool - gap) * 0.5f;
    F32 size = std::max(floor, own > 0.0f ? std::min(own, half) : half);
    if (rig.minimizedSource()) {
        size = strip;
    }
    std::unordered_map<std::string, Rect> expected;
    for (const auto name : {"a", "b", "c", "d", "e"}) {
        if (name != moved && reference.probe(name).visible()) {
            expected[name] = reference.tile(reference.probe(name));
        }
    }
    const auto inside = [&](const std::vector<std::string>& listed, const std::vector<std::string>& near,
                            const std::string& name) {
        if (std::find(listed.begin(), listed.end(), name) == listed.end() || near.empty()) {
            return false;
        }
        const auto [low, extent] = span(reference, near, vertical);
        const Rect tile = reference.tile(reference.probe(name));
        const F32 start = vertical ? tile.y : tile.x;
        const F32 end = vertical ? tile.bottom() : tile.right();
        return start >= low - margin && end <= low + extent + margin;
    };
    const auto inGroup = [&](const std::string& name) {
        return inside(gapCase.before, leading, name) || inside(gapCase.after, trailing, name);
    };
    for (const auto& [name, rect] : expected) {
        if (inGroup(name)) {
            continue;
        }
        CAPTURE(name);
        REQUIRE(rig.probe(name).visible());
        CHECK(rig.tile(rig.probe(name)).x == Catch::Approx(rect.x).margin(margin));
        CHECK(rig.tile(rig.probe(name)).y == Catch::Approx(rect.y).margin(margin));
        CHECK(rig.tile(rig.probe(name)).width == Catch::Approx(rect.width).margin(margin));
        CHECK(rig.tile(rig.probe(name)).height == Catch::Approx(rect.height).margin(margin));
    }

    const auto* leaf = FindLeaf(*rig.emissions.last(), moved);
    REQUIRE(leaf != nullptr);
    const auto [leadStart, afterLead] = span(rig, leading, vertical);
    const auto [trailStart, afterTrail] = span(rig, trailing, vertical);
    const F32 taken = size + gap;
    CHECK(afterLead == Catch::Approx(lead - taken * lead / pool).margin(margin));
    CHECK(afterTrail == Catch::Approx(trail - taken * trail / pool).margin(margin));
    CHECK(Along(*preview, vertical) == Catch::Approx(size).margin(margin));
    const F32 start = vertical ? preview->y : preview->x;
    if (afterLead > 0.0f) {
        CHECK(start == Catch::Approx(leadStart + afterLead + gap).margin(margin));
    }
    if (afterTrail > 0.0f) {
        CHECK(start + size + gap == Catch::Approx(trailStart).margin(margin));
    }
    if (rig.minimizedSource()) {
        CHECK(leaf->minimized);
        CHECK_FALSE(rig.probe(moved).visible());
        return;
    }
    CHECK_FALSE(leaf->minimized);
    REQUIRE(rig.probe(moved).visible());
    const Rect landed = rig.tile(rig.probe(moved));
    CHECK(preview->x == Catch::Approx(landed.x).margin(margin));
    CHECK(preview->y == Catch::Approx(landed.y).margin(margin));
    CHECK(preview->width == Catch::Approx(landed.width).margin(margin));
    CHECK(preview->height == Catch::Approx(landed.height).margin(margin));
    CHECK(Along(landed, vertical) == Catch::Approx(size).margin(margin));
}

TEST_CASE("A minimized tile split beside a full height column keeps half of it for when it is restored",
          "[core][sakura][dock][columns][drop-rules]") {
    const F32 scale = GENERATE(1.0f, 2.0f);
    CAPTURE(scale);
    MoveRig rig(scale, true, MoveSource::MinimizedStacked);
    const Rect target = rig.tile(rig.d);
    rig.start();
    REQUIRE(rig.host->drag.active);
    const auto at = rig.point(MoveDrop::Right, target);
    REQUIRE(rig.hover(at).has_value());
    rig.host->send(Mouse(MouseEventType::Release, at.x, at.y));
    REQUIRE(rig.emissions.trees.size() == 1);
    REQUIRE(rig.emissions.last().has_value());
    const auto& row = *rig.emissions.last();
    REQUIRE(row.children.size() == row.sizes.size());
    REQUIRE(row.children.back().pane == "b");
    CHECK(row.children.back().minimized);
    CHECK(row.sizes.back() == Catch::Approx((target.width - Dock::ContentPadding * scale) * 0.5f / scale).margin(1.0f));
    rig.settle();
    CHECK(rig.tile(rig.d).width == Catch::Approx(target.width - Dock::ContentPadding * scale - 32.0f * scale).margin(1.0f));
}

TEST_CASE("A tile is not offered as a drop target for itself or while it is minimized",
          "[core][sakura][dock][columns][drop-rules]") {
    MoveRig own(1.0f, true, MoveSource::Stacked);
    own.start();
    REQUIRE(own.host->drag.active);
    for (const auto drop : {MoveDrop::Left, MoveDrop::Right, MoveDrop::Top, MoveDrop::Bottom}) {
        CHECK_FALSE(own.hover(own.point(drop, own.tile(own.b))).has_value());
    }

    MoveRig folded(1.0f, true, MoveSource::MinimizedStacked);
    const Rect strip = {folded.tile(folded.c).x, folded.viewport().y, folded.tile(folded.c).width * 2.0f, 32.0f};
    folded.host->send(Mouse(MouseEventType::Click, folded.tile(folded.d).x + 40.0f, folded.tile(folded.d).y + 10.0f));
    folded.host->send(Mouse(MouseEventType::Move, folded.tile(folded.d).x + 50.0f, folded.tile(folded.d).y + 10.0f));
    REQUIRE(folded.host->drag.active);
    CHECK_FALSE(folded.hover({strip.x + 100.0f, strip.y + 16.0f}).has_value());
}

TEST_CASE("The gap band reaches twelve pixels into each neighbour and tile splits take over beyond it",
          "[core][sakura][dock][columns][drop-rules]") {
    const F32 scale = GENERATE(1.0f, 2.0f);
    CAPTURE(scale);
    MoveRig rig(scale, true, MoveSource::Foreign, "", true);
    const Rect left = rig.tile(rig.a);
    const Rect right = rig.tile(rig.b);
    const F32 band = 12.0f * scale;
    const F32 y = left.y + left.height * 0.5f;
    rig.start();
    REQUIRE(rig.host->drag.active);
    const auto center = rig.hover({(left.right() + right.x) * 0.5f, y});
    REQUIRE(center.has_value());
    for (const F32 x : {left.right() - band + 1.0f, right.x + band - 1.0f}) {
        CAPTURE(x);
        const auto inside = rig.hover({x, y});
        REQUIRE(inside.has_value());
        CHECK(*inside == *center);
    }
    const auto before = rig.hover({left.right() - band - 3.0f * scale, y});
    REQUIRE(before.has_value());
    CHECK(*before != *center);
    CHECK(before->right() == Catch::Approx(left.right()).margin(1.0f));
    const auto after = rig.hover({right.x + band + 3.0f * scale, y});
    REQUIRE(after.has_value());
    CHECK(*after != *center);
    CHECK(after->x == Catch::Approx(right.x).margin(1.0f));
}

TEST_CASE("Dividers inside a nested row trade width between their neighbours and double clicking evens them",
          "[core][sakura][dock][columns][drop-rules]") {
    MoveRig rig(1.0f, true, MoveSource::Lone);
    const Rect c = rig.tile(rig.c);
    const Rect e = rig.tile(rig.e);
    const Rect b = rig.tile(rig.b);
    const Extent2D<F32> divider = {(c.right() + e.x) * 0.5f, c.y + c.height * 0.5f};
    rig.host->send(Mouse(MouseEventType::Click, divider.x, divider.y));
    rig.host->send(Mouse(MouseEventType::Move, divider.x + 40.0f, divider.y));
    rig.host->send(Mouse(MouseEventType::Release, divider.x + 40.0f, divider.y));
    REQUIRE(rig.emissions.trees.size() == 1);
    rig.settle();
    CHECK(rig.tile(rig.c).width == Catch::Approx(c.width + 40.0f).margin(1.0f));
    CHECK(rig.tile(rig.e).width == Catch::Approx(e.width - 40.0f).margin(1.0f));
    CHECK(rig.tile(rig.b) == b);

    const F32 moved = (rig.tile(rig.c).right() + rig.tile(rig.e).x) * 0.5f;
    rig.host->send(Mouse(MouseEventType::Click, moved, divider.y, MouseButton::Left, 2));
    rig.host->send(Mouse(MouseEventType::Release, moved, divider.y));
    REQUIRE(rig.emissions.trees.size() == 2);
    rig.settle();
    CHECK(rig.tile(rig.c).width == Catch::Approx(rig.tile(rig.e).width).margin(1.0f));
    CHECK(rig.tile(rig.b) == b);
}

TEST_CASE("Between two full height tiles the gap band splits the tile on the pointer's side",
          "[core][sakura][dock][columns][drop-rules]") {
    const bool trailing = GENERATE(false, true);
    CAPTURE(trailing);
    Probe a;
    Probe b;
    Probe t;
    for (auto* probe : {&a, &b, &t}) {
        probe->consume = false;
    }
    Dock tabs;
    Dock dock;
    Emissions emissions;
    auto tab = MakePane("t", &t);
    tab.group = "terminal";
    tabs.update({.tabs = true, .accepts = "main", .panes = {tab}});
    Dock::Config config = {
        .accepts = "panel",
        .adopts = {"terminal"},
        .panes = {MakePane("a", &a), MakePane("b", &b)},
        .tiles = ColumnRow({Leaf("a"), Leaf("b")}, {600.0f, 600.0f}),
        .onTiles = emissions.sink(),
        .columns = true,
    };
    dock.update(config);
    Columns columns(tabs, dock);
    Layer layer;
    layer.update({.content = &columns});
    LayerHost host({&layer});
    const auto ctx = MakeContext({2600.0f, 600.0f});
    const Rect bounds = {0.0f, 0.0f, 2600.0f, 600.0f};
    host.place(ctx, bounds);
    const Rect left = a.laidFrame;
    const Rect right = b.laidFrame;
    const auto rect = tabs.tabRect("t");
    REQUIRE(rect.has_value());
    host.send(Mouse(MouseEventType::Click, rect->x + 5.0f, rect->y + 12.0f));
    host.send(Mouse(MouseEventType::Move, rect->x + 15.0f, rect->y + 12.0f));
    const F32 center = (left.right() + right.x) * 0.5f;
    const Extent2D<F32> point = {trailing ? center + 2.0f : center - 2.0f, 300.0f};
    host.send(Mouse(MouseEventType::Move, point.x, point.y));
    REQUIRE(host.drag.active);
    auto hovered = ctx;
    hovered.mouse = point;
    host.place(hovered, bounds);
    REQUIRE(host.drag.target.has_value());
    const Rect preview = host.drag.target->preview;
    if (trailing) {
        CHECK(preview.x == Catch::Approx(right.x).margin(1.0f));
    } else {
        CHECK(preview.right() == Catch::Approx(left.right()).margin(1.0f));
    }
    host.send(Mouse(MouseEventType::Release, point.x, point.y));
    REQUIRE(emissions.last().has_value());
    CHECK(Leaves(*emissions.last()) == std::vector<std::string>{"a", "t", "b"});
    config.tiles = emissions.last();
    dock.update(config);
    host.place(ctx, bounds);
    CHECK((trailing ? a : b).laidFrame.width == Catch::Approx((trailing ? left : right).width).margin(1.0f));
}

TEST_CASE("A gap that would narrow a neighbour below the width floor is not offered and other columns stay alone",
          "[core][sakura][dock][columns][drop-rules]") {
    const F32 first = GENERATE(64.0f, 100.0f, 400.0f);
    CAPTURE(first);
    Probe a;
    Probe a2;
    Probe b;
    Probe b2;
    Probe c;
    Probe t;
    Dock tabs;
    Dock dock;
    Emissions emissions;
    for (auto* probe : {&a, &a2, &b, &b2, &c, &t}) {
        probe->consume = false;
    }
    auto tab = MakePane("t", &t);
    tab.group = "terminal";
    tabs.update({.tabs = true, .accepts = "main", .panes = {tab}});
    Dock::Config config = {
        .accepts = "panel",
        .adopts = {"terminal"},
        .panes = {MakePane("a", &a), MakePane("a2", &a2), MakePane("b", &b), MakePane("b2", &b2), MakePane("c", &c)},
        .tiles = ColumnRow({Column({500.0f, 500.0f}, {Leaf("a"), Leaf("a2")}), Column({500.0f, 500.0f}, {Leaf("b"), Leaf("b2")}),
                            Leaf("c")},
                           {first, 600.0f, 500.0f}),
        .onTiles = emissions.sink(),
        .columns = true,
    };
    dock.update(config);
    Columns columns(tabs, dock);
    Layer layer;
    layer.update({.content = &columns});
    LayerHost host({&layer});
    const F32 width = first + 600.0f + 500.0f + 4.0f * Dock::ContentPadding;
    const auto ctx = MakeContext({2.0f * width, 600.0f});
    const Rect bounds = {0.0f, 0.0f, 2.0f * width, 600.0f};
    host.place(ctx, bounds);
    REQUIRE(a.laidFrame.width == Catch::Approx(first).margin(1.0f));
    REQUIRE(c.laidFrame.width == Catch::Approx(500.0f).margin(1.0f));
    const auto rect = tabs.tabRect("t");
    REQUIRE(rect.has_value());
    host.send(Mouse(MouseEventType::Click, rect->x + 5.0f, rect->y + 12.0f));
    host.send(Mouse(MouseEventType::Move, rect->x + 15.0f, rect->y + 12.0f));
    const Extent2D<F32> point = {(a.laidFrame.right() + b.laidFrame.x) * 0.5f, 300.0f};
    host.send(Mouse(MouseEventType::Move, point.x, point.y));
    REQUIRE(host.drag.active);
    auto hovered = ctx;
    hovered.mouse = point;
    host.place(hovered, bounds);
    if (first < 400.0f) {
        CHECK_FALSE(host.drag.target.has_value());
        return;
    }
    REQUIRE(host.drag.target.has_value());
    const Rect preview = host.drag.target->preview;
    host.send(Mouse(MouseEventType::Release, point.x, point.y));
    REQUIRE(emissions.trees.size() == 1);
    REQUIRE(emissions.last().has_value());
    config.tiles = emissions.last();
    config.panes.push_back(tab);
    tabs.update({.tabs = true, .accepts = "main"});
    dock.update(config);
    host.place(ctx, bounds);
    const F32 pool = first + 600.0f;
    const F32 size = (pool - 6.0f) * 0.5f;
    CHECK(a.laidFrame.width == Catch::Approx(first - (size + 6.0f) * first / pool).margin(1.0f));
    CHECK(b.laidFrame.width == Catch::Approx(600.0f - (size + 6.0f) * 600.0f / pool).margin(1.0f));
    CHECK(c.laidFrame.width == Catch::Approx(500.0f).margin(1.0f));
    CHECK(t.laidFrame.width == Catch::Approx(size).margin(1.0f));
    CHECK(t.laidFrame.x == Catch::Approx(preview.x).margin(1.0f));
    CHECK(t.laidFrame.width == Catch::Approx(preview.width).margin(1.0f));
}

TEST_CASE("Gaps whose neighbours are all minimized columns are not offered",
          "[core][sakura][dock][columns][drop-rules]") {
    Probe a;
    Probe b;
    Probe c;
    Probe d;
    for (auto* probe : {&a, &b, &c, &d}) {
        probe->consume = false;
    }
    Dock dock;
    Emissions emissions;
    auto second = Leaf("b");
    second.minimized = true;
    auto third = Leaf("c");
    third.minimized = true;
    dock.update({
        .accepts = "panel",
        .panes = {MakePane("a", &a), MakePane("b", &b), MakePane("c", &c), MakePane("d", &d)},
        .tiles = ColumnRow({Leaf("d"), Leaf("a"), second, third}),
        .onTiles = emissions.sink(),
        .columns = true,
        .flexPane = "a",
        .persistView = true,
    });
    Layer layer;
    layer.update({.content = &dock});
    LayerHost host({&layer});
    host.place(MakeContext({1600.0f, 600.0f}), kWideDock);
    REQUIRE_FALSE(b.visible());
    REQUIRE_FALSE(c.visible());
    const Rect source = d.laidFrame;
    host.send(Mouse(MouseEventType::Click, source.x + 100.0f, source.y - kHeader * 0.5f));
    host.send(Mouse(MouseEventType::Move, source.x + 110.0f, source.y - kHeader * 0.5f));
    REQUIRE(host.drag.active);
    const F32 strip = 32.0f;
    const F32 last = 1594.0f - strip;
    for (const F32 x : {last - 3.0f, 1599.0f}) {
        CAPTURE(x);
        host.send(Mouse(MouseEventType::Move, x, 300.0f));
        host.place(WideAt(x, 300.0f), kWideDock);
        CHECK_FALSE(host.drag.target.has_value());
    }
}

TEST_CASE("A tile narrower than the width floor becomes a column at the floor taken only from its target",
          "[core][sakura][dock][columns][drop-rules]") {
    Probe a;
    Probe b;
    Probe c;
    Probe e;
    for (auto* probe : {&a, &b, &c, &e}) {
        probe->consume = false;
    }
    Dock dock;
    Emissions emissions;
    Dock::Config config = {
        .accepts = "panel",
        .panes = {MakePane("a", &a), MakePane("b", &b), MakePane("c", &c), MakePane("e", &e)},
        .tiles = ColumnRow({Leaf("a"), Column({500.0f, 500.0f}, {Leaf("b"), Row({60.0f, 940.0f}, {Leaf("c"), Leaf("e")})})},
                           {900.0f, 0.0f}),
        .onTiles = emissions.sink(),
        .columns = true,
        .flexPane = "b",
    };
    dock.update(config);
    Layer layer;
    layer.update({.content = &dock});
    LayerHost host({&layer});
    host.place(MakeContext({1600.0f, 600.0f}), kWideDock);
    const Rect source = c.laidFrame;
    const Rect target = a.laidFrame;
    REQUIRE(source.width < 64.0f);
    host.drag.active = true;
    host.drag.pane = "c";
    host.drag.group = "panel";
    const Extent2D<F32> point = {target.x + 20.0f, target.y + target.height * 0.5f};
    Sakura::Access::invalidateLayout(dock);
    host.place(WideAt(point.x, point.y), kWideDock);
    REQUIRE(host.drag.target.has_value());
    const auto accept = host.drag.target->accept;
    host.drag.reset();
    accept("c");
    REQUIRE(emissions.trees.size() == 1);
    config.tiles = emissions.last();
    dock.update(config);
    host.place(MakeContext({1600.0f, 600.0f}), kWideDock);
    CHECK(c.laidFrame.width == Catch::Approx(64.0f).margin(1.0f));
    CHECK(c.laidFrame.x == Catch::Approx(target.x).margin(1.0f));
    CHECK(a.laidFrame.width == Catch::Approx(target.width - 64.0f - 6.0f).margin(1.0f));
    CHECK(a.laidFrame.x == Catch::Approx(target.x + 64.0f + 6.0f).margin(1.0f));
}

TEST_CASE("A tile already narrower than the width floor does not block a drop that leaves it alone",
          "[core][sakura][dock][columns][drop-rules]") {
    Probe a;
    Probe b;
    Probe c;
    Probe e;
    Probe t;
    for (auto* probe : {&a, &b, &c, &e, &t}) {
        probe->consume = false;
    }
    Dock tabs;
    Dock dock;
    Emissions emissions;
    auto tab = MakePane("t", &t);
    tab.group = "terminal";
    tabs.update({.tabs = true, .accepts = "main", .panes = {tab}});
    dock.update({
        .accepts = "panel",
        .adopts = {"terminal"},
        .panes = {MakePane("a", &a), MakePane("b", &b), MakePane("c", &c), MakePane("e", &e)},
        .tiles = ColumnRow({Leaf("a"), Column({500.0f, 500.0f}, {Leaf("b"), Row({60.0f, 940.0f}, {Leaf("c"), Leaf("e")})})},
                           {900.0f, 0.0f}),
        .onTiles = emissions.sink(),
        .columns = true,
        .flexPane = "b",
    });
    Columns columns(tabs, dock);
    Layer layer;
    layer.update({.content = &columns});
    LayerHost host({&layer});
    const auto ctx = MakeContext({3200.0f, 600.0f});
    const Rect bounds = {0.0f, 0.0f, 3200.0f, 600.0f};
    host.place(ctx, bounds);
    REQUIRE(c.laidFrame.width < 64.0f);
    const auto rect = tabs.tabRect("t");
    REQUIRE(rect.has_value());
    host.send(Mouse(MouseEventType::Click, rect->x + 5.0f, rect->y + 12.0f));
    host.send(Mouse(MouseEventType::Move, rect->x + 15.0f, rect->y + 12.0f));
    const Extent2D<F32> point = {a.laidFrame.x + a.laidFrame.width * 0.5f, a.laidFrame.y + 20.0f};
    host.send(Mouse(MouseEventType::Move, point.x, point.y));
    REQUIRE(host.drag.active);
    auto hovered = ctx;
    hovered.mouse = point;
    host.place(hovered, bounds);
    CHECK(host.drag.target.has_value());
    host.drag.reset();
}

TEST_CASE("A stacked tile is not offered a column split it would no longer fit once the dragged tile leaves",
          "[core][sakura][dock][columns][drop-rules]") {
    Probe a;
    Probe b;
    Probe c;
    for (auto* probe : {&a, &b, &c}) {
        probe->consume = false;
    }
    Dock dock;
    Emissions emissions;
    dock.update({
        .accepts = "panel",
        .panes = {MakePane("a", &a), MakePane("b", &b), MakePane("c", &c)},
        .tiles = ColumnRow({Column({500.0f, 500.0f}, {Leaf("a"), Leaf("b")}), Leaf("c")}, {100.0f, 500.0f}),
        .onTiles = emissions.sink(),
        .columns = true,
    });
    Layer layer;
    layer.update({.content = &dock});
    LayerHost host({&layer});
    const F32 width = 100.0f + 500.0f + 3.0f * Dock::ContentPadding;
    const auto ctx = MakeContext({width, 600.0f});
    const Rect bounds = {0.0f, 0.0f, width, 600.0f};
    host.place(ctx, bounds);
    REQUIRE(b.laidFrame.width == Catch::Approx(100.0f).margin(1.0f));
    host.drag.active = true;
    host.drag.pane = "a";
    host.drag.group = "panel";
    auto hovered = ctx;
    hovered.mouse = Extent2D<F32>{b.laidFrame.right() - 20.0f, b.laidFrame.y + b.laidFrame.height * 0.5f};
    Sakura::Access::invalidateLayout(dock);
    host.place(hovered, bounds);
    CHECK_FALSE(host.drag.target.has_value());
    CHECK(emissions.trees.empty());
    host.drag.reset();
}

TEST_CASE("Column gaps skip hidden columns and take space from the visible neighbours",
          "[core][sakura][dock][columns][drop-rules]") {
    Probe a;
    Probe a2;
    Probe h;
    Probe b;
    Probe b2;
    Probe t;
    for (auto* probe : {&a, &a2, &h, &b, &b2, &t}) {
        probe->consume = false;
    }
    Dock tabs;
    Dock dock;
    Emissions emissions;
    auto tab = MakePane("t", &t);
    tab.group = "terminal";
    tabs.update({.tabs = true, .accepts = "main", .panes = {tab}});
    auto hidden = MakePane("h", &h);
    hidden.hidden = true;
    Dock::Config config = {
        .accepts = "panel",
        .adopts = {"terminal"},
        .panes = {MakePane("a", &a), MakePane("a2", &a2), hidden, MakePane("b", &b), MakePane("b2", &b2)},
        .tiles = ColumnRow({Column({500.0f, 500.0f}, {Leaf("a"), Leaf("a2")}), Leaf("h"),
                            Column({500.0f, 500.0f}, {Leaf("b"), Leaf("b2")})},
                           {500.0f, 200.0f, 300.0f}),
        .onTiles = emissions.sink(),
        .columns = true,
    };
    dock.update(config);
    Columns columns(tabs, dock);
    Layer layer;
    layer.update({.content = &columns});
    LayerHost host({&layer});
    const F32 width = 500.0f + 300.0f + 3.0f * Dock::ContentPadding;
    const auto ctx = MakeContext({2.0f * width, 600.0f});
    const Rect bounds = {0.0f, 0.0f, 2.0f * width, 600.0f};
    host.place(ctx, bounds);
    REQUIRE_FALSE(h.visible());
    REQUIRE(a.laidFrame.width == Catch::Approx(500.0f).margin(1.0f));
    REQUIRE(b.laidFrame.width == Catch::Approx(300.0f).margin(1.0f));
    const auto rect = tabs.tabRect("t");
    REQUIRE(rect.has_value());
    host.send(Mouse(MouseEventType::Click, rect->x + 5.0f, rect->y + 12.0f));
    host.send(Mouse(MouseEventType::Move, rect->x + 15.0f, rect->y + 12.0f));
    const Extent2D<F32> point = {(a.laidFrame.right() + b.laidFrame.x) * 0.5f, 300.0f};
    host.send(Mouse(MouseEventType::Move, point.x, point.y));
    REQUIRE(host.drag.active);
    auto hovered = ctx;
    hovered.mouse = point;
    host.place(hovered, bounds);
    REQUIRE(host.drag.target.has_value());
    host.send(Mouse(MouseEventType::Release, point.x, point.y));
    REQUIRE(emissions.trees.size() == 1);
    REQUIRE(emissions.last().has_value());
    CHECK(Leaves(*emissions.last()) == std::vector<std::string>{"a", "a2", "h", "t", "b", "b2"});
    config.tiles = emissions.last();
    config.panes.push_back(tab);
    tabs.update({.tabs = true, .accepts = "main"});
    dock.update(config);
    host.place(ctx, bounds);
    const F32 size = (800.0f - 6.0f) * 0.5f;
    CHECK(t.laidFrame.width == Catch::Approx(size).margin(1.0f));
    CHECK(b.laidFrame.width == Catch::Approx(300.0f - (size + 6.0f) * 300.0f / 800.0f).margin(1.0f));
    CHECK(a.laidFrame.width == Catch::Approx(500.0f - (size + 6.0f) * 500.0f / 800.0f).margin(1.0f));
}

TEST_CASE("Splitting a full height column into a row keeps the editor and the slot where they were",
          "[core][sakura][dock][columns][drop-rules]") {
    Probe editor;
    Probe one;
    Probe two;
    Dock dock;
    Host host(dock);
    Dock::Config config = {
        .accepts = "panel",
        .panes = {MakePane("editor", &editor), MakePane("one", &one)},
        .tiles = ColumnRow({Leaf("editor"), Leaf("one")}, {0.0f, 600.0f}),
        .columns = true,
        .flexPane = "editor",
    };
    dock.update(config);
    host.place(MakeContext({1600.0f, 600.0f}), kWideDock);
    const Rect before = editor.laidFrame;
    const Rect slot = one.laidFrame;
    config.panes.push_back(MakePane("two", &two));
    config.tiles = ColumnRow({Leaf("editor"), Row({500.0f, 500.0f}, {Leaf("one"), Leaf("two")})}, {0.0f, 600.0f});
    dock.update(config);
    host.place(MakeContext({1600.0f, 600.0f}), kWideDock);
    CHECK(editor.laidFrame == before);
    CHECK(one.laidFrame.x == Catch::Approx(slot.x).margin(1.0f));
    CHECK(two.laidFrame.right() == Catch::Approx(slot.right()).margin(1.0f));
    CHECK(one.laidFrame.width == Catch::Approx((slot.width - 6.0f) * 0.5f).margin(1.0f));
    CHECK(two.laidFrame.width == Catch::Approx((slot.width - 6.0f) * 0.5f).margin(1.0f));
}

TEST_CASE("A column split is validated after the dragged tile leaves, not before",
          "[core][sakura][dock][columns][drop-rules]") {
    Probe a;
    Probe b;
    for (auto* probe : {&a, &b}) {
        probe->consume = false;
    }
    Dock dock;
    Emissions emissions;
    Dock::Config config = {
        .accepts = "panel",
        .panes = {MakePane("a", &a), MakePane("b", &b)},
        .tiles = ColumnRow({Leaf("a"), Leaf("b")}, {100.0f, 500.0f}),
        .onTiles = emissions.sink(),
        .columns = true,
    };
    dock.update(config);
    Layer layer;
    layer.update({.content = &dock});
    LayerHost host({&layer});
    const F32 width = 100.0f + 500.0f + 3.0f * Dock::ContentPadding;
    const auto ctx = MakeContext({width, 600.0f});
    const Rect bounds = {0.0f, 0.0f, width, 600.0f};
    host.place(ctx, bounds);
    REQUIRE(a.laidFrame.width == Catch::Approx(100.0f).margin(1.0f));
    host.drag.active = true;
    host.drag.pane = "b";
    host.drag.group = "panel";
    auto hovered = ctx;
    hovered.mouse = Extent2D<F32>{a.laidFrame.right() - 20.0f, a.laidFrame.y + a.laidFrame.height * 0.5f};
    Sakura::Access::invalidateLayout(dock);
    host.place(hovered, bounds);
    REQUIRE(host.drag.target.has_value());
    const auto accept = host.drag.target->accept;
    host.drag.reset();
    accept("b");
    REQUIRE(emissions.trees.size() == 1);
    config.tiles = emissions.last();
    dock.update(config);
    host.place(ctx, bounds);
    CHECK(a.laidFrame.width == Catch::Approx(b.laidFrame.width).margin(1.0f));
    CHECK(a.laidFrame.right() < b.laidFrame.x);
}

TEST_CASE("Splitting a narrow column keeps each half at the width floor instead of the default width",
          "[core][sakura][dock][columns][drop-rules]") {
    Probe editor;
    Probe one;
    Probe two;
    Dock dock;
    Host host(dock);
    Dock::Config config = {
        .accepts = "panel",
        .panes = {MakePane("editor", &editor), MakePane("one", &one)},
        .tiles = ColumnRow({Leaf("editor"), Leaf("one")}, {0.0f, 100.0f}),
        .columns = true,
        .flexPane = "editor",
    };
    dock.update(config);
    host.place(MakeContext({1600.0f, 600.0f}), kWideDock);
    const F32 before = editor.laidFrame.width;
    config.panes.push_back(MakePane("two", &two));
    config.tiles = ColumnRow({Leaf("editor"), Row({500.0f, 500.0f}, {Leaf("one"), Leaf("two")})}, {0.0f, 100.0f});
    dock.update(config);
    host.place(MakeContext({1600.0f, 600.0f}), kWideDock);
    CHECK(one.laidFrame.width == Catch::Approx(64.0f).margin(1.0f));
    CHECK(two.laidFrame.width == Catch::Approx(64.0f).margin(1.0f));
    CHECK(editor.laidFrame.width == Catch::Approx(before - (64.0f + 6.0f + 64.0f - 100.0f)).margin(1.0f));
}

TEST_CASE("A minimized tile splits a narrow full height column when its strip fits",
          "[core][sakura][dock][columns][drop-rules]") {
    Probe a;
    Probe b;
    Probe m;
    for (auto* probe : {&a, &b, &m}) {
        probe->consume = false;
    }
    Dock dock;
    Emissions emissions;
    auto folded = Leaf("m");
    folded.minimized = true;
    Dock::Config config = {
        .accepts = "panel",
        .panes = {MakePane("a", &a), MakePane("b", &b), MakePane("m", &m)},
        .tiles = ColumnRow({Leaf("a"), Leaf("b"), folded}, {0.0f, 120.0f, 300.0f}),
        .onTiles = emissions.sink(),
        .columns = true,
        .flexPane = "a",
        .persistView = true,
    };
    dock.update(config);
    Layer layer;
    layer.update({.content = &dock});
    LayerHost host({&layer});
    host.place(MakeContext({1600.0f, 600.0f}), kWideDock);
    REQUIRE_FALSE(m.visible());
    REQUIRE(b.laidFrame.width == Catch::Approx(120.0f).margin(1.0f));
    host.drag.active = true;
    host.drag.pane = "m";
    host.drag.group = "panel";
    const Extent2D<F32> point = {b.laidFrame.right() - 20.0f, b.laidFrame.y + b.laidFrame.height * 0.5f};
    Sakura::Access::invalidateLayout(dock);
    host.place(WideAt(point.x, point.y), kWideDock);
    REQUIRE(host.drag.target.has_value());
    const auto accept = host.drag.target->accept;
    host.drag.reset();
    accept("m");
    REQUIRE(emissions.trees.size() == 1);
    config.tiles = emissions.last();
    dock.update(config);
    host.place(MakeContext({1600.0f, 600.0f}), kWideDock);
    CHECK_FALSE(m.visible());
    CHECK(b.laidFrame.width == Catch::Approx(120.0f - 32.0f - 6.0f).margin(1.0f));
}

TEST_CASE("A hidden tile is never picked as the visible neighbour that pays for a gap",
          "[core][sakura][dock][columns][drop-rules]") {
    Probe s;
    Probe a;
    Probe h;
    Probe b;
    Probe b2;
    Probe z;
    for (auto* probe : {&s, &a, &h, &b, &b2, &z}) {
        probe->consume = false;
    }
    Dock dock;
    Emissions emissions;
    auto hidden = MakePane("h", &h);
    hidden.hidden = true;
    Dock::Config config = {
        .accepts = "panel",
        .panes = {MakePane("s", &s), MakePane("a", &a), hidden, MakePane("b", &b), MakePane("b2", &b2),
                  MakePane("z", &z)},
        .tiles = ColumnRow({Column({400.0f, 600.0f}, {Leaf("s"), Row({500.0f, 500.0f}, {Leaf("a"), Leaf("h")})}),
                            Column({500.0f, 500.0f}, {Leaf("b"), Leaf("b2")}), Leaf("z")},
                           {700.0f, 300.0f, 0.0f}),
        .onTiles = emissions.sink(),
        .columns = true,
        .flexPane = "z",
    };
    dock.update(config);
    Layer layer;
    layer.update({.content = &dock});
    LayerHost host({&layer});
    host.place(MakeContext({1600.0f, 600.0f}), kWideDock);
    REQUIRE_FALSE(h.visible());
    REQUIRE(b.laidFrame.width == Catch::Approx(300.0f).margin(1.0f));
    const F32 zone = z.laidFrame.width;
    host.drag.active = true;
    host.drag.pane = "x";
    host.drag.group = "panel";
    const Extent2D<F32> point = {(a.laidFrame.right() + b.laidFrame.x) * 0.5f, 300.0f};
    Sakura::Access::invalidateLayout(dock);
    host.place(WideAt(point.x, point.y), kWideDock);
    REQUIRE(host.drag.target.has_value());
    const auto accept = host.drag.target->accept;
    host.drag.reset();
    accept("x");
    REQUIRE(emissions.trees.size() == 1);
    Probe x;
    config.panes.push_back(MakePane("x", &x));
    config.tiles = emissions.last();
    dock.update(config);
    host.place(MakeContext({1600.0f, 600.0f}), kWideDock);
    const F32 pool = 700.0f + 300.0f;
    const F32 size = (pool - 6.0f) * 0.5f;
    CHECK(x.laidFrame.width == Catch::Approx(size).margin(2.0f));
    CHECK(s.laidFrame.width == Catch::Approx(700.0f - (size + 6.0f) * 700.0f / pool).margin(2.0f));
    CHECK(b.laidFrame.width == Catch::Approx(300.0f - (size + 6.0f) * 300.0f / pool).margin(2.0f));
    CHECK(z.laidFrame.width == Catch::Approx(zone).margin(1.0f));
    CHECK_FALSE(h.visible());
}

TEST_CASE("A minimized tile is offered any tile that keeps room after its strip and no smaller one",
          "[core][sakura][dock][columns][drop-rules]") {
    const F32 share = GENERATE(0.05f, 0.14f);
    CAPTURE(share);
    Probe a;
    Probe b;
    Probe m;
    for (auto* probe : {&a, &b, &m}) {
        probe->consume = false;
    }
    Dock dock;
    Emissions emissions;
    auto folded = Leaf("m");
    folded.minimized = true;
    dock.update({
        .accepts = "panel",
        .panes = {MakePane("a", &a), MakePane("b", &b), MakePane("m", &m)},
        .tiles = ColumnRow({Column({share, 1.0f - share}, {Leaf("a"), Leaf("b")}), folded}, {0.0f, 300.0f}),
        .onTiles = emissions.sink(),
        .columns = true,
        .flexPane = "b",
        .persistView = true,
    });
    Layer layer;
    layer.update({.content = &dock});
    LayerHost host({&layer});
    host.place(MakeContext({1600.0f, 600.0f}), kWideDock);
    REQUIRE_FALSE(m.visible());
    const Rect small = {a.laidFrame.x, a.laidFrame.y - kHeader, a.laidFrame.width, a.laidFrame.height + kHeader};
    const F32 strip = 32.0f;
    host.drag.active = true;
    host.drag.pane = "m";
    host.drag.group = "panel";
    Sakura::Access::invalidateLayout(dock);
    host.place(WideAt(small.x + small.width * 0.5f, small.y + 3.0f), kWideDock);
    if (small.height - 6.0f - strip <= 0.0f) {
        CHECK_FALSE(host.drag.target.has_value());
        CHECK(emissions.trees.empty());
        host.drag.reset();
        return;
    }
    REQUIRE(small.height < 100.0f);
    REQUIRE(host.drag.target.has_value());
    host.drag.reset();
}

TEST_CASE("A maximized tile offers no drop targets, not even a column gap at its edges",
          "[core][sakura][dock][columns][drop-rules]") {
    Probe a;
    Probe b;
    for (auto* probe : {&a, &b}) {
        probe->consume = false;
    }
    Dock dock;
    Emissions emissions;
    dock.update({
        .accepts = "panel",
        .panes = {MakePane("a", &a), MakePane("b", &b)},
        .tiles = ColumnRow({Leaf("a"), Leaf("b")}, {0.0f, 400.0f}),
        .onTiles = emissions.sink(),
        .columns = true,
        .flexPane = "a",
    });
    Layer layer;
    layer.update({.content = &dock});
    LayerHost host({&layer});
    const auto ctx = MakeContext({1600.0f, 600.0f});
    host.place(ctx, kWideDock);
    const Rect card = {a.laidFrame.x, a.laidFrame.y - kHeader, a.laidFrame.width, a.laidFrame.height + kHeader};
    const auto maximize = MaximizeOf(card);
    host.send(Mouse(MouseEventType::Click, maximize.x, maximize.y));
    host.send(Mouse(MouseEventType::Release, maximize.x, maximize.y));
    host.place(ctx, kWideDock);
    REQUIRE_FALSE(b.visible());
    const Rect full = {a.laidFrame.x, a.laidFrame.y - kHeader, a.laidFrame.width, a.laidFrame.height + kHeader};
    REQUIRE(full.width > 1500.0f);
    host.drag.active = true;
    host.drag.pane = "b";
    host.drag.group = "panel";
    for (const Extent2D<F32> point : {Extent2D<F32>{full.x + full.width * 0.5f, full.y + 20.0f},
                                      Extent2D<F32>{full.x + 3.0f, full.y + full.height * 0.5f},
                                      Extent2D<F32>{full.right() - 3.0f, full.y + full.height * 0.5f},
                                      Extent2D<F32>{full.x + full.width * 0.5f, full.y + full.height * 0.5f}}) {
        CAPTURE(point.x, point.y);
        host.send(Mouse(MouseEventType::Move, point.x, point.y));
        Sakura::Access::invalidateLayout(dock);
        host.place(WideAt(point.x, point.y), kWideDock);
        CHECK_FALSE(host.drag.target.has_value());
    }
    host.drag.reset();
}

TEST_CASE("A drop is refused when the columns left after removing the dragged tile cannot hold their widths",
          "[core][sakura][dock][columns][drop-rules]") {
    Probe s;
    Probe a;
    Probe b;
    Probe c;
    Probe d;
    Probe t;
    for (auto* probe : {&s, &a, &b, &c, &d, &t}) {
        probe->consume = false;
    }
    Dock dock;
    Emissions emissions;
    dock.update({
        .accepts = "panel",
        .panes = {MakePane("s", &s), MakePane("a", &a), MakePane("b", &b), MakePane("c", &c), MakePane("d", &d),
                  MakePane("t", &t)},
        .tiles = ColumnRow({Column({400.0f, 600.0f}, {Leaf("s"), Row({250.0f, 250.0f, 250.0f, 250.0f},
                                                                 {Leaf("a"), Leaf("b"), Leaf("c"), Leaf("d")})}),
                            Leaf("t")},
                           {100.0f, 240.0f}),
        .onTiles = emissions.sink(),
        .columns = true,
    });
    Layer layer;
    layer.update({.content = &dock});
    LayerHost host({&layer});
    const F32 width = 100.0f + 240.0f + 3.0f * Dock::ContentPadding;
    const auto ctx = MakeContext({width, 600.0f});
    const Rect bounds = {0.0f, 0.0f, width, 600.0f};
    host.place(ctx, bounds);
    host.drag.active = true;
    host.drag.pane = "s";
    host.drag.group = "panel";
    auto hovered = ctx;
    hovered.mouse = Extent2D<F32>{t.laidFrame.right() - 20.0f, t.laidFrame.y + t.laidFrame.height * 0.5f};
    Sakura::Access::invalidateLayout(dock);
    host.place(hovered, bounds);
    CHECK_FALSE(host.drag.target.has_value());
    CHECK(emissions.trees.empty());
    host.drag.reset();
}

TEST_CASE("A drop is judged on the columns left after the dragged tile leaves, not on the current ones",
          "[core][sakura][dock][columns][drop-rules]") {
    Probe a;
    Probe b;
    Probe c;
    for (auto* probe : {&a, &b, &c}) {
        probe->consume = false;
    }
    Dock dock;
    Emissions emissions;
    dock.update({
        .accepts = "panel",
        .panes = {MakePane("a", &a), MakePane("b", &b), MakePane("c", &c)},
        .tiles = ColumnRow({Leaf("a"), Leaf("b"), Leaf("c")}, {3000.0f, 480.0f, 420.0f}),
        .onTiles = emissions.sink(),
        .columns = true,
        .flexPane = "a",
    });
    Layer layer;
    layer.update({.content = &dock});
    LayerHost host({&layer});
    const auto ctx = MakeContext({600.0f, 600.0f});
    const Rect bounds = {0.0f, 0.0f, 600.0f, 600.0f};
    host.place(ctx, bounds);
    REQUIRE(c.laidFrame.width < 64.0f);
    host.drag.active = true;
    host.drag.pane = "a";
    host.drag.group = "panel";
    auto hovered = ctx;
    hovered.mouse = Extent2D<F32>{b.laidFrame.x + b.laidFrame.width * 0.5f, b.laidFrame.y + 20.0f};
    Sakura::Access::invalidateLayout(dock);
    host.place(hovered, bounds);
    CHECK(host.drag.target.has_value());
    host.drag.reset();
}

TEST_CASE("A gap beside the dragged tile stays blocked when its only other tile is hidden",
          "[core][sakura][dock][columns][drop-rules]") {
    Probe s;
    Probe h;
    Probe t;
    for (auto* probe : {&s, &h, &t}) {
        probe->consume = false;
    }
    Dock dock;
    Emissions emissions;
    auto hidden = MakePane("h", &h);
    hidden.hidden = true;
    dock.update({
        .accepts = "panel",
        .panes = {MakePane("s", &s), hidden, MakePane("t", &t)},
        .tiles = ColumnRow({Column({500.0f, 500.0f}, {Leaf("s"), Leaf("h")}), Leaf("t")}, {600.0f, 0.0f}),
        .onTiles = emissions.sink(),
        .columns = true,
        .flexPane = "t",
    });
    Layer layer;
    layer.update({.content = &dock});
    LayerHost host({&layer});
    host.place(MakeContext({1600.0f, 600.0f}), kWideDock);
    REQUIRE_FALSE(h.visible());
    host.drag.active = true;
    host.drag.pane = "s";
    host.drag.group = "panel";
    Sakura::Access::invalidateLayout(dock);
    host.place(WideAt((s.laidFrame.right() + t.laidFrame.x) * 0.5f, 300.0f), kWideDock);
    CHECK_FALSE(host.drag.target.has_value());
    CHECK(emissions.trees.empty());
    host.drag.reset();
}

TEST_CASE("Tiles inside a fully minimized group line up along the collapsed strip",
          "[core][sakura][dock][columns]") {
    const bool whole = GENERATE(false, true);
    CAPTURE(whole);
    Probe z;
    Probe a;
    Probe b;
    Probe c;
    Dock dock;
    auto top = Leaf("a");
    top.minimized = whole;
    auto left = Leaf("b");
    left.minimized = true;
    auto right = Leaf("c");
    right.minimized = true;
    dock.update({
        .accepts = "panel",
        .panes = {MakePane("z", &z), MakePane("a", &a), MakePane("b", &b), MakePane("c", &c)},
        .tiles = ColumnRow({Leaf("z"), Column({500.0f, 500.0f}, {top, Row({500.0f, 500.0f}, {left, right})})}, {0.0f, 600.0f}),
        .columns = true,
        .flexPane = "z",
        .persistView = true,
    });
    Host host(dock);
    host.place(MakeContext({1600.0f, 600.0f}), kWideDock);
    REQUIRE_FALSE(b.visible());
    REQUIRE_FALSE(c.visible());
    if (!whole) {
        REQUIRE(a.visible());
        const F32 x = 1594.0f - 600.0f;
        const F32 width = (600.0f - 6.0f) * 0.5f;
        CHECK(BoxWith(dock, {x, 562.0f, width, 32.0f}) != nullptr);
        CHECK(BoxWith(dock, {x + width + 6.0f, 562.0f, width, 32.0f}) != nullptr);
        CHECK(a.laidFrame.height == Catch::Approx(588.0f - 32.0f - 6.0f - 20.0f).margin(1.0f));
    } else {
        REQUIRE_FALSE(a.visible());
        const F32 x = 1594.0f - 32.0f;
        CHECK(BoxWith(dock, {x, 303.0f, 32.0f, 143.0f}) != nullptr);
        CHECK(BoxWith(dock, {x, 452.0f, 32.0f, 142.0f}) != nullptr);
    }
}

TEST_CASE("Hovering every drop target and cancelling leaves the column layout untouched",
          "[core][sakura][dock][columns][drop-rules]") {
    const F32 scale = GENERATE(1.0f, 2.0f);
    const bool flex = GENERATE(false, true);
    const auto source = GENERATE(MoveSource::Lone, MoveSource::Stacked, MoveSource::Row,
                                 MoveSource::MinimizedStacked, MoveSource::MinimizedLone, MoveSource::Foreign);
    CAPTURE(scale, flex, static_cast<int>(source));
    MoveRig rig(scale, flex, source);
    std::vector<Rect> before;
    std::vector<bool> shown;
    for (const auto name : {"a", "b", "c", "d", "e"}) {
        before.push_back(rig.probe(name).laidFrame);
        shown.push_back(rig.probe(name).visible());
    }
    std::vector<Rect> targets;
    for (const auto name : {"a", "b", "e"}) {
        if (rig.probe(name).visible()) {
            targets.push_back(rig.tile(rig.probe(name)));
        }
    }

    rig.start();
    REQUIRE(rig.host->drag.active);
    U64 offered = 0;
    for (const auto& target : targets) {
        for (const auto drop : {MoveDrop::Left, MoveDrop::Right, MoveDrop::Top, MoveDrop::Bottom}) {
            offered += rig.hover(rig.point(drop, target)).has_value() ? 1 : 0;
        }
    }
    offered += rig.hover(rig.point(MoveDrop::DockLeft, {})).has_value() ? 1 : 0;
    offered += rig.hover(rig.point(MoveDrop::DockRight, {})).has_value() ? 1 : 0;
    const Rect left = rig.rect("a");
    const Rect middle = rig.rect("c");
    const Rect side = rig.rect("e");
    const Rect top = rig.rect("b");
    const Rect last = rig.rect("d");
    for (const Extent2D<F32> point : {Extent2D<F32>{(left.right() + middle.x) * 0.5f, 300.0f * scale},
                                      Extent2D<F32>{(side.right() + last.x) * 0.5f, 300.0f * scale},
                                      Extent2D<F32>{(middle.right() + side.x) * 0.5f, middle.y + middle.height * 0.5f},
                                      Extent2D<F32>{middle.right(), (top.bottom() + middle.y) * 0.5f}}) {
        offered += rig.hover(point).has_value() ? 1 : 0;
    }
    CHECK(offered > 0);

    rig.host->send(Mouse(MouseEventType::Leave, 0.0f, 0.0f));
    CHECK_FALSE(rig.host->drag.active);
    rig.host->place(rig.ctx, rig.bounds);
    CHECK(rig.emissions.trees.empty());
    U64 index = 0;
    for (const auto name : {"a", "b", "c", "d", "e"}) {
        CAPTURE(name);
        CHECK(rig.probe(name).visible() == shown[index]);
        if (shown[index]) {
            CHECK(rig.probe(name).laidFrame == before[index]);
        }
        ++index;
    }
}

namespace {

enum class SweepLayout : U8 { Plain, Stacked, Folded, Hidden, Pairs };

struct SweepRig {
    F32 scale = 1.0f;
    std::unordered_map<std::string, std::unique_ptr<Probe>> probes;
    std::vector<std::string> names;
    std::vector<std::vector<std::string>> groups;
    Dock tabs;
    Dock dock;
    Emissions emissions;
    Dock::Config config;
    Dock::Pane tab;
    Columns columns{tabs, dock};
    Layer layer;
    std::unique_ptr<LayerHost> host;
    Sakura::Context ctx;
    Rect bounds;

    SweepRig(F32 scale, SweepLayout fixture, bool folded, const std::string& skip) : scale(scale) {
        for (const auto name : {"a", "b", "c", "d", "e", "f", "h", "m", "t"}) {
            probes[name] = std::make_unique<Probe>();
            probes[name]->consume = false;
        }
        tab = MakePane("t", probes["t"].get());
        tab.group = "terminal";
        tabs.update({.tabs = true, .accepts = "main", .panes = {tab}});
        const bool stacked = fixture == SweepLayout::Stacked || fixture == SweepLayout::Hidden;
        auto upper = Leaf("b");
        upper.minimized = folded;
        auto strip = Leaf("m");
        strip.minimized = true;
        std::vector<DockTiles> row;
        std::vector<F32> widths;
        if (fixture == SweepLayout::Pairs) {
            row = {Leaf("a"), Column({600.0f, 400.0f}, {upper, Leaf("c")}), Column({500.0f, 500.0f}, {Leaf("e"), Leaf("d")})};
            widths = {0.0f, 0.0f, 360.0f};
            groups = {{"a"}, {"b", "c"}, {"e", "d"}};
        } else {
            row = {stacked ? Column({500.0f, 500.0f}, {Leaf("a"), Leaf("f")}) : Leaf("a"),
                   Column({600.0f, 400.0f}, {upper, Row({500.0f, 500.0f}, {Leaf("c"), Leaf("e")})}), Leaf("d")};
            widths = {0.0f, 0.0f, 360.0f};
            groups = {stacked ? std::vector<std::string>{"a", "f"} : std::vector<std::string>{"a"},
                      {"b", "c", "e"}, {"d"}};
        }
        if (fixture == SweepLayout::Folded) {
            row.insert(row.begin() + 2, strip);
            widths.insert(widths.begin() + 2, 300.0f);
            groups.insert(groups.begin() + 2, {"m"});
        }
        if (fixture == SweepLayout::Hidden) {
            row.insert(row.begin() + 1, Leaf("h"));
            widths.insert(widths.begin() + 1, 300.0f);
            groups.insert(groups.begin() + 1, {"h"});
        }
        for (const auto& group : groups) {
            names.insert(names.end(), group.begin(), group.end());
        }
        std::erase(names, skip);
        std::optional<DockTiles> tiles = ColumnRow(std::move(row), std::move(widths));
        std::vector<Dock::Pane> panes;
        for (const auto& name : names) {
            auto pane = MakePane(name, probes[name].get());
            pane.hidden = name == "h";
            panes.push_back(std::move(pane));
        }
        if (!skip.empty()) {
            tiles = WithoutPane(*tiles, skip);
        }
        config = {
            .accepts = "panel",
            .adopts = {"terminal"},
            .panes = std::move(panes),
            .tiles = tiles,
            .onTiles = emissions.sink(),
            .columns = true,
            .flexPane = "a",
            .persistView = true,
        };
        dock.update(config);
        layer.update({.content = &columns});
        host = std::make_unique<LayerHost>(std::vector<Layer*>{&layer});
        ctx = MakeContext({3200.0f * scale, 600.0f * scale});
        ctx.pixelRatio = scale;
        bounds = {0.0f, 0.0f, 3200.0f * scale, 600.0f * scale};
        host->place(ctx, bounds);
    }

    Rect viewport() const {
        const F32 inset = Dock::ContentPadding * scale;
        return {1600.0f * scale + inset, inset, 1600.0f * scale - 2.0f * inset, 600.0f * scale - 2.0f * inset};
    }

    std::unordered_map<std::string, Rect> tiles(const std::string& except) {
        std::unordered_map<std::string, Rect> result;
        for (const auto& name : names) {
            if (name != except && name != "h" && probes[name]->visible()) {
                const Rect body = probes[name]->laidFrame;
                result[name] = {body.x, body.y - kHeader * scale, body.width, body.height + kHeader * scale};
            }
        }
        return result;
    }

    std::optional<std::pair<F32, F32>> extent(const std::vector<std::string>& panes) {
        const Rect area = viewport();
        std::optional<std::pair<F32, F32>> result;
        for (F32 x = area.x; x < area.right(); x += 1.0f) {
            for (const F32 v : {0.02f, 0.25f, 0.5f, 0.75f, 0.98f}) {
                const std::string pane = dock.paneAt({x + 0.5f, area.y + area.height * v});
                if (std::find(panes.begin(), panes.end(), pane) != panes.end()) {
                    result = result ? std::make_pair(std::min(result->first, x), std::max(result->second, x + 1.0f))
                                    : std::make_pair(x, x + 1.0f);
                }
            }
        }
        return result;
    }

    std::optional<Rect> hover(const std::string& moved, Extent2D<F32> at) {
        host->drag.active = true;
        host->drag.pane = moved;
        host->drag.group = moved == "t" ? "terminal" : "panel";
        host->send(Mouse(MouseEventType::Move, at.x, at.y));
        auto hovered = ctx;
        hovered.mouse = at;
        Sakura::Access::invalidateLayout(dock);
        host->place(hovered, bounds);
        if (!host->drag.target.has_value()) {
            return std::nullopt;
        }
        return host->drag.target->preview;
    }

    void drop(const std::string& moved) {
        const auto accept = host->drag.target->accept;
        host->drag.reset();
        accept(moved);
        config.tiles = emissions.last();
        if (moved == "t") {
            names.push_back("t");
            config.panes.push_back(tab);
            tabs.update({.tabs = true, .accepts = "main"});
        }
        dock.update(config);
        Sakura::Access::invalidateLayout(dock);
        host->place(ctx, bounds);
    }
};

bool Near(const Rect& a, const Rect& b, F32 margin) {
    return std::abs(a.x - b.x) <= margin && std::abs(a.y - b.y) <= margin && std::abs(a.right() - b.right()) <= margin &&
           std::abs(a.bottom() - b.bottom()) <= margin;
}

struct SweepExpect {
    bool offered = false;
    std::string split;
    Dock::Side side = Dock::Side::Left;
    std::vector<std::string> payers;
    F32 size = 0.0f;
};

}  // namespace

TEST_CASE("Every offered drop lands on its preview, keeps the floor and leaves tiles that do not pay alone",
          "[core][sakura][dock][columns][drop-rules]") {
    struct Source {
        const char* pane;
        bool folded;
    };
    const F32 scale = GENERATE(1.0f, 1.25f, 1.5f, 2.0f);
    const auto fixture = GENERATE(SweepLayout::Plain, SweepLayout::Stacked, SweepLayout::Folded, SweepLayout::Hidden,
                                  SweepLayout::Pairs);
    const auto source = GENERATE(Source{"d", false}, Source{"b", false}, Source{"b", true}, Source{"t", false},
                                 Source{"a", false});
    const std::string moved = source.pane;
    CAPTURE(scale, static_cast<int>(fixture), moved, source.folded);

    SweepRig rig(scale, fixture, source.folded, "");
    const auto before = rig.tiles(moved);
    SweepRig reference(scale, fixture, false, moved == "t" ? "" : moved);
    const auto kept = reference.tiles(moved);
    const Rect area = rig.viewport();
    const F32 gap = Dock::ContentPadding * scale;
    const F32 strip = 32.0f * scale;
    const F32 floor = 64.0f * scale;
    const F32 margin = scale == std::floor(scale) ? 1.0f : 2.0f;
    const auto ownAlong = [&](bool vertical) {
        const auto all = rig.tiles("");
        const auto it = all.find(moved);
        return it == all.end() ? 0.0f : Along(it->second, vertical);
    };

    struct Shown {
        std::vector<std::string> panes;
        F32 left = 0.0f;
        F32 right = 0.0f;
    };
    std::vector<Shown> shown;
    for (const auto& group : rig.groups) {
        if (const auto extent = rig.extent(group)) {
            shown.push_back({group, extent->first, extent->second});
        }
    }
    std::vector<std::string> order;
    for (const auto& group : rig.groups) {
        order.insert(order.end(), group.begin(), group.end());
    }
    const auto minimizedIn = [&](SweepRig& from, const std::string& name) {
        return name != "h" && name != "t" && std::find(from.names.begin(), from.names.end(), name) != from.names.end() &&
               !from.probes[name]->visible();
    };
    const auto whole = [&](const std::string& name) {
        if (!kept.contains(name)) {
            return false;
        }
        const Rect& tile = kept.at(name);
        return std::abs(tile.y - area.y) <= margin && std::abs(tile.height - area.height) <= margin;
    };
    const auto columnWidth = [&](const std::vector<std::string>& panes) {
        F32 low = std::numeric_limits<F32>::max();
        F32 high = std::numeric_limits<F32>::lowest();
        for (const auto& name : panes) {
            if (kept.contains(name)) {
                low = std::min(low, kept.at(name).x);
                high = std::max(high, kept.at(name).right());
            }
        }
        return high > low ? high - low : 0.0f;
    };
    const auto expect = [&](Extent2D<F32> at) {
        SweepExpect result;
        const F32 band = 12.0f * scale;
        std::optional<U64> best;
        F32 distance = 0.0f;
        const U64 count = shown.size();
        for (U64 index = 0; index <= count; ++index) {
            const F32 left = index == 0 ? area.x : shown[index - 1].right;
            const F32 right = index == count ? area.right() : shown[index].left;
            const F32 center = index == 0 ? area.x : index == count ? area.right() : (left + right) * 0.5f;
            if (at.x < left - band || at.x > right + band || (best && distance <= std::abs(at.x - center))) {
                continue;
            }
            best = index;
            distance = std::abs(at.x - center);
        }
        if (best) {
            const U64 index = *best;
            const auto rep = [&](U64 slot, bool last) {
                auto panes = shown[slot].panes;
                if (last) {
                    std::reverse(panes.begin(), panes.end());
                }
                for (const auto& name : panes) {
                    if (name != moved && name != "h") {
                        return name;
                    }
                }
                return std::string();
            };
            const std::string lead = index > 0 ? rep(index - 1, true) : std::string();
            const std::string trail = index < count ? rep(index, false) : std::string();
            const bool leadOpen = !lead.empty() && !minimizedIn(reference, lead);
            const bool trailOpen = !trail.empty() && !minimizedIn(reference, trail);
            if ((index > 0 && lead.empty()) || (index < count && trail.empty()) || (!leadOpen && !trailOpen)) {
                return result;
            }
            const std::string leadWhole = leadOpen && whole(lead) ? lead : std::string();
            const std::string trailWhole = trailOpen && whole(trail) ? trail : std::string();
            const F32 center = index == 0 ? area.x : index == count ? area.right()
                                                                    : (shown[index - 1].right + shown[index].left) * 0.5f;
            if (!leadWhole.empty() || !trailWhole.empty()) {
                const bool useTrail = !trailWhole.empty() && (leadWhole.empty() || at.x >= center);
                result.offered = true;
                result.split = useTrail ? trailWhole : leadWhole;
                result.side = useTrail ? Dock::Side::Left : Dock::Side::Right;
            } else {
                result.offered = true;
                F32 pool = 0.0f;
                for (const auto& [slot, open] : {std::make_pair(index - 1, leadOpen), std::make_pair(index, trailOpen)}) {
                    if (!open || slot >= count) {
                        continue;
                    }
                    std::vector<std::string> column;
                    for (const auto& name : shown[slot].panes) {
                        if (name != moved) {
                            column.push_back(name);
                        }
                    }
                    pool += columnWidth(column);
                    result.payers.insert(result.payers.end(), column.begin(), column.end());
                }
                const F32 half = (pool - gap) * 0.5f;
                const F32 mine = source.folded ? 0.0f : ownAlong(false);
                result.size = source.folded ? strip : std::max(floor, mine > 0.0f ? std::min(mine, half) : half);
                return result;
            }
        } else {
            const F32 slack = gap * 0.5f + 1.0f;
            std::string hit = rig.dock.paneAt(at);
            if (hit.empty()) {
                for (const auto& name : order) {
                    const auto all = rig.tiles("");
                    if (all.contains(name) && Sakura::Inflate(all.at(name), slack).contains(at.x, at.y)) {
                        hit = name;
                        break;
                    }
                }
            }
            if (hit.empty() || hit == moved || !before.contains(hit)) {
                return result;
            }
            const Rect& tile = before.at(hit);
            const F32 u = (at.x - tile.x) / tile.width;
            const F32 v = (at.y - tile.y) / tile.height;
            const F32 nearest = std::min({u, 1.0f - u, v, 1.0f - v});
            result.offered = true;
            result.split = hit;
            result.side = nearest == u ? Dock::Side::Left : nearest == 1.0f - u ? Dock::Side::Right
                                       : nearest == v                           ? Dock::Side::Top
                                                                                : Dock::Side::Bottom;
        }
        const bool vertical = result.side == Dock::Side::Top || result.side == Dock::Side::Bottom;
        const F32 slot = Along(kept.at(result.split), vertical);
        const F32 half = (slot - gap) * 0.5f;
        const F32 mine = source.folded ? 0.0f : ownAlong(vertical);
        F32 size = mine > 0.0f ? std::min(mine, half) : half;
        if (!vertical && whole(result.split)) {
            size = std::max(floor, size);
        }
        result.size = source.folded ? strip : size;
        result.payers = {result.split};
        return result;
    };

    std::vector<std::tuple<Rect, Extent2D<F32>, SweepExpect>> offers;
    U64 refused = 0;
    for (F32 x = 1600.0f * scale + 1.0f; x < 3200.0f * scale; x += 25.0f * scale) {
        for (const F32 v : {0.04f, 0.3f, 0.5f, 0.7f, 0.96f}) {
            const Extent2D<F32> at = {x, 600.0f * scale * v};
            CAPTURE(at.x, at.y);
            const auto wanted = expect(at);
            const auto preview = rig.hover(moved, at);
            if (!wanted.offered) {
                CHECK_FALSE(preview.has_value());
                continue;
            }
            if (!preview) {
                ++refused;
                continue;
            }
            if (!wanted.split.empty()) {
                const bool vertical = wanted.side == Dock::Side::Top || wanted.side == Dock::Side::Bottom;
                const Rect& tile = before.at(wanted.split);
                const F32 thickness = std::min(wanted.size, Along(tile, vertical));
                CHECK(Along(*preview, vertical) == Catch::Approx(thickness).margin(margin));
                CHECK(Along(*preview, !vertical) == Catch::Approx(Along(tile, !vertical)).margin(margin));
                const bool end = wanted.side == Dock::Side::Right || wanted.side == Dock::Side::Bottom;
                const F32 edge = end ? (vertical ? preview->bottom() : preview->right()) : (vertical ? preview->y : preview->x);
                const F32 target = end ? (vertical ? tile.bottom() : tile.right()) : (vertical ? tile.y : tile.x);
                CHECK(edge == Catch::Approx(target).margin(margin));
            } else {
                CHECK(preview->width == Catch::Approx(wanted.size).margin(margin));
            }
            if (std::none_of(offers.begin(), offers.end(), [&](const auto& offer) {
                    return std::get<0>(offer) == *preview;
                })) {
                offers.emplace_back(*preview, at, wanted);
            }
        }
    }
    CHECK(offers.size() > 4);
    CHECK(refused < offers.size() + 8);

    for (const auto& [preview, at, wanted] : offers) {
        CAPTURE(preview.x, preview.y, preview.width, preview.height, at.x, at.y, wanted.split);
        SweepRig drop(scale, fixture, source.folded, "");
        const auto offered = drop.hover(moved, at);
        REQUIRE(offered.has_value());
        REQUIRE(*offered == preview);
        drop.drop(moved);
        REQUIRE(drop.emissions.trees.size() == 1);
        REQUIRE(drop.emissions.last().has_value());
        const DockTiles& tree = *drop.emissions.last();
        const auto* leaf = FindLeaf(tree, moved);
        REQUIRE(leaf != nullptr);
        CHECK(leaf->minimized == source.folded);
        CHECK(drop.probes[moved]->visible() == !source.folded);

        const auto after = drop.tiles("");
        if (!source.folded) {
            REQUIRE(after.contains(moved));
            const Rect landed = after.at(moved);
            if (wanted.split.empty()) {
                CHECK(Near(landed, preview, margin));
            } else {
                const bool vertical = wanted.side == Dock::Side::Top || wanted.side == Dock::Side::Bottom;
                const Rect& slot = kept.at(wanted.split);
                CHECK(Along(landed, vertical) == Catch::Approx(wanted.size).margin(margin));
                CHECK(Along(landed, !vertical) == Catch::Approx(Along(slot, !vertical)).margin(margin));
                const bool end = wanted.side == Dock::Side::Right || wanted.side == Dock::Side::Bottom;
                const F32 edge = end ? (vertical ? landed.bottom() : landed.right()) : (vertical ? landed.y : landed.x);
                const F32 target = end ? (vertical ? slot.bottom() : slot.right()) : (vertical ? slot.y : slot.x);
                CHECK(edge == Catch::Approx(target).margin(margin));
            }
        }
        for (const auto& [name, rect] : after) {
            CAPTURE(name);
            CHECK(rect.width >= floor - margin);
            if (name == moved || std::find(wanted.payers.begin(), wanted.payers.end(), name) != wanted.payers.end()) {
                continue;
            }
            REQUIRE(kept.contains(name));
            CHECK(Near(rect, kept.at(name), margin));
        }
    }
}

TEST_CASE("A hidden tile beside a split keeps its stored pixel size and the tiles that do not pay keep their rects",
          "[core][sakura][dock][columns][drop-rules]") {
    Probe a;
    Probe b;
    Probe c;
    Probe h;
    Probe e;
    Probe t;
    for (auto* probe : {&a, &b, &c, &h, &e, &t}) {
        probe->consume = false;
    }
    Dock tabs;
    Dock dock;
    Emissions emissions;
    auto tab = MakePane("t", &t);
    tab.group = "terminal";
    tabs.update({.tabs = true, .accepts = "main", .panes = {tab}});
    auto hidden = MakePane("h", &h);
    hidden.hidden = true;
    Dock::Config config = {
        .accepts = "panel",
        .adopts = {"terminal"},
        .panes = {MakePane("a", &a), MakePane("b", &b), MakePane("c", &c), hidden, MakePane("e", &e)},
        .tiles = ColumnRow({Leaf("a"), Column({500.0f, 500.0f}, {Leaf("b"), Row({250.0f, 250.0f, 500.0f},
                                                                              {Leaf("c"), Leaf("h"), Leaf("e")})})},
                           {0.0f, 900.0f}),
        .onTiles = emissions.sink(),
        .columns = true,
        .flexPane = "a",
    };
    dock.update(config);
    Columns columns(tabs, dock);
    Layer layer;
    layer.update({.content = &columns});
    LayerHost host({&layer});
    const auto ctx = MakeContext({3200.0f, 600.0f});
    const Rect bounds = {0.0f, 0.0f, 3200.0f, 600.0f};
    host.place(ctx, bounds);
    REQUIRE_FALSE(h.visible());
    const Rect slot = c.laidFrame;
    const auto rect = tabs.tabRect("t");
    REQUIRE(rect.has_value());
    host.send(Mouse(MouseEventType::Click, rect->x + 5.0f, rect->y + 12.0f));
    host.send(Mouse(MouseEventType::Move, rect->x + 15.0f, rect->y + 12.0f));
    const Extent2D<F32> point = {e.laidFrame.x + 20.0f, e.laidFrame.y + e.laidFrame.height * 0.5f};
    host.send(Mouse(MouseEventType::Move, point.x, point.y));
    REQUIRE(host.drag.active);
    auto hovered = ctx;
    hovered.mouse = point;
    host.place(hovered, bounds);
    REQUIRE(host.drag.target.has_value());
    host.send(Mouse(MouseEventType::Release, point.x, point.y));
    REQUIRE(emissions.trees.size() == 1);
    REQUIRE(emissions.last().has_value());
    std::function<const DockTiles*(const DockTiles&)> parent = [&](const DockTiles& node) -> const DockTiles* {
        for (const auto& child : node.children) {
            if (child.children.empty() && child.pane == "h") {
                return &node;
            }
            if (const auto* found = parent(child)) {
                return found;
            }
        }
        return nullptr;
    };
    const auto* row = parent(*emissions.last());
    REQUIRE(row != nullptr);
    CHECK(Leaves(*row) == std::vector<std::string>{"c", "h", "t", "e"});
    REQUIRE(row->sizes.size() == 4);
    CHECK(row->sizes[1] == 250.0f);
    config.tiles = emissions.last();
    config.panes.push_back(tab);
    tabs.update({.tabs = true, .accepts = "main"});
    dock.update(config);
    host.place(ctx, bounds);
    CHECK(t.visible());
    CHECK(c.laidFrame == slot);
}

TEST_CASE("Dragging a pane into an empty column dock offers its whole viewport and moves the pane over",
          "[core][sakura][dock][columns]") {
    ColumnRig rig;
    rig.tiles = ColumnRow({Leaf("a"), Leaf("b"), Leaf("c")});
    Emissions target;
    Dock empty;
    Columns columns(rig.dock, empty);
    Layer layer;
    layer.update({.content = &columns});
    LayerHost host({&layer});
    rig.update();
    empty.update({.accepts = "panel", .onTiles = target.sink(), .columns = true});
    host.place(MakeContext(), kScreen);

    host.send(Mouse(MouseEventType::Click, 30.0f, 16.0f));
    host.send(Mouse(MouseEventType::Move, 40.0f, 16.0f));
    host.send(Mouse(MouseEventType::Move, 600.0f, 300.0f));
    REQUIRE(host.drag.active);
    host.place(MouseAt(600.0f, 300.0f), kScreen);
    REQUIRE(host.drag.target.has_value());
    CHECK(host.drag.target->owner == static_cast<const void*>(&empty));
    CHECK(host.drag.target->preview == Rect{406.0f, 6.0f, 388.0f, 588.0f});

    host.send(Mouse(MouseEventType::Release, 600.0f, 300.0f));
    REQUIRE(rig.emitted.size() == 1);
    CHECK(rig.emitted[0] == ColumnRow({Leaf("b"), Leaf("c")}));
    REQUIRE(target.trees.size() == 1);
    CHECK(target.last() == Leaf("a"));
}

TEST_CASE("Dragging the divider between stacked panes resizes their rows, emits on release, and double clicking evens them",
          "[core][sakura][dock][columns]") {
    const auto ctx = MakeContext({1600.0f, 600.0f});
    ColumnRig rig;
    rig.tiles = ColumnRow({Leaf("a"), Column({500.0f, 500.0f}, {Leaf("b"), Leaf("c")})});
    Host host(rig.dock);
    rig.update();
    host.place(ctx, kWideDock);
    CHECK(rig.b.laidFrame == Body({1174.0f, 6.0f, 420.0f, 291.0f}));
    CHECK(rig.c.laidFrame == Body({1174.0f, 303.0f, 420.0f, 291.0f}));
    CHECK(Peek::CursorAt(rig.dock, {1300.0f, 300.0f}) == SurfaceCursor::ResizeNS);

    host.send(Mouse(MouseEventType::Click, 1300.0f, 300.0f));
    host.send(Mouse(MouseEventType::Move, 1300.0f, 153.0f));
    host.place(ctx, kWideDock);
    CHECK(rig.b.laidFrame == Body({1174.0f, 6.0f, 420.0f, 144.0f}));
    CHECK(rig.c.laidFrame == Body({1174.0f, 156.0f, 420.0f, 438.0f}));
    CHECK(rig.a.laidFrame.width == 1162.0f);
    CHECK(rig.emitted.empty());
    host.send(Mouse(MouseEventType::Release, 1300.0f, 153.0f));
    REQUIRE(rig.emitted.size() == 1);
    REQUIRE(rig.emitted[0]->children.size() == 2);
    CHECK(rig.emitted[0]->sizes == std::vector<F32>{0.0f, 0.0f});
    CHECK(rig.emitted[0]->children[0] == Leaf("a"));
    CheckApprox(rig.emitted[0]->children[1], Column({144.0f, 438.0f}, {Leaf("b"), Leaf("c")}));

    rig.update();
    host.place(ctx, kWideDock);
    CHECK(rig.b.laidFrame == Body({1174.0f, 6.0f, 420.0f, 144.0f}));
    host.send(Mouse(MouseEventType::Click, 1300.0f, 153.0f, MouseButton::Left, 2));
    host.send(Mouse(MouseEventType::Release, 1300.0f, 153.0f));
    REQUIRE(rig.emitted.size() == 2);
    CheckApprox(rig.emitted[1]->children[1], Column({291.0f, 291.0f}, {Leaf("b"), Leaf("c")}));
    rig.update();
    host.place(ctx, kWideDock);
    CHECK(rig.b.laidFrame == Body({1174.0f, 6.0f, 420.0f, 291.0f}));
    CHECK(rig.c.laidFrame == Body({1174.0f, 303.0f, 420.0f, 291.0f}));
}

TEST_CASE("A divider left of the flex column trades width with it and keeps its left neighbour above the minimum",
          "[core][sakura][dock][columns]") {
    const auto ctx = MakeContext({1600.0f, 600.0f});
    ColumnRig rig;
    rig.flex = "b";
    rig.tiles = ColumnRow({Leaf("a"), Leaf("b"), Leaf("c")});
    Host host(rig.dock);
    rig.update();
    host.place(ctx, kWideDock);
    CHECK(rig.a.laidFrame.width == 420.0f);
    CHECK(rig.b.laidFrame == Rect{432.0f, 26.0f, 736.0f, 568.0f});

    host.send(Mouse(MouseEventType::Click, 429.0f, 300.0f));
    host.send(Mouse(MouseEventType::Move, 529.0f, 300.0f));
    host.place(ctx, kWideDock);
    CHECK(rig.a.laidFrame.width == 520.0f);
    CHECK(rig.b.laidFrame == Rect{532.0f, 26.0f, 636.0f, 568.0f});
    CHECK(rig.c.laidFrame.x == 1174.0f);
    host.send(Mouse(MouseEventType::Release, 529.0f, 300.0f));
    REQUIRE(rig.emitted.size() == 1);
    CHECK(rig.emitted[0]->sizes == std::vector<F32>{520.0f, 0.0f, 0.0f});

    rig.update();
    host.place(ctx, kWideDock);
    host.send(Mouse(MouseEventType::Click, 529.0f, 300.0f));
    host.send(Mouse(MouseEventType::Move, 29.0f, 300.0f));
    host.send(Mouse(MouseEventType::Release, 29.0f, 300.0f));
    REQUIRE(rig.emitted.size() == 2);
    CHECK(rig.emitted[1]->sizes == std::vector<F32>{240.0f, 0.0f, 0.0f});
    rig.update();
    host.place(ctx, kWideDock);
    CHECK(rig.a.laidFrame.width == 240.0f);
    CHECK(rig.b.laidFrame == Rect{252.0f, 26.0f, 916.0f, 568.0f});
}

TEST_CASE("Without a flex column a divider fixes every column width and trades with its neighbour down to the minimum",
          "[core][sakura][dock][columns]") {
    const auto ctx = MakeContext({1600.0f, 600.0f});
    ColumnRig rig;
    rig.flex.clear();
    rig.tiles = ColumnRow({Leaf("a"), Leaf("b"), Leaf("c")});
    Host host(rig.dock);
    rig.update();
    host.place(ctx, kWideDock);

    host.send(Mouse(MouseEventType::Click, 534.0f, 300.0f));
    host.send(Mouse(MouseEventType::Move, 634.0f, 300.0f));
    host.send(Mouse(MouseEventType::Release, 634.0f, 300.0f));
    REQUIRE(rig.emitted.size() == 1);
    CheckSizes(rig.emitted[0]->sizes, {625.33331f, 425.33331f, 525.33331f});

    rig.update();
    host.place(ctx, kWideDock);
    CHECK(rig.a.laidFrame.width == 625.0f);
    CHECK(rig.b.laidFrame == Rect{637.0f, 26.0f, 426.0f, 568.0f});
    CHECK(rig.c.laidFrame == Rect{1069.0f, 26.0f, 525.0f, 568.0f});

    host.send(Mouse(MouseEventType::Click, 634.0f, 300.0f));
    host.send(Mouse(MouseEventType::Move, 234.0f, 300.0f));
    host.place(ctx, kWideDock);
    CHECK(rig.a.laidFrame == Rect{6.0f, 26.0f, 240.0f, 568.0f});
    CHECK(rig.b.laidFrame == Rect{252.0f, 26.0f, 811.0f, 568.0f});
    CHECK(rig.c.laidFrame.right() == 1594.0f);
    host.send(Mouse(MouseEventType::Release, 234.0f, 300.0f));
    REQUIRE(rig.emitted.size() == 2);
    CheckSizes(rig.emitted[1]->sizes, {240.0f, 810.66663f, 525.33331f});
}

TEST_CASE("Double clicking a divider evens its neighbours beside the flex column, without one, and in a narrow window",
          "[core][sakura][dock][columns]") {
    struct Case {
        std::string flex;
        Rect bounds;
        F32 x;
        std::vector<F32> widths;
    };
    const auto test = GENERATE(Case{"b", {0.0f, 0.0f, 1600.0f, 600.0f}, 429.0f, {578.0f, 0.0f, 0.0f}},
                               Case{"", {0.0f, 0.0f, 1600.0f, 600.0f}, 1066.0f, {525.33331f, 525.33331f, 525.33331f}},
                               Case{"a", {0.0f, 0.0f, 800.0f, 600.0f}, 291.0f, {264.54547f, 264.54547f, 246.90909f}});
    CAPTURE(test.flex, test.x);

    const auto ctx = MakeContext({test.bounds.width, test.bounds.height});
    ColumnRig rig;
    rig.flex = test.flex;
    rig.tiles = ColumnRow({Leaf("a"), Leaf("b"), Leaf("c")});
    Host host(rig.dock);
    rig.update();
    host.place(ctx, test.bounds);

    host.send(Mouse(MouseEventType::Click, test.x, 300.0f, MouseButton::Left, 2));
    host.send(Mouse(MouseEventType::Release, test.x, 300.0f));
    REQUIRE(rig.emitted.size() == 1);
    CheckSizes(rig.emitted[0]->sizes, test.widths);
}

TEST_CASE("Cancelling a column divider drag keeps the dragged widths and reports them like a release",
          "[core][sakura][dock][columns]") {
    const auto ctx = MakeContext({1600.0f, 600.0f});
    ColumnRig rig;
    rig.tiles = ColumnRow({Leaf("a"), Leaf("b"), Leaf("c")});
    Host host(rig.dock);
    rig.update();
    host.place(ctx, kWideDock);

    host.send(Mouse(MouseEventType::Click, 745.0f, 300.0f));
    host.send(Mouse(MouseEventType::Move, 645.0f, 300.0f));
    host.place(ctx, kWideDock);
    CHECK(rig.b.laidFrame.width == 520.0f);
    host.send(Mouse(MouseEventType::Cancel, 645.0f, 300.0f));
    REQUIRE(rig.emitted.size() == 1);
    CHECK(rig.emitted[0]->sizes == std::vector<F32>{0.0f, 520.0f, 0.0f});

    host.send(Mouse(MouseEventType::Move, 545.0f, 300.0f));
    host.send(Mouse(MouseEventType::Release, 545.0f, 300.0f));
    CHECK(rig.emitted.size() == 1);
    rig.update();
    host.place(ctx, kWideDock);
    CHECK(rig.b.laidFrame == Rect{648.0f, 26.0f, 520.0f, 568.0f});
}

TEST_CASE("Leaving during a tile splitter drag keeps the dragged sizes and reports them like a release",
          "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Emissions emissions;
    Dock dock;
    Host host(dock);
    dock.update({.panes = {MakePane("a"), MakePane("b"), MakePane("c")},
                 .tiles = Row({500.0f, 250.0f, 250.0f}, {Leaf("a"), Leaf("b"), Leaf("c")}),
                 .onTiles = emissions.sink()});
    host.place(ctx, kScreen);

    host.send(Mouse(MouseEventType::Click, 597.0f, 300.0f));
    host.send(Mouse(MouseEventType::Move, 498.5f, 300.0f));
    host.send(Mouse(MouseEventType::Leave, 498.5f, 300.0f));
    REQUIRE(emissions.trees.size() == 1);
    REQUIRE(emissions.last().has_value());
    CheckApprox(*emissions.last(), Row({388.0f, 95.5f, 292.5f}, {Leaf("a"), Leaf("b"), Leaf("c")}));

    host.send(Mouse(MouseEventType::Move, 400.0f, 300.0f));
    host.send(Mouse(MouseEventType::Release, 400.0f, 300.0f));
    CHECK(emissions.trees.size() == 1);
}

TEST_CASE("Column mode adopts a vertical root as one stacked column", "[core][sakura][dock][columns]") {
    ColumnRig rig;
    rig.withC = false;
    rig.tiles = Column({500.0f, 500.0f}, {Leaf("a"), Leaf("b")});
    Host host(rig.dock);
    rig.update();
    host.place(MakeContext({1600.0f, 600.0f}), kWideDock);

    CHECK(rig.a.laidFrame == Body({6.0f, 6.0f, 1588.0f, 291.0f}));
    CHECK(rig.b.laidFrame == Body({6.0f, 303.0f, 1588.0f, 291.0f}));
    CHECK(rig.emitted.empty());
}

TEST_CASE("Column mode flattens a row nested in a row into columns that keep the nested row's width",
          "[core][sakura][dock][columns]") {
    ColumnRig rig;
    rig.tiles = Row({700.0f, 600.0f}, {Leaf("a"), Row({300.0f, 300.0f}, {Leaf("b"), Leaf("c")})});
    Host host(rig.dock);
    rig.update();
    host.place(MakeContext({1600.0f, 600.0f}), kWideDock);

    CHECK(rig.a.laidFrame == Rect{6.0f, 26.0f, 982.0f, 568.0f});
    CHECK(rig.b.laidFrame == Rect{994.0f, 26.0f, 297.0f, 568.0f});
    CHECK(rig.c.laidFrame == Rect{1297.0f, 26.0f, 297.0f, 568.0f});
    CHECK(rig.emitted.empty());
}

TEST_CASE("Column mode keeps a row nested inside a column side by side within its share of the column",
          "[core][sakura][dock][columns]") {
    Probe a;
    Probe b;
    Probe c;
    Probe d;
    Emissions emissions;
    Dock dock;
    Host host(dock);
    dock.update({
        .accepts = "panel",
        .panes = {MakePane("a", &a), MakePane("b", &b), MakePane("c", &c), MakePane("d", &d)},
        .tiles = ColumnRow({Leaf("a"), Column({500.0f, 500.0f}, {Leaf("b"), Row({500.0f, 500.0f}, {Leaf("c"), Leaf("d")})})}),
        .onTiles = emissions.sink(),
        .columns = true,
        .flexPane = "a",
    });
    host.place(MakeContext({1600.0f, 600.0f}), kWideDock);

    CHECK(a.laidFrame == Rect{6.0f, 26.0f, 1162.0f, 568.0f});
    CHECK(b.laidFrame == Body({1174.0f, 6.0f, 420.0f, 291.0f}));
    CHECK(c.laidFrame == Body({1174.0f, 303.0f, 207.0f, 291.0f}));
    CHECK(d.laidFrame == Body({1387.0f, 303.0f, 207.0f, 291.0f}));
    CHECK(emissions.trees.empty());
}

TEST_CASE("Closing a stacked pane leaves its column-mate alone in the column",
          "[core][sakura][dock][columns]") {
    const auto ctx = MakeContext({1600.0f, 600.0f});
    ColumnRig rig;
    rig.tiles = ColumnRow({Leaf("a"), Column({500.0f, 500.0f}, {Leaf("b"), Leaf("c")})});
    Host host(rig.dock);
    rig.update();
    host.place(ctx, kWideDock);

    const auto close = CloseOf({1174.0f, 303.0f, 420.0f, 291.0f});
    host.click(close.x, close.y);
    CHECK(rig.closed == std::vector<std::string>{"c"});
    REQUIRE(rig.emitted.size() == 1);
    CHECK(rig.emitted[0] == ColumnRow({Leaf("a"), Leaf("b")}));

    rig.withC = false;
    rig.update();
    host.place(ctx, kWideDock);
    CHECK(rig.b.laidFrame == Body({1174.0f, 6.0f, 420.0f, 588.0f}));
    CHECK(rig.emitted.size() == 1);
}

TEST_CASE("Closing the flex pane shares the spare width between the remaining columns",
          "[core][sakura][dock][columns]") {
    const auto ctx = MakeContext({1600.0f, 600.0f});
    ColumnRig rig;
    rig.tiles = ColumnRow({Leaf("a"), Leaf("b"), Leaf("c")});
    Host host(rig.dock);
    rig.update();
    host.place(ctx, kWideDock);

    const auto close = CloseOf({6.0f, 6.0f, 736.0f, 588.0f});
    host.click(close.x, close.y);
    CHECK(rig.closed == std::vector<std::string>{"a"});
    REQUIRE(rig.emitted.size() == 1);
    CHECK(rig.emitted[0] == ColumnRow({Leaf("b"), Leaf("c")}));

    rig.withA = false;
    rig.update();
    host.place(ctx, kWideDock);
    CHECK(rig.b.laidFrame == Rect{6.0f, 26.0f, 791.0f, 568.0f});
    CHECK(rig.c.laidFrame == Rect{803.0f, 26.0f, 791.0f, 568.0f});
    CHECK(rig.emitted.size() == 1);
}

TEST_CASE("Unknown leaves in column mode hold a centered loading placeholder until their pane arrives",
          "[core][sakura][dock][columns]") {
    const auto ctx = MakeContext({1600.0f, 600.0f});
    ColumnRig rig;
    rig.withC = false;
    rig.tiles = ColumnRow({Leaf("a"), Leaf("c"), Leaf("b")});
    Host host(rig.dock);
    rig.update();
    host.place(ctx, kWideDock);

    const auto loading = [&]() -> std::optional<Rect> {
        for (auto* stack : Direct<Sakura::Stack>(rig.dock)) {
            const auto spinners = Find<Sakura::Spinner>(*stack);
            auto* label = ShownText(*stack, "Loading");
            if (!spinners.empty() && label != nullptr) {
                CHECK(label->config().align == Sakura::Text::Align::Center);
                return Peek::Frame(*spinners.front());
            }
        }
        return std::nullopt;
    };
    const auto placeholder = loading();
    REQUIRE(placeholder.has_value());
    CHECK(placeholder->x + placeholder->width * 0.5f == Catch::Approx(958.0f).margin(1.0f));
    CHECK(placeholder->height == 18.0f);
    CHECK(placeholder->y > 280.0f);
    CHECK(placeholder->bottom() < 310.0f);

    host.place(ctx, {kWideDock.x, kWideDock.y, kWideDock.width, 60.0f});
    CHECK_FALSE(loading().has_value());
    host.place(ctx, kWideDock);
    REQUIRE(loading().has_value());
    CHECK(rig.b.laidFrame.x == 1174.0f);
    CHECK(rig.emitted.empty());

    rig.withC = true;
    rig.update();
    host.place(ctx, kWideDock);
    CHECK_FALSE(loading().has_value());
    CHECK(rig.c.laidFrame == Rect{748.0f, 26.0f, 420.0f, 568.0f});
    CHECK(rig.emitted.empty());
}

TEST_CASE("Saved view flags and widths survive late arriving dock content and smaller windows",
          "[core][sakura][dock][columns][recovery]") {
    ColumnRig rig;
    rig.persistView = true;
    rig.withC = false;
    auto late = Leaf("c");
    SECTION("minimized") { late.minimized = true; }
    SECTION("maximized") { late.maximized = true; }
    const auto saved = ColumnRow({Leaf("a"), late, Leaf("b")}, {0.0f, 510.0f, 350.0f});
    rig.tiles = saved;
    Host host(rig.dock);
    rig.update();
    host.place(MakeContext({1600.0f, 600.0f}), kWideDock);
    CHECK(rig.emitted.empty());

    rig.withC = true;
    rig.update();
    host.place(MakeContext({1600.0f, 600.0f}), kWideDock);
    CHECK(rig.c.visible() == !late.minimized);
    CHECK(rig.a.visible() == !late.maximized);
    if (late.maximized) {
        CHECK(rig.c.laidFrame == Body({6.0f, 6.0f, 1588.0f, 588.0f}));
    }
    host.place(MakeContext({600.0f, 400.0f}), {0.0f, 0.0f, 600.0f, 400.0f});
    CHECK(rig.tiles == saved);
    CHECK(rig.emitted.empty());
    host.place(MakeContext({1600.0f, 600.0f}), kWideDock);
    CHECK(rig.tiles == saved);
}

TEST_CASE("A pane added while a column is maximized restores the columns and shows the new one",
          "[core][sakura][dock][columns]") {
    const auto ctx = MakeContext({1600.0f, 600.0f});
    ColumnRig rig;
    rig.withC = false;
    rig.tiles = ColumnRow({Leaf("a"), Leaf("b")});
    Host host(rig.dock);
    rig.update();
    host.place(ctx, kWideDock);

    const auto maximize = MaximizeOf({6.0f, 6.0f, 1162.0f, 588.0f});
    host.click(maximize.x, maximize.y);
    host.place(ctx, kWideDock);
    CHECK(rig.a.laidFrame == Body({6.0f, 6.0f, 1588.0f, 588.0f}));
    CHECK_FALSE(rig.b.visible());

    rig.withC = true;
    rig.update();
    host.place(ctx, kWideDock);
    REQUIRE(rig.emitted.size() == 1);
    CHECK(Leaves(*rig.emitted[0]) == std::vector<std::string>{"a", "b", "c"});
    CHECK(rig.a.laidFrame == Rect{6.0f, 26.0f, 736.0f, 568.0f});
    CHECK(rig.b.visible());
    CHECK(rig.c.laidFrame == Rect{1174.0f, 26.0f, 420.0f, 568.0f});
}

TEST_CASE("End chrome sits at the right edge centered on the pill track and presses over it are not bar presses",
          "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    Probe b;
    Probe leading({40.0f, 20.0f});
    Probe trailing({24.0f, 30.0f});
    Probe end({30.0f, 20.0f});
    std::vector<U8> presses;
    Dock dock;
    Host host(dock);
    const auto update = [&](F32 height) {
        dock.update({.tabs = true, .panes = {MakePane("a", &a), MakePane("b", &b)}, .leading = &leading,
                     .trailing = &trailing, .end = &end, .tabBarHeight = height, .leadingInset = 70.0f,
                     .onBarPress = [&](U8 clicks) { presses.push_back(clicks); }});
        host.place(ctx, kScreen);
    };
    update(0.0f);

    CHECK(end.laidFrame == Rect{764.0f, 10.0f, 30.0f, 20.0f});
    CHECK(trailing.laidFrame == Rect{616.0f, 5.0f, 24.0f, 30.0f});
    CHECK(BoxWith(dock, {122.0f, 4.0f, 488.0f, 32.0f}) != nullptr);

    host.click(779.0f, 20.0f);
    CHECK(presses.empty());
    CHECK(end.count(MouseEventType::Click) == 1);
    host.click(700.0f, 19.0f);
    CHECK(presses == std::vector<U8>{1});

    update(48.0f);
    CHECK(end.laidFrame == Rect{764.0f, 15.0f, 30.0f, 20.0f});
    CHECK(BoxWith(dock, {122.0f, 9.0f, 488.0f, 32.0f}) != nullptr);
    CHECK(b.laidFrame.y == 42.0f);
}

TEST_CASE("Chrome components destroyed before the dock drops them are forgotten safely",
          "[core][sakura][dock]") {
    Probe a;
    Dock dock;
    auto leading = std::make_unique<Probe>(Extent2D<F32>{40.0f, 20.0f});
    auto trailing = std::make_unique<Probe>(Extent2D<F32>{30.0f, 20.0f});
    dock.update({.tabs = true, .panes = {MakePane("a", &a)}, .leading = leading.get(), .trailing = trailing.get()});
    Host host(dock);
    host.place(MakeContext(), kScreen);

    leading.reset();
    trailing.reset();
    dock.update({.tabs = true, .panes = {MakePane("a", &a)}});
    host.place(MakeContext(), kScreen);
    CHECK(a.visible());

    Probe replacement({40.0f, 20.0f});
    dock.update({.tabs = true, .panes = {MakePane("a", &a)}, .leading = &replacement});
    host.place(MakeContext(), kScreen);
    CHECK(replacement.laidFrame.width == 40.0f);
}

TEST_CASE("Activating a tab programmatically focuses it, so clicking the previous tab focuses that one again",
          "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    Probe b;
    U64 focusA = 0;
    U64 focusB = 0;
    Dock dock;
    Host host(dock);
    auto paneA = MakePane("a", &a);
    paneA.onFocus = [&] { ++focusA; };
    auto paneB = MakePane("b", &b);
    paneB.onFocus = [&] { ++focusB; };
    dock.update({.tabs = true, .panes = {paneA, paneB}});
    host.place(ctx, kScreen);
    host.click(10.0f, 12.0f);
    host.place(ctx, kScreen);
    REQUIRE(focusA == 1);

    dock.update({.tabs = true, .panes = {paneA, paneB}, .activate = "b"});
    host.place(ctx, kScreen);
    CHECK(focusB == 1);
    dock.update({.tabs = true, .panes = {paneA, paneB}});
    host.place(ctx, kScreen);
    CHECK(focusB == 1);

    host.click(10.0f, 12.0f);
    CHECK(focusA == 2);
}

TEST_CASE("Closing the active tab focuses the tab the dock falls back to", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    Probe b;
    Probe c;
    std::vector<std::string> focused;
    Dock dock;
    Host host(dock);
    const auto panes = [&](bool withB) {
        std::vector<Dock::Pane> list;
        for (auto [id, probe] : {std::pair{"a", &a}, std::pair{"b", &b}, std::pair{"c", &c}}) {
            if (std::string(id) == "b" && !withB) {
                continue;
            }
            auto pane = MakePane(id, probe);
            pane.onFocus = [&focused, id] { focused.emplace_back(id); };
            list.push_back(pane);
        }
        return list;
    };
    dock.update({.tabs = true, .panes = panes(true), .activate = "b"});
    host.place(ctx, kScreen);
    REQUIRE(focused == std::vector<std::string>{"b"});

    dock.update({.tabs = true, .panes = panes(false)});
    host.place(ctx, kScreen);
    REQUIRE(focused.size() == 2);
    CHECK(c.visible() != a.visible());
    CHECK(focused[1] == (c.visible() ? "c" : "a"));

    dock.update({.tabs = true, .panes = panes(false)});
    host.place(ctx, kScreen);
    CHECK(focused.size() == 2);
}

TEST_CASE("A newly added tab counts as focused, so clicking back to the previous tab notifies it",
          "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    Probe b;
    Probe c;
    std::vector<std::string> focused;
    Dock dock;
    Host host(dock);
    const auto pane = [&](const char* id, Probe* probe) {
        auto item = MakePane(id, probe);
        item.onFocus = [&focused, id] { focused.emplace_back(id); };
        return item;
    };
    dock.update({.tabs = true, .panes = {pane("a", &a), pane("b", &b)}});
    host.place(ctx, kScreen);
    REQUIRE(b.visible());
    host.click(10.0f, 12.0f);
    host.place(ctx, kScreen);
    REQUIRE(focused == std::vector<std::string>{"a"});

    dock.update({.tabs = true, .panes = {pane("a", &a), pane("b", &b), pane("c", &c)}});
    host.place(ctx, kScreen);
    CHECK(c.visible());
    CHECK(focused == std::vector<std::string>{"a"});

    host.click(10.0f, 12.0f);
    host.place(ctx, kScreen);
    CHECK(a.visible());
    CHECK(focused == std::vector<std::string>{"a", "a"});
}

TEST_CASE("Selecting another tab before a dropped tab is configured cancels the dropped tab's focus",
          "[core][sakura][dock]") {
    Probe b;
    Probe c;
    U64 focusA = 0;
    U64 focusC = 0;
    Dock left;
    Dock right;
    Columns columns(left, right);
    Layer layer;
    layer.update({.content = &columns});
    LayerHost host({&layer});
    c.consume = false;
    auto paneA = MakePane("a");
    paneA.onFocus = [&] { ++focusA; };
    auto paneC = MakePane("c", &c);
    paneC.onFocus = [&] { ++focusC; };
    left.update({.tabs = true, .accepts = "panel", .panes = {MakePane("a"), MakePane("b", &b)}, .activate = "a"});
    right.update({.tabs = true, .accepts = "panel", .panes = {paneC}});
    host.place(MakeContext(), kScreen);

    host.send(Mouse(MouseEventType::Click, 10.0f, 12.0f));
    host.send(Mouse(MouseEventType::Move, 405.0f, 12.0f));
    host.place(MouseAt(405.0f, 12.0f), kScreen);
    REQUIRE(host.drag.target.has_value());
    host.send(Mouse(MouseEventType::Release, 405.0f, 12.0f));
    host.place(MakeContext(), kScreen);

    host.click(697.0f, 20.0f);
    host.place(MakeContext(), kScreen);
    REQUIRE(c.visible());

    left.update({.tabs = true, .accepts = "panel", .panes = {MakePane("b", &b)}});
    right.update({.tabs = true, .accepts = "panel", .panes = {paneC, paneA}});
    host.place(MakeContext(), kScreen);
    CHECK(focusA == 0);
    CHECK(c.visible());
}

TEST_CASE("A tile dragged into an empty dock and back focuses on each arrival", "[core][sakura][dock]") {
    Probe a;
    U64 focusA = 0;
    Dock left;
    Dock right;
    Columns columns(left, right);
    Layer layer;
    layer.update({.content = &columns});
    LayerHost host({&layer});
    auto paneA = MakePane("a", &a);
    paneA.onFocus = [&] { ++focusA; };
    left.update({.accepts = "panel", .panes = {paneA}, .tiles = Leaf("a")});
    right.update({.accepts = "panel"});
    host.place(MakeContext(), kScreen);
    host.click(100.0f, 12.0f);
    host.place(MakeContext(), kScreen);
    REQUIRE(focusA == 1);

    const auto drag = [&](F32 fromX, F32 toX) {
        host.send(Mouse(MouseEventType::Click, fromX, 12.0f));
        host.send(Mouse(MouseEventType::Move, toX, 300.0f));
        REQUIRE(host.drag.active);
        host.place(MouseAt(toX, 300.0f), kScreen);
        REQUIRE(host.drag.target.has_value());
        host.send(Mouse(MouseEventType::Release, toX, 300.0f));
    };

    drag(100.0f, 600.0f);
    left.update({.accepts = "panel"});
    right.update({.accepts = "panel", .panes = {paneA}, .tiles = Leaf("a")});
    host.place(MakeContext(), kScreen);
    CHECK(focusA == 2);

    drag(500.0f, 200.0f);
    right.update({.accepts = "panel"});
    left.update({.accepts = "panel", .panes = {paneA}, .tiles = Leaf("a")});
    host.place(MakeContext(), kScreen);
    CHECK(focusA == 3);
}

TEST_CASE("Adding tabs while another tab wins activation keeps one focus notification per change",
          "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    Probe b;
    Probe c;
    std::vector<std::string> focused;
    Dock dock;
    Host host(dock);
    const auto pane = [&](const char* id, Probe* probe) {
        auto item = MakePane(id, probe);
        item.onFocus = [&focused, id] { focused.emplace_back(id); };
        return item;
    };
    dock.update({.tabs = true, .panes = {pane("a", &a), pane("b", &b)}, .activate = "a"});
    host.place(ctx, kScreen);
    REQUIRE(focused == std::vector<std::string>{"a"});

    dock.update({.tabs = true, .panes = {pane("a", &a), pane("b", &b), pane("c", &c)}, .activate = "a"});
    host.place(ctx, kScreen);
    CHECK(a.visible());
    CHECK(focused == std::vector<std::string>{"a"});
}

TEST_CASE("A tab added in the same update that configures a dropped tab wins focus over the drop",
          "[core][sakura][dock]") {
    Probe b;
    Probe c;
    Probe d;
    U64 focusA = 0;
    Dock left;
    Dock right;
    Columns columns(left, right);
    Layer layer;
    layer.update({.content = &columns});
    LayerHost host({&layer});
    c.consume = false;
    auto paneA = MakePane("a");
    paneA.onFocus = [&] { ++focusA; };
    left.update({.tabs = true, .accepts = "panel", .panes = {MakePane("a"), MakePane("b", &b)}, .activate = "a"});
    right.update({.tabs = true, .accepts = "panel", .panes = {MakePane("c", &c)}});
    host.place(MakeContext(), kScreen);

    host.send(Mouse(MouseEventType::Click, 10.0f, 12.0f));
    host.send(Mouse(MouseEventType::Move, 405.0f, 12.0f));
    host.place(MouseAt(405.0f, 12.0f), kScreen);
    REQUIRE(host.drag.target.has_value());
    host.send(Mouse(MouseEventType::Release, 405.0f, 12.0f));
    host.place(MakeContext(), kScreen);

    left.update({.tabs = true, .accepts = "panel", .panes = {MakePane("b", &b)}});
    right.update({.tabs = true, .accepts = "panel", .panes = {paneA, MakePane("d", &d)}});
    host.place(MakeContext(), kScreen);
    CHECK(d.visible());
    CHECK(focusA == 0);

    host.click(503.0f, 20.0f);
    CHECK(focusA == 1);
}

TEST_CASE("Tab mode shows the owner's activated pane again when the unchanged config is re-applied after a click",
          "[core][sakura][dock]") {
    Probe a;
    Probe b;
    Dock dock;
    Host host(dock);
    Frames frames(host);
    frames.verify = false;
    const auto update = [&] {
        dock.update({.tabs = true, .panes = {MakePane("a", &a), MakePane("b", &b)}, .activate = "a"});
    };
    REQUIRE(frames.settle(update));
    REQUIRE(a.visible());
    REQUIRE_FALSE(b.visible());

    host.click(372.0f, 20.0f);
    frames.run();
    REQUIRE(b.visible());
    REQUIRE_FALSE(a.visible());

    frames.run(update);
    CHECK(a.visible());
    CHECK_FALSE(b.visible());
    CHECK(frames.run(update).clean());
}

TEST_CASE("A settled dock redraws an unknown leaf's header when only its description changes",
          "[core][sakura][dock]") {
    Scene scene;
    Probe a;
    Dock dock;
    Host host(dock);
    Frames frames(host, kScreen, scene.context());
    frames.verify = false;
    std::string title = "Ghost Alpha";
    const auto tiles = Row({500.0f, 500.0f}, {Leaf("a"), Leaf("ghost")});
    const auto update = [&] {
        dock.update({.panes = {MakePane("a", &a)},
                     .tiles = tiles,
                     .describe = [&](const std::string&) { return title; }});
    };
    const auto draws = [&](const std::string& str) {
        for (const auto& leaf : Drawn(host)) {
            if (const auto* label = std::get_if<Sakura::Label::Config>(&leaf.config)) {
                for (const auto& instance : label->instances) {
                    if (instance.visible && instance.str == str) {
                        return true;
                    }
                }
            }
        }
        return false;
    };
    REQUIRE(frames.settle(update));
    REQUIRE(draws("Ghost Alpha"));

    title = "Ghost Beta";
    frames.run(update);
    CHECK(draws("Ghost Beta"));
    CHECK_FALSE(draws("Ghost Alpha"));
    CHECK(frames.run(update).clean());
}

TEST_CASE("Releasing another button keeps a splitter drag going", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Emissions emissions;
    Dock dock;
    Host host(dock);
    dock.update({.panes = {MakePane("a"), MakePane("b")},
                 .tiles = Row({500.0f, 500.0f}, {Leaf("a"), Leaf("b")}),
                 .onTiles = emissions.sink()});
    host.place(ctx, kScreen);
    const F32 splitter = 400.0f;
    CHECK(host.send(Mouse(MouseEventType::Click, splitter, 300.0f)));
    REQUIRE(dock.resizing());
    host.send(Mouse(MouseEventType::Release, splitter, 300.0f, MouseButton::Right));
    host.send(Mouse(MouseEventType::Release, splitter, 300.0f, MouseButton::Middle));
    CHECK(dock.resizing());
    CHECK(emissions.trees.empty());
    CHECK(host.send(Mouse(MouseEventType::Release, splitter, 300.0f)));
    CHECK_FALSE(dock.resizing());
}

namespace {

std::optional<Sakura::Label::Instance> Spine(Sakura::Component& dock, const std::string& title) {
    for (auto* label : Find<Sakura::Label>(dock)) {
        for (const auto& instance : label->config().instances) {
            if (instance.visible && instance.rotation != 0.0f && instance.str == title) {
                return instance;
            }
        }
    }
    return std::nullopt;
}

}  // namespace

TEST_CASE("Minimizing a lone column collapses it to a strip with a vertical title and a click on the title restores it",
          "[core][sakura][dock][columns]") {
    const auto ctx = MakeContext({1600.0f, 600.0f});
    ColumnRig rig;
    rig.tiles = ColumnRow({Leaf("a"), Leaf("b"), Leaf("c")});
    Host host(rig.dock);
    rig.update();
    host.place(ctx, kWideDock);
    const Rect cardC = {1174.0f, 6.0f, 420.0f, 588.0f};
    REQUIRE(GlyphIn(rig.dock, cardC, ICON_LC_MINUS));

    const auto minimize = MinimizeOf(cardC);
    host.click(minimize.x, minimize.y);
    host.place(ctx, kWideDock);
    CHECK_FALSE(rig.c.visible());
    CHECK(rig.a.laidFrame == Body({6.0f, 6.0f, 1124.0f, 588.0f}));
    CHECK(rig.b.laidFrame == Body({1136.0f, 6.0f, 420.0f, 588.0f}));
    const auto spine = Spine(rig.dock, "c");
    REQUIRE(spine.has_value());
    CHECK(spine->rotation == -90.0f);
    CHECK(spine->rect.y == 6.0f);
    CHECK(spine->rect.height == 588.0f);
    CHECK(spine->rect.x + spine->rect.width * 0.5f == Catch::Approx(1578.0f).margin(1.5f));
    CHECK_FALSE(GlyphIn(rig.dock, {1562.0f, 6.0f, 32.0f, 588.0f}, ICON_LC_MINUS));
    const auto* cross = GlyphIn(rig.dock, {1562.0f, 6.0f, 32.0f, 588.0f}, ICON_LC_X);
    REQUIRE(cross != nullptr);
    const Rect crossFrame = Peek::Frame(*cross);
    CHECK(crossFrame.y < spine->rect.y + 40.0f);
    CHECK(Peek::CursorAt(rig.dock, {1578.0f, 300.0f}) == SurfaceCursor::Hand);
    CHECK(rig.emitted.empty());

    host.click(1578.0f, 300.0f);
    host.place(ctx, kWideDock);
    CHECK(rig.c.visible());
    CHECK(rig.c.laidFrame == Body(cardC));
    CHECK_FALSE(Spine(rig.dock, "c").has_value());
    CHECK(rig.emitted.empty());

    host.click(minimize.x, minimize.y);
    host.place(ctx, kWideDock);
    REQUIRE(Spine(rig.dock, "c").has_value());
    const auto* strip = GlyphIn(rig.dock, {1562.0f, 6.0f, 32.0f, 588.0f}, ICON_LC_X);
    REQUIRE(strip != nullptr);
    const Rect close = Peek::Frame(*strip);
    host.click(close.x + close.width * 0.5f, close.y + close.height * 0.5f);
    host.place(ctx, kWideDock);
    CHECK(rig.closed == std::vector<std::string>{"c"});
    CHECK_FALSE(Spine(rig.dock, "c").has_value());
    CHECK_FALSE(rig.c.visible());
}

TEST_CASE("A pane that closes only when minimized hides its header close until it collapses to a strip",
          "[core][sakura][dock][columns]") {
    const auto ctx = MakeContext({1600.0f, 600.0f});
    ColumnRig rig;
    rig.tiles = ColumnRow({Leaf("a"), Leaf("b"), Leaf("c")});
    rig.closeOnlyMinimized = "c";
    Host host(rig.dock);
    rig.update();
    host.place(ctx, kWideDock);
    const Rect cardC = {1174.0f, 6.0f, 420.0f, 588.0f};
    const auto* minus = GlyphIn(rig.dock, cardC, ICON_LC_MINUS);
    REQUIRE(minus != nullptr);
    CHECK_FALSE(GlyphIn(rig.dock, cardC, ICON_LC_X));
    CHECK(GlyphIn(rig.dock, {1136.0f, 6.0f, 420.0f, 588.0f}, ICON_LC_X));

    const Rect minimize = Peek::Frame(*minus);
    host.click(minimize.x + minimize.width * 0.5f, minimize.y + minimize.height * 0.5f);
    host.place(ctx, kWideDock);
    REQUIRE(Spine(rig.dock, "c").has_value());
    const auto* strip = GlyphIn(rig.dock, {1562.0f, 6.0f, 32.0f, 588.0f}, ICON_LC_X);
    REQUIRE(strip != nullptr);
    const Rect close = Peek::Frame(*strip);
    host.click(close.x + close.width * 0.5f, close.y + close.height * 0.5f);
    host.place(ctx, kWideDock);
    CHECK(rig.closed == std::vector<std::string>{"c"});
}

TEST_CASE("The first minimize draws the vertical title in the same frame without a resource rebuild",
          "[core][sakura][dock][columns][retention]") {
    auto ctx = MakeContext({1600.0f, 600.0f});
    ColumnRig rig;
    rig.tiles = ColumnRow({Leaf("a"), Leaf("b"), Leaf("c")});
    Host host(rig.dock);
    Frames frames(host, kWideDock, ctx);
    const auto update = [&] {
        rig.update();
    };
    REQUIRE(frames.settle(update));

    const auto minimize = MinimizeOf({1174.0f, 6.0f, 420.0f, 588.0f});
    host.click(minimize.x, minimize.y);
    const auto first = frames.run(update);
    CHECK_FALSE(first.rebuilt);
    CHECK(first.repainted);
    CHECK(Spine(rig.dock, "c").has_value());
    CHECK_FALSE(Sakura::Access::resourceDirty(host));
    CHECK_FALSE(frames.run(update).rebuilt);
}

TEST_CASE("A pane that starts minimized on the left shows a bare leftmost strip once and restores on click",
          "[core][sakura][dock][columns]") {
    const auto ctx = MakeContext({1600.0f, 600.0f});
    Probe a;
    Probe notes;
    Dock dock;
    std::optional<DockTiles> tiles;
    const auto update = [&](bool withNotes) {
        std::vector<Dock::Pane> panes = {MakePane("a", &a)};
        if (withNotes) {
            auto pane = MakePane("notes", &notes);
            pane.closable = false;
            pane.maximizable = false;
            pane.leftmost = true;
            pane.startMinimized = true;
            pane.width = 520.0f;
            panes.push_back(std::move(pane));
        }
        dock.update({
            .accepts = "panel",
            .panes = std::move(panes),
            .tiles = tiles,
            .onTiles = [&](std::optional<DockTiles> next) { tiles = std::move(next); },
            .columns = true,
            .flexPane = "a",
        });
    };
    Host host(dock);
    update(true);
    host.place(ctx, kWideDock);

    CHECK_FALSE(notes.visible());
    const auto spine = Spine(dock, "notes");
    REQUIRE(spine.has_value());
    const F32 center = spine->rect.x + spine->rect.width * 0.5f;
    CHECK(center == Catch::Approx(22.0f).margin(1.5f));
    CHECK(spine->rect.y + spine->rect.height * 0.5f == Catch::Approx(300.0f).margin(1.5f));
    CHECK(a.laidFrame.x > 38.0f);
    CHECK_FALSE(GlyphIn(dock, {6.0f, 6.0f, 32.0f, 588.0f}, ICON_LC_MAXIMIZE_2));
    CHECK_FALSE(GlyphIn(dock, {6.0f, 6.0f, 32.0f, 588.0f}, ICON_LC_X));

    host.click(center, 300.0f);
    host.place(ctx, kWideDock);
    CHECK(notes.visible());
    CHECK(notes.laidFrame.x < a.laidFrame.x);
    CHECK(notes.laidFrame.width == Catch::Approx(520.0f).margin(1.0f));

    update(true);
    host.place(ctx, kWideDock);
    CHECK(notes.visible());

    update(false);
    host.place(ctx, kWideDock);
    update(true);
    host.place(ctx, kWideDock);
    CHECK_FALSE(notes.visible());
    CHECK(Spine(dock, "notes").has_value());
}

TEST_CASE("Minimizing and restoring a column settles and its minimize button repaints once and only acts on a release",
          "[core][sakura][dock][columns][retention]") {
    auto ctx = MakeContext({1600.0f, 600.0f});
    ctx.palette = DistinctPalette();
    ColumnRig rig;
    rig.tiles = ColumnRow({Leaf("a"), Leaf("b"), Leaf("c")});
    Host host(rig.dock);
    Frames frames(host, kWideDock, ctx);
    const auto update = [&] {
        rig.update();
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));

    const Rect cardC = {1174.0f, 6.0f, 420.0f, 588.0f};
    const auto minimize = MinimizeOf(cardC);
    const Extent2D<F32> body = {1300.0f, 400.0f};
    CheckHoverRepaintsOnce(frames, host, update, minimize, body);
    CheckLeaveRepaintsOnce(frames, host, update, minimize);

    host.send(Mouse(MouseEventType::Click, minimize.x, minimize.y));
    host.send(Mouse(MouseEventType::Cancel, minimize.x, minimize.y));
    host.send(Mouse(MouseEventType::Release, minimize.x, minimize.y));
    REQUIRE(frames.settle(update));
    CHECK(rig.c.visible());
    host.send(Mouse(MouseEventType::Click, minimize.x, minimize.y));
    host.send(Mouse(MouseEventType::Release, body.x, body.y));
    REQUIRE(frames.settle(update));
    CHECK(rig.c.visible());

    host.click(minimize.x, minimize.y);
    REQUIRE(frames.settle(update));
    CHECK_FALSE(rig.c.visible());
    REQUIRE(Spine(rig.dock, "c").has_value());
    CHECK(DirtyFrames(frames, update, 10) == 0);

    host.click(1578.0f, 300.0f);
    REQUIRE(frames.settle(update));
    CHECK(rig.c.visible());
    CHECK(Retains(frames, update));

    const U64 restructures = frames.restructures;
    const U64 rebuilds = frames.rebuilds;
    host.click(minimize.x, minimize.y);
    CHECK(RepaintOnly(frames.run(update)));
    REQUIRE(frames.settle(update));
    CHECK_FALSE(rig.c.visible());
    host.click(1578.0f, 300.0f);
    CHECK(RepaintOnly(frames.run(update)));
    REQUIRE(frames.settle(update));
    CHECK(rig.c.visible());
    CHECK(frames.restructures == restructures);
    CHECK(frames.rebuilds == rebuilds);
    CHECK(rig.emitted.empty());
}

TEST_CASE("Collapsed strips scale their width and title size with the pixel ratio", "[core][sakura][dock][columns]") {
    auto ctx = MakeContext({3200.0f, 1200.0f});
    ctx.pixelRatio = 2.0f;
    const Rect bounds = {0.0f, 0.0f, 3200.0f, 1200.0f};
    ColumnRig rig;
    rig.tiles = ColumnRow({Leaf("a"), Leaf("b"), Leaf("c")});
    Host host(rig.dock);
    rig.update();
    host.place(ctx, bounds);
    const Rect body = rig.c.laidFrame;
    const Rect card = {body.x, body.y - 2.0f * kHeader, body.width, body.height + 2.0f * kHeader};
    const auto* glyph = GlyphIn(rig.dock, card, ICON_LC_MINUS);
    REQUIRE(glyph != nullptr);
    const Rect button = Peek::Frame(*glyph);
    host.click(button.x + button.width * 0.5f, button.y + button.height * 0.5f);
    host.place(ctx, bounds);
    CHECK_FALSE(rig.c.visible());
    const auto spine = Spine(rig.dock, "c");
    REQUIRE(spine.has_value());
    CHECK(spine->fontSize == 34.0f);
    CHECK(spine->rect.width == 64.0f);
    CHECK(spine->rect.y == card.y);
    CHECK(spine->rect.height == card.height);
    CHECK(spine->rect.x + spine->rect.width * 0.5f == Catch::Approx(bounds.right() - 12.0f - 32.0f).margin(3.0f));
}

TEST_CASE("Minimizing a stacked pane shrinks it to a header the width of a strip and its neighbour takes the height",
          "[core][sakura][dock][columns]") {
    const auto ctx = MakeContext({1600.0f, 600.0f});
    ColumnRig rig;
    rig.tiles = ColumnRow({Leaf("a"), Column({500.0f, 500.0f}, {Leaf("b"), Leaf("c")})});
    Host host(rig.dock);
    rig.update();
    host.place(ctx, kWideDock);

    const auto minimize = MinimizeOf({1174.0f, 303.0f, 420.0f, 291.0f});
    host.click(minimize.x, minimize.y);
    host.place(ctx, kWideDock);
    CHECK_FALSE(rig.c.visible());
    CHECK(rig.b.laidFrame == Body({1174.0f, 6.0f, 420.0f, 550.0f}));
    const Rect shade = {1174.0f, 562.0f, 420.0f, 32.0f};
    CHECK(BoxWith(rig.dock, shade) != nullptr);
    CHECK(GlyphIn(rig.dock, shade, ICON_LC_X));
    CHECK(GlyphIn(rig.dock, shade, ICON_LC_MAXIMIZE_2));
    CHECK_FALSE(GlyphIn(rig.dock, shade, ICON_LC_MINUS));
    CHECK(Peek::CursorAt(rig.dock, {1300.0f, 559.0f}) != SurfaceCursor::ResizeNS);

    host.click(1300.0f, 578.0f);
    host.place(ctx, kWideDock);
    CHECK(rig.c.visible());
    CHECK(rig.c.laidFrame == Body({1174.0f, 303.0f, 420.0f, 291.0f}));
    CHECK(rig.emitted.empty());
}

TEST_CASE("A minimized tinted pane falls back to the default tile fill and outline", "[core][sakura][dock][columns]") {
    constexpr ColorRGBA<F32> kTint = {0.3f, 0.0f, 0.0f, 1.0f};
    const auto ctx = MakeContext({1600.0f, 600.0f});
    ColumnRig rig;
    rig.fillC = kTint;
    rig.tiles = ColumnRow({Leaf("a"), Column({500.0f, 500.0f}, {Leaf("b"), Leaf("c")})});
    Host host(rig.dock);
    rig.update();
    host.place(ctx, kWideDock);
    const auto tinted = [&] {
        U64 count = 0;
        for (auto* box : Direct<Sakura::Box>(rig.dock)) {
            const auto& config = box->config();
            for (const auto& instance : config.instances) {
                if (instance.visible && (instance.backgroundColor == kTint ||
                                         (config.borderWidth > 0.0f && config.borderColor.r > config.borderColor.g &&
                                          config.borderColor.r > kTint.r))) {
                    ++count;
                }
            }
        }
        return count;
    };
    CHECK(tinted() == 3);

    const auto minimize = MinimizeOf({1174.0f, 303.0f, 420.0f, 291.0f});
    host.click(minimize.x, minimize.y);
    host.place(ctx, kWideDock);
    REQUIRE_FALSE(rig.c.visible());
    CHECK(tinted() == 0);
    CHECK(BoxWith(rig.dock, {1174.0f, 562.0f, 420.0f, 32.0f}) != nullptr);
}

TEST_CASE("Minimizing a pane cancels a press held in its content and releases focus inside it",
          "[core][sakura][dock][columns]") {
    const auto ctx = MakeContext({1600.0f, 600.0f});
    ColumnRig rig;
    rig.tiles = ColumnRow({Leaf("a"), Column({500.0f, 500.0f}, {Leaf("b"), Leaf("c")})});
    Host host(rig.dock);
    rig.update();
    host.place(ctx, kWideDock);

    const auto content = rig.c.laidFrame;
    host.send(Mouse(MouseEventType::Click, content.x + 10.0f, content.y + 10.0f));
    REQUIRE(rig.c.count(MouseEventType::Click) == 1);
    REQUIRE(Sakura::Access::focus(rig.c));
    REQUIRE(FocusOwner(host) == &rig.c);
    REQUIRE(rig.c.count(MouseEventType::Cancel) == 0);

    const auto minimize = MinimizeOf({1174.0f, 303.0f, 420.0f, 291.0f});
    host.click(minimize.x, minimize.y);
    host.place(ctx, kWideDock);
    CHECK_FALSE(rig.c.visible());
    CHECK(rig.c.count(MouseEventType::Cancel) == 1);
    CHECK(FocusOwner(host) == nullptr);

    host.place(ctx, kWideDock);
    CHECK(rig.c.count(MouseEventType::Cancel) == 1);
}

TEST_CASE("Dividers beside a collapsed strip resize the open columns on either side of it",
          "[core][sakura][dock][columns]") {
    const auto ctx = MakeContext({1600.0f, 600.0f});
    ColumnRig rig;
    rig.tiles = ColumnRow({Leaf("a"), Leaf("b"), Leaf("c")});
    Host host(rig.dock);
    rig.update();
    host.place(ctx, kWideDock);
    const auto minimize = MinimizeOf({748.0f, 6.0f, 420.0f, 588.0f});
    host.click(minimize.x, minimize.y);
    host.place(ctx, kWideDock);
    REQUIRE_FALSE(rig.b.visible());
    CHECK(rig.a.laidFrame.width == 1124.0f);
    CHECK(rig.c.laidFrame.x == 1174.0f);

    for (const F32 gap : {1133.0f, 1171.0f}) {
        CAPTURE(gap);
        CHECK(Peek::CursorAt(rig.dock, {gap, 300.0f}) == SurfaceCursor::ResizeEW);
    }
    host.send(Mouse(MouseEventType::Click, 1171.0f, 300.0f));
    host.send(Mouse(MouseEventType::Move, 1071.0f, 300.0f));
    host.place(ctx, kWideDock);
    CHECK(rig.c.laidFrame.width == 520.0f);
    CHECK(rig.a.laidFrame.width == 1024.0f);
    host.send(Mouse(MouseEventType::Release, 1071.0f, 300.0f));
    REQUIRE(rig.emitted.size() == 1);
    CHECK(rig.emitted[0]->sizes[1] == 0.0f);
    CHECK(rig.emitted[0]->sizes[2] == 520.0f);
}

TEST_CASE("Fullscreen from a collapsed strip returns to the strip and the last open pane cannot be minimized",
          "[core][sakura][dock][columns]") {
    const auto ctx = MakeContext({1600.0f, 600.0f});
    ColumnRig rig;
    rig.tiles = ColumnRow({Leaf("a"), Leaf("b"), Leaf("c")});
    Host host(rig.dock);
    rig.update();
    host.place(ctx, kWideDock);
    for (const Rect card : {Rect{1174.0f, 6.0f, 420.0f, 588.0f}, Rect{1136.0f, 6.0f, 420.0f, 588.0f}}) {
        const auto minimize = MinimizeOf(card);
        host.click(minimize.x, minimize.y);
        host.place(ctx, kWideDock);
    }
    REQUIRE_FALSE(rig.b.visible());
    REQUIRE_FALSE(rig.c.visible());
    const Rect cardA = {6.0f, 6.0f, 1512.0f, 588.0f};
    CHECK(rig.a.laidFrame == Body(cardA));
    CHECK_FALSE(GlyphIn(rig.dock, cardA, ICON_LC_MINUS));
    CHECK(GlyphIn(rig.dock, cardA, ICON_LC_MAXIMIZE_2));

    const Rect strip = {1562.0f, 6.0f, 32.0f, 588.0f};
    const auto* fullscreen = GlyphIn(rig.dock, strip, ICON_LC_MAXIMIZE_2);
    REQUIRE(fullscreen);
    const Rect button = Peek::Frame(*fullscreen);
    CHECK(button.bottom() > strip.bottom() - 26.0f);
    host.click(button.x + button.width * 0.5f, button.y + button.height * 0.5f);
    host.place(ctx, kWideDock);
    CHECK(rig.c.visible());
    CHECK_FALSE(rig.a.visible());
    const Rect full = {6.0f, 6.0f, 1588.0f, 588.0f};
    CHECK(rig.c.laidFrame == Body(full));
    CHECK_FALSE(GlyphIn(rig.dock, full, ICON_LC_MINUS));

    const auto exit = MaximizeOf(full);
    host.click(exit.x, exit.y);
    host.place(ctx, kWideDock);
    CHECK_FALSE(rig.c.visible());
    CHECK(rig.a.laidFrame == Body(cardA));
    CHECK(Spine(rig.dock, "c").has_value());
    CHECK(rig.emitted.empty());
}

TEST_CASE("Dividers beside a minimized row resize the open rows above and below it", "[core][sakura][dock][columns]") {
    const auto ctx = MakeContext({1600.0f, 600.0f});
    ColumnRig rig;
    rig.tiles = Column({1000.0f, 1000.0f, 1000.0f}, {Leaf("a"), Leaf("b"), Leaf("c")});
    Host host(rig.dock);
    rig.update();
    host.place(ctx, kWideDock);
    REQUIRE(rig.b.laidFrame == Body({6.0f, 204.0f, 1588.0f, 192.0f}));

    const auto minimize = MinimizeOf({6.0f, 204.0f, 1588.0f, 192.0f});
    host.click(minimize.x, minimize.y);
    host.place(ctx, kWideDock);
    REQUIRE_FALSE(rig.b.visible());
    CHECK(rig.a.laidFrame == Body({6.0f, 6.0f, 1588.0f, 272.0f}));
    CHECK(rig.c.laidFrame == Body({6.0f, 322.0f, 1588.0f, 272.0f}));

    for (const F32 gap : {281.0f, 319.0f}) {
        CAPTURE(gap);
        CHECK(Peek::CursorAt(rig.dock, {800.0f, gap}) == SurfaceCursor::ResizeNS);
    }
    host.send(Mouse(MouseEventType::Click, 800.0f, 281.0f));
    host.send(Mouse(MouseEventType::Move, 800.0f, 181.0f));
    host.place(ctx, kWideDock);
    CHECK(rig.a.laidFrame.bottom() == Catch::Approx(178.0f).margin(2.0f));
    CHECK(rig.c.laidFrame.bottom() == 594.0f);
    host.send(Mouse(MouseEventType::Release, 800.0f, 181.0f));
    REQUIRE(rig.emitted.size() == 1);
}

namespace {

struct CancelDeletingProbe : Probe {
    std::function<void()> onCancel;

    bool mouseEvent(const MouseEvent& event) override {
        if (event.type == MouseEventType::Cancel && onCancel) {
            auto callback = std::move(onCancel);
            onCancel = {};
            callback();
            return false;
        }
        return Probe::mouseEvent(event);
    }
};

struct BlurDeletingProbe : Probe {
    std::function<void()> onBlur;

    void focusChanged(bool focused) override {
        if (!focused && onBlur) {
            auto callback = std::move(onBlur);
            onBlur = {};
            callback();
        }
    }
};

}  // namespace

TEST_CASE("Hiding an aside survives the aside destroying itself when its mouse is cancelled",
          "[core][sakura][dock][retention]") {
    const auto ctx = MakeContext();
    Probe content;
    auto aside = std::make_unique<CancelDeletingProbe>();
    Dock dock;
    Host host(dock);
    dock.update({.tabs = true, .panes = {MakePane("a", &content)}, .aside = aside.get(), .asideWidth = 200.0f});
    host.place(ctx, kScreen);
    REQUIRE(aside->visible());
    aside->onCancel = [&] {
        aside.reset();
    };
    dock.update({.tabs = true, .panes = {MakePane("a", &content)}});
    host.place(ctx, kScreen);
    CHECK(aside == nullptr);
    CHECK(content.visible());
    host.place(ctx, kScreen);
    CHECK(content.visible());
}

TEST_CASE("Hiding an aside survives the aside destroying itself when it loses focus",
          "[core][sakura][dock][retention]") {
    const auto ctx = MakeContext();
    Probe content;
    auto aside = std::make_unique<BlurDeletingProbe>();
    Dock dock;
    Host host(dock);
    dock.update({.tabs = true, .panes = {MakePane("a", &content)}, .aside = aside.get(), .asideWidth = 200.0f});
    host.place(ctx, kScreen);
    REQUIRE(Sakura::Access::focus(*aside));
    aside->onBlur = [&] {
        aside.reset();
    };
    dock.update({.tabs = true, .panes = {MakePane("a", &content)}});
    host.place(ctx, kScreen);
    CHECK(aside == nullptr);
    CHECK(content.visible());
    host.place(ctx, kScreen);
    CHECK(content.visible());
}

TEST_CASE("Hiding an aside leaves it alone when it moves to another parent while its mouse is cancelled",
          "[core][sakura][dock][retention]") {
    const auto ctx = MakeContext();
    Probe content;
    Probe destination;
    CancelDeletingProbe aside;
    Dock dock;
    Host host(dock);
    dock.update({.tabs = true, .panes = {MakePane("a", &content)}, .aside = &aside, .asideWidth = 200.0f});
    host.place(ctx, kScreen);
    REQUIRE(aside.visible());
    aside.onCancel = [&] {
        destination.adopt(aside);
    };
    dock.update({.tabs = true, .panes = {MakePane("a", &content)}});
    host.place(ctx, kScreen);
    const auto adopted = Peek::Children(destination);
    REQUIRE(std::find(adopted.begin(), adopted.end(), &aside) != adopted.end());
    CHECK(aside.visible());
}

TEST_CASE("Dock columns ease a hidden pane out and back while its content keeps its width",
          "[core][sakura][dock][columns]") {
    auto ctx = MakeContext();
    Probe side;
    Probe page;
    Dock dock;
    Host host(dock);
    bool hidden = false;
    const auto place = [&](F64 time) {
        auto pane = MakePane("side", &side);
        pane.hidden = hidden;
        dock.update({
            .panes = {pane, MakePane("page", &page)},
            .tiles = DockTiles{.sizes = {300.0f, 0.0f}, .children = {{.pane = "side"}, {.pane = "page"}}},
            .columns = true,
            .flexPane = "page",
            .headers = false,
            .resizable = false,
        });
        ctx.time = time;
        host.place(ctx, kScreen);
    };

    place(0.0);
    REQUIRE(side.visible());
    const Rect open = side.laidFrame;
    const Rect wide = page.laidFrame;
    CHECK(open.width == 300.0f);

    hidden = true;
    place(1.0);
    place(1.125);
    CHECK(side.visible());
    CHECK(side.laidFrame.width == open.width);
    CHECK(side.laidFrame.x < open.x);
    CHECK(page.laidFrame.x < wide.x);
    CHECK(page.laidFrame.width > wide.width);

    place(1.5);
    CHECK_FALSE(side.visible());
    CHECK(page.laidFrame.x < open.x + open.width);
    CHECK(page.laidFrame.width == Catch::Approx(wide.right() - page.laidFrame.x));

    hidden = false;
    place(2.0);
    place(2.5);
    CHECK(side.visible());
    CHECK(side.laidFrame == open);
    CHECK(page.laidFrame == wide);
}

TEST_CASE("Dock leading reach stretches the leading slot to an edge and eases it back",
          "[core][sakura][dock]") {
    auto ctx = MakeContext();
    Probe a;
    Probe leading({40.0f, 20.0f});
    Dock dock;
    Host host(dock);
    F32 reach = 0.0f;
    const auto place = [&](F64 time) {
        dock.update({.tabs = true, .hideTabs = true, .panes = {MakePane("a", &a)}, .leading = &leading,
                     .leadingInset = 70.0f, .leadingReach = reach});
        ctx.time = time;
        host.place(ctx, kScreen);
    };

    place(0.0);
    CHECK(leading.laidFrame.width == 40.0f);

    reach = 336.0f;
    place(1.0);
    place(1.125);
    CHECK(leading.laidFrame.width > 40.0f);
    CHECK(leading.laidFrame.right() < 336.0f);
    place(1.5);
    CHECK(leading.laidFrame.right() == 336.0f);

    reach = 0.0f;
    place(2.0);
    place(2.5);
    CHECK(leading.laidFrame.width == 40.0f);
}

TEST_CASE("Dock hidden panes settle once their slide ends", "[core][sakura][dock][retention]") {
    Probe side;
    Probe page;
    Dock dock;
    Place place(dock, kScreen);
    Frames frames(place, kScreen, MakeContext());
    bool hidden = false;
    const auto update = [&] {
        auto pane = MakePane("side", &side);
        pane.hidden = hidden;
        dock.update({
            .panes = {pane, MakePane("page", &page)},
            .tiles = DockTiles{.sizes = {300.0f, 0.0f}, .children = {{.pane = "side"}, {.pane = "page"}}},
            .columns = true,
            .flexPane = "page",
            .headers = false,
            .resizable = false,
        });
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));

    hidden = true;
    for (U64 frame = 0; frame < 30; ++frame) {
        frames.ctx.time += 1.0 / 60.0;
        frames.run(update);
    }
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
    CHECK_FALSE(side.visible());
}

TEST_CASE("Dock bar presses reach the window through the unused part of the leading slot",
          "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    Probe leading({40.0f, 20.0f});
    Dock dock;
    Host host(dock);
    std::vector<U8> presses;
    dock.update({.tabs = true, .hideTabs = true, .panes = {MakePane("a", &a)}, .leading = &leading,
                 .leadingInset = 70.0f, .leadingReach = 336.0f,
                 .onBarPress = [&](U8 clicks) { presses.push_back(clicks); }});
    host.place(ctx, kScreen);
    REQUIRE(leading.laidFrame.right() == 336.0f);
    const F32 y = leading.laidFrame.y + leading.laidFrame.height * 0.5f;

    host.click(leading.laidFrame.x + 10.0f, y);
    CHECK(presses.empty());

    leading.consume = false;
    host.click(leading.laidFrame.x + 10.0f, y);
    CHECK(presses == std::vector<U8>{1});
}

TEST_CASE("Dock leading reach snaps without easing when tabs appear or disappear", "[core][sakura][dock]") {
    auto ctx = MakeContext();
    Probe a;
    Probe b;
    Probe leading({40.0f, 20.0f});
    Dock dock;
    Host host(dock);
    bool hideTabs = true;
    const auto place = [&](F64 time) {
        dock.update({.tabs = true, .hideTabs = hideTabs, .panes = {MakePane("a", &a), MakePane("b", &b)},
                     .leading = &leading, .leadingInset = 70.0f, .leadingReach = 336.0f});
        ctx.time = time;
        host.place(ctx, kScreen);
    };

    place(0.0);
    REQUIRE(leading.laidFrame.right() == 336.0f);

    hideTabs = false;
    place(1.0);
    CHECK(leading.laidFrame.width == 40.0f);

    hideTabs = true;
    place(2.0);
    CHECK(leading.laidFrame.right() == 336.0f);
}

TEST_CASE("Dock leading reach stops short of the right side controls in narrow bars", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    Probe leading({40.0f, 20.0f});
    Probe end({60.0f, 20.0f});
    Dock dock;
    Host host(dock);
    dock.update({.tabs = true, .hideTabs = true, .panes = {MakePane("a", &a)}, .leading = &leading, .end = &end,
                 .leadingInset = 70.0f, .leadingReach = 336.0f});
    host.place(ctx, {0.0f, 0.0f, 348.0f, 300.0f});

    CHECK(leading.laidFrame.width >= 40.0f);
    CHECK(leading.laidFrame.right() < end.laidFrame.x);
}

TEST_CASE("Dock hides a pane inside a stacked column and gives its height to the rest",
          "[core][sakura][dock][columns]") {
    auto ctx = MakeContext();
    Probe top;
    Probe bottom;
    Probe page;
    Dock dock;
    Host host(dock);
    bool hidden = false;
    const auto place = [&](F64 time) {
        auto pane = MakePane("bottom", &bottom);
        pane.hidden = hidden;
        dock.update({
            .panes = {MakePane("top", &top), pane, MakePane("page", &page)},
            .tiles = DockTiles{.sizes = {300.0f, 0.0f},
                               .children = {{.vertical = true,
                                             .sizes = {300.0f, 300.0f},
                                             .children = {{.pane = "top"}, {.pane = "bottom"}}},
                                            {.pane = "page"}}},
            .columns = true,
            .flexPane = "page",
            .headers = false,
            .resizable = false,
        });
        ctx.time = time;
        host.place(ctx, kScreen);
    };

    place(0.0);
    REQUIRE(bottom.visible());
    const F32 split = top.laidFrame.height;
    const F32 column = bottom.laidFrame.bottom() - top.laidFrame.y;

    hidden = true;
    place(1.0);
    place(1.5);
    CHECK_FALSE(bottom.visible());
    CHECK(top.visible());
    CHECK(top.laidFrame.height > split);
    CHECK(top.laidFrame.height == Catch::Approx(column));
    CHECK(page.visible());
}

TEST_CASE("Dock hidden columns hand their space to visible columns and keep one divider between them",
          "[core][sakura][dock][columns]") {
    auto ctx = MakeContext();
    Probe left;
    Probe middle;
    Probe page;
    CursorDock dock;
    Host host(dock);
    std::string hiddenPane;
    const auto place = [&](F64 time) {
        std::vector<Dock::Pane> panes = {MakePane("left", &left), MakePane("middle", &middle),
                                         MakePane("page", &page)};
        for (auto& pane : panes) {
            pane.hidden = pane.id == hiddenPane;
        }
        dock.update({
            .panes = std::move(panes),
            .tiles = DockTiles{.sizes = {250.0f, 250.0f, 0.0f},
                               .children = {{.pane = "left"}, {.pane = "middle"}, {.pane = "page"}}},
            .columns = true,
            .flexPane = "page",
            .headers = false,
        });
        ctx.time = time;
        host.place(ctx, kScreen);
    };

    place(0.0);
    const F32 gap = middle.laidFrame.x - left.laidFrame.right();
    REQUIRE(gap > 0.0f);

    hiddenPane = "middle";
    place(1.0);
    place(1.5);
    CHECK_FALSE(middle.visible());
    CHECK(page.laidFrame.x - left.laidFrame.right() == gap);
    CHECK(dock.cursorAt({left.laidFrame.right() + gap * 0.5f, 300.0f}) == SurfaceCursor::ResizeEW);

    hiddenPane = "page";
    place(2.0);
    place(2.5);
    CHECK_FALSE(page.visible());
    CHECK(middle.visible());
    CHECK(middle.laidFrame.right() == Catch::Approx(kScreen.right() - gap));
}

TEST_CASE("Dock stacked columns keep a divider between the rows around a hidden row",
          "[core][sakura][dock][columns]") {
    auto ctx = MakeContext();
    Probe top;
    Probe middle;
    Probe bottom;
    Probe page;
    CursorDock dock;
    Host host(dock);
    bool hidden = false;
    const auto place = [&](F64 time) {
        auto pane = MakePane("middle", &middle);
        pane.hidden = hidden;
        dock.update({
            .panes = {MakePane("top", &top), pane, MakePane("bottom", &bottom), MakePane("page", &page)},
            .tiles = DockTiles{.sizes = {300.0f, 0.0f},
                               .children = {{.vertical = true,
                                             .sizes = {200.0f, 200.0f, 200.0f},
                                             .children = {{.pane = "top"}, {.pane = "middle"}, {.pane = "bottom"}}},
                                            {.pane = "page"}}},
            .columns = true,
            .flexPane = "page",
            .headers = false,
        });
        ctx.time = time;
        host.place(ctx, kScreen);
    };

    place(0.0);
    hidden = true;
    place(1.0);
    place(1.5);
    REQUIRE_FALSE(middle.visible());
    const F32 seam = (top.laidFrame.bottom() + bottom.laidFrame.y) * 0.5f;
    CHECK(dock.cursorAt({top.laidFrame.x + top.laidFrame.width * 0.5f, seam}) == SurfaceCursor::ResizeNS);
}

TEST_CASE("Dock column resizes keep the stored width of hidden columns",
          "[core][sakura][dock][columns]") {
    auto ctx = MakeContext();
    Probe left;
    Probe middle;
    Probe right;
    Dock dock;
    Host host(dock);
    bool hidden = false;
    std::optional<DockTiles> tiles = DockTiles{.sizes = {250.0f, 250.0f, 250.0f},
                                               .children = {{.pane = "left"}, {.pane = "middle"}, {.pane = "right"}}};
    const auto place = [&](F64 time) {
        auto pane = MakePane("middle", &middle);
        pane.hidden = hidden;
        dock.update({
            .panes = {MakePane("left", &left), pane, MakePane("right", &right)},
            .tiles = tiles,
            .onTiles = [&](std::optional<DockTiles> next) { tiles = std::move(next); },
            .columns = true,
            .headers = false,
        });
        ctx.time = time;
        host.place(ctx, kScreen);
    };

    place(0.0);
    const Rect open = middle.laidFrame;
    const F32 leftWidth = left.laidFrame.width;

    hidden = true;
    place(1.0);
    place(1.5);
    REQUIRE_FALSE(middle.visible());
    const F32 seam = (left.laidFrame.right() + right.laidFrame.x) * 0.5f;
    host.send(Mouse(MouseEventType::Click, seam, 300.0f));
    host.send(Mouse(MouseEventType::Move, seam + 8.0f, 300.0f));
    host.send(Mouse(MouseEventType::Release, seam + 8.0f, 300.0f));
    place(1.6);
    CHECK(left.laidFrame.width > leftWidth);

    hidden = false;
    place(2.0);
    place(2.5);
    CHECK(middle.visible());
    CHECK(middle.laidFrame.width <= open.width);
}

TEST_CASE("Dock hidden flex columns give their space to open columns, not to minimized strips",
          "[core][sakura][dock][columns]") {
    auto ctx = MakeContext();
    Probe left;
    Probe page;
    Probe notes;
    Dock dock;
    Host host(dock);
    bool hidden = false;
    const auto place = [&](F64 time) {
        auto flex = MakePane("page", &page);
        flex.hidden = hidden;
        auto strip = MakePane("notes", &notes);
        strip.startMinimized = true;
        dock.update({
            .panes = {MakePane("left", &left), flex, strip},
            .tiles = DockTiles{.sizes = {250.0f, 0.0f, 250.0f},
                               .children = {{.pane = "left"}, {.pane = "page"}, {.pane = "notes"}}},
            .columns = true,
            .flexPane = "page",
        });
        ctx.time = time;
        host.place(ctx, kScreen);
    };

    place(0.0);
    REQUIRE_FALSE(notes.visible());
    const auto strip = Spine(dock, "notes");
    REQUIRE(strip.has_value());
    const F32 stripWidth = strip->rect.width;

    hidden = true;
    place(1.0);
    place(1.5);
    REQUIRE_FALSE(page.visible());
    const auto after = Spine(dock, "notes");
    REQUIRE(after.has_value());
    CHECK(after->rect.width == Catch::Approx(stripWidth).margin(1.0f));
    CHECK(left.laidFrame.width > 250.0f);
}

TEST_CASE("Dock column layouts leave a maximized pane once it is hidden", "[core][sakura][dock][columns]") {
    auto ctx = MakeContext();
    Probe left;
    Probe middle;
    Probe page;
    Dock dock;
    Host host(dock);
    bool hidden = false;
    const auto place = [&](F64 time) {
        auto pane = MakePane("middle", &middle);
        pane.hidden = hidden;
        dock.update({
            .panes = {MakePane("left", &left), pane, MakePane("page", &page)},
            .tiles = DockTiles{.sizes = {250.0f, 250.0f, 0.0f},
                               .children = {{.pane = "left"}, {.pane = "middle"}, {.pane = "page"}}},
            .columns = true,
            .flexPane = "page",
        });
        ctx.time = time;
        host.place(ctx, kScreen);
    };
    place(0.0);
    const Rect body = middle.laidFrame;
    const auto maximize = MaximizeOf({body.x, body.y - kHeader, body.width, body.height + kHeader});
    host.click(maximize.x, maximize.y);
    place(0.1);
    REQUIRE_FALSE(left.visible());
    REQUIRE(middle.visible());

    hidden = true;
    place(1.0);
    place(1.5);
    CHECK_FALSE(middle.visible());
    CHECK(left.visible());
    CHECK(page.visible());

    const Rect card = {left.laidFrame.x, left.laidFrame.y - kHeader, left.laidFrame.width,
                       left.laidFrame.height + kHeader};
    const auto minimize = MinimizeOf(card);
    host.click(minimize.x, minimize.y);
    place(1.6);
    CHECK_FALSE(left.visible());
}

TEST_CASE("Dock dividers next to a column that took hidden space use its visible width for drag limits",
          "[core][sakura][dock][columns]") {
    auto ctx = MakeContext();
    Probe left;
    Probe middle;
    Probe page;
    Dock dock;
    Host host(dock);
    std::optional<DockTiles> tiles = DockTiles{.sizes = {250.0f, 250.0f, 0.0f},
                                               .children = {{.pane = "left"}, {.pane = "middle"}, {.pane = "page"}}};
    bool hidden = false;
    const auto place = [&](F64 time) {
        auto pane = MakePane("middle", &middle);
        pane.hidden = hidden;
        dock.update({
            .panes = {MakePane("left", &left), pane, MakePane("page", &page)},
            .tiles = tiles,
            .onTiles = [&](std::optional<DockTiles> next) { tiles = std::move(next); },
            .columns = true,
            .flexPane = "page",
            .headers = false,
        });
        ctx.time = time;
        host.place(ctx, kScreen);
    };

    place(0.0);
    hidden = true;
    place(1.0);
    place(1.5);
    REQUIRE_FALSE(middle.visible());
    const F32 before = left.laidFrame.width;
    const F32 seam = (left.laidFrame.right() + page.laidFrame.x) * 0.5f;
    host.send(Mouse(MouseEventType::Click, seam, 300.0f));
    host.send(Mouse(MouseEventType::Move, seam + 100.0f, 300.0f));
    host.send(Mouse(MouseEventType::Release, seam + 100.0f, 300.0f));
    place(1.6);
    CHECK(left.laidFrame.width == Catch::Approx(before + 100.0f).margin(1.0f));
}

TEST_CASE("Dock equalizing columns around a hidden column keeps them equal once it returns",
          "[core][sakura][dock][columns]") {
    auto ctx = MakeContext();
    Probe left;
    Probe middle;
    Probe right;
    Dock dock;
    Host host(dock);
    std::optional<DockTiles> tiles = DockTiles{.sizes = {300.0f, 250.0f, 250.0f},
                                               .children = {{.pane = "left"}, {.pane = "middle"}, {.pane = "right"}}};
    bool hidden = false;
    const auto place = [&](F64 time) {
        auto pane = MakePane("middle", &middle);
        pane.hidden = hidden;
        dock.update({
            .panes = {MakePane("left", &left), pane, MakePane("right", &right)},
            .tiles = tiles,
            .onTiles = [&](std::optional<DockTiles> next) { tiles = std::move(next); },
            .columns = true,
            .headers = false,
        });
        ctx.time = time;
        host.place(ctx, kScreen);
    };

    place(0.0);
    REQUIRE(left.laidFrame.width > right.laidFrame.width + 20.0f);
    hidden = true;
    place(1.0);
    place(1.5);
    REQUIRE_FALSE(middle.visible());
    const F32 seam = (left.laidFrame.right() + right.laidFrame.x) * 0.5f;
    host.send(Mouse(MouseEventType::Click, seam, 300.0f, MouseButton::Left, 2));
    host.send(Mouse(MouseEventType::Release, seam, 300.0f));
    place(1.6);

    hidden = false;
    place(2.0);
    place(2.5);
    REQUIRE(middle.visible());
    CHECK(left.laidFrame.width == Catch::Approx(right.laidFrame.width).margin(1.0f));
}

TEST_CASE("Dock columns animate out from their fitted width in narrow docks", "[core][sakura][dock][columns]") {
    auto ctx = MakeContext();
    Probe side;
    Probe page;
    Dock dock;
    Host host(dock);
    bool hidden = false;
    const Rect narrow = {0.0f, 0.0f, 600.0f, 400.0f};
    const auto place = [&](F64 time) {
        auto pane = MakePane("side", &side);
        pane.hidden = hidden;
        dock.update({
            .panes = {pane, MakePane("page", &page)},
            .tiles = DockTiles{.sizes = {330.0f, 0.0f}, .children = {{.pane = "side"}, {.pane = "page"}}},
            .columns = true,
            .flexPane = "page",
            .headers = false,
            .resizable = false,
        });
        ctx.time = time;
        host.place(ctx, narrow);
    };

    place(0.0);
    const F32 fitted = side.laidFrame.width;
    const F32 pageWidth = page.laidFrame.width;
    REQUIRE(fitted < 330.0f);

    hidden = true;
    place(1.0);
    place(1.02);
    CHECK(side.laidFrame.width == Catch::Approx(fitted).margin(1.0f));
    CHECK(page.laidFrame.width >= pageWidth);
}

TEST_CASE("Dock keeps one visible pane open when the others are minimized or hidden",
          "[core][sakura][dock][columns]") {
    auto ctx = MakeContext();
    Probe left;
    Probe middle;
    Probe page;
    Dock dock;
    Host host(dock);
    const auto place = [&](F64 time) {
        auto pane = MakePane("middle", &middle);
        pane.hidden = true;
        dock.update({
            .panes = {MakePane("left", &left), pane, MakePane("page", &page)},
            .tiles = DockTiles{.sizes = {250.0f, 250.0f, 0.0f},
                               .children = {{.pane = "left"}, {.pane = "middle"}, {.pane = "page"}}},
            .columns = true,
            .flexPane = "page",
        });
        ctx.time = time;
        host.place(ctx, kScreen);
    };
    place(0.0);
    REQUIRE_FALSE(middle.visible());

    const auto card = [](const Rect& body) {
        return Rect{body.x, body.y - kHeader, body.width, body.height + kHeader};
    };
    const auto minimizeLeft = MinimizeOf(card(left.laidFrame));
    host.click(minimizeLeft.x, minimizeLeft.y);
    place(0.1);
    REQUIRE_FALSE(left.visible());
    REQUIRE(page.visible());

    const auto minimizePage = MinimizeOf(card(page.laidFrame));
    host.click(minimizePage.x, minimizePage.y);
    place(0.2);
    CHECK(page.visible());
}

TEST_CASE("Dock collapses a column to a strip when its other panes are hidden", "[core][sakura][dock][columns]") {
    auto ctx = MakeContext();
    Probe notes;
    Probe extra;
    Probe page;
    Dock dock;
    Host host(dock);
    const auto place = [&](F64 time) {
        auto strip = MakePane("notes", &notes);
        strip.startMinimized = true;
        auto gone = MakePane("extra", &extra);
        gone.hidden = true;
        dock.update({
            .panes = {strip, gone, MakePane("page", &page)},
            .tiles = DockTiles{.sizes = {250.0f, 0.0f},
                               .children = {{.vertical = true,
                                             .sizes = {300.0f, 300.0f},
                                             .children = {{.pane = "notes"}, {.pane = "extra"}}},
                                            {.pane = "page"}}},
            .columns = true,
            .flexPane = "page",
        });
        ctx.time = time;
        host.place(ctx, kScreen);
    };
    place(0.0);

    CHECK_FALSE(notes.visible());
    CHECK_FALSE(extra.visible());
    const auto spine = Spine(dock, "notes");
    REQUIRE(spine.has_value());
    CHECK(spine->rect.width < 40.0f);
    CHECK(page.laidFrame.x < 60.0f);
}

TEST_CASE("Dock dividers resize only their neighbors while the flex column is hidden",
          "[core][sakura][dock][columns]") {
    auto ctx = MakeContext();
    Probe a;
    Probe b;
    Probe c;
    Probe page;
    Dock dock;
    Host host(dock);
    std::optional<DockTiles> tiles = DockTiles{
        .sizes = {250.0f, 250.0f, 250.0f, 0.0f},
        .children = {{.pane = "a"}, {.pane = "b"}, {.pane = "c"}, {.pane = "page"}},
    };
    const auto place = [&](F64 time) {
        auto flex = MakePane("page", &page);
        flex.hidden = true;
        dock.update({
            .panes = {MakePane("a", &a), MakePane("b", &b), MakePane("c", &c), flex},
            .tiles = tiles,
            .onTiles = [&](std::optional<DockTiles> next) { tiles = std::move(next); },
            .columns = true,
            .flexPane = "page",
            .headers = false,
        });
        ctx.time = time;
        host.place(ctx, kScreen);
    };
    place(0.0);
    REQUIRE_FALSE(page.visible());
    const F32 cWidth = c.laidFrame.width;
    const F32 aWidth = a.laidFrame.width;

    const F32 seam = (a.laidFrame.right() + b.laidFrame.x) * 0.5f;
    host.send(Mouse(MouseEventType::Click, seam, 300.0f));
    host.send(Mouse(MouseEventType::Move, seam + 10.0f, 300.0f));
    host.send(Mouse(MouseEventType::Release, seam + 10.0f, 300.0f));
    place(0.1);
    CHECK(a.laidFrame.width > aWidth);
    CHECK(c.laidFrame.width == Catch::Approx(cWidth).margin(1.0f));
}

namespace {

struct ScaleRig {
    F32 scale = 1.0f;
    Extent2D<F32> size = {1600.0f, 600.0f};
    std::unordered_map<std::string, std::unique_ptr<Probe>> probes;
    Emissions emissions;
    Dock::Config config;
    Dock dock;
    Layer layer;
    std::unique_ptr<LayerHost> host;
    Sakura::Context ctx;
    Rect bounds;

    ScaleRig(F32 scale, std::optional<DockTiles> tiles, const std::vector<std::string>& names,
             const std::string& flex = "a", Extent2D<F32> size = {1600.0f, 600.0f}, bool columns = true)
        : scale(scale), size(size) {
        for (const auto& name : names) {
            probes[name] = std::make_unique<Probe>();
            probes[name]->consume = false;
            config.panes.push_back(MakePane(name, probes[name].get()));
        }
        config.accepts = "panel";
        config.tiles = std::move(tiles);
        config.onTiles = emissions.sink();
        config.columns = columns;
        config.flexPane = flex;
        config.persistView = true;
        dock.update(config);
        layer.update({.content = &dock});
        host = std::make_unique<LayerHost>(std::vector<Layer*>{&layer});
        place(scale, size);
    }

    void place(F32 next, Extent2D<F32> logical) {
        scale = next;
        size = logical;
        ctx = MakeContext({size.x * scale, size.y * scale});
        ctx.pixelRatio = scale;
        bounds = {0.0f, 0.0f, size.x * scale, size.y * scale};
        Sakura::Access::invalidateLayout(dock);
        host->place(ctx, bounds);
    }

    void adopt() {
        config.tiles = emissions.last();
        dock.update(config);
        Sakura::Access::invalidateLayout(dock);
        host->place(ctx, bounds);
    }

    void add(const std::string& name, std::optional<Dock::Placement> placement = std::nullopt) {
        probes[name] = std::make_unique<Probe>();
        probes[name]->consume = false;
        auto pane = MakePane(name, probes[name].get());
        pane.placement = std::move(placement);
        config.panes.push_back(std::move(pane));
        dock.update(config);
        Sakura::Access::invalidateLayout(dock);
        host->place(ctx, bounds);
    }

    Rect tile(const std::string& name) const {
        const Rect body = probes.at(name)->laidFrame;
        return {body.x, body.y - kHeader * scale, body.width, body.height + kHeader * scale};
    }

    Rect logical(const std::string& name) const {
        const Rect card = tile(name);
        return {card.x / scale, card.y / scale, card.width / scale, card.height / scale};
    }

    Extent2D<F32> at(F32 x, F32 y) const {
        return {x * scale, y * scale};
    }

    std::optional<Rect> hover(const std::string& moved, Extent2D<F32> logicalPoint) {
        const Extent2D<F32> point = at(logicalPoint.x, logicalPoint.y);
        host->drag.active = true;
        host->drag.pane = moved;
        host->drag.group = "panel";
        host->send(Mouse(MouseEventType::Move, point.x, point.y));
        auto hovered = ctx;
        hovered.mouse = point;
        Sakura::Access::invalidateLayout(dock);
        host->place(hovered, bounds);
        if (!host->drag.target.has_value()) {
            host->drag.reset();
            return std::nullopt;
        }
        const Rect preview = host->drag.target->preview;
        return Rect{preview.x / scale, preview.y / scale, preview.width / scale, preview.height / scale};
    }

    void drop(const std::string& moved) {
        const auto accept = host->drag.target->accept;
        host->drag.reset();
        accept(moved);
        config.tiles = emissions.last();
        if (!probes.contains(moved)) {
            add(moved);
        }
        adopt();
    }

    void divide(Extent2D<F32> from, Extent2D<F32> to) {
        host->send(Mouse(MouseEventType::Click, from.x * scale, from.y * scale));
        host->send(Mouse(MouseEventType::Move, to.x * scale, to.y * scale));
        host->send(Mouse(MouseEventType::Release, to.x * scale, to.y * scale));
        adopt();
    }
};

DockTiles ScaleLayout() {
    auto strip = Leaf("m");
    strip.minimized = true;
    return ColumnRow({Leaf("a"), Column({200.0f, 382.0f}, {Leaf("b"), Leaf("c")}), strip, Leaf("d")},
                     {0.0f, 420.0f, 300.0f, 360.0f});
}

}  // namespace

TEST_CASE("Dragging a tile out of a stack beside another stack splits the full height tile it leaves and only that tile pays",
          "[core][sakura][dock][columns][drop-rules]") {
    const F32 scale = GENERATE(1.0f, 1.25f, 2.0f);
    CAPTURE(scale);
    ScaleRig rig(scale,
                 ColumnRow({Column({300.0f, 282.0f}, {Leaf("a"), Leaf("b")}), Column({300.0f, 282.0f}, {Leaf("c"), Leaf("d")}),
                            Leaf("e")},
                           {500.0f, 500.0f, 0.0f}),
                 {"a", "b", "c", "d", "e"}, "e");
    const Rect b = rig.logical("b");
    const Rect c = rig.logical("c");
    const Rect d = rig.logical("d");
    const Rect e = rig.logical("e");
    const F32 center = (b.right() + c.x) * 0.5f;
    const auto preview = rig.hover("a", {center - 1.0f, 300.0f});
    REQUIRE(preview.has_value());
    const F32 half = (500.0f - 6.0f) * 0.5f;
    CHECK(preview->width == Catch::Approx(half).margin(1.0f));
    CHECK(preview->right() == Catch::Approx(b.right()).margin(1.0f));
    CHECK(preview->height == Catch::Approx(b.height).margin(1.0f));
    rig.drop("a");

    CHECK(Near(rig.logical("c"), c, 1.0f));
    CHECK(Near(rig.logical("d"), d, 1.0f));
    CHECK(Near(rig.logical("e"), e, 1.0f));
    CHECK(rig.logical("b").x == Catch::Approx(b.x).margin(1.0f));
    CHECK(rig.logical("b").width == Catch::Approx(half).margin(1.0f));
    CHECK(rig.logical("b").height == Catch::Approx(588.0f).margin(1.0f));
    CHECK(rig.logical("a").x == Catch::Approx(b.x + half + 6.0f).margin(1.0f));
    CHECK(rig.logical("a").width == Catch::Approx(half).margin(1.0f));
    CHECK(rig.logical("a").height == Catch::Approx(588.0f).margin(1.0f));
    REQUIRE(rig.emissions.last().has_value());
    CHECK(Leaves(*rig.emissions.last()) == std::vector<std::string>{"b", "a", "c", "d", "e"});
}

TEST_CASE("Window resizes give the slack to the flex column, keep the other widths and shrink together once its base no longer fits",
          "[core][sakura][dock][columns][policy]") {
    const F32 scale = GENERATE(1.0f, 1.5f);
    CAPTURE(scale);
    ScaleRig rig(scale, ColumnRow({Leaf("a"), Leaf("b"), Leaf("c")}, {0.0f, 300.0f, 400.0f}), {"a", "b", "c"});
    for (const F32 width : {1600.0f, 1300.0f, 2400.0f, 1600.0f}) {
        CAPTURE(width);
        rig.place(scale, {width, 600.0f});
        CHECK(rig.logical("b").width == Catch::Approx(300.0f).margin(1.0f));
        CHECK(rig.logical("c").width == Catch::Approx(400.0f).margin(1.0f));
        CHECK(rig.logical("a").width == Catch::Approx(width - 12.0f - 12.0f - 700.0f).margin(1.0f));
    }
    rig.place(scale, {1000.0f, 600.0f});
    const F32 room = 1000.0f - 12.0f - 12.0f;
    CHECK(rig.logical("a").width == Catch::Approx(room * 480.0f / 1180.0f).margin(1.0f));
    CHECK(rig.logical("b").width == Catch::Approx(room * 300.0f / 1180.0f).margin(1.0f));
    CHECK(rig.logical("c").width == Catch::Approx(room * 400.0f / 1180.0f).margin(1.0f));
    rig.place(scale, {1600.0f, 600.0f});
    CHECK(rig.logical("b").width == Catch::Approx(300.0f).margin(1.0f));
    CHECK(rig.emissions.trees.empty());
}

TEST_CASE("Window resizes scale proportional splits together in column and tile docks",
          "[core][sakura][dock][policy]") {
    const F32 scale = GENERATE(1.0f, 1.25f);
    CAPTURE(scale);
    ScaleRig column(scale, ColumnRow({Leaf("a"), Column({100.0f, 300.0f}, {Leaf("b"), Leaf("c")})}, {0.0f, 420.0f}),
                    {"a", "b", "c"});
    ScaleRig tiles(scale, Row({200.0f, 600.0f}, {Leaf("a"), Leaf("b")}), {"a", "b"}, "", {800.0f, 600.0f}, false);
    for (const F32 extent : {600.0f, 900.0f, 450.0f}) {
        CAPTURE(extent);
        column.place(scale, {1600.0f, extent});
        const F32 room = extent - 12.0f - 6.0f;
        CHECK(column.logical("b").height == Catch::Approx(room * 0.25f).margin(1.0f));
        CHECK(column.logical("c").height == Catch::Approx(room * 0.75f).margin(1.0f));
        CHECK(column.logical("b").width == Catch::Approx(420.0f).margin(1.0f));

        tiles.place(scale, {extent + 200.0f, 600.0f});
        const F32 across = extent + 200.0f - 12.0f - 6.0f;
        CHECK(tiles.logical("a").width == Catch::Approx(across * 0.25f).margin(1.0f));
        CHECK(tiles.logical("b").width == Catch::Approx(across * 0.75f).margin(1.0f));
    }
    CHECK(column.emissions.trees.empty());
    CHECK(tiles.emissions.trees.empty());
}

TEST_CASE("Columns without a flex column grow together and shrink down to the minimum width",
          "[core][sakura][dock][columns][policy]") {
    ScaleRig rig(1.0f, ColumnRow({Leaf("a"), Leaf("b")}, {600.0f, 200.0f}), {"a", "b"}, "");
    rig.place(1.0f, {1612.0f + 6.0f, 600.0f});
    CHECK(rig.logical("a").width == Catch::Approx(1200.0f).margin(1.0f));
    CHECK(rig.logical("b").width == Catch::Approx(400.0f).margin(1.0f));
    rig.place(1.0f, {700.0f, 600.0f});
    CHECK(rig.logical("b").width == Catch::Approx(200.0f).margin(1.0f));
    CHECK(rig.logical("a").width == Catch::Approx(700.0f - 12.0f - 6.0f - 200.0f).margin(1.0f));
    CHECK(rig.emissions.trees.empty());
}

TEST_CASE("A pane placed beside another splits that pane's slot with the same sizing as a drag",
          "[core][sakura][dock][columns][placement]") {
    const F32 scale = GENERATE(1.0f, 1.5f);
    const auto side = GENERATE(Dock::Side::Right, Dock::Side::Bottom, Dock::Side::Left, Dock::Side::Top);
    CAPTURE(scale, static_cast<int>(side));
    const auto layout = ColumnRow({Leaf("a"), Leaf("t")}, {0.0f, 500.0f});
    ScaleRig placed(scale, layout, {"a", "t"});
    ScaleRig dragged(scale, layout, {"a", "t"});
    const Rect a = placed.logical("a");
    const Rect slot = placed.logical("t");

    placed.add("u", Dock::Placement{.anchor = "t", .side = side});
    REQUIRE(placed.emissions.trees.size() == 1);
    placed.adopt();
    const Extent2D<F32> edge = side == Dock::Side::Right    ? Extent2D<F32>{slot.right() - 20.0f, slot.y + slot.height * 0.5f}
                               : side == Dock::Side::Left   ? Extent2D<F32>{slot.x + 20.0f, slot.y + slot.height * 0.5f}
                               : side == Dock::Side::Top    ? Extent2D<F32>{slot.x + slot.width * 0.5f, slot.y + 20.0f}
                                                            : Extent2D<F32>{slot.x + slot.width * 0.5f, slot.bottom() - 20.0f};
    REQUIRE(dragged.hover("u", edge).has_value());
    dragged.drop("u");

    const bool vertical = side == Dock::Side::Top || side == Dock::Side::Bottom;
    const F32 half = (Along(slot, vertical) - 6.0f) * 0.5f;
    CHECK(Along(placed.logical("u"), vertical) == Catch::Approx(half).margin(1.0f));
    CHECK(Along(placed.logical("t"), vertical) == Catch::Approx(half).margin(1.0f));
    CHECK(Near(placed.logical("u"), dragged.logical("u"), 1.0f));
    CHECK(Near(placed.logical("t"), dragged.logical("t"), 1.0f));
    CHECK(Near(placed.logical("a"), a, 1.0f));
    REQUIRE(placed.emissions.last().has_value());
    REQUIRE(dragged.emissions.last().has_value());
    CHECK(Leaves(*placed.emissions.last()) == Leaves(*dragged.emissions.last()));
    placed.place(scale, {1600.0f, 600.0f});
    CHECK(placed.emissions.trees.size() == 1);
}

TEST_CASE("A placement in a tile dock splits its anchor and a missing anchor appends the pane instead",
          "[core][sakura][dock][placement]") {
    ScaleRig tiles(1.0f, Leaf("t"), {"t"}, "", {800.0f, 600.0f}, false);
    tiles.add("u", Dock::Placement{.anchor = "t", .side = Dock::Side::Right});
    REQUIRE(tiles.emissions.trees.size() == 1);
    CheckApprox(*tiles.emissions.last(), Row({391.0f, 391.0f}, {Leaf("t"), Leaf("u")}));
    tiles.adopt();
    CHECK(tiles.logical("t").width == Catch::Approx(391.0f).margin(1.0f));
    CHECK(tiles.logical("u").x == Catch::Approx(403.0f).margin(1.0f));

    ScaleRig columns(1.0f, ColumnRow({Leaf("a"), Leaf("t")}, {0.0f, 500.0f}), {"a", "t"});
    columns.add("u", Dock::Placement{.anchor = "missing", .side = Dock::Side::Bottom});
    REQUIRE(columns.emissions.trees.size() == 1);
    CHECK(columns.emissions.last() == ColumnRow({Leaf("a"), Leaf("t"), Leaf("u")}, {0.0f, 500.0f, 0.0f}));
    columns.adopt();
    CHECK(columns.logical("u").width == Catch::Approx(420.0f).margin(1.0f));
    CHECK(columns.logical("t").width == Catch::Approx(500.0f).margin(1.0f));
}

TEST_CASE("Splitting a column keeps the default width of the columns that do not pay",
          "[core][sakura][dock][columns][placement]") {
    const F32 scale = GENERATE(1.0f, 1.5f);
    CAPTURE(scale);
    ScaleRig rig(scale, ColumnRow({Leaf("a"), Leaf("b"), Leaf("c")}, {0.0f, 0.0f, 0.0f}), {"a", "b", "c"});
    const Rect c = rig.logical("c");
    rig.add("u", Dock::Placement{.anchor = "b", .side = Dock::Side::Right});
    REQUIRE(rig.emissions.last().has_value());
    const auto& sizes = rig.emissions.last()->sizes;
    REQUIRE(Leaves(*rig.emissions.last()) == std::vector<std::string>{"a", "b", "u", "c"});
    CHECK(sizes[0] == 0.0f);
    CHECK(sizes[3] == 0.0f);
    rig.adopt();
    CHECK(Near(rig.logical("c"), c, 1.0f));
}

TEST_CASE("A placement beside a minimized or too narrow anchor appends the pane as a column",
          "[core][sakura][dock][columns][placement]") {
    auto strip = Leaf("t");
    strip.minimized = true;
    ScaleRig minimized(1.0f, ColumnRow({Leaf("a"), strip}, {0.0f, 500.0f}), {"a", "t"});
    const Rect folded = minimized.logical("t");
    minimized.add("u", Dock::Placement{.anchor = "t", .side = Dock::Side::Bottom});
    REQUIRE(minimized.emissions.trees.size() == 1);
    CHECK(Leaves(*minimized.emissions.last()) == std::vector<std::string>{"a", "t", "u"});
    minimized.adopt();
    CHECK(Near(minimized.logical("t"), folded, 1.0f));
    CHECK(minimized.logical("u").width == Catch::Approx(420.0f).margin(1.0f));
    CHECK(minimized.logical("u").height == Catch::Approx(minimized.logical("a").height).margin(1.0f));

    ScaleRig narrow(1.0f, ColumnRow({Leaf("a"), Leaf("n")}, {0.0f, 100.0f}), {"a", "n"});
    const Rect slim = narrow.logical("n");
    narrow.add("u", Dock::Placement{.anchor = "n", .side = Dock::Side::Right});
    REQUIRE(narrow.emissions.trees.size() == 1);
    CHECK(Leaves(*narrow.emissions.last()) == std::vector<std::string>{"a", "n", "u"});
    narrow.adopt();
    CHECK(narrow.logical("n").width == Catch::Approx(slim.width).margin(1.0f));
    CHECK(narrow.logical("u").width == Catch::Approx(420.0f).margin(1.0f));
}

TEST_CASE("A layout gives the same logical rects, strips and gaps at every pixel ratio",
          "[core][sakura][dock][columns][scale]") {
    std::unordered_map<std::string, Rect> reference;
    for (const F32 scale : {1.0f, 1.25f, 1.5f, 2.0f, 3.0f}) {
        CAPTURE(scale);
        ScaleRig rig(scale, ScaleLayout(), {"a", "b", "c", "m", "d"});
        CHECK_FALSE(rig.probes.at("m")->visible());
        for (const auto name : {"a", "b", "c", "d"}) {
            CAPTURE(name);
            const Rect rect = rig.logical(name);
            if (scale == 1.0f) {
                reference[name] = rect;
            }
            CHECK(Near(rect, reference.at(name), 1.0f));
        }
        CHECK(rig.logical("c").y - rig.logical("b").bottom() == Catch::Approx(6.0f).margin(1.0f));
        CHECK(rig.logical("d").x - rig.logical("c").right() == Catch::Approx(6.0f + 32.0f + 6.0f).margin(1.0f));
        CHECK(rig.logical("b").x - rig.logical("a").right() == Catch::Approx(6.0f).margin(1.0f));
        CHECK(rig.emissions.trees.empty());
    }
}

TEST_CASE("Gap bands, the width floor and drop sizes are judged in logical pixels at every pixel ratio",
          "[core][sakura][dock][columns][scale][drop-rules]") {
    std::optional<Rect> band;
    std::optional<Rect> beyond;
    std::optional<Rect> fits;
    for (const F32 scale : {1.0f, 1.25f, 1.5f, 2.0f, 3.0f}) {
        CAPTURE(scale);
        ScaleRig rig(scale, ColumnRow({Leaf("a"), Leaf("n"), Leaf("w")}, {0.0f, 140.0f, 130.0f}), {"a", "n", "w"});
        const Rect n = rig.logical("n");
        const Rect w = rig.logical("w");

        const auto inside = rig.hover("x", {n.x + 11.0f, n.y + 4.0f});
        REQUIRE(inside.has_value());
        CHECK(inside->height == Catch::Approx(n.height).margin(1.0f));
        CHECK(inside->width == Catch::Approx(67.0f).margin(1.0f));
        const auto outside = rig.hover("x", {n.x + 13.5f, n.y + 4.0f});
        REQUIRE(outside.has_value());
        CHECK(outside->width == Catch::Approx(n.width).margin(1.0f));
        CHECK(outside->height < n.height * 0.6f);

        const auto right = rig.hover("x", {n.right() - 30.0f, n.y + n.height * 0.5f});
        REQUIRE(right.has_value());
        CHECK(right->width == Catch::Approx(67.0f).margin(1.0f));
        CHECK_FALSE(rig.hover("x", {w.x + 30.0f, w.y + w.height * 0.5f}).has_value());

        if (scale == 1.0f) {
            band = inside;
            beyond = outside;
            fits = right;
        }
        CHECK(Near(*inside, *band, 1.0f));
        CHECK(Near(*outside, *beyond, 1.0f));
        CHECK(Near(*right, *fits, 1.0f));
    }
}

TEST_CASE("A layout saved at one pixel ratio and window size keeps its fixed widths and proportions at another",
          "[core][sakura][dock][columns][scale]") {
    ScaleRig saved(1.5f, ScaleLayout(), {"a", "b", "c", "m", "d"});
    const Rect d = saved.logical("d");
    saved.divide({d.x - 3.0f, 300.0f}, {d.x - 43.0f, 300.0f});
    const Rect b = saved.logical("b");
    saved.divide({b.x + 100.0f, b.bottom() + 3.0f}, {b.x + 100.0f, b.bottom() + 63.0f});
    REQUIRE(saved.emissions.trees.size() == 2);
    const auto tiles = saved.emissions.last();
    const F32 dWidth = saved.logical("d").width;
    const F32 bWidth = saved.logical("b").width;
    const F32 share = saved.logical("b").height / (saved.logical("b").height + saved.logical("c").height);

    for (const F32 scale : {1.0f, 1.25f, 2.0f}) {
        CAPTURE(scale);
        ScaleRig loaded(scale, tiles, {"a", "b", "c", "m", "d"}, "a", {1400.0f, 900.0f});
        CHECK(loaded.logical("d").width == Catch::Approx(dWidth).margin(1.0f));
        CHECK(loaded.logical("b").width == Catch::Approx(bWidth).margin(1.0f));
        const F32 loadedShare = loaded.logical("b").height / (loaded.logical("b").height + loaded.logical("c").height);
        CHECK(loadedShare == Catch::Approx(share).margin(0.005f));
        CHECK_FALSE(loaded.probes.at("m")->visible());
        CHECK(loaded.emissions.trees.empty());
    }
}

TEST_CASE("A divider drag in a window too narrow for every column moves only the two columns beside it",
          "[core][sakura][dock][columns][policy]") {
    const F32 scale = GENERATE(1.0f, 1.25f);
    const F32 step = GENERATE(1.0f, 5.0f, -20.0f);
    CAPTURE(scale, step);
    ScaleRig rig(scale, ColumnRow({Leaf("a"), Leaf("b"), Leaf("c")}, {0.0f, 0.0f, 0.0f}), {"a", "b", "c"}, "a",
                 {800.0f, 600.0f});
    const Rect a = rig.logical("a");
    const Rect b = rig.logical("b");
    const Rect c = rig.logical("c");
    REQUIRE(a.width < 480.0f);
    const Extent2D<F32> seam{a.right() + 3.0f, 300.0f};
    rig.divide(seam, {seam.x + step, 300.0f});
    CHECK(rig.logical("a").width == Catch::Approx(a.width + step).margin(1.0f));
    CHECK(rig.logical("b").width == Catch::Approx(b.width - step).margin(1.0f));
    CHECK(Near(rig.logical("c"), c, 1.0f));
    rig.divide({seam.x + step, 300.0f}, seam);
    CHECK(Near(rig.logical("a"), a, 1.0f));
    CHECK(Near(rig.logical("b"), b, 1.0f));
    CHECK(Near(rig.logical("c"), c, 1.0f));
}

TEST_CASE("Repeated divider moves, stack moves, minimize and restore and window resizes at a fractional ratio do not drift",
          "[core][sakura][dock][columns][scale]") {
    ScaleRig rig(1.25f, ScaleLayout(), {"a", "b", "c", "m", "d"});
    const Rect d = rig.logical("d");
    rig.divide({d.x - 3.0f, 300.0f}, {d.x - 40.0f, 300.0f});
    rig.divide({d.x - 40.0f, 300.0f}, {d.x - 3.0f, 300.0f});
    REQUIRE(rig.emissions.last().has_value());
    const DockTiles baseline = *rig.emissions.last();
    const auto same = [&](const DockTiles& tiles) {
        CheckApprox(tiles, baseline);
    };

    for (int round = 0; round < 12; ++round) {
        CAPTURE(round);
        const Rect seam = rig.logical("d");
        rig.divide({seam.x - 3.0f, 300.0f}, {seam.x - 37.3f, 300.0f});
        rig.divide({seam.x - 37.3f, 300.0f}, {seam.x - 3.0f, 300.0f});
        same(*rig.emissions.last());

        const Rect c = rig.logical("c");
        REQUIRE(rig.hover("b", {c.x + c.width * 0.5f, c.y + 15.0f}).has_value());
        rig.drop("b");
        same(*rig.emissions.last());

        const Rect card = rig.logical("c");
        const auto minimize = rig.at(card.right() - 59.0f, card.y + 15.0f);
        rig.host->click(minimize.x, minimize.y);
        rig.adopt();
        REQUIRE_FALSE(rig.probes.at("c")->visible());
        const Rect upper = rig.logical("b");
        const auto restore = rig.at(upper.x + upper.width * 0.5f, upper.bottom() + 6.0f + 16.0f);
        rig.host->click(restore.x, restore.y);
        rig.adopt();
        REQUIRE(rig.probes.at("c")->visible());
        same(*rig.emissions.last());
        rig.place(1.25f, {1100.0f + 40.0f * static_cast<F32>(round), 700.0f});
        rig.place(1.25f, {1600.0f, 600.0f});
    }
    CHECK(rig.logical("d").width == Catch::Approx(d.width).margin(1.0f));
}

TEST_CASE("Changing the pixel ratio at runtime keeps stored sizes and logical rects",
          "[core][sakura][dock][columns][scale]") {
    ScaleRig rig(1.0f, ScaleLayout(), {"a", "b", "c", "m", "d"});
    std::unordered_map<std::string, Rect> before;
    for (const auto name : {"a", "b", "c", "d"}) {
        before[name] = rig.logical(name);
    }
    for (const F32 scale : {2.0f, 1.5f, 1.25f, 3.0f, 1.0f, 1.5f}) {
        CAPTURE(scale);
        rig.place(scale, {1600.0f, 600.0f});
        for (const auto name : {"a", "b", "c", "d"}) {
            CAPTURE(name);
            CHECK(Near(rig.logical(name), before.at(name), 1.0f));
        }
    }
    CHECK(rig.emissions.trees.empty());

    const Rect d = rig.logical("d");
    rig.divide({d.x - 3.0f, 300.0f}, {d.x - 3.0f - 40.0f / 1.5f, 300.0f});
    REQUIRE(rig.emissions.trees.size() == 1);
    REQUIRE(rig.emissions.last().has_value());
    const auto& sizes = rig.emissions.last()->sizes;
    REQUIRE(sizes.size() == 4);
    CHECK(sizes[1] == Catch::Approx(420.0f - 40.0f / 1.5f).margin(0.01f));
    CHECK(sizes[2] == 300.0f);
    CHECK(sizes[3] == Catch::Approx(360.0f + 40.0f / 1.5f).margin(0.01f));
    rig.place(2.0f, {1600.0f, 600.0f});
    CHECK(rig.logical("d").width == Catch::Approx(360.0f + 40.0f / 1.5f).margin(1.0f));
    CHECK(rig.emissions.trees.size() == 1);
}

namespace {

const std::vector<std::string> kGutterPanes = {"a", "b", "c", "d"};

ColorRGBA<F32> GutterFill(U64 index) {
    return {0.1f + 0.2f * static_cast<F32>(index), 0.3f, 0.7f, 1.0f};
}

struct GutterRig {
    Dock dock;
    Host host{dock};
    Sakura::Context ctx;
    Rect bounds;

    GutterRig(F32 scale, Rect bounds) : ctx(MakeContext({bounds.width, bounds.height})), bounds(bounds) {
        ctx.pixelRatio = scale;
    }

    void place(const DockTiles& tiles, const std::vector<std::string>& hidden = {}, F64 time = 0.0) {
        std::vector<Dock::Pane> panes;
        for (U64 index = 0; index < kGutterPanes.size(); ++index) {
            if (!Contains(tiles, kGutterPanes[index])) {
                continue;
            }
            auto pane = MakePane(kGutterPanes[index]);
            pane.fill = GutterFill(index);
            pane.hidden = std::find(hidden.begin(), hidden.end(), kGutterPanes[index]) != hidden.end();
            panes.push_back(pane);
        }
        dock.update({.panes = panes, .tiles = tiles, .headers = false});
        ctx.time = time;
        host.place(ctx, bounds);
    }

    std::optional<Rect> tile(const std::string& pane) {
        const auto index = std::find(kGutterPanes.begin(), kGutterPanes.end(), pane) - kGutterPanes.begin();
        for (const auto* box : Find<Sakura::Box>(dock)) {
            for (const auto& instance : box->config().instances) {
                if (instance.visible && instance.backgroundColor == GutterFill(index)) {
                    return instance.rect;
                }
            }
        }
        return std::nullopt;
    }

    static bool Contains(const DockTiles& tiles, const std::string& pane) {
        if (tiles.children.empty()) {
            return tiles.pane == pane;
        }
        return std::any_of(tiles.children.begin(), tiles.children.end(), [&](const auto& child) {
            return Contains(child, pane);
        });
    }
};

F32 StartAlong(const Rect& rect, bool vertical) {
    return vertical ? rect.y : rect.x;
}

F32 EndAlong(const Rect& rect, bool vertical) {
    return vertical ? rect.bottom() : rect.right();
}

}  // namespace

TEST_CASE("Dock gutters stay uniform and tiles reach the same edges at fractional pixel ratios",
          "[core][sakura][dock]") {
    const F32 scale = GENERATE(0.9f, 1.25f, 1.5f, 1.75f, 2.0f);
    const bool vertical = GENERATE(false, true);
    CAPTURE(scale, vertical);
    const Rect bounds = {0.0f, 0.0f, 802.0f, 802.0f};

    GutterRig single(scale, bounds);
    single.place(Leaf("a"));
    const auto whole = single.tile("a");
    REQUIRE(whole.has_value());

    GutterRig rig(scale, bounds);
    std::vector<DockTiles> leaves;
    for (const auto& pane : kGutterPanes) {
        leaves.push_back(Leaf(pane));
    }
    const auto tiles = vertical ? Column({1.0f, 1.0f, 1.0f, 1.0f}, leaves) : Row({1.0f, 1.0f, 1.0f, 1.0f}, leaves);
    rig.place(tiles);

    std::vector<Rect> painted;
    for (const auto& pane : kGutterPanes) {
        const auto rect = rig.tile(pane);
        REQUIRE(rect.has_value());
        painted.push_back(*rect);
    }
    CAPTURE(painted);
    const F32 gap = StartAlong(painted[1], vertical) - EndAlong(painted[0], vertical);
    CHECK(gap >= 1.0f);
    F32 smallest = Along(painted[0], vertical);
    F32 largest = smallest;
    for (U64 index = 1; index < painted.size(); ++index) {
        CHECK(StartAlong(painted[index], vertical) - EndAlong(painted[index - 1], vertical) == gap);
        CHECK(StartAlong(painted[index], !vertical) == StartAlong(*whole, !vertical));
        CHECK(EndAlong(painted[index], !vertical) == EndAlong(*whole, !vertical));
        smallest = std::min(smallest, Along(painted[index], vertical));
        largest = std::max(largest, Along(painted[index], vertical));
    }
    CHECK(largest - smallest <= 1.0f);
    CHECK(StartAlong(painted.front(), vertical) == StartAlong(*whole, vertical));
    CHECK(EndAlong(painted.back(), vertical) == EndAlong(*whole, vertical));
}

TEST_CASE("Nested dock splits share one gutter across both axes at fractional pixel ratios", "[core][sakura][dock]") {
    const F32 scale = GENERATE(0.9f, 1.25f, 1.5f, 1.75f);
    CAPTURE(scale);
    const Rect bounds = {0.0f, 0.0f, 802.0f, 802.0f};
    GutterRig single(scale, bounds);
    single.place(Leaf("a"));
    const auto whole = single.tile("a");
    REQUIRE(whole.has_value());

    GutterRig rig(scale, bounds);
    rig.place(Row({1.0f, 2.0f}, {Leaf("a"), Column({1.0f, 1.0f, 1.0f}, {Leaf("b"), Leaf("c"), Leaf("d")})}));
    const auto a = rig.tile("a");
    const auto b = rig.tile("b");
    const auto c = rig.tile("c");
    const auto d = rig.tile("d");
    REQUIRE((a && b && c && d));
    const F32 gap = b->x - a->right();
    CHECK(gap >= 1.0f);
    CHECK(c->y - b->bottom() == gap);
    CHECK(d->y - c->bottom() == gap);
    for (const auto& tile : {*b, *c, *d}) {
        CHECK(tile.x == b->x);
        CHECK(tile.right() == whole->right());
    }
    CHECK(a->x == whole->x);
    CHECK(a->y == whole->y);
    CHECK(b->y == whole->y);
    CHECK(a->bottom() == whole->bottom());
    CHECK(d->bottom() == whole->bottom());
}

TEST_CASE("Splits nested inside an inner tile end on that tile's snapped edge at fractional pixel ratios",
          "[core][sakura][dock]") {
    const F32 scale = GENERATE(0.9f, 1.25f, 1.5f, 1.75f);
    const F32 extent = GENERATE(802.0f, 803.0f, 805.0f);
    CAPTURE(scale, extent);
    GutterRig rig(scale, {0.0f, 0.0f, extent, extent});
    rig.place(Row({1.0f, 1.0f}, {Column({1.0f, 1.0f}, {Row({1.0f, 1.0f}, {Leaf("a"), Leaf("b")}), Leaf("c")}), Leaf("d")}));
    const auto a = rig.tile("a");
    const auto b = rig.tile("b");
    const auto c = rig.tile("c");
    const auto d = rig.tile("d");
    REQUIRE((a && b && c && d));
    const F32 gap = b->x - a->right();
    CHECK(gap >= 1.0f);
    CHECK(c->y - a->bottom() == gap);
    CHECK(c->y - b->bottom() == gap);
    CHECK(d->x - b->right() == gap);
    CHECK(d->x - c->right() == gap);
    CHECK(b->right() == c->right());
    CHECK(a->x == c->x);
}

TEST_CASE("Hiding a dock tile at a fractional pixel ratio never overlaps neighbours and settles on uniform gutters",
          "[core][sakura][dock]") {
    const F32 scale = GENERATE(0.9f, 1.25f, 1.75f);
    const F32 extent = GENERATE(802.0f, 803.0f, 805.0f);
    CAPTURE(scale, extent);
    GutterRig rig(scale, {0.0f, 0.0f, extent, extent});
    std::vector<DockTiles> leaves;
    for (const auto& pane : kGutterPanes) {
        leaves.push_back(Leaf(pane));
    }
    const auto tiles = Row({1.0f, 1.0f, 1.0f, 1.0f}, leaves);
    rig.place(tiles);
    std::vector<Rect> open;
    for (const auto& pane : kGutterPanes) {
        open.push_back(*rig.tile(pane));
    }

    const auto shown = [&] {
        std::vector<Rect> rects;
        for (const auto& pane : kGutterPanes) {
            if (const auto rect = rig.tile(pane); rect.has_value() && !rect->empty()) {
                rects.push_back(*rect);
            }
        }
        return rects;
    };
    rig.place(tiles, {"b"}, 1.0);
    for (const F64 time : {1.05, 1.1, 1.15, 1.2, 1.3}) {
        CAPTURE(time);
        rig.place(tiles, {"b"}, time);
        const auto rects = shown();
        REQUIRE(rects.size() >= 3);
        for (U64 index = 1; index < rects.size(); ++index) {
            CHECK(rects[index].x >= rects[index - 1].right());
        }
        CHECK(rects.front().x == open.front().x);
        CHECK(rects.back().right() == open.back().right());
    }

    rig.place(tiles, {"b"}, 3.0);
    const auto settled = shown();
    REQUIRE(settled.size() == 3);
    const F32 gap = open[1].x - open[0].right();
    CHECK(settled[1].x - settled[0].right() == gap);
    CHECK(settled[2].x - settled[1].right() == gap);
    CHECK(settled.front().x == open.front().x);
    CHECK(settled.back().right() == open.back().right());

    rig.place(tiles, {}, 4.0);
    rig.place(tiles, {}, 6.0);
    for (U64 index = 0; index < kGutterPanes.size(); ++index) {
        CHECK(*rig.tile(kGutterPanes[index]) == open[index]);
    }
}
