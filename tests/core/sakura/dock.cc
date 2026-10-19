#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <jetstream/render/base/texture.hh>
#include <jetstream/render/sakura/components/dock.hh>
#include <jetstream/render/sakura/components/image.hh>
#include <jetstream/render/sakura/components/polyline.hh>

#include <jetstream/render/tools/imgui_icons_ext.hh>

#include "harness.hh"
#include "fonts.hh"
#include "component_harness.hh"
#include "scene.hh"

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

using namespace Jetstream;
using namespace SakuraTest;

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

void CheckApprox(const DockTiles& actual, const DockTiles& expected) {
    CHECK(actual.pane == expected.pane);
    CHECK(actual.vertical == expected.vertical);
    REQUIRE(actual.weights.size() == expected.weights.size());
    for (U64 index = 0; index < expected.weights.size(); ++index) {
        CHECK(actual.weights[index] == Catch::Approx(expected.weights[index]));
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
    return str == ICON_FA_XMARK || str == ICON_FA_UP_RIGHT_AND_DOWN_LEFT_FROM_CENTER ||
           str == ICON_FA_DOWN_LEFT_AND_UP_RIGHT_TO_CENTER;
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
    for (auto* box : Direct<Sakura::Box>(dock)) {
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

TEST_CASE("Tile mode appends panes into a horizontal row and reports each tree", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Emissions emissions;
    Dock dock;
    Host host(dock);

    dock.update({.panes = {MakePane("a")}, .onTiles = emissions.sink()});
    REQUIRE(emissions.trees.size() == 1);
    CHECK(emissions.last() == Leaf("a"));

    dock.update({.panes = {MakePane("a"), MakePane("b")}, .tiles = emissions.last(), .onTiles = emissions.sink()});
    REQUIRE(emissions.trees.size() == 2);
    CHECK(emissions.last() == Row({0.5f, 0.5f}, {Leaf("a"), Leaf("b")}));

    dock.update({.panes = {MakePane("a"), MakePane("b"), MakePane("c")},
                 .tiles = emissions.last(),
                 .onTiles = emissions.sink()});
    REQUIRE(emissions.trees.size() == 3);
    REQUIRE(emissions.last().has_value());
    const F32 third = 1.0f / 3.0f;
    CheckApprox(*emissions.last(), Row({third, third, third}, {Leaf("a"), Leaf("b"), Leaf("c")}));
}

TEST_CASE("Tile mode appends to a vertical tree by wrapping it in a new row", "[core][sakura][dock]") {
    Emissions emissions;
    Dock dock;
    const auto column = Column({0.5f, 0.5f}, {Leaf("a"), Leaf("b")});
    dock.update({.panes = {MakePane("a"), MakePane("b"), MakePane("c")}, .tiles = column, .onTiles = emissions.sink()});

    REQUIRE(emissions.trees.size() == 1);
    CHECK(emissions.last() == Row({0.5f, 0.5f}, {column, Leaf("c")}));
}

TEST_CASE("Tile mode adopts owner tiles without emitting them back", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    Probe b;
    Emissions emissions;
    Dock dock;
    Host host(dock);
    const auto tiles = Row({0.3f, 0.7f}, {Leaf("a"), Leaf("b")});
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
    dock.update({.panes = panes, .tiles = Row({0.5f, 0.5f}, {Leaf("a"), Leaf("b")}), .onTiles = emissions.sink()});
    host.place(ctx, kScreen);
    CHECK(a.laidFrame.x == 6.0f);

    dock.update({.panes = panes, .tiles = Column({0.5f, 0.5f}, {Leaf("b"), Leaf("a")}), .onTiles = emissions.sink()});
    host.place(ctx, kScreen);
    CHECK(emissions.trees.empty());
    CHECK(b.laidFrame == Body({6.0f, 6.0f, 788.0f, 291.0f}));
    CHECK(a.laidFrame == Body({6.0f, 303.0f, 788.0f, 291.0f}));
}

TEST_CASE("Tile mode flattens nested rows along the same axis", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    Probe b;
    Probe c;
    Emissions emissions;
    Dock dock;
    Host host(dock);
    const auto nested = Row({0.5f, 0.5f}, {Leaf("a"), Row({0.5f, 0.5f}, {Leaf("b"), Leaf("c")})});
    dock.update({.panes = {MakePane("a", &a), MakePane("b", &b), MakePane("c", &c)},
                 .tiles = nested,
                 .onTiles = emissions.sink()});
    host.place(ctx, kScreen);

    CHECK(emissions.trees.empty());
    CHECK(a.laidFrame == Body({6.0f, 6.0f, 388.0f, 588.0f}));
    CHECK(b.laidFrame == Body({400.0f, 6.0f, 194.0f, 588.0f}));
    CHECK(c.laidFrame == Body({600.0f, 6.0f, 194.0f, 588.0f}));

    host.click(397.0f, 300.0f);
    REQUIRE(emissions.trees.size() == 1);
    CHECK(emissions.last() == Row({0.5f, 0.25f, 0.25f}, {Leaf("a"), Leaf("b"), Leaf("c")}));
}

TEST_CASE("Tile mode keeps nested splits along the other axis", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    Probe b;
    Probe c;
    Emissions emissions;
    Dock dock;
    Host host(dock);
    const auto tiles = Row({0.5f, 0.5f}, {Leaf("a"), Column({0.5f, 0.5f}, {Leaf("b"), Leaf("c")})});
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

TEST_CASE("Tile mode normalizes owner weights that do not sum to one", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Emissions emissions;
    Dock dock;
    Host host(dock);
    dock.update({.panes = {MakePane("a"), MakePane("b")},
                 .tiles = Row({3.0f, 1.0f}, {Leaf("a"), Leaf("b")}),
                 .onTiles = emissions.sink()});
    host.place(ctx, kScreen);
    host.click(599.0f, 300.0f);

    REQUIRE(emissions.trees.size() == 1);
    CHECK(emissions.last() == Row({0.75f, 0.25f}, {Leaf("a"), Leaf("b")}));
}

TEST_CASE("Tile mode treats missing weights as equal shares", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Emissions emissions;
    Dock dock;
    Host host(dock);
    dock.update({.panes = {MakePane("a"), MakePane("b")},
                 .tiles = Row({}, {Leaf("a"), Leaf("b")}),
                 .onTiles = emissions.sink()});
    host.place(ctx, kScreen);
    host.click(400.0f, 300.0f);

    REQUIRE(emissions.trees.size() == 1);
    CHECK(emissions.last() == Row({0.5f, 0.5f}, {Leaf("a"), Leaf("b")}));
}

TEST_CASE("Unknown leaves keep their place and are described for their header", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    Probe ghost;
    Emissions emissions;
    std::vector<std::string> described;
    Dock dock;
    Host host(dock);
    const auto tiles = Row({0.5f, 0.5f}, {Leaf("a"), Leaf("ghost")});
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
                 .tiles = Row({0.5f, 0.5f}, {Leaf("a"), Leaf("b")}),
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

TEST_CASE("Closing a header renormalizes the remaining weights", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Emissions emissions;
    Dock dock;
    Host host(dock);
    dock.update({.panes = {MakePane("a"), MakePane("b"), MakePane("c")},
                 .tiles = Row({0.5f, 0.25f, 0.25f}, {Leaf("a"), Leaf("b"), Leaf("c")}),
                 .onTiles = emissions.sink()});
    host.place(ctx, kScreen);

    const auto closeA = CloseOf({6.0f, 6.0f, 388.0f, 588.0f});
    host.click(closeA.x, closeA.y);
    REQUIRE(emissions.trees.size() == 1);
    CHECK(emissions.last() == Row({0.5f, 0.5f}, {Leaf("b"), Leaf("c")}));
}

TEST_CASE("Closing a tile inside a nested split collapses its parent", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Emissions emissions;
    Dock dock;
    Host host(dock);
    dock.update({.panes = {MakePane("a"), MakePane("b"), MakePane("c")},
                 .tiles = Row({0.4f, 0.6f}, {Leaf("a"), Column({0.5f, 0.5f}, {Leaf("b"), Leaf("c")})}),
                 .onTiles = emissions.sink()});
    host.place(ctx, kScreen);

    const auto closeB = CloseOf({325.0f, 6.0f, 469.0f, 291.0f});
    host.click(closeB.x, closeB.y);
    REQUIRE(emissions.trees.size() == 1);
    REQUIRE(emissions.last().has_value());
    CheckApprox(*emissions.last(), Row({0.4f, 0.6f}, {Leaf("a"), Leaf("c")}));
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
                 .tiles = Row({0.5f, 0.5f}, {Leaf("a"), Leaf("b")}),
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
                 .tiles = Row({0.5f, 0.5f}, {Leaf("a"), Leaf("b")}),
                 .onTiles = emissions.sink()});
    host.place(ctx, kScreen);

    const auto close = CloseOf({403.0f, 6.0f, 391.0f, 588.0f});
    host.click(close.x, close.y);
    CHECK(emissions.trees.empty());
    CHECK(closed == 0);
}

TEST_CASE("Dragging a splitter changes only its neighbouring weights and emits on release",
          "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Emissions emissions;
    Dock dock;
    Host host(dock);
    dock.update({.panes = {MakePane("a"), MakePane("b"), MakePane("c")},
                 .tiles = Row({0.5f, 0.25f, 0.25f}, {Leaf("a"), Leaf("b"), Leaf("c")}),
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
    CheckApprox(*emissions.last(), Row({0.5f, 0.125f, 0.375f}, {Leaf("a"), Leaf("b"), Leaf("c")}));
}

TEST_CASE("Double clicking a splitter evens out its two neighbours and leaves the rest alone",
          "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Emissions emissions;
    Dock dock;
    Host host(dock);
    dock.update({.panes = {MakePane("a"), MakePane("b"), MakePane("c")},
                 .tiles = Row({0.5f, 0.125f, 0.375f}, {Leaf("a"), Leaf("b"), Leaf("c")}),
                 .onTiles = emissions.sink()});
    host.place(ctx, kScreen);

    CHECK(host.send(Mouse(MouseEventType::Click, 498.5f, 300.0f, MouseButton::Left, 2)));
    CHECK(host.send(Mouse(MouseEventType::Release, 498.5f, 300.0f)));
    REQUIRE(emissions.trees.size() == 1);
    REQUIRE(emissions.last().has_value());
    CheckApprox(*emissions.last(), Row({0.5f, 0.25f, 0.25f}, {Leaf("a"), Leaf("b"), Leaf("c")}));
}

TEST_CASE("Splitter drags keep a minimum share for each neighbour", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Emissions emissions;
    Dock dock;
    Host host(dock);
    dock.update({.panes = {MakePane("a"), MakePane("b")},
                 .tiles = Row({0.5f, 0.5f}, {Leaf("a"), Leaf("b")}),
                 .onTiles = emissions.sink()});
    host.place(ctx, kScreen);

    host.send(Mouse(MouseEventType::Click, 400.0f, 300.0f));
    host.send(Mouse(MouseEventType::Move, -50.0f, 300.0f));
    host.send(Mouse(MouseEventType::Release, -50.0f, 300.0f));
    REQUIRE(emissions.trees.size() == 1);
    REQUIRE(emissions.last().has_value());
    const F32 minimum = 48.0f / 788.0f;
    CheckApprox(*emissions.last(), Row({minimum, 1.0f - minimum}, {Leaf("a"), Leaf("b")}));

    dock.update({.panes = {MakePane("a"), MakePane("b")}, .tiles = emissions.last(), .onTiles = emissions.sink()});
    host.place(ctx, kScreen);
    const F32 splitter = std::round(6.0f + 782.0f * minimum) + 3.0f;
    host.send(Mouse(MouseEventType::Click, splitter, 300.0f));
    host.send(Mouse(MouseEventType::Move, 900.0f, 300.0f));
    host.send(Mouse(MouseEventType::Release, 900.0f, 300.0f));
    REQUIRE(emissions.trees.size() == 2);
    REQUIRE(emissions.last().has_value());
    CheckApprox(*emissions.last(), Row({1.0f - minimum, minimum}, {Leaf("a"), Leaf("b")}));
}

TEST_CASE("Vertical splitters resize along the y axis", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    Probe b;
    Emissions emissions;
    Dock dock;
    Host host(dock);
    dock.update({.panes = {MakePane("a", &a), MakePane("b", &b)},
                 .tiles = Column({0.5f, 0.5f}, {Leaf("a"), Leaf("b")}),
                 .onTiles = emissions.sink()});
    host.place(ctx, kScreen);

    host.send(Mouse(MouseEventType::Click, 400.0f, 300.0f));
    host.send(Mouse(MouseEventType::Move, 400.0f, 153.0f));
    host.place(ctx, kScreen);
    CHECK(a.laidFrame == Body({6.0f, 6.0f, 788.0f, 146.0f}));
    CHECK(b.laidFrame == Body({6.0f, 158.0f, 788.0f, 436.0f}));

    host.send(Mouse(MouseEventType::Release, 400.0f, 153.0f));
    REQUIRE(emissions.trees.size() == 1);
    REQUIRE(emissions.last().has_value());
    CheckApprox(*emissions.last(), Column({0.25f, 0.75f}, {Leaf("a"), Leaf("b")}));
}

TEST_CASE("Splitter cursors follow the split axis", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    CursorDock dock;
    Host host(dock);
    dock.update({.panes = {MakePane("a"), MakePane("b"), MakePane("c")},
                 .tiles = Row({0.5f, 0.5f}, {Leaf("a"), Column({0.5f, 0.5f}, {Leaf("b"), Leaf("c")})})});
    host.place(ctx, kScreen);

    CHECK(dock.cursorAt({400.0f, 100.0f}) == SurfaceCursor::ResizeEW);
    CHECK(dock.cursorAt({600.0f, 300.0f}) == SurfaceCursor::ResizeNS);
    CHECK_FALSE(dock.cursorAt({200.0f, 300.0f}).has_value());
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

TEST_CASE("Header drags offer a split on the nearest edge of another leaf", "[core][sakura][dock]") {
    struct Case {
        Extent2D<F32> point;
        Rect preview;
        DockTiles tiles;
    };
    const auto test = GENERATE(
        Case{{420.0f, 300.0f}, {403.0f, 6.0f, 195.5f, 588.0f}, Row({0.5f, 0.5f}, {Leaf("a"), Leaf("b")})},
        Case{{790.0f, 300.0f}, {598.5f, 6.0f, 195.5f, 588.0f}, Row({0.5f, 0.5f}, {Leaf("b"), Leaf("a")})},
        Case{{600.0f, 10.0f}, {403.0f, 6.0f, 391.0f, 294.0f}, Column({0.5f, 0.5f}, {Leaf("a"), Leaf("b")})},
        Case{{600.0f, 590.0f}, {403.0f, 300.0f, 391.0f, 294.0f}, Column({0.5f, 0.5f}, {Leaf("b"), Leaf("a")})});
    CAPTURE(test.point.x, test.point.y);

    Emissions emissions;
    Dock dock;
    Layer layer;
    layer.update({.content = &dock});
    LayerHost host({&layer});
    dock.update({.accepts = "panel",
                 .panes = {MakePane("a"), MakePane("b")},
                 .tiles = Row({0.5f, 0.5f}, {Leaf("a"), Leaf("b")}),
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

TEST_CASE("Splitting along the row axis inserts beside the target and halves its weight",
          "[core][sakura][dock]") {
    Emissions emissions;
    Dock dock;
    Layer layer;
    layer.update({.content = &dock});
    LayerHost host({&layer});
    dock.update({.accepts = "panel",
                 .panes = {MakePane("a"), MakePane("b"), MakePane("c")},
                 .tiles = Row({0.5f, 0.25f, 0.25f}, {Leaf("a"), Leaf("b"), Leaf("c")}),
                 .onTiles = emissions.sink()});
    host.place(MakeContext(), kScreen);

    host.send(Mouse(MouseEventType::Click, 200.0f, 12.0f));
    host.send(Mouse(MouseEventType::Move, 590.0f, 300.0f));
    host.place(MouseAt(590.0f, 300.0f), kScreen);
    REQUIRE(host.drag.target.has_value());
    CHECK(host.drag.target->preview == Rect{497.0f, 6.0f, 97.0f, 588.0f});

    host.send(Mouse(MouseEventType::Release, 590.0f, 300.0f));
    REQUIRE(emissions.trees.size() == 1);
    CHECK(emissions.last() == Row({0.25f, 0.25f, 0.5f}, {Leaf("b"), Leaf("a"), Leaf("c")}));
}

TEST_CASE("Dragging a header over its own leaf offers no target", "[core][sakura][dock]") {
    Emissions emissions;
    Dock dock;
    Layer layer;
    layer.update({.content = &dock});
    LayerHost host({&layer});
    dock.update({.accepts = "panel",
                 .panes = {MakePane("a"), MakePane("b")},
                 .tiles = Row({0.5f, 0.5f}, {Leaf("a"), Leaf("b")}),
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
                 .tiles = Row({0.5f, 0.5f}, {Leaf("a"), Leaf("b")}),
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
                 .tiles = Row({0.5f, 0.5f}, {Leaf("a"), Leaf("b")}),
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
                 .tiles = Row({0.5f, 0.5f}, {Leaf("a"), Leaf("b")}),
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
                 .tiles = Row({0.5f, 0.5f}, {Leaf("a"), Leaf("b")}),
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
                 .tiles = Row({0.5f, 0.5f}, {Leaf("a"), Leaf("b")}),
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
                 .tiles = Row({0.5f, 0.5f}, {Leaf("a"), Leaf("b")}),
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

TEST_CASE("Strip tab icons sit centered in a badge before the title", "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Probe a;
    Probe b;
    std::vector<U8> presses;
    Dock dock;
    Host host(dock);
    auto paneA = MakePane("a", &a);
    paneA.icon = ICON_FA_TERMINAL;
    dock.update({.tabs = true, .panes = {paneA, MakePane("b", &b)}, .activate = "a",
                 .onBarPress = [&](U8 clicks) { presses.push_back(clicks); }});
    host.place(ctx, kScreen);

    const auto texts = TextFrames(dock);
    REQUIRE(texts.size() == 3);
    CHECK(texts[0].x == 21.0f);
    CHECK(texts[0].width == 15.0f);
    CHECK(texts[1].x == 50.0f);
    CHECK(texts[2].x == 262.0f);

    host.click(600.0f, 19.0f);
    CHECK(presses == std::vector<U8>{1});
    host.click(300.0f, 19.0f);
    host.place(ctx, kScreen);
    CHECK(presses.size() == 1);
    CHECK(b.layouts == 1);
}

TEST_CASE("Strip tabs share one track with a bordered active capsule, dividers and icon badges",
          "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Dock dock;
    Host host(dock);
    auto paneB = MakePane("b");
    paneB.icon = ICON_FA_TERMINAL;
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
            const bool edge = config.instances[0].rect == Rect{8.0f, 6.0f, 193.0f, 28.0f} && config.cornerRadius == 14.0f;
            const bool rim = config.instances[0].rect == Rect{9.0f, 7.0f, 191.0f, 26.0f} && config.cornerRadius == 13.0f;
            CHECK((edge || rim));
            ++litEdges;
        }
    }
    CHECK(litEdges == 2);

    const auto* dividers = BoxWith(dock, {597.0f, 12.0f, 1.0f, 15.0f});
    REQUIRE(dividers);
    REQUIRE(dividers->instances.size() == 2);
    CHECK(dividers->instances[0].rect == Rect{400.0f, 12.0f, 1.0f, 15.0f});

    const auto* badges = BoxWith(dock, {215.0f, 9.0f, 22.0f, 22.0f});
    REQUIRE(badges);
    CHECK(badges->instances.size() == 1);
    CHECK(badges->cornerRadius == 6.0f);
    CHECK(badges->borderWidth == 1.0f);

    const auto texts = TextFrames(dock);
    REQUIRE(texts.size() == 5);
    CHECK(texts[0].x == 18.0f);
    CHECK(texts[1].x == 218.0f);
    CHECK(texts[2].x == 247.0f);
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

TEST_CASE("Tile cards cast a soft dropped shadow and carry a top lit edge and rim without painting corner masks",
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
    for (auto* box : Direct<Sakura::Box>(dock)) {
        if (box->config().topLit && !box->config().instances.empty()) {
            lit.push_back(&box->config());
        }
    }
    REQUIRE(lit.size() == 2);
    std::sort(lit.begin(), lit.end(), [](const auto* a, const auto* b) { return a->cornerRadius > b->cornerRadius; });
    CHECK(lit[0]->cornerRadius == 12.0f);
    CHECK(lit[0]->borderWidth == 1.0f);
    REQUIRE(lit[0]->instances.size() == 2);
    CHECK(lit[0]->instances[0].rect == cardA);
    CHECK(lit[0]->instances[1].rect == cardB);
    CHECK(lit[1]->cornerRadius == 11.0f);
    CHECK(lit[1]->borderWidth == 1.0f);
    REQUIRE(lit[1]->instances.size() == 2);
    CHECK(lit[1]->instances[0].rect == Rect{7.0f, 7.0f, 389.0f, 586.0f});
    CHECK(lit[1]->instances[1].rect == Rect{404.0f, 7.0f, 389.0f, 586.0f});

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
    paneA.icon = ICON_FA_CHART_LINE;
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
    const auto* cross = GlyphIn(dock, card, ICON_FA_XMARK);
    REQUIRE(cross);
    const Rect glyph = Peek::Frame(*cross);
    CHECK(glyph.right() == Catch::Approx(card.right() - (texts[0].x - card.x)));
    CHECK(glyph.y + glyph.height * 0.5f == Catch::Approx(texts[0].y + texts[0].height * 0.5f));
    const auto* crossB = GlyphIn(dock, {403.0f, 6.0f, 391.0f, 588.0f}, ICON_FA_XMARK);
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

TEST_CASE("Header buttons are centered icon glyphs that recolor on hover and flip when maximized",
          "[core][sakura][dock]") {
    const auto ctx = MakeContext();
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
    const auto expand = std::string(ICON_FA_UP_RIGHT_AND_DOWN_LEFT_FROM_CENTER);
    const auto compress = std::string(ICON_FA_DOWN_LEFT_AND_UP_RIGHT_TO_CENTER);
    CHECK(glyph(card, ICON_FA_XMARK).colorKey == "tile_header_text");
    CHECK(glyph(card, expand).colorKey == "tile_header_text");
    CHECK(glyph(card, expand).scale < glyph(card, ICON_FA_XMARK).scale);
    const Rect expandFrame = Peek::Frame(*GlyphIn(dock, card, expand));
    const Rect crossFrame = Peek::Frame(*GlyphIn(dock, card, ICON_FA_XMARK));
    CHECK(expandFrame.y + expandFrame.height * 0.5f == Catch::Approx(crossFrame.y + crossFrame.height * 0.5f));
    CHECK(expandFrame.right() < crossFrame.x);

    const auto point = MaximizeOf(card);
    host.send(Mouse(MouseEventType::Move, point.x, point.y));
    host.place(ctx, kScreen);
    CHECK(glyph(card, expand).colorKey == "tile_button_hover");
    CHECK(glyph(card, ICON_FA_XMARK).colorKey == "tile_header_text");
    CHECK(Direct<Sakura::Box>(dock).size() > 0);
    for (auto* box : Direct<Sakura::Box>(dock)) {
        for (const auto& instance : box->config().instances) {
            CHECK_FALSE((instance.rect.width == 20.0f && instance.rect.height == 20.0f));
        }
    }

    host.click(point.x, point.y);
    host.place(ctx, kScreen);
    const Rect full = {6.0f, 6.0f, 788.0f, 588.0f};
    CHECK(glyph(full, compress).str == compress);
    CHECK_FALSE(GlyphIn(dock, full, expand));

    const auto closePoint = CloseOf(full);
    host.send(Mouse(MouseEventType::Move, closePoint.x, closePoint.y));
    host.place(ctx, kScreen);
    CHECK(glyph(full, compress).colorKey == "tile_header_text");
    CHECK(glyph(full, ICON_FA_XMARK).colorKey == "tile_button_hover");
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
    const auto tiles = Row({1.0f, 1.0f, 1.0f}, {Leaf("a"), Leaf("b"), Leaf("c")});
    const auto update = [&] {
        dock.update({.panes = {MakePane("a", &a), MakePane("b", &b), MakePane("c", &c)}, .tiles = tiles,
                     .onTiles = emissions.sink()});
    };
    update();
    host.place(ctx, kScreen);
    const Rect cardB = {271.0f, 6.0f, 259.0f, 588.0f};
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
    std::optional<DockTiles> tiles = Row({0.5f, 0.5f}, {Leaf("a"), Leaf("b")});
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
    tiles = Row({0.5f, 0.5f}, {Leaf("b"), Leaf("c")});
    dock.update({.panes = {MakePane("b", &b), MakePane("c", &c)}, .tiles = tiles, .onTiles = sink});
    host.place(ctx, kScreen);
    CHECK(b.visible());
    CHECK(c.visible());
    CHECK(b.laidFrame == Body({6.0f, 6.0f, 391.0f, 588.0f}));
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
    std::optional<DockTiles> tiles = Row({0.5f, 0.5f}, {Leaf("a"), Leaf("b")});
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
                 .tiles = Row({0.5f, 0.5f}, {Leaf("a"), Column({0.5f, 0.5f}, {Leaf("b"), Leaf("c")})}),
                 .onTiles = emissions.sink()});
    host.place(MakeContext(), kScreen);

    host.send(Mouse(MouseEventType::Click, 200.0f, 21.0f));
    host.send(Mouse(MouseEventType::Move, 600.0f, 299.0f));
    REQUIRE(host.drag.active);
    host.place(MouseAt(600.0f, 299.0f), kScreen);
    REQUIRE(host.drag.target.has_value());
    CHECK(host.drag.target->preview == Rect{403.0f, 151.5f, 391.0f, 145.5f});
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
    return DockTiles{.weights = std::move(widths), .children = std::move(children)};
}

struct ColumnRig {
    Probe a;
    Probe b;
    Probe c;
    std::string flex = "a";
    std::optional<DockTiles> tiles;
    std::vector<std::optional<DockTiles>> emitted;
    std::vector<std::string> closed;
    bool withA = true;
    bool withC = true;
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
    CHECK(rig.emitted[0]->weights == std::vector<F32>{0.0f, 0.0f, 0.0f});
    CHECK(Leaves(*rig.emitted[0]) == std::vector<std::string>{"a", "b", "c"});

    rig.flex.clear();
    rig.update();
    host.place(ctx, kWideDock);
    CHECK(rig.a.laidFrame == Rect{6.0f, 26.0f, 525.0f, 568.0f});
    CHECK(rig.b.laidFrame == Rect{537.0f, 26.0f, 525.0f, 568.0f});
    CHECK(rig.c.laidFrame == Rect{1068.0f, 26.0f, 526.0f, 568.0f});
    CHECK(rig.emitted.size() == 1);
}

TEST_CASE("Column mode keeps stored widths and falls back to defaults for widths below the floor",
          "[core][sakura][dock][columns]") {
    ColumnRig rig;
    rig.tiles = ColumnRow({Leaf("a"), Leaf("b"), Leaf("c")}, {0.0f, 300.0f, 20.0f});
    Host host(rig.dock);
    rig.update();
    host.place(MakeContext({1600.0f, 600.0f}), kWideDock);

    CHECK(rig.a.laidFrame == Rect{6.0f, 26.0f, 856.0f, 568.0f});
    CHECK(rig.b.laidFrame == Rect{868.0f, 26.0f, 300.0f, 568.0f});
    CHECK(rig.c.laidFrame == Rect{1174.0f, 26.0f, 420.0f, 568.0f});
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
    CHECK(rig.emitted[0]->weights == std::vector<F32>{0.0f, 520.0f, 0.0f});

    rig.update();
    host.place(ctx, kWideDock);
    host.send(Mouse(MouseEventType::Click, 1171.0f, 300.0f));
    host.send(Mouse(MouseEventType::Move, 1221.0f, 300.0f));
    host.send(Mouse(MouseEventType::Release, 1221.0f, 300.0f));
    rig.update();
    host.place(ctx, kWideDock);
    REQUIRE(rig.emitted.size() == 2);
    CHECK(rig.emitted[1]->weights == std::vector<F32>{0.0f, 570.0f, 370.0f});
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
    CHECK(rig.emitted[2]->weights == std::vector<F32>{0.0f, 240.0f, 700.0f});
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
    CHECK(rig.emitted[0]->weights == std::vector<F32>{240.0f, 916.0f, 0.0f});
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
    CHECK(rig.emitted[0]->weights == std::vector<F32>{0.0f, 578.0f, 0.0f});
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

TEST_CASE("Dropping on a tile's top or bottom stacks it in that column and on its sides makes a new column",
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
    CHECK(host.drag.target->preview == Rect{748.0f, 300.0f, 420.0f, 294.0f});
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
    CHECK(host.drag.target->preview == Rect{1384.0f, 6.0f, 210.0f, 588.0f});
    host.send(Mouse(MouseEventType::Release, 1580.0f, 150.0f));
    REQUIRE(rig.emitted.size() == 2);
    CHECK(rig.emitted[1]->children.size() == 3);

    rig.update();
    host.place(MakeContext({1600.0f, 600.0f}), kWideDock);
    CHECK(rig.b.laidFrame == Rect{748.0f, 26.0f, 420.0f, 568.0f});
    CHECK(rig.c.laidFrame == Rect{1174.0f, 26.0f, 420.0f, 568.0f});
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
        paneA.actions = {{.icon = ICON_FA_SLIDERS, .active = active, .onClick = [&] { ++clicks; }}};
        dock.update({.panes = {paneA, MakePane("b")}});
        host.place(ctx, kScreen);
    };
    update();

    const Rect card = {6.0f, 6.0f, 391.0f, 588.0f};
    const auto* action = GlyphIn(dock, card, ICON_FA_SLIDERS);
    const auto* expand = GlyphIn(dock, card, ICON_FA_UP_RIGHT_AND_DOWN_LEFT_FROM_CENTER);
    REQUIRE(action);
    REQUIRE(expand);
    const Rect actionFrame = Peek::Frame(*action);
    const Rect expandFrame = Peek::Frame(*expand);
    CHECK(actionFrame.right() < expandFrame.x);
    CHECK(actionFrame.y + actionFrame.height * 0.5f == Catch::Approx(expandFrame.y + expandFrame.height * 0.5f));
    CHECK(action->config().colorKey == "tile_header_text");
    CHECK_FALSE(GlyphIn(dock, {403.0f, 6.0f, 391.0f, 588.0f}, ICON_FA_SLIDERS));

    active = true;
    update();
    CHECK(GlyphIn(dock, card, ICON_FA_SLIDERS)->config().colorKey == "cyber_blue");

    const Extent2D<F32> center = {actionFrame.x + actionFrame.width * 0.5f, actionFrame.y + actionFrame.height * 0.5f};
    host.send(Mouse(MouseEventType::Move, center.x, center.y));
    update();
    CHECK(GlyphIn(dock, card, ICON_FA_SLIDERS)->config().colorKey == "tile_button_hover");
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
            if (text->config().str == ICON_FA_XMARK && text->config().opacity > 0.0f) {
                values.push_back(text->config().opacity);
            }
        }
        return values;
    };
    CHECK(opacities() == std::vector<F32>{0.35f});
    for (auto* text : Direct<Sakura::Text>(dock)) {
        if (text->config().str == ICON_FA_XMARK && text->config().opacity > 0.0f) {
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
        if (text->config().str == ICON_FA_XMARK && text->config().opacity > 0.0f) {
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
    CHECK(rig.emitted[0]->weights == std::vector<F32>{0.0f, 520.0f, 0.0f});

    rig.update();
    host.place(ctx, bounds);
    CHECK(rig.b.laidFrame == Rect{1296.0f, 52.0f, 1040.0f, 1136.0f});
    host.send(Mouse(MouseEventType::Click, 2342.0f, 600.0f, MouseButton::Left, 2));
    host.send(Mouse(MouseEventType::Release, 2342.0f, 600.0f));
    REQUIRE(rig.emitted.size() == 2);
    CHECK(rig.emitted[1]->weights == std::vector<F32>{0.0f, 470.0f, 470.0f});

    rig.flex.clear();
    rig.update();
    host.place(ctx, bounds);
    CHECK(rig.a.laidFrame.width == 973.0f);
    CHECK(rig.b.laidFrame.width == 1089.0f);
    CHECK(rig.c.laidFrame.width == 1090.0f);
    host.send(Mouse(MouseEventType::Click, 991.0f, 600.0f));
    host.send(Mouse(MouseEventType::Move, 1091.0f, 600.0f));
    host.send(Mouse(MouseEventType::Release, 1091.0f, 600.0f));
    REQUIRE(rig.emitted.size() == 3);
    CHECK(rig.emitted[2]->weights == std::vector<F32>{536.5f, 494.5f, 545.0f});
    rig.update();
    host.place(ctx, bounds);
    CHECK(rig.a.laidFrame.width == 1073.0f);
    CHECK(rig.b.laidFrame.width == 989.0f);
    CHECK(rig.c.laidFrame == Rect{2098.0f, 52.0f, 1090.0f, 1136.0f});
}

TEST_CASE("Moving a single pane column beside another column or past the last one keeps its stored width",
          "[core][sakura][dock][columns]") {
    ColumnRig rig;
    rig.flex = "b";
    rig.tiles = ColumnRow({Leaf("a"), Leaf("b"), Leaf("c")}, {500.0f, 0.0f, 0.0f});
    Layer layer;
    layer.update({.content = &rig.dock});
    LayerHost host({&layer});
    rig.update();
    host.place(MakeContext({1600.0f, 600.0f}), kWideDock);
    CHECK(rig.a.laidFrame.width == 500.0f);
    CHECK(rig.c.laidFrame.x == 1174.0f);

    host.send(Mouse(MouseEventType::Click, 200.0f, 16.0f));
    host.send(Mouse(MouseEventType::Move, 210.0f, 16.0f));
    host.send(Mouse(MouseEventType::Move, 1580.0f, 300.0f));
    host.place(WideAt(1580.0f, 300.0f), kWideDock);
    REQUIRE(host.drag.target.has_value());
    CHECK(host.drag.target->preview == Rect{1384.0f, 6.0f, 210.0f, 588.0f});
    host.send(Mouse(MouseEventType::Release, 1580.0f, 300.0f));
    REQUIRE(rig.emitted.size() == 1);
    CHECK(rig.emitted[0] == ColumnRow({Leaf("b"), Leaf("c"), Leaf("a")}, {0.0f, 0.0f, 500.0f}));

    rig.update();
    host.place(MakeContext({1600.0f, 600.0f}), kWideDock);
    CHECK(rig.a.laidFrame == Rect{1094.0f, 26.0f, 500.0f, 568.0f});

    host.send(Mouse(MouseEventType::Click, 1300.0f, 16.0f));
    host.send(Mouse(MouseEventType::Move, 1310.0f, 16.0f));
    host.send(Mouse(MouseEventType::Move, 20.0f, 300.0f));
    host.place(WideAt(20.0f, 300.0f), kWideDock);
    REQUIRE(host.drag.target.has_value());
    CHECK(host.drag.target->preview == Rect{6.0f, 6.0f, 328.0f, 588.0f});
    host.send(Mouse(MouseEventType::Release, 20.0f, 300.0f));
    REQUIRE(rig.emitted.size() == 2);
    CHECK(rig.emitted[1] == ColumnRow({Leaf("a"), Leaf("b"), Leaf("c")}, {500.0f, 0.0f, 0.0f}));

    rig.update();
    host.place(MakeContext({1600.0f, 600.0f}), kWideDock);
    host.send(Mouse(MouseEventType::Click, 800.0f, 16.0f));
    host.send(Mouse(MouseEventType::Move, 810.0f, 16.0f));
    host.send(Mouse(MouseEventType::Move, 1599.0f, 300.0f));
    host.place(WideAt(1599.0f, 300.0f), kWideDock);
    REQUIRE(host.drag.target.has_value());
    CHECK(host.drag.target->preview.x >= 1594.0f);
    host.send(Mouse(MouseEventType::Release, 1599.0f, 300.0f));
    REQUIRE(rig.emitted.size() == 3);
    CHECK(rig.emitted[2] == ColumnRow({Leaf("a"), Leaf("c"), Leaf("b")}, {500.0f, 0.0f, 0.0f}));
}

TEST_CASE("Dropping on a tile's top stacks above it, and a stacked pane dropped on its own side gets its own column",
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
    CHECK(host.drag.target->preview == Rect{748.0f, 6.0f, 420.0f, 294.0f});
    host.send(Mouse(MouseEventType::Release, 958.0f, 20.0f));
    REQUIRE(rig.emitted.size() == 1);
    CHECK(rig.emitted[0] == ColumnRow({Leaf("a"), Column({0.5f, 0.5f}, {Leaf("c"), Leaf("b")})}));

    rig.update();
    host.place(MakeContext({1600.0f, 600.0f}), kWideDock);
    CHECK(rig.c.laidFrame == Body({1174.0f, 6.0f, 420.0f, 291.0f}));
    CHECK(rig.b.laidFrame == Body({1174.0f, 303.0f, 420.0f, 291.0f}));

    host.send(Mouse(MouseEventType::Click, 1300.0f, 16.0f));
    host.send(Mouse(MouseEventType::Move, 1310.0f, 16.0f));
    host.send(Mouse(MouseEventType::Move, 1300.0f, 280.0f));
    host.place(WideAt(1300.0f, 280.0f), kWideDock);
    CHECK_FALSE(host.drag.target.has_value());
    host.send(Mouse(MouseEventType::Move, 1580.0f, 150.0f));
    host.place(WideAt(1580.0f, 150.0f), kWideDock);
    REQUIRE(host.drag.target.has_value());
    CHECK(host.drag.target->preview == Rect{1384.0f, 6.0f, 210.0f, 588.0f});
    host.send(Mouse(MouseEventType::Release, 1580.0f, 150.0f));
    REQUIRE(rig.emitted.size() == 2);
    CHECK(rig.emitted[1] == ColumnRow({Leaf("a"), Leaf("b"), Leaf("c")}));
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
    rig.tiles = ColumnRow({Leaf("a"), Column({0.5f, 0.5f}, {Leaf("b"), Leaf("c")})});
    Host host(rig.dock);
    rig.update();
    host.place(ctx, kWideDock);
    CHECK(rig.b.laidFrame == Body({1174.0f, 6.0f, 420.0f, 291.0f}));
    CHECK(rig.c.laidFrame == Body({1174.0f, 303.0f, 420.0f, 291.0f}));
    CHECK(Peek::CursorAt(rig.dock, {1300.0f, 300.0f}) == SurfaceCursor::ResizeNS);

    host.send(Mouse(MouseEventType::Click, 1300.0f, 300.0f));
    host.send(Mouse(MouseEventType::Move, 1300.0f, 153.0f));
    host.place(ctx, kWideDock);
    CHECK(rig.b.laidFrame == Body({1174.0f, 6.0f, 420.0f, 146.0f}));
    CHECK(rig.c.laidFrame == Body({1174.0f, 158.0f, 420.0f, 436.0f}));
    CHECK(rig.a.laidFrame.width == 1162.0f);
    CHECK(rig.emitted.empty());
    host.send(Mouse(MouseEventType::Release, 1300.0f, 153.0f));
    REQUIRE(rig.emitted.size() == 1);
    REQUIRE(rig.emitted[0]->children.size() == 2);
    CHECK(rig.emitted[0]->weights == std::vector<F32>{0.0f, 0.0f});
    CHECK(rig.emitted[0]->children[0] == Leaf("a"));
    CheckApprox(rig.emitted[0]->children[1], Column({0.25f, 0.75f}, {Leaf("b"), Leaf("c")}));

    rig.update();
    host.place(ctx, kWideDock);
    CHECK(rig.b.laidFrame == Body({1174.0f, 6.0f, 420.0f, 146.0f}));
    host.send(Mouse(MouseEventType::Click, 1300.0f, 155.0f, MouseButton::Left, 2));
    host.send(Mouse(MouseEventType::Release, 1300.0f, 155.0f));
    REQUIRE(rig.emitted.size() == 2);
    CheckApprox(rig.emitted[1]->children[1], Column({0.5f, 0.5f}, {Leaf("b"), Leaf("c")}));
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
    CHECK(rig.emitted[0]->weights == std::vector<F32>{520.0f, 0.0f, 0.0f});

    rig.update();
    host.place(ctx, kWideDock);
    host.send(Mouse(MouseEventType::Click, 529.0f, 300.0f));
    host.send(Mouse(MouseEventType::Move, 29.0f, 300.0f));
    host.send(Mouse(MouseEventType::Release, 29.0f, 300.0f));
    REQUIRE(rig.emitted.size() == 2);
    CHECK(rig.emitted[1]->weights == std::vector<F32>{240.0f, 0.0f, 0.0f});
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
    CHECK(rig.emitted[0]->weights == std::vector<F32>{625.0f, 425.0f, 526.0f});

    rig.update();
    host.place(ctx, kWideDock);
    CHECK(rig.a.laidFrame.width == 625.0f);
    CHECK(rig.b.laidFrame == Rect{637.0f, 26.0f, 425.0f, 568.0f});
    CHECK(rig.c.laidFrame == Rect{1068.0f, 26.0f, 526.0f, 568.0f});

    host.send(Mouse(MouseEventType::Click, 634.0f, 300.0f));
    host.send(Mouse(MouseEventType::Move, 234.0f, 300.0f));
    host.place(ctx, kWideDock);
    CHECK(rig.a.laidFrame == Rect{6.0f, 26.0f, 240.0f, 568.0f});
    CHECK(rig.b.laidFrame == Rect{252.0f, 26.0f, 810.0f, 568.0f});
    CHECK(rig.c.laidFrame.right() == 1594.0f);
    host.send(Mouse(MouseEventType::Release, 234.0f, 300.0f));
    REQUIRE(rig.emitted.size() == 2);
    CHECK(rig.emitted[1]->weights == std::vector<F32>{240.0f, 810.0f, 526.0f});
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
                               Case{"", {0.0f, 0.0f, 1600.0f, 600.0f}, 1065.0f, {525.0f, 525.5f, 525.5f}},
                               Case{"a", {0.0f, 0.0f, 800.0f, 600.0f}, 291.0f, {0.0f, 264.5f, 0.0f}});
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
    CHECK(rig.emitted[0]->weights == test.widths);
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
    CHECK(rig.emitted[0]->weights == std::vector<F32>{0.0f, 520.0f, 0.0f});

    host.send(Mouse(MouseEventType::Move, 545.0f, 300.0f));
    host.send(Mouse(MouseEventType::Release, 545.0f, 300.0f));
    CHECK(rig.emitted.size() == 1);
    rig.update();
    host.place(ctx, kWideDock);
    CHECK(rig.b.laidFrame == Rect{648.0f, 26.0f, 520.0f, 568.0f});
}

TEST_CASE("Leaving during a tile splitter drag keeps the dragged weights and reports them like a release",
          "[core][sakura][dock]") {
    const auto ctx = MakeContext();
    Emissions emissions;
    Dock dock;
    Host host(dock);
    dock.update({.panes = {MakePane("a"), MakePane("b"), MakePane("c")},
                 .tiles = Row({0.5f, 0.25f, 0.25f}, {Leaf("a"), Leaf("b"), Leaf("c")}),
                 .onTiles = emissions.sink()});
    host.place(ctx, kScreen);

    host.send(Mouse(MouseEventType::Click, 597.0f, 300.0f));
    host.send(Mouse(MouseEventType::Move, 498.5f, 300.0f));
    host.send(Mouse(MouseEventType::Leave, 498.5f, 300.0f));
    REQUIRE(emissions.trees.size() == 1);
    REQUIRE(emissions.last().has_value());
    CheckApprox(*emissions.last(), Row({0.5f, 0.125f, 0.375f}, {Leaf("a"), Leaf("b"), Leaf("c")}));

    host.send(Mouse(MouseEventType::Move, 400.0f, 300.0f));
    host.send(Mouse(MouseEventType::Release, 400.0f, 300.0f));
    CHECK(emissions.trees.size() == 1);
}

TEST_CASE("Column mode adopts a vertical root as one stacked column", "[core][sakura][dock][columns]") {
    ColumnRig rig;
    rig.withC = false;
    rig.tiles = Column({0.5f, 0.5f}, {Leaf("a"), Leaf("b")});
    Host host(rig.dock);
    rig.update();
    host.place(MakeContext({1600.0f, 600.0f}), kWideDock);

    CHECK(rig.a.laidFrame == Body({6.0f, 6.0f, 1588.0f, 291.0f}));
    CHECK(rig.b.laidFrame == Body({6.0f, 303.0f, 1588.0f, 291.0f}));
    CHECK(rig.emitted.empty());
}

TEST_CASE("Column mode spreads a row nested in a row into columns and ignores the nested widths",
          "[core][sakura][dock][columns]") {
    ColumnRig rig;
    rig.tiles = Row({700.0f, 1.0f}, {Leaf("a"), Row({300.0f, 300.0f}, {Leaf("b"), Leaf("c")})});
    Host host(rig.dock);
    rig.update();
    host.place(MakeContext({1600.0f, 600.0f}), kWideDock);

    CHECK(rig.a.laidFrame == Rect{6.0f, 26.0f, 736.0f, 568.0f});
    CHECK(rig.b.laidFrame == Rect{748.0f, 26.0f, 420.0f, 568.0f});
    CHECK(rig.c.laidFrame == Rect{1174.0f, 26.0f, 420.0f, 568.0f});
    CHECK(rig.emitted.empty());
}

TEST_CASE("Column mode stacks a split nested inside a column as extra rows sharing its weight",
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
        .tiles = ColumnRow({Leaf("a"), Column({0.5f, 0.5f}, {Leaf("b"), Row({0.5f, 0.5f}, {Leaf("c"), Leaf("d")})})}),
        .onTiles = emissions.sink(),
        .columns = true,
        .flexPane = "a",
    });
    host.place(MakeContext({1600.0f, 600.0f}), kWideDock);

    CHECK(a.laidFrame == Rect{6.0f, 26.0f, 1162.0f, 568.0f});
    CHECK(b.laidFrame == Body({1174.0f, 6.0f, 420.0f, 288.0f}));
    CHECK(c.laidFrame == Body({1174.0f, 300.0f, 420.0f, 144.0f}));
    CHECK(d.laidFrame == Body({1174.0f, 450.0f, 420.0f, 144.0f}));
    CHECK(emissions.trees.empty());
}

TEST_CASE("Closing a stacked pane leaves its column-mate alone in the column",
          "[core][sakura][dock][columns]") {
    const auto ctx = MakeContext({1600.0f, 600.0f});
    ColumnRig rig;
    rig.tiles = ColumnRow({Leaf("a"), Column({0.5f, 0.5f}, {Leaf("b"), Leaf("c")})});
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
        for (auto* text : Direct<Sakura::Text>(rig.dock)) {
            if (text->config().str == "Loading...") {
                return Peek::Frame(*text);
            }
        }
        return std::nullopt;
    };
    const auto placeholder = loading();
    REQUIRE(placeholder.has_value());
    CHECK(placeholder->x == 748.0f);
    CHECK(placeholder->width == 420.0f);
    CHECK(placeholder->y + placeholder->height * 0.5f == Catch::Approx(310.0f).margin(1.0f));
    CHECK(rig.b.laidFrame.x == 1174.0f);
    CHECK(rig.emitted.empty());

    rig.withC = true;
    rig.update();
    host.place(ctx, kWideDock);
    CHECK_FALSE(loading().has_value());
    CHECK(rig.c.laidFrame == Rect{748.0f, 26.0f, 420.0f, 568.0f});
    CHECK(rig.emitted.empty());
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
    const auto tiles = Row({0.5f, 0.5f}, {Leaf("a"), Leaf("ghost")});
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
                 .tiles = Row({0.5f, 0.5f}, {Leaf("a"), Leaf("b")}),
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
