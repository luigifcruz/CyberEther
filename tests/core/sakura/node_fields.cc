#include <catch2/catch_test_macros.hpp>

#include "fonts.hh"
#include "component_harness.hh"
#include "typist.hh"

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/label.hh>
#include <jetstream/render/sakura/components/layer.hh>
#include <jetstream/render/sakura/components/node/bool_input.hh>
#include <jetstream/render/sakura/components/node/combo.hh>
#include <jetstream/render/sakura/components/node/content_stack.hh>
#include <jetstream/render/sakura/components/node/field.hh>
#include <jetstream/render/sakura/components/node/field_grid.hh>
#include <jetstream/render/sakura/components/node/field_panel.hh>
#include <jetstream/render/sakura/components/node/float_input.hh>
#include <jetstream/render/sakura/components/node/input_row.hh>
#include <jetstream/render/sakura/components/node/int_input.hh>
#include <jetstream/render/sakura/components/node/label.hh>
#include <jetstream/render/sakura/components/node/path_input.hh>
#include <jetstream/render/sakura/components/node/progress_bar.hh>
#include <jetstream/render/sakura/components/node/range_input.hh>
#include <jetstream/render/sakura/components/node/surface.hh>
#include <jetstream/render/sakura/components/node/table.hh>
#include <jetstream/render/sakura/components/node/text_input.hh>
#include <jetstream/render/sakura/components/node/uint_input.hh>
#include <jetstream/render/sakura/components/text_grid.hh>

#include "harness.hh"
#include "compositor/default/views/components/tooltip.hh"
#include "compositor/default/views/flowgraph/editor/field_tooltip.hh"
#include "render/sakura/metrics.hh"
#include "render/sakura/detail/text_metrics.hh"
#include "scene.hh"

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

using namespace Jetstream;
using namespace SakuraTest;

namespace {


constexpr const char* kTags = "[core][sakura][node-fields]";

Extent2D<F32> MeasureOf(const Sakura::Context& ctx, Sakura::Component& component, Extent2D<F32> available) {
    Host host(component);
    return host.measureOf(ctx, available);
}

constexpr const char* kRetentionTags = "[core][sakura][node-fields][retention]";

Sakura::Context NodeContext(Render::Window* window) {
    auto ctx = MakeContext();
    ctx.render = window;
    ctx.palette = DistinctPalette();
    ctx.fontSize = Sakura::kNodeFont;
    ctx.mouse = Extent2D<F32>{790.0f, 590.0f};
    return ctx;
}

struct Pointer {
    LayerHost& host;
    Frames& frames;

    bool send(const MouseEvent& event) {
        if (event.type != MouseEventType::Leave && event.type != MouseEventType::Cancel) {
            frames.ctx.mouse = event.position;
        }
        return host.send(event);
    }
};

struct Bench {
    MonoFontWindow window;
    LayerSlot slot;
    LayerHost host{std::vector<Sakura::Layer*>{&slot}};
    Frames frames{host, kScreen, NodeContext(&window)};
    Pointer pointer{host, frames};
    U64 rebuilds = 0;
    U64 restructures = 0;

    Bench(Sakura::Component& component, Rect at) {
        slot.hold(component, at);
    }

    std::function<bool(const MouseEvent&)> sender() {
        return [this](const MouseEvent& event) {
            return pointer.send(event);
        };
    }

    template<typename Update>
    bool settle(Update update) {
        if (!frames.settle(update) || !Retains(frames, update)) {
            return false;
        }
        rebuilds = frames.rebuilds;
        restructures = frames.restructures;
        return true;
    }

    bool unchanged() const {
        return frames.rebuilds == rebuilds && frames.restructures == restructures;
    }
};

template<typename Update>
void CheckRepaintOnly(Frames& frames, Update update) {
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
}

template<typename Update>
void CheckRestructuresOnce(Frames& frames, Update update) {
    const U64 restructures = frames.restructures;
    const U64 rebuilds = frames.rebuilds;
    const auto stats = frames.run(update);
    CHECK(stats.restructured);
    CHECK_FALSE(stats.rebuilt);
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
    CHECK(frames.restructures == restructures + 1);
    CHECK(frames.rebuilds == rebuilds);
}

template<typename T>
std::vector<T*> ChildrenOf(const Sakura::Component& component) {
    std::vector<T*> result;
    for (auto* child : Peek::Children(component)) {
        if (auto* typed = dynamic_cast<T*>(child)) {
            result.push_back(typed);
        }
    }
    return result;
}

bool Mounted(const Sakura::Component& parent, const Sakura::Component& child) {
    const auto& children = Peek::Children(parent);
    return std::find(children.begin(), children.end(), &child) != children.end();
}

}  // namespace

TEST_CASE("Node fields place their child below a card label and keep the card height as a floor", kTags) {
    const auto ctx = MakeContext();
    Probe child({0.0f, 21.0f});
    Sakura::NodeField field;
    field.update({.id = "gain", .label = "Gain", .content = &child});
    Host host(field);

    CHECK(host.measureOf(ctx, {300.0f, kInfinite}) == Extent2D<F32>{300.0f, 34.25f});
    host.place(ctx, {10.0f, 20.0f, 300.0f, 34.25f});
    CHECK(child.laidFrame == Rect{10.0f, 33.25f, 300.0f, 21.0f});

    child.resize({0.0f, 5.0f});
    CHECK(host.measureOf(ctx, {300.0f, kInfinite}) == Extent2D<F32>{300.0f, 29.0f});
    host.place(ctx, {10.0f, 20.0f, 300.0f, 29.0f});
    CHECK(child.laidFrame == Rect{10.0f, 33.25f, 300.0f, 15.75f});
}

TEST_CASE("Node fields without a card or title move the child up and drop the height floor", kTags) {
    const auto ctx = MakeContext();
    Probe child({0.0f, 5.0f});
    Sakura::NodeField field;
    Host host(field);

    field.update({.id = "plain", .label = "Gain", .background = false, .content = &child});
    CHECK(host.measureOf(ctx, {200.0f, kInfinite}) == Extent2D<F32>{200.0f, 20.25f});
    host.place(ctx, {0.0f, 0.0f, 200.0f, 20.25f});
    CHECK(child.laidFrame == Rect{0.0f, 15.25f, 200.0f, 5.0f});

    field.update({.id = "plain", .label = "Gain", .title = false, .content = &child});
    CHECK(host.measureOf(ctx, {200.0f, kInfinite}) == Extent2D<F32>{200.0f, 5.0f});
    host.place(ctx, {0.0f, 0.0f, 200.0f, 5.0f});
    CHECK(child.laidFrame == Rect{0.0f, 0.0f, 200.0f, 5.0f});

    field.update({.id = "plain", .content = &child});
    CHECK(host.measureOf(ctx, {200.0f, kInfinite}) == Extent2D<F32>{200.0f, 5.0f});
}

TEST_CASE("Node field label offsets and card floor scale with the pixel ratio", kTags) {
    auto ctx = MakeContext();
    ctx.pixelRatio = 2.0f;
    Probe child({0.0f, 21.0f});
    Sakura::NodeField field;
    field.update({.id = "gain", .label = "Gain", .content = &child});
    Host host(field);

    CHECK(host.measureOf(ctx, {300.0f, kInfinite}) == Extent2D<F32>{300.0f, 58.0f});
    child.resize({0.0f, 40.0f});
    CHECK(host.measureOf(ctx, {300.0f, kInfinite}) == Extent2D<F32>{300.0f, 66.5f});
    host.place(ctx, {0.0f, 0.0f, 300.0f, 66.5f});
    CHECK(child.laidFrame == Rect{0.0f, 26.5f, 300.0f, 40.0f});
}

TEST_CASE("Node fields show only their configured content", kTags) {
    const auto ctx = MakeContext();
    Probe first({0.0f, 21.0f});
    Probe second({0.0f, 40.0f});
    Sakura::NodeField field;
    Host host(field);

    field.update({.id = "swap", .label = "Value", .content = &first});
    CHECK(first.visible());
    CHECK(host.measureOf(ctx, {100.0f, kInfinite}).y == 34.25f);

    field.update({.id = "swap", .label = "Value", .content = &second});
    CHECK_FALSE(first.visible());
    CHECK(second.visible());
    CHECK(host.measureOf(ctx, {100.0f, kInfinite}).y == 53.25f);
    host.place(ctx, {0.0f, 0.0f, 100.0f, 53.25f});
    CHECK(second.layouts == 1);
    CHECK(first.layouts == 0);

    field.update({.id = "swap", .label = "Value", .content = &first});
    CHECK(first.visible());
    CHECK_FALSE(second.visible());

    field.update({.id = "swap", .label = "Value"});
    CHECK_FALSE(first.visible());
    CHECK(host.measureOf(ctx, {100.0f, kInfinite}).y == 29.0f);
}

TEST_CASE("Node field labels request their tooltip at the mouse after the hover delay", kTags) {
    const Scene scene;
    auto ctx = scene.ctx;
    ctx.hoverDelay = 0.5f;
    Probe tip({100.0f, 20.0f});
    Probe child({0.0f, 21.0f});
    Sakura::NodeField field;
    field.update({.id = "gain", .label = "Gain", .tooltip = {.content = [&] { return &tip; }}, .content = &child});
    LayerSlot slot;
    slot.hold(field, {100.0f, 100.0f, 300.0f, 34.25f});
    LayerHost host({&slot});

    const auto hover = [&](F32 x, F32 y) {
        ctx.mouse = Extent2D<F32>{x, y};
        host.send(Mouse(MouseEventType::Move, x, y));
    };
    host.place(ctx, kScreen);
    hover(110.0f, 106.0f);
    ctx.time = 1.0;
    host.place(ctx, kScreen);
    CHECK(tip.layouts == 0);

    ctx.time = 1.6;
    host.place(ctx, kScreen);
    REQUIRE(tip.layouts == 1);
    CHECK(tip.laidFrame == Rect{122.0f, 142.0f, 100.0f, 20.0f});

    hover(110.0f, 125.0f);
    host.place(ctx, kScreen);
    CHECK_FALSE(tip.visible());

    hover(110.0f, 106.0f);
    ctx.time = 1.7;
    host.place(ctx, kScreen);
    CHECK_FALSE(tip.visible());
    ctx.time = 2.2;
    host.place(ctx, kScreen);
    CHECK(tip.visible());
}

TEST_CASE("Node field labels with a lazily prepared tooltip show it after the delay without further input", kTags) {
    Probe tip({100.0f, 20.0f});
    Probe child({0.0f, 21.0f});
    Tooltip tooltip;
    Sakura::NodeField field;
    Bench bench(field, {100.0f, 100.0f, 300.0f, 34.25f});
    bench.frames.verify = false;
    bench.frames.timeStep = 0.1;
    bench.frames.ctx.hoverDelay = 0.25f;
    const auto update = [&] {
        if (tooltip.requested()) {
            tooltip.update({.id = "gain", .content = &tip});
        }
        field.update({.id = "gain", .label = "Gain", .tooltip = {.content = [&] { return tooltip.show(); }}, .content = &child});
    };
    const auto shown = [&] {
        const auto probes = Find<Probe>(bench.host);
        return std::find(probes.begin(), probes.end(), &tip) != probes.end();
    };
    REQUIRE(bench.settle(update));

    bench.pointer.send(Mouse(MouseEventType::Move, 110.0f, 106.0f));
    for (U64 frame = 0; frame < 8; ++frame) {
        bench.frames.run(update);
    }
    CHECK(shown());
    CHECK(tip.layouts > 0);
    CHECK(Retains(bench.frames, update));

    bench.pointer.send(Mouse(MouseEventType::Move, 110.0f, 125.0f));
    bench.frames.run(update);
    CHECK_FALSE(shown());
    CHECK(Retains(bench.frames, update));
}

TEST_CASE("Node field tooltips follow the pointer along the label and stay clean while it rests", kTags) {
    Probe tip({100.0f, 20.0f});
    Probe child({0.0f, 21.0f});
    Sakura::NodeField field;
    Bench bench(field, {100.0f, 100.0f, 300.0f, 34.25f});
    bench.frames.verify = false;
    bench.frames.timeStep = 0.1;
    bench.frames.ctx.hoverDelay = 0.25f;
    const auto update = [&] {
        field.update({.id = "gain", .label = "Gain", .tooltip = {.content = [&] { return &tip; }}, .content = &child});
    };
    REQUIRE(bench.settle(update));

    bench.pointer.send(Mouse(MouseEventType::Move, 110.0f, 106.0f));
    for (U64 frame = 0; frame < 5; ++frame) {
        bench.frames.run(update);
    }
    REQUIRE(tip.visible());
    const Rect entered = tip.currentFrame();
    bench.pointer.send(Mouse(MouseEventType::Move, 110.0f, 106.0f));
    CHECK(bench.frames.run(update).clean());

    bench.pointer.send(Mouse(MouseEventType::Move, 120.0f, 107.0f));
    bench.frames.run(update);
    CHECK(tip.visible());
    CHECK(tip.currentFrame().x == entered.x + 10.0f);
    CHECK(tip.currentFrame().y == entered.y + 1.0f);
    CHECK(Retains(bench.frames, update));

    bench.pointer.send(Mouse(MouseEventType::Move, 110.0f, 125.0f));
    bench.frames.run(update);
    CHECK_FALSE(tip.visible());
    CHECK(QuietMoves(bench.frames, update, {{120.0f, 126.0f}, {130.0f, 127.0f}}, bench.sender()));
}

TEST_CASE("Node field tooltips follow the pointer while help is missing and show when help returns under it", kTags) {
    Probe tip({100.0f, 20.0f});
    Probe child({0.0f, 21.0f});
    Sakura::NodeField field;
    Bench bench(field, {100.0f, 100.0f, 300.0f, 34.25f});
    bench.frames.verify = false;
    bench.frames.timeStep = 0.1;
    bench.frames.ctx.hoverDelay = 0.25f;
    bool help = true;
    const auto update = [&] {
        field.update({.id = "gain",
                      .label = "Gain",
                      .tooltip = help ? Sakura::NodeOverlay{.content = [&] { return &tip; }} : Sakura::NodeOverlay{},
                      .content = &child});
    };
    const auto shown = [&] {
        const auto probes = Find<Probe>(bench.host);
        return std::find(probes.begin(), probes.end(), &tip) != probes.end();
    };
    const auto wait = [&] {
        for (U64 frame = 0; frame < 6; ++frame) {
            bench.frames.run(update);
        }
    };
    REQUIRE(bench.settle(update));

    bench.pointer.send(Mouse(MouseEventType::Move, 110.0f, 106.0f));
    wait();
    REQUIRE(shown());
    help = false;
    wait();
    REQUIRE_FALSE(shown());
    bench.pointer.send(Mouse(MouseEventType::Move, 110.0f, 125.0f));
    wait();
    help = true;
    wait();
    CHECK_FALSE(shown());

    help = false;
    wait();
    bench.pointer.send(Mouse(MouseEventType::Move, 110.0f, 106.0f));
    wait();
    REQUIRE_FALSE(shown());
    help = true;
    wait();
    CHECK(shown());
}

TEST_CASE("Node path inputs follow the pointer while their tooltip is unavailable", kTags) {
    Probe tip({100.0f, 20.0f});
    Sakura::NodePathInput path;
    Bench bench(path, {0.0f, 100.0f, 300.0f, 21.0f});
    bench.frames.verify = false;
    bench.frames.timeStep = 0.1;
    bench.frames.ctx.hoverDelay = 0.25f;
    std::string value = "/tmp/capture.bin";
    const auto update = [&] {
        path.update({.id = "path", .value = value, .tooltip = {.content = [&] { return &tip; }}});
    };
    const auto shown = [&] {
        const auto probes = Find<Probe>(bench.host);
        return std::find(probes.begin(), probes.end(), &tip) != probes.end();
    };
    const auto wait = [&] {
        for (U64 frame = 0; frame < 6; ++frame) {
            bench.frames.run(update);
        }
    };
    REQUIRE(bench.settle(update));

    value.clear();
    wait();
    bench.pointer.send(Mouse(MouseEventType::Move, 50.0f, 110.0f));
    wait();
    REQUIRE_FALSE(shown());
    value = "/tmp/capture.bin";
    wait();
    CHECK(shown());

    value.clear();
    wait();
    bench.pointer.send(Mouse(MouseEventType::Move, 50.0f, 300.0f));
    wait();
    value = "/tmp/capture.bin";
    wait();
    CHECK_FALSE(shown());
}

TEST_CASE("Node field tooltips and hover clear when the pointer is cancelled", kTags) {
    Probe tip({100.0f, 20.0f});
    Probe child({0.0f, 21.0f});
    Sakura::NodeField field;
    Bench bench(field, {100.0f, 100.0f, 300.0f, 34.25f});
    bench.frames.verify = false;
    bench.frames.timeStep = 0.1;
    bench.frames.ctx.hoverDelay = 0.25f;
    const auto update = [&] {
        field.update({.id = "gain", .label = "Gain", .tooltip = {.content = [&] { return &tip; }}, .content = &child});
    };
    const auto shown = [&] {
        const auto probes = Find<Probe>(bench.host);
        return std::find(probes.begin(), probes.end(), &tip) != probes.end();
    };
    REQUIRE(bench.settle(update));
    bench.pointer.send(Mouse(MouseEventType::Move, 110.0f, 106.0f));
    for (U64 frame = 0; frame < 6; ++frame) {
        bench.frames.run(update);
    }
    REQUIRE(shown());

    bench.pointer.send(Mouse(MouseEventType::Cancel, 110.0f, 106.0f));
    bench.frames.run(update);
    CHECK_FALSE(shown());
    CHECK(bench.frames.run(update).clean());
}

TEST_CASE("Node field grids split into as many 140 pixel columns as fit and stretch rows to the tallest item", kTags) {
    const auto ctx = MakeContext();
    Probe a({0.0f, 21.0f});
    Probe b({0.0f, 30.0f});
    Probe c({0.0f, 10.0f});
    Sakura::NodeFieldGrid grid;
    grid.update({.id = "grid", .items = {{.child = &a}, {.child = &b}, {.child = &c}}});
    Host host(grid);

    CHECK(host.measureOf(ctx, {300.0f, kInfinite}) == Extent2D<F32>{300.0f, 42.0f});
    host.place(ctx, {0.0f, 0.0f, 300.0f, 42.0f});
    CHECK(a.laidFrame == Rect{0.0f, 0.0f, 149.0f, 30.0f});
    CHECK(b.laidFrame == Rect{151.0f, 0.0f, 149.0f, 30.0f});
    CHECK(c.laidFrame == Rect{0.0f, 32.0f, 149.0f, 10.0f});

    CHECK(host.measureOf(ctx, {424.0f, kInfinite}) == Extent2D<F32>{424.0f, 30.0f});
    host.place(ctx, {0.0f, 0.0f, 424.0f, 30.0f});
    CHECK(a.laidFrame == Rect{0.0f, 0.0f, 140.0f, 30.0f});
    CHECK(b.laidFrame == Rect{142.0f, 0.0f, 140.0f, 30.0f});
    CHECK(c.laidFrame == Rect{284.0f, 0.0f, 140.0f, 30.0f});

    CHECK(host.measureOf(ctx, {282.0f, kInfinite}).y == 42.0f);
    CHECK(host.measureOf(ctx, {281.0f, kInfinite}) == Extent2D<F32>{281.0f, 65.0f});
    host.place(ctx, {0.0f, 0.0f, 281.0f, 65.0f});
    CHECK(a.laidFrame == Rect{0.0f, 0.0f, 281.0f, 21.0f});
    CHECK(b.laidFrame == Rect{0.0f, 23.0f, 281.0f, 30.0f});
    CHECK(c.laidFrame == Rect{0.0f, 55.0f, 281.0f, 10.0f});
}

TEST_CASE("Full width node grid items break the column groups around them", kTags) {
    const auto ctx = MakeContext();
    Probe a({0.0f, 21.0f});
    Probe b({0.0f, 30.0f});
    Probe c({0.0f, 10.0f});
    Probe d({0.0f, 21.0f});
    Sakura::NodeFieldGrid grid;
    grid.update({.id = "grid", .items = {
        {.child = &a},
        {.child = &b, .fullWidth = true},
        {.child = &c},
        {.child = &d},
    }});
    Host host(grid);

    CHECK(host.measureOf(ctx, {300.0f, kInfinite}) == Extent2D<F32>{300.0f, 76.0f});
    host.place(ctx, {0.0f, 0.0f, 300.0f, 76.0f});
    CHECK(a.laidFrame == Rect{0.0f, 0.0f, 300.0f, 21.0f});
    CHECK(b.laidFrame == Rect{0.0f, 23.0f, 300.0f, 30.0f});
    CHECK(c.laidFrame == Rect{0.0f, 55.0f, 149.0f, 21.0f});
    CHECK(d.laidFrame == Rect{151.0f, 55.0f, 149.0f, 21.0f});

    grid.update({.id = "grid"});
    CHECK(host.measureOf(ctx, {300.0f, kInfinite}) == Extent2D<F32>{300.0f, 0.0f});
}

TEST_CASE("Node field grid columns and full width rows preserve the parent clip at every pixel ratio", kTags) {
    const F32 scale = GENERATE(1.0f, 2.0f);
    const U64 columns = GENERATE(1, 2, 3);
    const bool fullWidth = GENERATE(false, true);
    auto ctx = MakeContext();
    ctx.pixelRatio = scale;
    std::vector<std::unique_ptr<Probe>> probes;
    Sakura::NodeFieldGrid grid;
    Sakura::NodeFieldGrid::Config config{.id = "clipped-grid"};
    for (U64 i = 0; i < columns * 2; ++i) {
        probes.push_back(std::make_unique<Probe>(Extent2D<F32>{0.0f, 40.0f * scale}));
        config.items.push_back({.child = probes.back().get(), .fullWidth = fullWidth});
    }
    grid.update(std::move(config));
    const F32 width = 160.0f * columns * scale;
    Place placement(grid, {100.0f, 100.0f, width, 300.0f * scale});
    Host host(placement, true);
    const Rect clip{100.0f, 100.0f + 5.0f * scale, width - 20.0f * scale, 50.0f * scale};
    host.place(ctx, clip);
    CHECK(Peek::Clip(host) == clip);
    REQUIRE_FALSE(probes.front()->laidClip.empty());
    for (const auto& probe : probes) {
        if (!probe->laidClip.empty()) {
            CHECK(probe->laidClip.x >= clip.x);
            CHECK(probe->laidClip.y >= clip.y);
            CHECK(probe->laidClip.right() <= clip.right());
            CHECK(probe->laidClip.bottom() <= clip.bottom());
        }
    }
}

TEST_CASE("Node content stacks space items by default, custom and trailing gaps and skip hidden items", kTags) {
    const auto ctx = MakeContext();
    Probe a({0.0f, 20.0f});
    Probe b({0.0f, 30.0f});
    Probe c({0.0f, 10.0f});
    Probe d({0.0f, 15.0f});
    Sakura::NodeContentStack stack;
    stack.update({.id = "stack", .items = {
        {.child = &a},
        {.child = &b, .gap = 4.0f},
        {.child = &c, .trailingGap = 6.0f},
        {.child = &d},
    }});
    Host host(stack);

    CHECK(Sakura::NodeContentStack::Spacing == 8.0f);
    CHECK(host.measureOf(ctx, {300.0f, kInfinite}) == Extent2D<F32>{300.0f, 101.0f});
    host.place(ctx, {0.0f, 10.0f, 300.0f, 101.0f});
    CHECK(a.laidFrame == Rect{0.0f, 10.0f, 300.0f, 20.0f});
    CHECK(b.laidFrame == Rect{0.0f, 34.0f, 300.0f, 30.0f});
    CHECK(c.laidFrame == Rect{0.0f, 72.0f, 300.0f, 10.0f});
    CHECK(d.laidFrame == Rect{0.0f, 96.0f, 300.0f, 15.0f});
    CHECK_FALSE(stack.empty());
    CHECK(stack.leadingGap() == 8.0f);
    CHECK_FALSE(stack.hasFlexible());

    b.setVisible(false);
    CHECK(host.measureOf(ctx, {300.0f, kInfinite}).y == 67.0f);

    a.setVisible(false);
    b.setVisible(true);
    CHECK(stack.leadingGap() == 4.0f);
    CHECK(host.measureOf(ctx, {300.0f, kInfinite}).y == 77.0f);
    host.place(ctx, {0.0f, 0.0f, 300.0f, 77.0f});
    CHECK(b.laidFrame == Rect{0.0f, 0.0f, 300.0f, 30.0f});

    for (auto* probe : {&b, &c, &d}) {
        probe->setVisible(false);
    }
    CHECK(stack.empty());
    CHECK(stack.leadingGap() == 8.0f);
    CHECK(host.measureOf(ctx, {300.0f, kInfinite}).y == 0.0f);
}

TEST_CASE("Flexible node stack items share the leftover height by grow and never shrink below their minimum",
          kTags) {
    const auto ctx = MakeContext();
    Probe a({0.0f, 20.0f});
    Probe b({0.0f, 10.0f});
    Probe c({0.0f, 40.0f});
    Sakura::NodeContentStack stack;
    stack.update({.id = "stack", .items = {
        {.child = &a},
        {.child = &b, .flex = Sakura::NodeContentStack::Flex{.minimum = 50.0f, .grow = 1.0f}},
        {.child = &c, .flex = Sakura::NodeContentStack::Flex{.minimum = 30.0f, .grow = 3.0f}},
    }});
    Host host(stack);

    CHECK(stack.hasFlexible());
    CHECK(host.measureOf(ctx, {300.0f, kInfinite}) == Extent2D<F32>{300.0f, 126.0f});

    host.place(ctx, {0.0f, 0.0f, 300.0f, 200.0f});
    CHECK(a.laidFrame == Rect{0.0f, 0.0f, 300.0f, 20.0f});
    CHECK(b.laidFrame == Rect{0.0f, 28.0f, 300.0f, 71.0f});
    CHECK(c.laidFrame == Rect{0.0f, 107.0f, 300.0f, 93.0f});

    host.place(ctx, {0.0f, 0.0f, 300.0f, 60.0f});
    CHECK(b.laidFrame == Rect{0.0f, 28.0f, 300.0f, 50.0f});
    CHECK(c.laidFrame == Rect{0.0f, 86.0f, 300.0f, 30.0f});
}

TEST_CASE("Node content stack metrics report natural, fixed, minimum and flexible basis heights", kTags) {
    const auto ctx = MakeContext();
    Probe a({0.0f, 20.0f});
    Probe b({0.0f, 10.0f});
    Probe c({0.0f, 40.0f});
    Sakura::NodeContentStack stack;
    stack.update({.id = "stack", .items = {
        {.child = &a},
        {.child = &b, .flex = Sakura::NodeContentStack::Flex{.minimum = 50.0f, .basis = 60.0f}},
        {.child = &c, .flex = Sakura::NodeContentStack::Flex{.minimum = 30.0f}},
    }});

    const auto metrics = stack.metrics(ctx, 300.0f);
    CHECK(metrics.natural == 126.0f);
    CHECK(metrics.fixed == 36.0f);
    CHECK(metrics.minimum == 116.0f);
    CHECK(metrics.flexibleBasis == 90.0f);

    auto doubled = ctx;
    doubled.pixelRatio = 2.0f;
    const auto scaled = stack.metrics(doubled, 300.0f);
    CHECK(scaled.fixed == 52.0f);
    CHECK(scaled.minimum == 212.0f);
    CHECK(scaled.flexibleBasis == 180.0f);
}

TEST_CASE("Node field panels measure their content naturally, by its minimum when flexible, or fill", kTags) {
    const auto ctx = MakeContext();
    Probe a({0.0f, 20.0f});
    Probe b({0.0f, 100.0f});
    Sakura::NodeContentStack stack;
    stack.update({.id = "stack", .items = {{.child = &a}, {.child = &b}}});
    Sakura::NodeFieldPanel panel;
    Host host(panel);

    panel.update({.id = "panel"});
    CHECK(host.measureOf(ctx, {300.0f, 400.0f}) == Extent2D<F32>{300.0f, 0.0f});

    panel.update({.id = "panel", .content = &stack});
    CHECK(host.measureOf(ctx, {300.0f, 400.0f}) == Extent2D<F32>{300.0f, 128.0f});
    CHECK(host.measureOf(ctx, {kInfinite, 400.0f}) == Extent2D<F32>{0.0f, 128.0f});

    stack.update({.id = "stack", .items = {
        {.child = &a},
        {.child = &b, .flex = Sakura::NodeContentStack::Flex{.minimum = 50.0f}},
    }});
    CHECK(host.measureOf(ctx, {300.0f, 400.0f}) == Extent2D<F32>{300.0f, 78.0f});

    panel.update({.id = "panel", .content = &stack, .height = Sakura::Fill});
    CHECK(host.measureOf(ctx, {300.0f, 400.0f}) == Extent2D<F32>{300.0f, 400.0f});
    CHECK(host.measureOf(ctx, {300.0f, kInfinite}) == Extent2D<F32>{300.0f, 0.0f});
}

TEST_CASE("Node field panels pass their parent's backdrop to their content", kTags) {
    struct BackdropProbe : Probe {
        std::string backdrop;

     protected:
        void layout(const Sakura::Context& ctx) override {
            Probe::layout(ctx);
            backdrop = ctx.backdropColorKey;
        }
    };

    BackdropProbe content;
    content.resize({0.0f, 30.0f});
    Sakura::NodeContentStack stack;
    stack.update({.id = "stack", .items = {{.child = &content}}});
    Sakura::NodeFieldPanel panel;
    panel.update({.id = "panel", .content = &stack});
    Host host(panel);

    auto ctx = MakeContext();
    host.place(ctx, {0.0f, 0.0f, 300.0f, 50.0f});
    CHECK(content.backdrop == "background");

    ctx.backdropColorKey = "tile_bg";
    host.place(ctx, {0.0f, 0.0f, 300.0f, 50.0f});
    CHECK(content.backdrop == "tile_bg");

    ctx.backdropColorKey = "node_background";
    host.place(ctx, {0.0f, 0.0f, 300.0f, 50.0f});
    CHECK(content.backdrop == "node_background");
}

TEST_CASE("Node field panels give content their bounds, let flexible items fill and clip overflow", kTags) {
    const auto ctx = MakeContext();
    Probe a({0.0f, 30.0f});
    Probe b({0.0f, 30.0f});
    Sakura::NodeContentStack stack;
    stack.update({.id = "stack", .items = {{.child = &a}, {.child = &b}}});
    Sakura::NodeFieldPanel panel;
    panel.update({.id = "panel", .content = &stack});
    LayerSlot slot;
    slot.hold(panel, {10.0f, 10.0f, 300.0f, 50.0f});
    LayerHost host({&slot});

    host.place(ctx, kScreen);
    CHECK(a.laidFrame == Rect{10.0f, 10.0f, 300.0f, 30.0f});
    CHECK(b.laidFrame == Rect{10.0f, 48.0f, 300.0f, 30.0f});
    CHECK(b.laidClip == Rect{10.0f, 10.0f, 300.0f, 50.0f});

    slot.hold(stack, {10.0f, 10.0f, 300.0f, 50.0f});
    host.place(ctx, kScreen);
    CHECK(b.laidClip == kScreen);

    stack.update({.id = "stack", .items = {
        {.child = &a},
        {.child = &b, .flex = Sakura::NodeContentStack::Flex{.minimum = 10.0f}},
    }});
    panel.update({.id = "panel", .content = &stack, .height = Sakura::Fill});
    slot.hold(panel, {0.0f, 0.0f, 300.0f, 200.0f});
    host.place(ctx, kScreen);
    CHECK(b.laidFrame == Rect{0.0f, 38.0f, 300.0f, 162.0f});
}

TEST_CASE("Node inputs fill the available width at one frame height", kTags) {
    const Scene scene;
    auto ctx = scene.ctx;
    Sakura::NodeTextInput text;
    text.update({.id = "text"});
    Sakura::NodeFloatInput plain;
    plain.update({.id = "plain"});
    Sakura::NodeFloatInput stepped;
    stepped.update({.id = "stepped", .step = 0.5f});
    Sakura::NodeIntInput integer;
    integer.update({.id = "int"});
    Sakura::NodeUIntInput unsignedInteger;
    unsignedInteger.update({.id = "uint"});
    Sakura::NodeBoolInput toggle;
    toggle.update({.id = "bool"});
    Sakura::NodeRangeInput range;
    range.update({.id = "range"});
    Sakura::NodeCombo combo;
    combo.update({.id = "combo", .options = {"a"}});
    Sakura::NodePathInput path;
    path.update({.id = "path"});

    for (Sakura::Component* input : std::vector<Sakura::Component*>{
             &text, &plain, &integer, &unsignedInteger, &toggle, &range, &combo, &path}) {
        CHECK(MeasureOf(ctx, *input, {240.0f, kInfinite}) == Extent2D<F32>{240.0f, 21.0f});
    }
    CHECK(MeasureOf(ctx, stepped, {240.0f, kInfinite}) == Extent2D<F32>{240.0f, 40.0f});

    text.update({.id = "text", .height = 60.0f});
    CHECK(MeasureOf(ctx, text, {240.0f, kInfinite}) == Extent2D<F32>{240.0f, 60.0f});

    ctx.pixelRatio = 2.0f;
    CHECK(MeasureOf(ctx, text, {240.0f, kInfinite}) == Extent2D<F32>{240.0f, 120.0f});
    CHECK(MeasureOf(ctx, toggle, {240.0f, kInfinite}) == Extent2D<F32>{240.0f, 42.0f});
    CHECK(MeasureOf(ctx, stepped, {240.0f, kInfinite}) == Extent2D<F32>{240.0f, 80.0f});

    ctx.pixelRatio = 1.0f;
    ctx.fontSize = 20.0f;
    CHECK(MeasureOf(ctx, combo, {240.0f, kInfinite}) == Extent2D<F32>{240.0f, 26.0f});
    CHECK(MeasureOf(ctx, stepped, {240.0f, kInfinite}) == Extent2D<F32>{240.0f, 50.0f});
}

TEST_CASE("Node bool inputs request the toggled value on a full click inside", kTags) {
    const Scene scene;
    std::vector<bool> changes;
    Sakura::NodeBoolInput toggle;
    toggle.update({.id = "bool", .value = false, .onChange = [&](bool value) { changes.push_back(value); }});
    Host host(toggle);
    host.place(scene.ctx, {0.0f, 0.0f, 200.0f, 21.0f});

    CHECK(host.send(Mouse(MouseEventType::Click, 100.0f, 10.0f)));
    CHECK(changes.empty());
    CHECK(host.send(Mouse(MouseEventType::Release, 150.0f, 15.0f)));
    CHECK(changes == std::vector<bool>{true});

    toggle.update({.id = "bool", .value = true, .onChange = [&](bool value) { changes.push_back(value); }});
    CHECK(host.click(10.0f, 10.0f));
    CHECK(changes == std::vector<bool>{true, false});

    CHECK(host.send(Mouse(MouseEventType::Click, 100.0f, 10.0f)));
    CHECK_FALSE(host.send(Mouse(MouseEventType::Release, 300.0f, 10.0f)));
    CHECK(host.send(Mouse(MouseEventType::Click, 100.0f, 10.0f)));
    host.send(Mouse(MouseEventType::Cancel, 100.0f, 10.0f));
    CHECK_FALSE(host.send(Mouse(MouseEventType::Release, 100.0f, 10.0f)));
    CHECK_FALSE(host.click(100.0f, 10.0f, MouseButton::Right));
    CHECK(changes.size() == 2);
}

TEST_CASE("Node range inputs map clicks and drags across the track to clamped values", kTags) {
    const Scene scene;
    std::vector<F32> changes;
    Sakura::NodeRangeInput range;
    const auto configure = [&](F32 value, bool integer) {
        range.update({.id = "range", .min = 0.0f, .max = 10.0f, .value = value, .integer = integer,
                      .onChange = [&](F32 next) { changes.push_back(next); }});
    };
    configure(0.0f, false);
    Host host(range);
    host.place(scene.ctx, {0.0f, 0.0f, 200.0f, 21.0f});

    CHECK(host.send(Mouse(MouseEventType::Click, 50.0f, 10.0f)));
    CHECK(host.send(Mouse(MouseEventType::Move, 150.0f, 10.0f)));
    CHECK(host.send(Mouse(MouseEventType::Move, 150.0f, 40.0f)));
    CHECK(host.send(Mouse(MouseEventType::Move, 500.0f, 10.0f)));
    CHECK(host.send(Mouse(MouseEventType::Move, -20.0f, 10.0f)));
    CHECK(host.send(Mouse(MouseEventType::Release, -20.0f, 10.0f)));
    CHECK(changes == std::vector<F32>{2.5f, 7.5f, 10.0f, 0.0f});

    CHECK_FALSE(host.send(Mouse(MouseEventType::Move, 100.0f, 10.0f)));
    CHECK_FALSE(host.send(Mouse(MouseEventType::Release, 100.0f, 10.0f)));
    CHECK_FALSE(host.send(Mouse(MouseEventType::Click, 100.0f, 10.0f, MouseButton::Right)));
    CHECK(changes.size() == 4);

    changes.clear();
    configure(5.0f, true);
    CHECK(host.send(Mouse(MouseEventType::Click, 50.0f, 10.0f)));
    CHECK(host.send(Mouse(MouseEventType::Move, 52.0f, 10.0f)));
    CHECK(host.send(Mouse(MouseEventType::Move, 109.0f, 10.0f)));
    host.send(Mouse(MouseEventType::Cancel, 109.0f, 10.0f));
    CHECK_FALSE(host.send(Mouse(MouseEventType::Move, 180.0f, 10.0f)));
    CHECK(changes == std::vector<F32>{3.0f, 5.0f});

    changes.clear();
    configure(5.0f, true);
    CHECK(host.send(Mouse(MouseEventType::Click, 100.0f, 10.0f)));
    CHECK(host.send(Mouse(MouseEventType::Release, 100.0f, 10.0f)));
    CHECK(changes.empty());
}

TEST_CASE("Node range inputs draw their dragged value when the owner does not echo it", kTags) {
    Sakura::NodeRangeInput range;
    Bench bench(range, {10.0f, 10.0f, 200.0f, 19.5f});
    bench.frames.verify = false;
    std::vector<F32> reported;
    const auto update = [&] {
        range.update({.id = "range", .min = 0.0f, .max = 10.0f, .value = 4.0f, .unit = "dB",
                      .onChange = [&](F32 next) { reported.push_back(next); }});
    };
    const auto shown = [&]() -> std::string {
        for (auto* label : Find<Sakura::Label>(range)) {
            for (const auto& instance : label->config().instances) {
                if (instance.str != "dB") {
                    return instance.str;
                }
            }
        }
        return {};
    };
    REQUIRE(bench.settle(update));
    REQUIRE(shown() == "4");

    bench.pointer.send(Mouse(MouseEventType::Click, 60.0f, 15.0f));
    CHECK(RepaintOnly(bench.frames.run(update)));
    CHECK(shown() == "2");
    bench.pointer.send(Mouse(MouseEventType::Move, 160.0f, 15.0f));
    CHECK(RepaintOnly(bench.frames.run(update)));
    CHECK(shown() == "8");
    CHECK(reported == std::vector<F32>{2.5f, 7.5f});

    bench.pointer.send(Mouse(MouseEventType::Release, 160.0f, 15.0f));
    bench.frames.run(update);
    bench.frames.run(update);
    CHECK(shown() == "4");
}

TEST_CASE("Node range inputs draw their value with the configured precision unless integer", kTags) {
    Sakura::NodeRangeInput range;
    Bench bench(range, {10.0f, 10.0f, 200.0f, 19.5f});
    I32 precision = 0;
    bool integer = false;
    const auto update = [&] {
        range.update({.id = "range", .min = 0.0f, .max = 1.0f, .value = 0.25f, .integer = integer,
                      .precision = precision});
    };
    const auto shown = [&]() -> std::string {
        for (auto* label : Find<Sakura::Label>(range)) {
            return label->config().instances.front().str;
        }
        return {};
    };
    REQUIRE(bench.settle(update));
    CHECK(shown() == "0");

    precision = 3;
    CHECK(RepaintOnly(bench.frames.run(update)));
    CHECK(shown() == "0.250");
    CHECK(Retains(bench.frames, update));

    integer = true;
    CHECK(RepaintOnly(bench.frames.run(update)));
    CHECK(shown() == "0");
    CHECK(bench.unchanged());
}

TEST_CASE("Node range inputs switch to a focused text editor on double click and commit typed values", kTags) {
    Typist typist;
    std::vector<F32> changes;
    Sakura::NodeRangeInput range;
    range.update({.id = "range-edit", .min = 0.0f, .max = 10.0f, .value = 4.0f,
                  .onChange = [&](F32 next) { changes.push_back(next); }});
    Host host(range);
    typist.place = [&] { host.place(typist.ctx, {0.0f, 0.0f, 200.0f, 21.0f}); };
    typist.root = &host;
    typist.frame();

    CHECK(host.send(Mouse(MouseEventType::Click, 50.0f, 10.0f, MouseButton::Left, 2)));
    CHECK(changes.empty());
    typist.frame();
    CHECK(FocusOwner(host) != nullptr);
    CHECK_FALSE(host.send(Mouse(MouseEventType::Move, 150.0f, 10.0f)));

    typist.type("12.5");
    typist.press(KeyCode::Enter);
    CHECK(changes == std::vector<F32>{12.5f});
    CHECK(FocusOwner(host) == nullptr);

    CHECK(host.send(Mouse(MouseEventType::Click, 100.0f, 10.0f)));
    CHECK(changes == std::vector<F32>{12.5f, 5.0f});
    host.send(Mouse(MouseEventType::Release, 100.0f, 10.0f));

    CHECK(host.send(Mouse(MouseEventType::Click, 50.0f, 10.0f, MouseButton::Left, 2)));
    typist.frame();
    typist.type("9");
    typist.press(KeyCode::Escape);
    CHECK(changes.size() == 2);
    CHECK(FocusOwner(host) == nullptr);
    CHECK(host.send(Mouse(MouseEventType::Click, 150.0f, 10.0f)));
    CHECK(changes.back() == 7.5f);
}

TEST_CASE("Stepped node float inputs nudge the value by the step when the buttons are released", kTags) {
    const Scene scene;
    std::vector<F32> changes;
    Sakura::NodeFloatInput input;
    input.update({.id = "stepped", .value = 1.0f, .step = 0.25f,
                  .onChange = [&](F32 next) { changes.push_back(next); }});
    Host host(input);
    host.place(scene.ctx, {0.0f, 0.0f, 200.0f, 40.0f});

    CHECK(host.click(20.0f, 30.0f));
    CHECK(host.click(180.0f, 30.0f));
    CHECK(changes == std::vector<F32>{0.75f, 1.25f});

    CHECK(host.send(Mouse(MouseEventType::Click, 20.0f, 30.0f)));
    CHECK(host.send(Mouse(MouseEventType::Release, 500.0f, 500.0f)));
    CHECK(changes == std::vector<F32>{0.75f, 1.25f, 0.75f});

    CHECK(host.send(Mouse(MouseEventType::Click, 180.0f, 30.0f)));
    host.send(Mouse(MouseEventType::Cancel, 180.0f, 30.0f));
    CHECK_FALSE(host.send(Mouse(MouseEventType::Release, 180.0f, 30.0f)));
    CHECK_FALSE(host.click(20.0f, 30.0f, MouseButton::Right));
    CHECK(changes.size() == 3);

    input.update({.id = "stepped", .value = 1.0f, .onChange = [&](F32 next) { changes.push_back(next); }});
    host.place(scene.ctx, {0.0f, 0.0f, 200.0f, 40.0f});
    CHECK(FocusOwner(host) == nullptr);
    CHECK(host.click(20.0f, 30.0f));
    CHECK(FocusOwner(host) != nullptr);
    CHECK(changes.size() == 3);
}

TEST_CASE("Stepped node float inputs edit the step size in place of the toggle label", kTags) {
    Typist typist;
    std::vector<F32> changes;
    std::vector<F32> steps;
    Sakura::NodeFloatInput input;
    input.update({.id = "step-edit", .value = 1.0f, .step = 0.25f,
                  .onChange = [&](F32 next) { changes.push_back(next); },
                  .onStepChange = [&](F32 next) { steps.push_back(next); }});
    Host host(input);
    typist.place = [&] { host.place(typist.ctx, {0.0f, 0.0f, 200.0f, 40.0f}); };
    typist.root = &host;
    typist.frame();

    CHECK(host.click(100.0f, 30.0f));
    CHECK(FocusOwner(host) == nullptr);
    typist.frame();
    CHECK(FocusOwner(host) != nullptr);

    typist.press(KeyCode::A, {.control = true});
    typist.type("0.5");
    typist.press(KeyCode::Enter);
    CHECK(steps == std::vector<F32>{0.5f});
    CHECK(changes.empty());
    CHECK(FocusOwner(host) == nullptr);

    CHECK(host.click(100.0f, 30.0f));
    typist.frame();
    typist.press(KeyCode::Escape);
    CHECK(steps.size() == 1);
    CHECK(host.click(180.0f, 30.0f));
    CHECK(changes == std::vector<F32>{1.25f});
}

TEST_CASE("Node float inputs select all on click and commit parsed values on enter", kTags) {
    Typist typist;
    std::vector<F32> changes;
    Sakura::NodeFloatInput input;
    input.update({.id = "float", .value = 1.5f, .unit = "dB", .onChange = [&](F32 next) { changes.push_back(next); }});
    Host host(input);
    typist.place = [&] { host.place(typist.ctx, {0.0f, 0.0f, 200.0f, 21.0f}); };
    typist.root = &host;
    typist.frame();

    const auto submit = [&](const std::string& text) {
        CHECK(host.click(50.0f, 10.0f));
        typist.frame();
        REQUIRE(FocusOwner(host) != nullptr);
        typist.type(text);
        typist.press(KeyCode::Enter);
        CHECK(FocusOwner(host) == nullptr);
    };

    submit("2.5e1");
    CHECK(changes == std::vector<F32>{25.0f});
    submit("4x2");
    CHECK(changes == std::vector<F32>{25.0f, 42.0f});
    submit("3*2/1");
    CHECK(changes == std::vector<F32>{25.0f, 42.0f, 321.0f});
    submit("1.50");
    submit("-");
    CHECK(changes.size() == 3);

    CHECK(host.click(50.0f, 10.0f));
    typist.frame();
    typist.type("7");
    typist.blur();
    CHECK(changes.size() == 3);
}

TEST_CASE("Node int and uint inputs commit parsed integers and reject what their type cannot hold", kTags) {
    Typist typist;
    std::vector<I64> ints;
    std::vector<U64> uints;
    Sakura::NodeIntInput integer;
    integer.update({.id = "int", .value = 64, .onChange = [&](I64 next) { ints.push_back(next); }});
    Sakura::NodeUIntInput unsignedInteger;
    unsignedInteger.update({.id = "uint", .value = 2048, .onChange = [&](U64 next) { uints.push_back(next); }});
    Host intHost(integer);
    Host uintHost(unsignedInteger);
    typist.place = [&] {
        intHost.place(typist.ctx, {0.0f, 0.0f, 200.0f, 21.0f});
        uintHost.place(typist.ctx, {0.0f, 40.0f, 200.0f, 21.0f});
    };
    typist.frame();

    const auto submit = [&](Host& host, F32 y, const std::string& text) {
        typist.root = &host;
        CHECK(host.click(50.0f, y));
        typist.frame();
        typist.type(text);
        typist.press(KeyCode::Enter);
    };

    submit(intHost, 10.0f, "-42");
    submit(intHost, 10.0f, "12.7");
    submit(intHost, 10.0f, "64");
    submit(intHost, 10.0f, "1e3");
    CHECK(ints == std::vector<I64>{-42, 12, 13});
    submit(intHost, 10.0f, "99999999999999999999");
    submit(intHost, 10.0f, "-99999999999999999999");
    CHECK(ints == std::vector<I64>{-42, 12, 13});

    submit(uintHost, 50.0f, "-5");
    submit(uintHost, 50.0f, "4096");
    submit(uintHost, 50.0f, "2048");
    CHECK(uints == std::vector<U64>{4096});
}

TEST_CASE("Node text inputs commit on enter or focus loss according to their submit mode", kTags) {
    Typist typist;
    std::vector<std::string> changes;
    U64 deactivated = 0;
    Sakura::NodeTextInput input;
    const auto configure = [&](Sakura::NodeTextInput::Submit submit) {
        input.update({.id = "name", .value = "alpha", .hint = "Name", .submit = submit,
                      .onChange = [&](const std::string& value) { changes.push_back(value); },
                      .onDeactivate = [&] { ++deactivated; }});
    };
    configure(Sakura::NodeTextInput::Submit::OnEnter);
    Host host(input);
    typist.place = [&] { host.place(typist.ctx, {0.0f, 0.0f, 200.0f, 21.0f}); };
    typist.root = &host;
    typist.frame();
    CHECK_FALSE(input.focused());

    input.focus(true);
    CHECK(input.focused());
    typist.frame();
    typist.type("beta");
    typist.blur();
    CHECK_FALSE(input.focused());
    CHECK(changes.empty());
    CHECK(deactivated == 1);

    input.focus(true);
    typist.frame();
    typist.press(KeyCode::Enter);
    CHECK(changes == std::vector<std::string>{"alpha"});
    CHECK(deactivated == 2);

    changes.clear();
    configure(Sakura::NodeTextInput::Submit::OnCommit);
    input.focus(true);
    typist.frame();
    typist.type("beta");
    typist.blur();
    CHECK(changes == std::vector<std::string>{"beta"});
    CHECK(deactivated == 3);

    input.focus(true);
    typist.frame();
    typist.blur();
    CHECK(changes.size() == 1);

    input.focus(true);
    typist.frame();
    typist.type("gamma");
    typist.press(KeyCode::Escape);
    CHECK_FALSE(input.focused());
    CHECK(changes.size() == 1);

    input.focus(true);
    typist.frame();
    typist.type("delta");
    typist.press(KeyCode::Enter);
    CHECK(changes == std::vector<std::string>{"beta", "delta"});
    CHECK(deactivated == 6);
}

TEST_CASE("Node text inputs drop characters their filter rejects", kTags) {
    Typist typist;
    std::vector<std::string> changes;
    Sakura::NodeTextInput input;
    input.update({.id = "filtered",
                  .characterFilter = [](U32 character) { return character >= '0' && character <= '9'; },
                  .onChange = [&](const std::string& value) { changes.push_back(value); }});
    Host host(input);
    typist.place = [&] { host.place(typist.ctx, {0.0f, 0.0f, 200.0f, 21.0f}); };
    typist.root = &host;
    typist.frame();

    CHECK(host.click(20.0f, 10.0f));
    CHECK(input.focused());
    typist.frame();
    typist.type("a1b2 3");
    typist.press(KeyCode::Enter);
    CHECK(changes == std::vector<std::string>{"123"});
}

TEST_CASE("A focused node text input applies a replaced character filter to the next keystroke", kTags) {
    U32 accepted = 'a';
    std::vector<std::string> changes;
    std::vector<std::string> mirrored;
    Sakura::NodeTextInput input;
    Sakura::NodeTextInput mirror;
    const auto apply = [&](Sakura::NodeTextInput& target, std::vector<std::string>& log) {
        target.update({.id = "filtered",
                       .characterFilter = [accepted](U32 character) { return character == accepted; },
                       .onChange = [&log](const std::string& value) { log.push_back(value); }});
    };
    const auto update = [&] {
        apply(input, changes);
        apply(mirror, mirrored);
    };
    Place live(input, {0.0f, 0.0f, 200.0f, 21.0f});
    Place reference(mirror, {0.0f, 0.0f, 200.0f, 21.0f});
    Frames frames(live, reference);
    REQUIRE(frames.settle(update));
    for (Place* root : {&live, &reference}) {
        CHECK(root->send(Mouse(MouseEventType::Click, 20.0f, 10.0f)));
        root->send(Mouse(MouseEventType::Release, 20.0f, 10.0f));
    }
    REQUIRE(frames.settle(update));
    REQUIRE(input.focused());
    Type(live, "ab");
    Type(reference, "ab");
    REQUIRE(frames.settle(update));

    accepted = 'b';
    CHECK(frames.run(update).clean());
    Type(live, "ab");
    Type(reference, "ab");
    frames.run(update);
    Tap(live, KeyCode::Enter);
    Tap(reference, KeyCode::Enter);
    frames.run(update);
    CHECK(changes == std::vector<std::string>{"ab"});
    CHECK(mirrored == changes);
}

TEST_CASE("Node field help tooltips refresh their text under a resting pointer", kTags) {
    Probe child({0.0f, 21.0f});
    FlowgraphFieldTooltip help;
    Sakura::NodeField field;
    Bench bench(field, {100.0f, 100.0f, 300.0f, 34.25f});
    bench.frames.verify = false;
    bench.frames.timeStep = 0.1;
    bench.frames.ctx.hoverDelay = 0.25f;
    std::string text = "Gain in decibels";
    const auto update = [&] {
        help.update("gain", text);
        field.update({.id = "gain", .label = "Gain", .tooltip = help.overlay(), .content = &child});
    };
    const auto draws = [&](const std::string& str) {
        for (const auto& leaf : Drawn(bench.host)) {
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
    REQUIRE(bench.settle(update));

    bench.pointer.send(Mouse(MouseEventType::Move, 110.0f, 106.0f));
    for (U64 frame = 0; frame < 8; ++frame) {
        bench.frames.run(update);
    }
    REQUIRE(draws("Gain in decibels"));
    REQUIRE(bench.frames.run(update).clean());

    text = "Gain in volts";
    bench.frames.run(update);
    bench.frames.run(update);
    CHECK(draws("Gain in volts"));
    CHECK_FALSE(draws("Gain in decibels"));
    CHECK(Retains(bench.frames, update));
}

TEST_CASE("Node combos open a frame height dropdown below and report the chosen row", kTags) {
    const Scene scene;
    std::vector<std::string> events;
    Sakura::NodeCombo combo;
    combo.update({
        .id = "mode",
        .options = {"FM", "AM", "USB"},
        .value = "FM",
        .onChange = [&](const std::string& value) { events.push_back("change:" + value); },
        .onSelect = [&](U64 index) { events.push_back("select:" + std::to_string(index)); },
    });
    LayerSlot slot;
    slot.hold(combo, {100.0f, 100.0f, 200.0f, 25.0f});
    LayerHost host({&slot});
    host.place(scene.ctx, kScreen);
    CHECK_FALSE(host.popups.covers({600.0f, 500.0f}));

    CHECK_FALSE(host.click(150.0f, 123.0f));
    host.place(scene.ctx, kScreen);
    CHECK_FALSE(host.popups.covers({600.0f, 500.0f}));

    CHECK(host.click(150.0f, 110.0f));
    host.place(scene.ctx, kScreen);
    CHECK(host.popups.interactiveAt({100.0f, 121.0f}));
    CHECK(host.popups.interactiveAt({299.0f, 195.0f}));
    CHECK_FALSE(host.popups.interactiveAt({150.0f, 197.0f}));
    CHECK_FALSE(host.popups.interactiveAt({150.0f, 120.0f}));

    CHECK(host.click(150.0f, 158.0f));
    CHECK(events == std::vector<std::string>{"select:1", "change:AM"});
    host.place(scene.ctx, kScreen);
    CHECK_FALSE(host.popups.covers({600.0f, 500.0f}));
}

TEST_CASE("Node combos toggle their dropdown on repeated clicks and close when dismissed outside", kTags) {
    const Scene scene;
    U64 changes = 0;
    Sakura::NodeCombo combo;
    combo.update({.id = "mode", .options = {"FM", "AM"}, .value = "AM",
                  .onChange = [&](const std::string&) { ++changes; }});
    LayerSlot slot;
    slot.hold(combo, {100.0f, 100.0f, 200.0f, 21.0f});
    LayerHost host({&slot});
    host.place(scene.ctx, kScreen);

    host.click(150.0f, 110.0f);
    host.place(scene.ctx, kScreen);
    REQUIRE(host.popups.covers({600.0f, 500.0f}));
    host.click(150.0f, 110.0f);
    host.place(scene.ctx, kScreen);
    CHECK_FALSE(host.popups.covers({600.0f, 500.0f}));

    CHECK_FALSE(host.click(150.0f, 110.0f, MouseButton::Right));
    host.place(scene.ctx, kScreen);
    CHECK_FALSE(host.popups.covers({600.0f, 500.0f}));

    host.click(150.0f, 110.0f);
    host.place(scene.ctx, kScreen);
    REQUIRE(host.popups.covers({600.0f, 500.0f}));
    host.click(600.0f, 500.0f);
    host.place(scene.ctx, kScreen);
    CHECK_FALSE(host.popups.covers({600.0f, 500.0f}));
    CHECK(changes == 0);
}

TEST_CASE("Node path inputs browse from the square button and accept typed paths", kTags) {
    Typist typist;
    U64 browsed = 0;
    std::vector<std::string> changes;
    Sakura::NodePathInput path;
    path.update({.id = "path", .onChange = [&](const std::string& value) { changes.push_back(value); },
                 .onBrowse = [&] { ++browsed; }});
    Host host(path);
    typist.place = [&] { host.place(typist.ctx, {0.0f, 0.0f, 300.0f, 21.0f}); };
    typist.root = &host;
    typist.frame();

    CHECK(host.click(289.0f, 10.0f));
    CHECK(browsed == 1);
    CHECK(FocusOwner(host) == nullptr);
    CHECK_FALSE(host.click(275.0f, 10.0f));
    CHECK(host.send(Mouse(MouseEventType::Click, 280.0f, 10.0f)));
    CHECK_FALSE(host.send(Mouse(MouseEventType::Release, 260.0f, 10.0f)));
    CHECK(browsed == 1);

    CHECK(host.click(100.0f, 10.0f));
    CHECK(FocusOwner(host) != nullptr);
    typist.frame();
    typist.type("/tmp/a.bin");
    typist.press(KeyCode::Enter);
    CHECK(changes == std::vector<std::string>{"/tmp/a.bin"});
    CHECK(browsed == 1);
}

TEST_CASE("Node path inputs show their tooltip only while hovered with a value", kTags) {
    const Scene scene;
    auto ctx = scene.ctx;
    Probe tip({100.0f, 20.0f});
    Sakura::NodePathInput path;
    LayerSlot slot;
    slot.hold(path, {0.0f, 100.0f, 300.0f, 21.0f});
    LayerHost host({&slot});

    path.update({.id = "path", .value = "/tmp/capture.bin", .tooltip = {.content = [&] { return &tip; }}});
    host.place(ctx, kScreen);
    const auto hover = [&](F32 x, F32 y) {
        ctx.mouse = Extent2D<F32>{x, y};
        host.send(Mouse(MouseEventType::Move, x, y));
        host.place(ctx, kScreen);
    };
    hover(50.0f, 110.0f);
    CHECK(tip.layouts == 1);
    CHECK(tip.visible());
    hover(290.0f, 110.0f);
    CHECK(tip.layouts == 2);
    CHECK(tip.laidFrame == Rect{302.0f, 146.0f, 100.0f, 20.0f});

    hover(50.0f, 130.0f);
    CHECK_FALSE(tip.visible());

    path.update({.id = "path", .tooltip = {.content = [&] { return &tip; }}});
    hover(50.0f, 110.0f);
    CHECK_FALSE(tip.visible());
}

TEST_CASE("Node path inputs show a lazily prepared tooltip and move it with the pointer", kTags) {
    Probe tip({100.0f, 20.0f});
    bool asked = false;
    bool ready = false;
    Sakura::NodePathInput path;
    Bench bench(path, {0.0f, 100.0f, 300.0f, 21.0f});
    bench.frames.verify = false;
    bench.frames.timeStep = 0.1;
    bench.frames.ctx.hoverDelay = 0.25f;
    const auto update = [&] {
        ready = ready || std::exchange(asked, false);
        path.update({.id = "path", .value = "/tmp/capture.bin", .tooltip = {.content = [&]() -> Sakura::Component* {
            asked = true;
            return ready ? &tip : nullptr;
        }}});
    };
    REQUIRE(bench.settle(update));

    bench.pointer.send(Mouse(MouseEventType::Move, 50.0f, 110.0f));
    for (U64 frame = 0; frame < 8; ++frame) {
        bench.frames.run(update);
    }
    REQUIRE(tip.visible());
    const Rect entered = tip.currentFrame();
    bench.pointer.send(Mouse(MouseEventType::Move, 60.0f, 110.0f));
    bench.frames.run(update);
    CHECK(tip.currentFrame().x == entered.x + 10.0f);
    CHECK(Retains(bench.frames, update));
}

TEST_CASE("Node input rows stack their fields when the minimum widths no longer fit", kTags) {
    const Scene scene;
    auto ctx = scene.ctx;
    Sakura::NodeInputRow row;
    const Sakura::NodeInputRow::Combo dtype{.options = {"CF32", "F32"}, .value = "CF32", .width = 64.0f};
    row.update({.id = "row", .onAxesChange = [](const std::string&) {}, .combos = {dtype}});
    Host host(row);

    CHECK(host.measureOf(ctx, {224.0f, kInfinite}) == Extent2D<F32>{224.0f, 21.0f});
    CHECK(host.measureOf(ctx, {223.0f, kInfinite}) == Extent2D<F32>{223.0f, 69.0f});

    row.update({.id = "row", .combos = {dtype}});
    CHECK(host.measureOf(ctx, {128.0f, kInfinite}) == Extent2D<F32>{128.0f, 21.0f});
    CHECK(host.measureOf(ctx, {127.0f, kInfinite}) == Extent2D<F32>{127.0f, 45.0f});

    row.update({.id = "row"});
    CHECK(host.measureOf(ctx, {56.0f, kInfinite}).y == 21.0f);
    CHECK(host.measureOf(ctx, {55.0f, kInfinite}).y == 21.0f);

    row.update({.id = "row", .shapeMinWidth = 100.0f, .onAxesChange = [](const std::string&) {}, .combos = {dtype}});
    CHECK(host.measureOf(ctx, {267.0f, kInfinite}).y == 69.0f);
    CHECK(host.measureOf(ctx, {268.0f, kInfinite}).y == 21.0f);

    ctx.pixelRatio = 2.0f;
    row.update({.id = "row", .onAxesChange = [](const std::string&) {}, .combos = {dtype}});
    CHECK(host.measureOf(ctx, {447.0f, kInfinite}) == Extent2D<F32>{447.0f, 138.0f});
    CHECK(host.measureOf(ctx, {448.0f, kInfinite}) == Extent2D<F32>{448.0f, 42.0f});
}

TEST_CASE("Node input row combos sit at the right end inline and on their own rows when stacked", kTags) {
    const Scene scene;
    std::vector<std::string> changes;
    Sakura::NodeInputRow row;
    row.update({.id = "row", .shape = "[8, 1024]", .axes = "[0]", .onAxesChange = [](const std::string&) {},
                .combos = {{.options = {"CF32", "F32"}, .value = "CF32", .width = 64.0f,
                            .onChange = [&](const std::string& value) { changes.push_back(value); }}}});
    LayerSlot slot;
    LayerHost host({&slot});

    slot.hold(row, {0.0f, 0.0f, 300.0f, 21.0f});
    host.place(scene.ctx, kScreen);
    CHECK_FALSE(host.click(230.0f, 10.0f));
    host.place(scene.ctx, kScreen);
    CHECK_FALSE(host.popups.covers({600.0f, 500.0f}));
    CHECK(host.click(240.0f, 10.0f));
    host.place(scene.ctx, kScreen);
    CHECK(host.popups.interactiveAt({240.0f, 22.0f}));
    CHECK(host.click(250.0f, 48.0f));
    CHECK(changes == std::vector<std::string>{"F32"});
    host.place(scene.ctx, kScreen);
    CHECK_FALSE(host.popups.covers({600.0f, 500.0f}));

    slot.hold(row, {0.0f, 0.0f, 200.0f, 69.0f});
    host.place(scene.ctx, kScreen);
    CHECK_FALSE(host.click(100.0f, 45.0f));
    host.place(scene.ctx, kScreen);
    CHECK_FALSE(host.popups.covers({600.0f, 500.0f}));
    CHECK(host.click(20.0f, 58.0f));
    host.place(scene.ctx, kScreen);
    CHECK(host.popups.interactiveAt({20.0f, 70.0f}));
    CHECK_FALSE(host.popups.interactiveAt({20.0f, 68.0f}));
}

TEST_CASE("Node input row shape and axes fields commit when they lose focus", kTags) {
    Typist typist;
    std::vector<std::string> shapes;
    std::vector<std::string> axes;
    Sakura::NodeInputRow row;
    row.update({.id = "row-commit", .shapeHint = "Shape",
                .onShapeChange = [&](const std::string& value) { shapes.push_back(value); },
                .axesHint = "Axes",
                .onAxesChange = [&](const std::string& value) { axes.push_back(value); }});
    Host host(row);
    typist.place = [&] { host.place(typist.ctx, {0.0f, 0.0f, 300.0f, 21.0f}); };
    typist.root = &host;
    typist.frame();

    CHECK(host.click(20.0f, 10.0f));
    typist.frame();
    typist.type("4,2");
    typist.blur();
    CHECK(shapes == std::vector<std::string>{"4,2"});
    CHECK(axes.empty());

    CHECK(host.click(250.0f, 10.0f));
    typist.frame();
    typist.type("1");
    typist.press(KeyCode::Enter);
    CHECK(axes == std::vector<std::string>{"1"});

    CHECK(host.click(20.0f, 10.0f));
    typist.frame();
    typist.type("8");
    typist.press(KeyCode::Escape);
    CHECK(shapes.size() == 1);
}

TEST_CASE("Node label values stay one line tall for long, unbroken and multi-line text at any width", kTags) {
    const Scene scene;
    Sakura::NodeLabel label;
    Host host(label);

    std::string value;
    SECTION("words") {
        value = "AAAA AAAA AAAA AAAA AAAA";
    }
    SECTION("unbroken identifier") {
        value = std::string(40, 'A');
    }
    SECTION("explicit lines and surrounding whitespace") {
        value = "AAAA AAAA AAAA   \nAA\n\nAAAA";
    }
    SECTION("device name") {
        value = "LimeSDR Mini [USB 3.0] 1D4249D8AEC2A9";
    }
    SECTION("short value") {
        value = "AAAA";
    }

    Sakura::TextMetrics metrics;
    metrics.setWindow(scene.ctx.render);

    for (const auto align : {Sakura::NodeLabel::Align::Left, Sakura::NodeLabel::Align::Right,
                             Sakura::NodeLabel::Align::Center}) {
        label.update({.id = "metric", .str = value, .align = align});
        for (const F32 width : {80.0f, 150.0f, 500.0f, 150.0f, 0.0f}) {
            CAPTURE(width);
            CHECK(host.measureOf(scene.ctx, {width, kInfinite}) == Extent2D<F32>{width, 15.0f});
            host.place(scene.ctx, {0.0f, 0.0f, width, 15.0f});
            const auto labels = Find<Sakura::Label>(label);
            REQUIRE(labels.size() == 1);
            const auto& config = labels.front()->config();
            REQUIRE(config.instances.size() == 1);
            const auto& instance = config.instances.front();
            CHECK(instance.str == value.substr(0, value.find('\n')));
            CHECK(instance.rect == Rect{0.0f, 0.0f, width, 15.0f});
            const bool overflows = metrics.measure(config.fontName, instance.str, instance.fontSize) > width;
            const I32 alignment = overflows || align == Sakura::NodeLabel::Align::Left ? 0
                                  : align == Sakura::NodeLabel::Align::Right ? 2 : 1;
            CHECK(instance.alignment == Extent2D<I32>{alignment, 1});
        }
    }
}

TEST_CASE("Node labels inside a node keep the node height when the value grows", kTags) {
    const Scene scene;
    Sakura::NodeLabel label;
    Sakura::NodeField field;
    field.update({.id = "field", .label = "Device", .background = false, .content = &label});
    Sakura::NodeContentStack stack;
    stack.update({.id = "stack", .items = {{.child = &field}}});
    Sakura::NodeFieldPanel panel;
    panel.update({.id = "panel", .content = &stack});
    Host host(panel);

    label.update({.id = "metric", .str = "AAAA", .align = Sakura::NodeLabel::Align::Right});
    const auto compact = host.measureOf(scene.ctx, {150.0f, kInfinite});
    CHECK(compact == Extent2D<F32>{150.0f, 30.25f});

    label.update({.id = "metric", .str = "LimeSDR Mini [USB 3.0] 1D4249D8AEC2A9",
                  .align = Sakura::NodeLabel::Align::Right});
    CHECK(host.measureOf(scene.ctx, {150.0f, kInfinite}) == compact);
    CHECK(host.measureOf(scene.ctx, {500.0f, kInfinite}) == Extent2D<F32>{500.0f, 30.25f});
    CHECK(host.measureOf(scene.ctx, {150.0f, kInfinite}) == compact);
}

TEST_CASE("Boxed node labels reserve an eight pixel inset around one line at every pixel ratio", kTags) {
    const Scene scene;
    for (const F32 scale : {1.0f, 2.0f}) {
        auto ctx = scene.ctx;
        ctx.pixelRatio = scale;
        Sakura::NodeLabel label;
        label.update({.id = "error", .str = "No metric", .colorKey = "text_secondary",
                      .align = Sakura::NodeLabel::Align::Center, .boxed = true});
        Host host(label);
        Sakura::TextMetrics metrics;
        metrics.setWindow(ctx.render);
        for (const F32 width : {320.0f, 120.0f, 40.0f, 320.0f}) {
            CAPTURE(scale, width);
            const auto size = host.measureOf(ctx, {width, kInfinite});
            CHECK(size == Extent2D<F32>{width, 15.0f * scale + 16.0f * scale});
            host.place(ctx, {0.0f, 0.0f, width, size.y});
            const auto labels = Find<Sakura::Label>(label);
            REQUIRE(labels.size() == 1);
            const auto& config = labels.front()->config();
            REQUIRE(config.instances.size() == 1);
            const auto& instance = config.instances.front();
            CHECK(instance.str == "No metric");
            CHECK(instance.rect == Rect{8.0f * scale, 8.0f * scale,
                                        width - 16.0f * scale, 15.0f * scale});
            CHECK(instance.rect.x + instance.rect.width * 0.5f == width * 0.5f);
            REQUIRE(config.clip.has_value());
            CHECK(*config.clip == instance.rect);
            const bool fits = metrics.measure(config.fontName, instance.str, instance.fontSize) <= instance.rect.width;
            CHECK(instance.alignment == Extent2D<I32>{fits ? 1 : 0, 1});
        }
    }
}

TEST_CASE("Node progress bars and tables measure to fixed row heights", kTags) {
    const Scene scene;
    auto ctx = scene.ctx;
    Sakura::NodeProgressBar bar;
    bar.update({.id = "progress", .value = 1.5f, .overlay = "150%"});
    CHECK(MeasureOf(ctx, bar, {200.0f, kInfinite}) == Extent2D<F32>{200.0f, 25.0f});

    Sakura::NodeTable table;
    Host host(table);
    table.update({.id = "table"});
    CHECK(host.measureOf(ctx, {300.0f, kInfinite}) == Extent2D<F32>{300.0f, 0.0f});
    table.update({.id = "table", .columns = {"Key", "Value"}, .rows = {{"rate", "2 MHz"}, {"gain"}}});
    CHECK(host.measureOf(ctx, {300.0f, kInfinite}) == Extent2D<F32>{300.0f, 69.0f});
    host.place(ctx, {0.0f, 0.0f, 300.0f, 69.0f});
    table.update({.id = "table", .columns = {"Key", "Value"}, .rows = {{"rate", "2 MHz"}, {"gain"}},
                  .showHeaders = false});
    CHECK(host.measureOf(ctx, {300.0f, kInfinite}) == Extent2D<F32>{300.0f, 46.0f});

    ctx.pixelRatio = 2.0f;
    CHECK(MeasureOf(ctx, bar, {200.0f, kInfinite}) == Extent2D<F32>{200.0f, 50.0f});
    CHECK(host.measureOf(ctx, {300.0f, kInfinite}) == Extent2D<F32>{300.0f, 92.0f});
    host.place(ctx, {0.0f, 0.0f, 300.0f, 92.0f});
}

TEST_CASE("Node surfaces report their size and detach only from the hovered corner button", kTags) {
    const Scene scene;
    auto ctx = scene.ctx;
    std::vector<Sakura::SurfaceResize> sizes;
    U64 detached = 0;
    Sakura::NodeSurface surface;
    const auto configure = [&](bool overlay) {
        surface.update({.id = "surface", .detachOverlay = overlay,
                        .onSize = [&](const Sakura::SurfaceResize& size) { sizes.push_back(size); },
                        .onDetach = [&] { ++detached; }});
    };
    configure(true);
    Host host(surface);

    CHECK(host.measureOf(ctx, {300.0f, kInfinite}) == Extent2D<F32>{300.0f, 150.0f});
    host.place(ctx, {0.0f, 0.0f, 300.0f, 150.0f});
    host.send(Mouse(MouseEventType::Move, 150.0f, 75.0f));
    host.place(ctx, {0.0f, 0.0f, 300.0f, 150.0f});
    REQUIRE(sizes.size() == 1);
    CHECK(sizes[0].logicalSize == Extent2D<U64>{300, 150});
    CHECK(sizes[0].framebufferSize == Extent2D<U64>{300, 150});
    host.place(ctx, {0.0f, 0.0f, 300.0f, 150.0f});
    CHECK(sizes.size() == 1);
    host.place(ctx, {0.0f, 0.0f, 240.0f, 150.0f});
    REQUIRE(sizes.size() == 2);
    CHECK(sizes[1].logicalSize == Extent2D<U64>{240, 150});
    host.place(ctx, {0.0f, 0.0f, 300.0f, 150.0f});

    CHECK_FALSE(host.send(Mouse(MouseEventType::Click, 150.0f, 75.0f)));
    CHECK_FALSE(host.send(Mouse(MouseEventType::Click, 280.0f, 20.0f, MouseButton::Right)));
    CHECK(detached == 0);
    CHECK(host.send(Mouse(MouseEventType::Click, 280.0f, 20.0f)));
    CHECK(detached == 1);

    host.send(Mouse(MouseEventType::Leave, 150.0f, 75.0f));
    host.place(ctx, {0.0f, 0.0f, 300.0f, 150.0f});
    CHECK_FALSE(host.send(Mouse(MouseEventType::Click, 280.0f, 20.0f)));

    host.send(Mouse(MouseEventType::Move, 150.0f, 75.0f));
    configure(false);
    host.place(ctx, {0.0f, 0.0f, 300.0f, 150.0f});
    CHECK_FALSE(host.send(Mouse(MouseEventType::Click, 280.0f, 20.0f)));
    CHECK(detached == 1);

    surface.update({.id = "surface", .minimumHeight = 90.0f});
    CHECK(host.measureOf(ctx, {300.0f, kInfinite}) == Extent2D<F32>{300.0f, 90.0f});
}

TEST_CASE("Node text ellipsizing keeps fitting text and cuts long text on a character boundary", kTags) {
    const Scene scene;
    Sakura::TextMetrics metrics;
    metrics.setWindow(scene.ctx.render);
    const std::string font = Sakura::Typography::MonoFont;
    const F32 glyph = metrics.measure(font, "A", 15.0f);
    REQUIRE(glyph > 0.0f);
    REQUIRE(metrics.measure(font, "AAAAA", 15.0f) == glyph * 5.0f);

    CHECK(metrics.ellipsize(font, "abc", 15.0f, 1000.0f) == "abc");
    CHECK(metrics.ellipsize(font, "abcde", 15.0f, glyph * 5.0f) == "abcde");

    const F32 width = glyph * 5.0f + 0.5f;
    const std::string cut = metrics.ellipsize(font, "abcdefghij", 15.0f, width);
    CHECK(cut == "ab...");
    CHECK(metrics.measure(font, cut, 15.0f) <= width);
    CHECK(metrics.ellipsize(font, "abcdefghij", 15.0f, glyph) == "...");

    const std::string accents = "\xC3\xA9\xC3\xA9\xC3\xA9\xC3\xA9\xC3\xA9\xC3\xA9";
    const std::string accented = metrics.ellipsize(font, accents, 15.0f, width);
    REQUIRE(accented.size() >= 3);
    CHECK(accented.substr(accented.size() - 3) == "...");
    CHECK((accented.size() - 3) % 2 == 0);
}

TEST_CASE("Hidden node grid items give up their column slot and gap until shown again", kTags) {
    const auto ctx = MakeContext();
    Probe a({0.0f, 21.0f});
    Probe b({0.0f, 30.0f});
    Probe c({0.0f, 10.0f});
    Sakura::NodeFieldGrid grid;
    grid.update({.id = "grid", .items = {{.child = &a}, {.child = &b}, {.child = &c}}});
    Host host(grid);

    b.setVisible(false);
    CHECK(host.measureOf(ctx, {300.0f, kInfinite}) == Extent2D<F32>{300.0f, 21.0f});
    host.place(ctx, {0.0f, 0.0f, 300.0f, 21.0f});
    CHECK(a.laidFrame == Rect{0.0f, 0.0f, 149.0f, 21.0f});
    CHECK(c.laidFrame == Rect{151.0f, 0.0f, 149.0f, 21.0f});
    CHECK(b.layouts == 0);

    CHECK(host.measureOf(ctx, {281.0f, kInfinite}) == Extent2D<F32>{281.0f, 33.0f});
    host.place(ctx, {0.0f, 0.0f, 281.0f, 33.0f});
    CHECK(a.laidFrame == Rect{0.0f, 0.0f, 281.0f, 21.0f});
    CHECK(c.laidFrame == Rect{0.0f, 23.0f, 281.0f, 10.0f});

    a.setVisible(false);
    CHECK(host.measureOf(ctx, {300.0f, kInfinite}) == Extent2D<F32>{300.0f, 10.0f});
    host.place(ctx, {0.0f, 0.0f, 300.0f, 10.0f});
    CHECK(c.laidFrame == Rect{0.0f, 0.0f, 300.0f, 10.0f});

    c.setVisible(false);
    CHECK(host.measureOf(ctx, {300.0f, kInfinite}) == Extent2D<F32>{300.0f, 0.0f});
    CHECK(host.measureOf(ctx, {281.0f, kInfinite}) == Extent2D<F32>{281.0f, 0.0f});

    for (auto* probe : {&a, &b, &c}) {
        probe->setVisible(true);
    }
    CHECK(host.measureOf(ctx, {300.0f, kInfinite}) == Extent2D<F32>{300.0f, 42.0f});
    host.place(ctx, {0.0f, 0.0f, 300.0f, 42.0f});
    CHECK(a.laidFrame == Rect{0.0f, 0.0f, 149.0f, 30.0f});
    CHECK(b.laidFrame == Rect{151.0f, 0.0f, 149.0f, 30.0f});
    CHECK(c.laidFrame == Rect{0.0f, 32.0f, 149.0f, 10.0f});
    CHECK(b.layouts == 1);
}

TEST_CASE("A hidden full width node grid item lets the columns on either side merge", kTags) {
    const auto ctx = MakeContext();
    Probe a({0.0f, 21.0f});
    Probe b({0.0f, 30.0f});
    Probe c({0.0f, 10.0f});
    Probe d({0.0f, 21.0f});
    Sakura::NodeFieldGrid grid;
    grid.update({.id = "grid", .items = {
        {.child = &a},
        {.child = &b, .fullWidth = true},
        {.child = &c},
        {.child = &d},
    }});
    Host host(grid);

    b.setVisible(false);
    CHECK(host.measureOf(ctx, {300.0f, kInfinite}) == Extent2D<F32>{300.0f, 44.0f});
    host.place(ctx, {0.0f, 0.0f, 300.0f, 44.0f});
    CHECK(a.laidFrame == Rect{0.0f, 0.0f, 149.0f, 21.0f});
    CHECK(c.laidFrame == Rect{151.0f, 0.0f, 149.0f, 21.0f});
    CHECK(d.laidFrame == Rect{0.0f, 23.0f, 149.0f, 21.0f});

    b.setVisible(true);
    CHECK(host.measureOf(ctx, {300.0f, kInfinite}) == Extent2D<F32>{300.0f, 76.0f});
    host.place(ctx, {0.0f, 0.0f, 300.0f, 76.0f});
    CHECK(a.laidFrame == Rect{0.0f, 0.0f, 300.0f, 21.0f});
    CHECK(b.laidFrame == Rect{0.0f, 23.0f, 300.0f, 30.0f});
    CHECK(c.laidFrame == Rect{0.0f, 55.0f, 149.0f, 21.0f});
    CHECK(d.laidFrame == Rect{151.0f, 55.0f, 149.0f, 21.0f});
}

TEST_CASE("Node text inputs settle, ignore mouse moves and repaint only when their value or hint changes",
          kRetentionTags) {
    Sakura::NodeTextInput input;
    Bench bench(input, {10.0f, 10.0f, 240.0f, 19.5f});
    std::string value = "alpha";
    std::string hint = "Name";
    const auto update = [&] {
        input.update({.id = "name", .value = value, .hint = hint, .unit = "Hz",
                      .onChange = [&](const std::string& next) { value = next; }});
    };
    REQUIRE(bench.settle(update));
    CHECK(QuietMoves(bench.frames, update, {{20.0f, 15.0f}, {200.0f, 20.0f}, {600.0f, 400.0f}}, bench.sender()));

    value = "beta";
    CheckRepaintOnly(bench.frames, update);
    value.clear();
    CheckRepaintOnly(bench.frames, update);
    hint = "Label";
    CheckRepaintOnly(bench.frames, update);
    CHECK(bench.unchanged());
}

TEST_CASE("Node float inputs settle and repaint only when their value or unit changes", kRetentionTags) {
    Sakura::NodeFloatInput input;
    Bench bench(input, {10.0f, 10.0f, 240.0f, 19.5f});
    F32 value = 1.5f;
    std::string unit = "dB";
    const auto update = [&] {
        input.update({.id = "gain", .value = value, .unit = unit, .onChange = [&](F32 next) { value = next; }});
    };
    REQUIRE(bench.settle(update));
    CHECK(QuietMoves(bench.frames, update, {{20.0f, 15.0f}, {600.0f, 400.0f}}, bench.sender()));

    value = 2.25f;
    CheckRepaintOnly(bench.frames, update);
    unit = "Hz";
    CheckRepaintOnly(bench.frames, update);
    CHECK(bench.unchanged());
}

TEST_CASE("Stepped node float input buttons repaint once on hover and once when a cancelled press clears it",
          kRetentionTags) {
    Sakura::NodeFloatInput input;
    Bench bench(input, {10.0f, 10.0f, 240.0f, 37.0f});
    F32 value = 1.0f;
    const auto update = [&] {
        input.update({.id = "stepped", .value = value, .step = 0.25f,
                      .onChange = [&](F32 next) { value = next; }});
    };
    REQUIRE(bench.settle(update));

    for (const F32 x : {30.0f, 130.0f, 230.0f}) {
        CAPTURE(x);
        CheckHoverRepaintsOnce(bench.frames, bench.pointer, update, {x, 38.0f}, {600.0f, 400.0f});
    }
    CheckLeaveRepaintsOnce(bench.frames, bench.pointer, update, {30.0f, 38.0f});

    bench.pointer.send(Mouse(MouseEventType::Move, 30.0f, 38.0f));
    REQUIRE(bench.frames.settle(update));
    bench.pointer.send(Mouse(MouseEventType::Click, 30.0f, 38.0f));
    CHECK(bench.frames.run(update).clean());
    bench.pointer.send(Mouse(MouseEventType::Cancel, 30.0f, 38.0f));
    CHECK(RepaintOnly(bench.frames.run(update)));
    CHECK(bench.frames.run(update).clean());
    bench.pointer.send(Mouse(MouseEventType::Release, 30.0f, 38.0f));
    CHECK(bench.frames.run(update).clean());
    CHECK(value == 1.0f);
    bench.pointer.send(Mouse(MouseEventType::Move, 30.0f, 38.0f));
    CHECK(RepaintOnly(bench.frames.run(update)));
    CHECK(bench.frames.run(update).clean());

    bench.pointer.send(Mouse(MouseEventType::Click, 30.0f, 38.0f));
    bench.pointer.send(Mouse(MouseEventType::Release, 30.0f, 38.0f));
    CHECK(value == 0.75f);
    CheckRepaintOnly(bench.frames, update);
    CHECK(bench.unchanged());
}

TEST_CASE("Node int and uint inputs settle and repaint only when their value changes", kRetentionTags) {
    Sakura::NodeIntInput integer;
    Sakura::NodeUIntInput unsignedInteger;
    Sakura::NodeContentStack stack;
    Bench bench(stack, {10.0f, 10.0f, 240.0f, 60.0f});
    I64 signedValue = -64;
    U64 unsignedValue = 2048;
    const auto update = [&] {
        integer.update({.id = "int", .value = signedValue, .onChange = [&](I64 next) { signedValue = next; }});
        unsignedInteger.update({.id = "uint", .value = unsignedValue,
                                .onChange = [&](U64 next) { unsignedValue = next; }});
        stack.update({.id = "stack", .items = {{.child = &integer}, {.child = &unsignedInteger}}});
    };
    REQUIRE(bench.settle(update));
    CHECK(QuietMoves(bench.frames, update, {{20.0f, 15.0f}, {20.0f, 45.0f}, {600.0f, 400.0f}}, bench.sender()));

    signedValue = 12;
    CheckRepaintOnly(bench.frames, update);
    unsignedValue = 4096;
    CheckRepaintOnly(bench.frames, update);
    CHECK(bench.unchanged());
}

TEST_CASE("Node bool inputs settle, show no press state and repaint only when toggled", kRetentionTags) {
    Sakura::NodeBoolInput toggle;
    Bench bench(toggle, {10.0f, 10.0f, 240.0f, 19.5f});
    bool value = false;
    const auto update = [&] {
        toggle.update({.id = "bool", .value = value, .onChange = [&](bool next) { value = next; }});
    };
    REQUIRE(bench.settle(update));
    CHECK(QuietMoves(bench.frames, update, {{20.0f, 15.0f}, {200.0f, 20.0f}, {600.0f, 400.0f}}, bench.sender()));

    bench.pointer.send(Mouse(MouseEventType::Click, 100.0f, 15.0f));
    CHECK(bench.frames.run(update).clean());
    bench.pointer.send(Mouse(MouseEventType::Cancel, 100.0f, 15.0f));
    CHECK(bench.frames.run(update).clean());
    CHECK_FALSE(value);

    bench.pointer.send(Mouse(MouseEventType::Click, 100.0f, 15.0f));
    bench.pointer.send(Mouse(MouseEventType::Release, 100.0f, 15.0f));
    REQUIRE(value);
    CheckRepaintOnly(bench.frames, update);
    CHECK(bench.unchanged());
}

TEST_CASE("Node range inputs settle and repaint only when a drag or the owner moves the value", kRetentionTags) {
    Sakura::NodeRangeInput range;
    Bench bench(range, {10.0f, 10.0f, 200.0f, 19.5f});
    F32 value = 4.0f;
    const auto update = [&] {
        range.update({.id = "range", .min = 0.0f, .max = 10.0f, .value = value, .unit = "dB",
                      .onChange = [&](F32 next) { value = next; }});
    };
    REQUIRE(bench.settle(update));
    CHECK(QuietMoves(bench.frames, update, {{60.0f, 15.0f}, {150.0f, 20.0f}, {600.0f, 400.0f}}, bench.sender()));

    bench.pointer.send(Mouse(MouseEventType::Click, 60.0f, 15.0f));
    CHECK(value == 2.5f);
    CheckRepaintOnly(bench.frames, update);
    bench.pointer.send(Mouse(MouseEventType::Move, 160.0f, 15.0f));
    CHECK(value == 7.5f);
    CheckRepaintOnly(bench.frames, update);
    bench.pointer.send(Mouse(MouseEventType::Move, 160.0f, 18.0f));
    CHECK(bench.frames.run(update).clean());
    bench.pointer.send(Mouse(MouseEventType::Cancel, 160.0f, 18.0f));
    CHECK(bench.frames.run(update).clean());
    CHECK(QuietMoves(bench.frames, update, {{60.0f, 15.0f}, {190.0f, 15.0f}}, bench.sender()));

    value = 1.0f;
    CheckRepaintOnly(bench.frames, update);
    CHECK(bench.unchanged());
}

TEST_CASE("Node combos settle, repaint once on hover enter and leave and when their value changes",
          kRetentionTags) {
    Sakura::NodeCombo combo;
    Bench bench(combo, {10.0f, 10.0f, 200.0f, 19.5f});
    std::string value = "AM";
    const auto update = [&] {
        combo.update({.id = "mode", .options = {"FM", "AM", "USB"}, .value = value,
                      .onChange = [&](const std::string& next) { value = next; }});
    };
    REQUIRE(bench.settle(update));

    CheckHoverRepaintsOnce(bench.frames, bench.pointer, update, {60.0f, 15.0f}, {600.0f, 400.0f});
    CheckLeaveRepaintsOnce(bench.frames, bench.pointer, update, {60.0f, 15.0f});

    bench.pointer.send(Mouse(MouseEventType::Move, 600.0f, 400.0f));
    REQUIRE(bench.frames.settle(update));
    value = "USB";
    CheckRepaintOnly(bench.frames, update);
    CHECK(bench.unchanged());
}

TEST_CASE("Node path inputs settle, ignore mouse moves without a tooltip and repaint only when their value changes",
          kRetentionTags) {
    Sakura::NodePathInput path;
    Bench bench(path, {10.0f, 10.0f, 240.0f, 19.5f});
    std::string value = "/tmp/a.bin";
    const auto update = [&] {
        path.update({.id = "path", .value = value, .onChange = [&](const std::string& next) { value = next; },
                     .onBrowse = [] {}});
    };
    REQUIRE(bench.settle(update));
    CHECK(QuietMoves(bench.frames, update, {{20.0f, 15.0f}, {240.0f, 15.0f}, {600.0f, 400.0f}}, bench.sender()));

    value = "/tmp/b.bin";
    CheckRepaintOnly(bench.frames, update);
    value.clear();
    CheckRepaintOnly(bench.frames, update);
    CHECK(bench.unchanged());
}

TEST_CASE("Node input rows settle, repaint once on combo hover and restructure once when a combo is added or removed",
          kRetentionTags) {
    Sakura::NodeInputRow row;
    Bench bench(row, {10.0f, 10.0f, 300.0f, 19.5f});
    std::string shape = "[8, 1024]";
    U64 combos = 1;
    const auto update = [&] {
        std::vector<Sakura::NodeInputRow::Combo> list;
        for (U64 index = 0; index < combos; ++index) {
            list.push_back({.options = {"CF32", "F32"}, .value = "CF32", .width = 64.0f});
        }
        row.update({.id = "row", .shape = shape, .axes = "[0]", .onAxesChange = [](const std::string&) {},
                    .combos = std::move(list)});
    };
    REQUIRE(bench.settle(update));
    CHECK(QuietMoves(bench.frames, update, {{20.0f, 15.0f}, {200.0f, 15.0f}, {600.0f, 400.0f}}, bench.sender()));
    CheckHoverRepaintsOnce(bench.frames, bench.pointer, update, {270.0f, 15.0f}, {600.0f, 400.0f});

    shape = "[4, 2]";
    CheckRepaintOnly(bench.frames, update);
    CHECK(bench.unchanged());

    combos = 2;
    CheckRestructuresOnce(bench.frames, update);
    combos = 1;
    CheckRepaintOnly(bench.frames, update);
    combos = 2;
    CheckRepaintOnly(bench.frames, update);
    CHECK(bench.frames.restructures == 1);
}

TEST_CASE("Node labels settle, repaint only for text and color changes and restructure once when boxed",
          kRetentionTags) {
    Sakura::NodeLabel label;
    Bench bench(label, {10.0f, 10.0f, 240.0f, 29.5f});
    std::string str = "Running";
    std::string colorKey = "text_primary";
    bool boxed = false;
    const auto update = [&] {
        label.update({.id = "status", .str = str, .colorKey = colorKey, .boxed = boxed});
    };
    REQUIRE(bench.settle(update));
    CHECK(QuietMoves(bench.frames, update, {{20.0f, 15.0f}, {600.0f, 400.0f}}, bench.sender()));

    str = "Stopped";
    CheckRepaintOnly(bench.frames, update);
    colorKey = "text_secondary";
    CheckRepaintOnly(bench.frames, update);
    CHECK(bench.unchanged());

    boxed = true;
    CheckRestructuresOnce(bench.frames, update);
}

TEST_CASE("Node progress bars settle and repaint only when their value, overlay or color changes",
          kRetentionTags) {
    Sakura::NodeProgressBar bar;
    Bench bench(bar, {10.0f, 10.0f, 200.0f, 23.5f});
    F32 value = 0.4f;
    std::string overlay = "40%";
    std::string colorKey = "button_active";
    const auto update = [&] {
        bar.update({.id = "progress", .value = value, .overlay = overlay, .colorKey = colorKey});
    };
    REQUIRE(bench.settle(update));
    CHECK(QuietMoves(bench.frames, update, {{20.0f, 15.0f}, {600.0f, 400.0f}}, bench.sender()));

    value = 0.6f;
    CheckRepaintOnly(bench.frames, update);
    overlay = "60%";
    CheckRepaintOnly(bench.frames, update);
    colorKey = "success_green";
    CheckRepaintOnly(bench.frames, update);
    CHECK(bench.unchanged());
}

TEST_CASE("Node tables settle, repaint only for cell and row changes and restructure once for a new column",
          kRetentionTags) {
    Sakura::NodeTable table;
    Bench bench(table, {10.0f, 10.0f, 300.0f, 120.0f});
    std::vector<std::string> columns = {"Key", "Value"};
    std::vector<std::vector<std::string>> rows = {{"rate", "2 MHz"}, {"gain", "10 dB"}};
    const auto update = [&] {
        table.update({.id = "table", .columns = columns, .rows = rows});
    };
    REQUIRE(bench.settle(update));
    CHECK(QuietMoves(bench.frames, update, {{20.0f, 15.0f}, {200.0f, 40.0f}, {600.0f, 400.0f}}, bench.sender()));

    rows[0][1] = "4 MHz";
    CheckRepaintOnly(bench.frames, update);
    rows.push_back({"mode", "FM"});
    CheckRepaintOnly(bench.frames, update);
    CHECK(bench.unchanged());

    columns.push_back("Unit");
    CheckRestructuresOnce(bench.frames, update);
}

TEST_CASE("Node fields build a new child in one restructure, swap children with a repaint and restructure once per title change",
          kRetentionTags) {
    Sakura::NodeLabel first;
    Sakura::NodeLabel second;
    Sakura::NodeField field;
    Bench bench(field, {10.0f, 10.0f, 240.0f, 40.0f});
    std::string label = "Gain";
    Sakura::Component* content = &first;
    const auto update = [&] {
        first.update({.id = "first", .str = "One"});
        second.update({.id = "second", .str = "Two"});
        field.update({.id = "field", .label = label, .content = content});
    };
    REQUIRE(bench.settle(update));
    CHECK(QuietMoves(bench.frames, update, {{20.0f, 15.0f}, {20.0f, 35.0f}, {600.0f, 400.0f}}, bench.sender()));

    label = "Level";
    CheckRepaintOnly(bench.frames, update);
    CHECK(bench.unchanged());

    content = &second;
    CheckRestructuresOnce(bench.frames, update);
    REQUIRE(bench.settle(update));
    content = &first;
    CheckRepaintOnly(bench.frames, update);
    content = &second;
    CheckRepaintOnly(bench.frames, update);
    CHECK(bench.unchanged());

    label.clear();
    CheckRestructuresOnce(bench.frames, update);
    label = "Gain";
    CheckRestructuresOnce(bench.frames, update);
}

TEST_CASE("Node field grids park dropped items with a repaint and restructure once for a new item",
          kRetentionTags) {
    Sakura::NodeLabel a;
    Sakura::NodeLabel b;
    Sakura::NodeLabel c;
    Sakura::NodeLabel d;
    Sakura::NodeFieldGrid grid;
    Bench bench(grid, {10.0f, 10.0f, 300.0f, 60.0f});
    bool withB = true;
    bool withD = false;
    const auto update = [&] {
        a.update({.id = "a", .str = "A"});
        b.update({.id = "b", .str = "B"});
        c.update({.id = "c", .str = "C"});
        d.update({.id = "d", .str = "D"});
        std::vector<Sakura::NodeFieldGrid::Item> items = {{.child = &a}};
        if (withB) {
            items.push_back({.child = &b});
        }
        items.push_back({.child = &c, .fullWidth = true});
        if (withD) {
            items.push_back({.child = &d});
        }
        grid.update({.id = "grid", .items = std::move(items)});
    };
    REQUIRE(bench.settle(update));

    withB = false;
    CheckRepaintOnly(bench.frames, update);
    CHECK_FALSE(b.visible());
    CHECK(Mounted(grid, b));
    withB = true;
    CheckRepaintOnly(bench.frames, update);
    CHECK(b.visible());
    CHECK(bench.unchanged());

    withD = true;
    CheckRestructuresOnce(bench.frames, update);
    REQUIRE(bench.settle(update));
    withD = false;
    CheckRepaintOnly(bench.frames, update);
    CHECK_FALSE(d.visible());
    CHECK(bench.unchanged());
}

TEST_CASE("Node content stacks park dropped items with a repaint and restructure once for a new item",
          kRetentionTags) {
    Sakura::NodeLabel a;
    Sakura::NodeLabel b;
    Sakura::NodeLabel c;
    Sakura::NodeContentStack stack;
    Bench bench(stack, {10.0f, 10.0f, 240.0f, 80.0f});
    bool withA = true;
    bool withC = false;
    const auto update = [&] {
        a.update({.id = "a", .str = "A"});
        b.update({.id = "b", .str = "B"});
        c.update({.id = "c", .str = "C"});
        std::vector<Sakura::NodeContentStack::Item> items;
        if (withA) {
            items.push_back({.child = &a});
        }
        items.push_back({.child = &b, .gap = 4.0f});
        if (withC) {
            items.push_back({.child = &c});
        }
        stack.update({.id = "stack", .items = std::move(items)});
    };
    REQUIRE(bench.settle(update));

    withA = false;
    CheckRepaintOnly(bench.frames, update);
    CHECK_FALSE(a.visible());
    CHECK(Mounted(stack, a));
    withA = true;
    CheckRepaintOnly(bench.frames, update);
    CHECK(a.visible());
    CHECK(bench.unchanged());

    withC = true;
    CheckRestructuresOnce(bench.frames, update);
    REQUIRE(bench.settle(update));
    withC = false;
    CheckRepaintOnly(bench.frames, update);
    CHECK_FALSE(c.visible());
    CHECK(bench.unchanged());
}

TEST_CASE("Node field panels park dropped content with a repaint and restructure once for new content",
          kRetentionTags) {
    Sakura::NodeLabel a;
    Sakura::NodeLabel b;
    Sakura::NodeContentStack first;
    Sakura::NodeContentStack second;
    Sakura::NodeFieldPanel panel;
    Bench bench(panel, {10.0f, 10.0f, 240.0f, 80.0f});
    bool withB = true;
    Sakura::NodeContentStack* content = &first;
    const auto update = [&] {
        a.update({.id = "a", .str = "A"});
        b.update({.id = "b", .str = "B"});
        std::vector<Sakura::NodeContentStack::Item> items = {{.child = &a}};
        if (withB) {
            items.push_back({.child = &b});
        }
        first.update({.id = "first", .items = std::move(items)});
        second.update({.id = "second"});
        panel.update({.id = "panel", .content = content});
    };
    REQUIRE(bench.settle(update));
    CHECK(QuietMoves(bench.frames, update, {{20.0f, 15.0f}, {600.0f, 400.0f}}, bench.sender()));

    withB = false;
    CheckRepaintOnly(bench.frames, update);
    withB = true;
    CheckRepaintOnly(bench.frames, update);
    content = nullptr;
    CheckRepaintOnly(bench.frames, update);
    CHECK_FALSE(first.visible());
    content = &first;
    CheckRepaintOnly(bench.frames, update);
    CHECK(first.visible());
    CHECK(bench.unchanged());

    content = &second;
    CheckRestructuresOnce(bench.frames, update);
    CHECK_FALSE(first.visible());
}

TEST_CASE("Node field panels with a fixed height measure it in logical units whatever their content", kTags) {
    const auto ctx = MakeContext();
    Probe a({0.0f, 20.0f});
    Probe b({0.0f, 100.0f});
    Sakura::NodeContentStack stack;
    stack.update({.id = "stack", .items = {{.child = &a}, {.child = &b}}});
    Sakura::NodeFieldPanel panel;
    panel.update({.id = "panel", .content = &stack, .height = 50.0f});
    Host host(panel);

    CHECK(host.measureOf(ctx, {300.0f, 400.0f}) == Extent2D<F32>{300.0f, 50.0f});
    CHECK(host.measureOf(ctx, {300.0f, kInfinite}) == Extent2D<F32>{300.0f, 50.0f});
    CHECK(host.measureOf(ctx, {300.0f, 30.0f}) == Extent2D<F32>{300.0f, 30.0f});

    panel.update({.id = "panel", .height = 50.0f});
    CHECK(host.measureOf(ctx, {300.0f, 400.0f}) == Extent2D<F32>{300.0f, 50.0f});
}

TEST_CASE("Node field panels scale their fixed height and content gaps with the pixel ratio", kTags) {
    auto ctx = MakeContext();
    ctx.pixelRatio = 2.0f;
    Probe a({0.0f, 20.0f});
    Probe b({0.0f, 100.0f});
    Sakura::NodeContentStack stack;
    stack.update({.id = "stack", .items = {{.child = &a}, {.child = &b}}});
    Sakura::NodeFieldPanel panel;
    Host host(panel);

    panel.update({.id = "panel", .content = &stack});
    CHECK(host.measureOf(ctx, {300.0f, 400.0f}) == Extent2D<F32>{300.0f, 136.0f});

    panel.update({.id = "panel", .content = &stack, .height = 50.0f});
    CHECK(host.measureOf(ctx, {300.0f, 400.0f}) == Extent2D<F32>{300.0f, 100.0f});
    host.place(ctx, {0.0f, 0.0f, 300.0f, 100.0f});
    CHECK(a.laidFrame == Rect{0.0f, 0.0f, 300.0f, 20.0f});
    CHECK(b.laidFrame == Rect{0.0f, 36.0f, 300.0f, 100.0f});
    CHECK(b.laidClip == Rect{0.0f, 0.0f, 300.0f, 100.0f});

    panel.update({.id = "panel", .content = &stack, .height = Sakura::Fill});
    CHECK(host.measureOf(ctx, {300.0f, 400.0f}) == Extent2D<F32>{300.0f, 400.0f});
}

TEST_CASE("Node text inputs with a fill height take the available height and one frame when unbounded", kTags) {
    const Scene scene;
    Sakura::NodeTextInput text;
    text.update({.id = "text", .height = Sakura::Fill});

    CHECK(MeasureOf(scene.ctx, text, {240.0f, 60.0f}) == Extent2D<F32>{240.0f, 60.0f});
    CHECK(MeasureOf(scene.ctx, text, {240.0f, 10.0f}) == Extent2D<F32>{240.0f, 10.0f});
    CHECK(MeasureOf(scene.ctx, text, {240.0f, kInfinite}) == Extent2D<F32>{240.0f, 21.0f});

    auto doubled = scene.ctx;
    doubled.pixelRatio = 2.0f;
    CHECK(MeasureOf(doubled, text, {240.0f, 60.0f}) == Extent2D<F32>{240.0f, 60.0f});
    CHECK(MeasureOf(doubled, text, {240.0f, kInfinite}) == Extent2D<F32>{240.0f, 42.0f});
}

TEST_CASE("Inline node input row combos take their fixed width and a natural width resolves to zero", kTags) {
    const Scene scene;
    Sakura::NodeInputRow row;
    Host host(row);
    const auto comboFrame = [&] {
        const auto combos = ChildrenOf<Sakura::NodeCombo>(row);
        REQUIRE(combos.size() == 1);
        return Peek::Frame(*combos[0]);
    };
    const auto configure = [&](F32 width) {
        row.update({.id = "row", .combos = {{.options = {"CF32", "F32"}, .value = "CF32", .width = width}}});
    };

    configure(64.0f);
    host.place(scene.ctx, {0.0f, 0.0f, 300.0f, 21.0f});
    CHECK(comboFrame() == Rect{236.0f, 0.0f, 64.0f, 21.0f});

    configure(0.0f);
    host.place(scene.ctx, {0.0f, 0.0f, 300.0f, 21.0f});
    CHECK(comboFrame() == Rect{300.0f, 0.0f, 0.0f, 21.0f});
    CHECK(host.measureOf(scene.ctx, {64.0f, kInfinite}).y == 21.0f);
    CHECK(host.measureOf(scene.ctx, {63.0f, kInfinite}).y == 45.0f);

    auto doubled = scene.ctx;
    doubled.pixelRatio = 2.0f;
    configure(64.0f);
    host.place(doubled, {0.0f, 0.0f, 600.0f, 42.0f});
    CHECK(comboFrame() == Rect{472.0f, 0.0f, 128.0f, 42.0f});
}

TEST_CASE("Node int and uint inputs scale their frame height with the pixel ratio", kTags) {
    const Scene scene;
    auto ctx = scene.ctx;
    ctx.pixelRatio = 2.0f;
    Sakura::NodeIntInput integer;
    integer.update({.id = "int", .value = 64});
    Sakura::NodeUIntInput unsignedInteger;
    unsignedInteger.update({.id = "uint", .value = 2048});

    for (Sakura::Component* input : std::vector<Sakura::Component*>{&integer, &unsignedInteger}) {
        CHECK(MeasureOf(ctx, *input, {240.0f, kInfinite}) == Extent2D<F32>{240.0f, 42.0f});
        Host host(*input);
        host.place(ctx, {0.0f, 0.0f, 240.0f, 42.0f});
        const auto fields = ChildrenOf<Sakura::NodeTextInput>(*input);
        REQUIRE(fields.size() == 1);
        CHECK(Peek::Frame(*fields[0]) == Rect{0.0f, 0.0f, 240.0f, 42.0f});
    }
}

TEST_CASE("Node range input knobs and unit insets scale with the pixel ratio", kTags) {
    const Scene scene;
    Sakura::TextMetrics metrics;
    metrics.setWindow(scene.ctx.render);
    for (const F32 scale : {1.0f, 2.0f}) {
        CAPTURE(scale);
        auto ctx = scene.ctx;
        ctx.pixelRatio = scale;
        Sakura::NodeRangeInput range;
        range.update({.id = "range", .min = 0.0f, .max = 10.0f, .value = 0.0f, .unit = "dB"});
        Host host(range);
        CHECK(host.measureOf(ctx, {200.0f, kInfinite}) == Extent2D<F32>{200.0f, 21.0f * scale});
        host.place(ctx, {0.0f, 0.0f, 200.0f, 21.0f * scale});

        const auto boxes = ChildrenOf<Sakura::Box>(range);
        const auto labels = ChildrenOf<Sakura::Label>(range);
        REQUIRE(boxes.size() == 3);
        REQUIRE(labels.size() == 1);
        REQUIRE(boxes[2]->config().instances.size() == 1);
        CHECK(boxes[2]->config().instances[0].rect == Rect{0.0f, 0.0f, 16.0f * scale, 21.0f * scale});
        REQUIRE(labels[0]->config().instances.size() == 2);
        const F32 unitWidth = metrics.measure(Sakura::Typography::MonoFont, "dB", 15.0f * scale);
        CHECK(labels[0]->config().instances[1].rect.x == 200.0f - unitWidth - 3.0f * scale);
        CHECK(labels[0]->config().instances[0].rect.x == 6.0f * scale);
    }
}

TEST_CASE("Node combos at a pixel ratio of two hit test a doubled frame and open a doubled dropdown", kTags) {
    const Scene scene;
    auto ctx = scene.ctx;
    ctx.pixelRatio = 2.0f;
    std::vector<std::string> changes;
    Sakura::NodeCombo combo;
    combo.update({.id = "mode", .options = {"FM", "AM", "USB"}, .value = "FM",
                  .onChange = [&](const std::string& value) { changes.push_back(value); }});
    CHECK(MeasureOf(ctx, combo, {300.0f, kInfinite}) == Extent2D<F32>{300.0f, 42.0f});
    LayerSlot slot;
    slot.hold(combo, {100.0f, 100.0f, 300.0f, 50.0f});
    LayerHost host({&slot});
    host.place(ctx, kScreen);

    CHECK_FALSE(host.click(150.0f, 145.0f));
    host.place(ctx, kScreen);
    CHECK_FALSE(host.popups.covers({600.0f, 500.0f}));

    CHECK(host.click(150.0f, 135.0f));
    host.place(ctx, kScreen);
    CHECK(host.popups.interactiveAt({150.0f, 143.0f}));
    CHECK_FALSE(host.popups.interactiveAt({150.0f, 141.0f}));
    CHECK(host.popups.interactiveAt({390.0f, 290.0f}));
    CHECK_FALSE(host.popups.interactiveAt({150.0f, 294.0f}));

    CHECK(host.click(150.0f, 217.0f));
    CHECK(changes == std::vector<std::string>{"AM"});
}

TEST_CASE("Node path input buttons and connectors scale with the pixel ratio", kTags) {
    const Scene scene;
    auto ctx = scene.ctx;
    ctx.pixelRatio = 2.0f;
    U64 browsed = 0;
    Sakura::NodePathInput path;
    path.update({.id = "path", .onBrowse = [&] { ++browsed; }});
    Host host(path);

    CHECK(host.measureOf(ctx, {300.0f, kInfinite}) == Extent2D<F32>{300.0f, 42.0f});
    host.place(ctx, {0.0f, 0.0f, 300.0f, 42.0f});
    const auto fields = ChildrenOf<Sakura::NodeTextInput>(path);
    REQUIRE(fields.size() == 1);
    CHECK(Peek::Frame(*fields[0]) == Rect{0.0f, 0.0f, 242.0f, 42.0f});

    CHECK_FALSE(host.click(250.0f, 20.0f));
    CHECK(browsed == 0);
    CHECK(host.click(265.0f, 20.0f));
    CHECK(browsed == 1);
}

TEST_CASE("Node field grid gaps and column minimums scale with the pixel ratio", kTags) {
    auto ctx = MakeContext();
    ctx.pixelRatio = 2.0f;
    Probe a({0.0f, 21.0f});
    Probe b({0.0f, 30.0f});
    Probe c({0.0f, 10.0f});
    Sakura::NodeFieldGrid grid;
    grid.update({.id = "grid", .items = {{.child = &a}, {.child = &b}, {.child = &c}}});
    Host host(grid);

    CHECK(host.measureOf(ctx, {600.0f, kInfinite}) == Extent2D<F32>{600.0f, 44.0f});
    host.place(ctx, {0.0f, 0.0f, 600.0f, 44.0f});
    CHECK(a.laidFrame == Rect{0.0f, 0.0f, 298.0f, 30.0f});
    CHECK(b.laidFrame == Rect{302.0f, 0.0f, 298.0f, 30.0f});
    CHECK(c.laidFrame == Rect{0.0f, 34.0f, 298.0f, 10.0f});

    CHECK(host.measureOf(ctx, {564.0f, kInfinite}).y == 44.0f);
    CHECK(host.measureOf(ctx, {563.0f, kInfinite}) == Extent2D<F32>{563.0f, 69.0f});
}

TEST_CASE("Node surfaces scale their minimum height and detach button with the pixel ratio", kTags) {
    const Scene scene;
    auto ctx = scene.ctx;
    ctx.pixelRatio = 2.0f;
    U64 detached = 0;
    Sakura::NodeSurface surface;
    surface.update({.id = "surface", .detachOverlay = true, .onDetach = [&] { ++detached; }});
    Host host(surface);

    CHECK(host.measureOf(ctx, {300.0f, kInfinite}) == Extent2D<F32>{300.0f, 300.0f});
    host.place(ctx, {0.0f, 0.0f, 300.0f, 300.0f});
    host.send(Mouse(MouseEventType::Move, 150.0f, 150.0f));
    host.place(ctx, {0.0f, 0.0f, 300.0f, 300.0f});

    CHECK_FALSE(host.send(Mouse(MouseEventType::Click, 290.0f, 10.0f)));
    CHECK_FALSE(host.send(Mouse(MouseEventType::Click, 240.0f, 66.0f)));
    CHECK(detached == 0);
    CHECK(host.send(Mouse(MouseEventType::Click, 240.0f, 60.0f)));
    CHECK(detached == 1);

    surface.update({.id = "surface", .minimumHeight = 90.0f});
    CHECK(host.measureOf(ctx, {300.0f, kInfinite}) == Extent2D<F32>{300.0f, 180.0f});
}

TEST_CASE("Node surfaces detach on the press itself and ignore releases and cancels", kTags) {
    const Scene scene;
    auto ctx = scene.ctx;
    U64 detached = 0;
    Sakura::NodeSurface surface;
    surface.update({.id = "surface", .detachOverlay = true, .onDetach = [&] { ++detached; }});
    Host host(surface);
    host.place(ctx, {0.0f, 0.0f, 300.0f, 150.0f});
    host.send(Mouse(MouseEventType::Move, 150.0f, 75.0f));
    host.place(ctx, {0.0f, 0.0f, 300.0f, 150.0f});

    CHECK(host.send(Mouse(MouseEventType::Click, 280.0f, 20.0f)));
    CHECK(detached == 1);
    CHECK_FALSE(host.send(Mouse(MouseEventType::Cancel, 280.0f, 20.0f)));
    CHECK_FALSE(host.send(Mouse(MouseEventType::Release, 280.0f, 20.0f)));
    CHECK(detached == 1);

    CHECK_FALSE(host.send(Mouse(MouseEventType::Release, 500.0f, 500.0f)));
    CHECK_FALSE(host.send(Mouse(MouseEventType::Release, 280.0f, 20.0f)));
    CHECK(detached == 1);
}

TEST_CASE("A cancelled press on a node combo row selects nothing and keeps the dropdown open", kTags) {
    const Scene scene;
    std::vector<std::string> changes;
    Sakura::NodeCombo combo;
    combo.update({.id = "mode", .options = {"FM", "AM", "USB"}, .value = "FM",
                  .onChange = [&](const std::string& value) { changes.push_back(value); }});
    LayerSlot slot;
    slot.hold(combo, {100.0f, 100.0f, 200.0f, 25.0f});
    LayerHost host({&slot});
    host.place(scene.ctx, kScreen);

    host.send(Mouse(MouseEventType::Click, 150.0f, 110.0f));
    host.send(Mouse(MouseEventType::Cancel, 150.0f, 110.0f));
    host.send(Mouse(MouseEventType::Release, 150.0f, 110.0f));
    host.place(scene.ctx, kScreen);
    REQUIRE(host.popups.covers({600.0f, 500.0f}));

    CHECK(host.send(Mouse(MouseEventType::Click, 150.0f, 158.0f)));
    host.send(Mouse(MouseEventType::Cancel, 150.0f, 158.0f));
    host.send(Mouse(MouseEventType::Release, 150.0f, 158.0f));
    host.place(scene.ctx, kScreen);
    CHECK(changes.empty());
    CHECK(host.popups.covers({600.0f, 500.0f}));

    CHECK(host.click(150.0f, 158.0f));
    CHECK(changes == std::vector<std::string>{"AM"});
    host.place(scene.ctx, kScreen);
    CHECK_FALSE(host.popups.covers({600.0f, 500.0f}));
}

TEST_CASE("Stepped node float inputs forget a press when the pointer leaves", kTags) {
    const auto ctx = MakeContext();
    U64 changes = 0;
    Sakura::NodeFloatInput input;
    input.update({.id = "leave", .value = 1.0f, .step = 0.25f, .onChange = [&](F32) { ++changes; }});
    Host host(input);
    host.place(ctx, {0.0f, 0.0f, 200.0f, 40.0f});

    host.send(Mouse(MouseEventType::Click, 10.0f, 30.0f));
    host.send(Mouse(MouseEventType::Leave, 10.0f, 30.0f));
    host.send(Mouse(MouseEventType::Release, 10.0f, 30.0f));
    CHECK(changes == 0);
}
