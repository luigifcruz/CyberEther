#include <catch2/catch_test_macros.hpp>

#include "harness.hh"
#include "component_harness.hh"

#include <jetstream/render/base/texture.hh>
#include <jetstream/render/sakura/components/background_layer.hh>
#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/dock.hh>
#include <jetstream/render/sakura/components/dropdown.hh>
#include <jetstream/render/sakura/components/floating_layer.hh>
#include <jetstream/render/sakura/components/hud_layer.hh>
#include <jetstream/render/sakura/components/image.hh>
#include <jetstream/render/sakura/components/layer.hh>
#include <jetstream/render/sakura/components/menu.hh>
#include <jetstream/render/sakura/components/modal_layer.hh>
#include <jetstream/render/sakura/components/node/bool_input.hh>
#include <jetstream/render/sakura/components/node/combo.hh>
#include <jetstream/render/sakura/components/node/content_stack.hh>
#include <jetstream/render/sakura/components/node/editor.hh>
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
#include <jetstream/render/sakura/components/node/table.hh>
#include <jetstream/render/sakura/components/node/text_input.hh>
#include <jetstream/render/sakura/components/node/uint_input.hh>
#include <jetstream/render/sakura/components/overlay.hh>
#include <jetstream/render/sakura/components/panel.hh>
#include <jetstream/render/sakura/components/popup.hh>
#include <jetstream/render/sakura/components/stack.hh>
#include <jetstream/render/sakura/components/surface_image.hh>
#include <jetstream/render/sakura/components/toast_layer.hh>
#include <jetstream/render/sakura/palette.hh>
#include <jetstream/render/sakura/toast.hh>

#include "render/sakura/toast_queue.hh"

#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

using namespace Jetstream;
using namespace SakuraTest;

namespace {


using Dock = Sakura::Dock;
using DockTiles = Sakura::DockTiles;
using NodeEditor = Sakura::NodeEditor;


Sakura::Context ShellContext(Render::Window* window = nullptr) {
    auto ctx = MakeContext();
    ctx.palette = DistinctPalette();
    ctx.render = window;
    return ctx;
}

struct Nest : Sakura::Component {
    Extent2D<F32> desired;
    Sakura::Component* inner = nullptr;

    Nest(Extent2D<F32> desired, Sakura::Component& inner) : desired(desired), inner(&inner) {}

 protected:
    Extent2D<F32> measure(const Sakura::Context&, Extent2D<F32>) override {
        return desired;
    }

    void layout(const Sakura::Context& ctx) override {
        Sakura::RequestPopup(ctx, {
            .content = inner,
            .anchor = frame(),
            .placement = Sakura::PopupPlacement::Side,
            .panel = true,
        });
    }
};

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

struct TabDock {
    Probe a;
    Probe b;
    Probe c;
    Dock dock;

    void update() {
        dock.update({.tabs = true, .accepts = "panel",
                     .panes = {MakePane("a", &a), MakePane("b", &b), MakePane("c", &c)}});
    }
};

struct TileDock {
    Probe a;
    Probe b;
    Probe c;
    Probe d;
    bool withD = false;
    Dock dock;
    std::optional<DockTiles> tiles = Row({0.5f, 0.5f}, {Leaf("a"), Column({0.5f, 0.5f}, {Leaf("b"), Leaf("c")})});

    void update() {
        std::vector<Dock::Pane> panes = {MakePane("a", &a), MakePane("b", &b), MakePane("c", &c)};
        if (withD) {
            panes.push_back(MakePane("d", &d));
        }
        dock.update({
            .accepts = "panel",
            .panes = std::move(panes),
            .tiles = tiles,
            .onTiles = [this](std::optional<DockTiles> next) {
                tiles = std::move(next);
            },
        });
    }
};

struct ColumnDock {
    Probe a;
    Probe b;
    Probe c;
    Dock dock;
    std::optional<DockTiles> tiles = Row({0.0f, 0.0f, 0.0f}, {Leaf("a"), Leaf("b"), Leaf("c")});

    void update() {
        dock.update({
            .accepts = "panel",
            .panes = {MakePane("a", &a), MakePane("b", &b), MakePane("c", &c)},
            .tiles = tiles,
            .onTiles = [this](std::optional<DockTiles> next) {
                tiles = std::move(next);
            },
            .columns = true,
            .flexPane = "a",
        });
    }
};

struct GraphNodes {
    std::unordered_map<std::string, NodeEditor::Geometry> geometry;

    NodeEditor::Node node(const std::string& id,
                          Extent2D<F32> position,
                          Sakura::NodeContentStack* content,
                          std::vector<NodeEditor::Pin> pins) {
        NodeEditor::Node result{
            .id = id,
            .title = id,
            .subtitle = "block",
            .pins = std::move(pins),
            .content = content,
            .dimensions = {160.0f, 0.0f},
            .gridPosition = position,
            .onGeometryChange = [this, id](const NodeEditor::Geometry& next) {
                geometry[id] = next;
            },
        };
        if (const auto it = geometry.find(id); it != geometry.end()) {
            result.gridPosition = it->second.gridPosition;
            result.dimensions = it->second.contentDimensions;
        }
        return result;
    }
};

}  // namespace

TEST_CASE("A tab dock repeating the same config every frame settles and stays clean",
          "[core][sakura][retention]") {
    TabDock tabs;
    Sakura::Layer layer;
    layer.update({.content = &tabs.dock});
    LayerHost host({&layer});
    Frames frames(host, kScreen, ShellContext());
    const auto update = [&] {
        tabs.update();
    };

    REQUIRE(frames.settle(update));
    CHECK(DirtyFrames(frames, update) == 0);
    CHECK(tabs.c.layouts > 0);
    CHECK(tabs.a.layouts == 0);
}

TEST_CASE("A tab dock repaints once when the hovered tab or close button changes",
          "[core][sakura][retention]") {
    TabDock tabs;
    Sakura::Layer layer;
    layer.update({.content = &tabs.dock});
    LayerHost host({&layer});
    Frames frames(host, kScreen, ShellContext());
    const auto update = [&] {
        tabs.update();
    };
    REQUIRE(frames.settle(update));

    Hover(host, frames, 400.0f, 300.0f);
    CHECK(frames.run(update).clean());
    Hover(host, frames, 420.0f, 320.0f);
    CHECK(frames.run(update).clean());

    Hover(host, frames, 60.0f, 12.0f);
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(DirtyFrames(frames, update) == 0);
    Hover(host, frames, 62.0f, 14.0f);
    CHECK(frames.run(update).clean());

    Hover(host, frames, 233.0f, 20.0f);
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(DirtyFrames(frames, update) == 0);

    Hover(host, frames, 400.0f, 300.0f);
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(DirtyFrames(frames, update) == 0);
}

TEST_CASE("Activating another tab never rebuilds and settles again", "[core][sakura][retention]") {
    TabDock tabs;
    Sakura::Layer layer;
    layer.update({.content = &tabs.dock});
    LayerHost host({&layer});
    Frames frames(host, kScreen, ShellContext());
    const auto update = [&] {
        tabs.update();
    };
    REQUIRE(frames.settle(update));
    const U64 rebuilds = frames.rebuilds;

    host.click(10.0f, 12.0f);
    const auto first = frames.run(update);
    CHECK(first.repainted);
    CHECK_FALSE(first.rebuilt);
    REQUIRE(frames.settle(update));
    CHECK(DirtyFrames(frames, update) == 0);
    CHECK(tabs.a.layouts > 0);

    host.click(372.0f, 20.0f);
    const auto back = frames.run(update);
    CHECK(back.repainted);
    CHECK_FALSE(back.rebuilt);
    REQUIRE(frames.settle(update));
    CHECK(DirtyFrames(frames, update) == 0);
    CHECK(frames.rebuilds == rebuilds);
}

TEST_CASE("A tile dock fed back its own tiles settles and stays clean", "[core][sakura][retention]") {
    TileDock tiles;
    Sakura::Layer layer;
    layer.update({.content = &tiles.dock});
    LayerHost host({&layer});
    Frames frames(host, kScreen, ShellContext());
    const auto update = [&] {
        tiles.update();
    };

    REQUIRE(frames.settle(update));
    CHECK(DirtyFrames(frames, update) == 0);
    CHECK(tiles.a.layouts > 0);
    CHECK(tiles.b.layouts > 0);
    CHECK(tiles.c.layouts > 0);
}

TEST_CASE("A tile dock repaints once when a header close or splitter becomes hovered",
          "[core][sakura][retention]") {
    TileDock tiles;
    Sakura::Layer layer;
    layer.update({.content = &tiles.dock});
    LayerHost host({&layer});
    Frames frames(host, kScreen, ShellContext());
    const auto update = [&] {
        tiles.update();
    };
    REQUIRE(frames.settle(update));

    Hover(host, frames, 200.0f, 300.0f);
    CHECK(frames.run(update).clean());
    Hover(host, frames, 210.0f, 320.0f);
    CHECK(frames.run(update).clean());

    Hover(host, frames, 779.0f, 21.0f);
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(DirtyFrames(frames, update) == 0);
    Hover(host, frames, 780.0f, 22.0f);
    CHECK(frames.run(update).clean());

    Hover(host, frames, 400.0f, 300.0f);
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(DirtyFrames(frames, update) == 0);

    Hover(host, frames, 200.0f, 300.0f);
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(DirtyFrames(frames, update) == 0);
}

TEST_CASE("Dragging a tile splitter repaints and settles once the owner adopts the new tiles",
          "[core][sakura][retention]") {
    TileDock tiles;
    Sakura::Layer layer;
    layer.update({.content = &tiles.dock});
    LayerHost host({&layer});
    Frames frames(host, kScreen, ShellContext());
    const auto update = [&] {
        tiles.update();
    };
    REQUIRE(frames.settle(update));
    const U64 rebuilds = frames.rebuilds;
    const U64 restructures = frames.restructures;
    const auto before = tiles.tiles;

    Press(host, frames, 400.0f, 300.0f);
    Hover(host, frames, 300.0f, 300.0f);
    CHECK(RepaintOnly(frames.run(update)));
    Hover(host, frames, 280.0f, 300.0f);
    CHECK(RepaintOnly(frames.run(update)));
    Release(host, frames, 280.0f, 300.0f);
    CHECK(tiles.tiles != before);

    REQUIRE(frames.settle(update));
    CHECK(DirtyFrames(frames, update) == 0);
    CHECK(frames.rebuilds == rebuilds);
    CHECK(frames.restructures == restructures);
}

TEST_CASE("A tile splitter held still under repeated moves stays clean until the pointer moves",
          "[core][sakura][retention]") {
    TileDock tiles;
    Sakura::Layer layer;
    layer.update({.content = &tiles.dock});
    LayerHost host({&layer});
    Frames frames(host, kScreen, ShellContext());
    const auto update = [&] {
        tiles.update();
    };
    REQUIRE(frames.settle(update));

    Press(host, frames, 400.0f, 300.0f);
    frames.run(update);
    Hover(host, frames, 400.0f, 300.0f);
    frames.run(update);
    REQUIRE(frames.settle(update));
    for (int frame = 0; frame < 3; ++frame) {
        Hover(host, frames, 400.0f, 300.0f);
        CHECK(frames.run(update).clean());
    }

    Hover(host, frames, 300.0f, 300.0f);
    CHECK(RepaintOnly(frames.run(update)));
    for (int frame = 0; frame < 3; ++frame) {
        Hover(host, frames, 300.0f, 300.0f);
        CHECK(frames.run(update).clean());
    }
    Release(host, frames, 300.0f, 300.0f);
}

TEST_CASE("A dock header drag held still stays clean and repaints only when the pointer moves",
          "[core][sakura][retention]") {
    TileDock tiles;
    Sakura::Layer layer;
    layer.update({.content = &tiles.dock});
    LayerHost host({&layer});
    Frames frames(host, kScreen, ShellContext());
    const auto update = [&] {
        tiles.update();
    };
    REQUIRE(frames.settle(update));
    const U64 rebuilds = frames.rebuilds;

    frames.timeStep = 0.05;
    Press(host, frames, 200.0f, 12.0f);
    frames.run(update);
    Hover(host, frames, 600.0f, 400.0f);
    REQUIRE(host.drag.active);
    CHECK(RepaintOnly(frames.run(update)));
    REQUIRE(frames.settle(update));
    Hover(host, frames, 600.0f, 400.0f);
    CHECK(frames.run(update).clean());
    Hover(host, frames, 610.0f, 410.0f);
    CHECK(RepaintOnly(frames.run(update)));
    REQUIRE(frames.settle(update));

    for (int frame = 0; frame < 3; ++frame) {
        Hover(host, frames, 610.0f, 410.0f);
        CHECK(frames.run(update).clean());
    }

    Hover(host, frames, 620.0f, 420.0f);
    CHECK(RepaintOnly(frames.run(update)));
    Hover(host, frames, 620.0f, 420.0f);
    CHECK(frames.run(update).clean());
    CHECK(frames.rebuilds == rebuilds);

    host.send(Mouse(MouseEventType::Cancel, 620.0f, 420.0f));
    CHECK_FALSE(host.drag.active);
}

TEST_CASE("Leaving or cancelling an idle dock stays clean", "[core][sakura][retention]") {
    TileDock tiles;
    Sakura::Layer layer;
    layer.update({.content = &tiles.dock});
    LayerHost host({&layer});
    Frames frames(host, kScreen, ShellContext());
    const auto update = [&] {
        tiles.update();
    };
    REQUIRE(frames.settle(update));

    for (int frame = 0; frame < 3; ++frame) {
        host.send(Mouse(MouseEventType::Leave, 200.0f, 300.0f));
        CHECK(frames.run(update).clean());
        host.send(Mouse(MouseEventType::Cancel, 200.0f, 300.0f));
        CHECK(frames.run(update).clean());
    }

    Hover(host, frames, 779.0f, 21.0f);
    CHECK(RepaintOnly(frames.run(update)));
    host.send(Mouse(MouseEventType::Leave, 779.0f, 21.0f));
    CHECK(RepaintOnly(frames.run(update)));
    host.send(Mouse(MouseEventType::Leave, 779.0f, 21.0f));
    CHECK(frames.run(update).clean());
    host.send(Mouse(MouseEventType::Cancel, 779.0f, 21.0f));
    CHECK(frames.run(update).clean());
}

TEST_CASE("A dock drag held still over the root stays clean and repaints only when the pointer moves",
          "[core][sakura][retention]") {
    Probe content;
    LayerSlot slot;
    slot.hold(content, kScreen);
    LayerHost host({&slot});
    Frames frames(host, kScreen, ShellContext());
    REQUIRE(frames.settle());

    host.drag.active = true;
    host.drag.pane = "a";
    host.drag.group = "panel";
    host.drag.title = "a";
    Hover(host, frames, 300.0f, 300.0f);
    CHECK(RepaintOnly(frames.run()));
    REQUIRE(frames.settle());

    for (int frame = 0; frame < 3; ++frame) {
        Hover(host, frames, 300.0f, 300.0f);
        CHECK(frames.run().clean());
    }

    Hover(host, frames, 320.0f, 310.0f);
    CHECK(RepaintOnly(frames.run()));
    Hover(host, frames, 320.0f, 310.0f);
    CHECK(frames.run().clean());

    Release(host, frames, 320.0f, 310.0f);
    CHECK_FALSE(host.drag.active);
}

TEST_CASE("The drop preview repaints only while it eases or fades and stays clean once settled",
          "[core][sakura][retention]") {
    TileDock tiles;
    Sakura::Layer layer;
    layer.update({.content = &tiles.dock});
    LayerHost host({&layer});
    Frames frames(host, kScreen, ShellContext());
    frames.ctx.time = 1.0;
    const auto update = [&] {
        tiles.update();
    };
    const auto at = [&](F64 time) {
        frames.ctx.time = time;
        return frames.run(update);
    };
    REQUIRE(frames.settle(update));
    const U64 rebuilds = frames.rebuilds;
    const Rect top = {403.0f, 303.0f, 391.0f, 145.5f};
    const Rect right = {598.5f, 303.0f, 195.5f, 291.0f};

    Press(host, frames, 200.0f, 21.0f);
    at(1.0);
    Hover(host, frames, 600.0f, 400.0f);
    REQUIRE(host.drag.active);
    CHECK(RepaintOnly(at(1.0)));
    REQUIRE(host.drag.target.has_value());
    CHECK(host.drag.target->preview == top);
    CHECK(host.motion.alpha == 0.0f);
    CHECK(RepaintOnly(at(1.04)));
    CHECK(host.motion.alpha > 0.0f);
    CHECK(host.motion.alpha < 1.0f);
    CHECK(host.motion.rect == top);
    CHECK(RepaintOnly(at(1.08)));
    CHECK(RepaintOnly(at(1.13)));
    CHECK(host.motion.alpha == 1.0f);
    for (const F64 time : {1.2, 1.3, 1.5}) {
        CAPTURE(time);
        CHECK(at(time).clean());
    }
    for (const F64 time : {1.6, 1.65, 1.7}) {
        Hover(host, frames, 600.0f, 400.0f);
        CHECK(at(time).clean());
    }

    Hover(host, frames, 780.0f, 450.0f);
    CHECK(RepaintOnly(at(2.0)));
    REQUIRE(host.drag.target.has_value());
    CHECK(host.drag.target->preview == right);
    CHECK(RepaintOnly(at(2.06)));
    CHECK(host.motion.rect.x > top.x);
    CHECK(host.motion.rect.x < right.x);
    CHECK(RepaintOnly(at(2.13)));
    CHECK(host.motion.rect == right);
    CHECK(at(2.2).clean());
    Hover(host, frames, 780.0f, 450.0f);
    CHECK(at(2.3).clean());

    Hover(host, frames, 20.0f, 300.0f);
    CHECK(RepaintOnly(at(3.0)));
    CHECK_FALSE(host.drag.target.has_value());
    CHECK(RepaintOnly(at(3.06)));
    CHECK(host.motion.alpha > 0.0f);
    CHECK(host.motion.alpha < 1.0f);
    CHECK(host.motion.rect == right);
    CHECK(RepaintOnly(at(3.13)));
    CHECK(host.motion.alpha == 0.0f);
    CHECK(at(3.2).clean());

    Hover(host, frames, 600.0f, 400.0f);
    CHECK(RepaintOnly(at(4.0)));
    CHECK(host.motion.rect == top);
    at(4.06);
    at(4.13);
    CHECK(at(4.2).clean());

    host.send(Mouse(MouseEventType::Cancel, 600.0f, 400.0f));
    CHECK_FALSE(host.drag.active);
    CHECK(RepaintOnly(at(5.0)));
    CHECK(RepaintOnly(at(5.06)));
    CHECK(host.motion.alpha > 0.0f);
    CHECK(RepaintOnly(at(5.13)));
    CHECK(host.motion.alpha == 0.0f);
    for (const F64 time : {5.2, 5.4, 6.0}) {
        CAPTURE(time);
        CHECK(at(time).clean());
    }
    CHECK(frames.rebuilds == rebuilds);
}

TEST_CASE("A pill tab dock with chrome settles and repaints once per pill hover change",
          "[core][sakura][retention]") {
    TabDock tabs;
    Probe leading({40.0f, 20.0f});
    Probe trailing({24.0f, 24.0f});
    Sakura::Layer layer;
    layer.update({.content = &tabs.dock});
    LayerHost host({&layer});
    Frames frames(host, kScreen, ShellContext());
    const auto update = [&] {
        tabs.dock.update({.tabs = true, .accepts = "panel",
                          .panes = {MakePane("a", &tabs.a), MakePane("b", &tabs.b), MakePane("c", &tabs.c)},
                          .leading = &leading, .trailing = &trailing, .leadingInset = 70.0f});
    };

    REQUIRE(frames.settle(update));
    CHECK(DirtyFrames(frames, update) == 0);
    CHECK(leading.layouts > 0);
    CHECK(trailing.layouts > 0);

    Hover(host, frames, 125.0f, 19.0f);
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(DirtyFrames(frames, update) == 0);
    Hover(host, frames, 127.0f, 20.0f);
    CHECK(frames.run(update).clean());

    Hover(host, frames, 400.0f, 19.0f);
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(DirtyFrames(frames, update) == 0);

    Hover(host, frames, 400.0f, 300.0f);
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(DirtyFrames(frames, update) == 0);
}

TEST_CASE("Maximizing and restoring a tile never rebuilds and settles again", "[core][sakura][retention]") {
    TileDock tiles;
    Sakura::Layer layer;
    layer.update({.content = &tiles.dock});
    LayerHost host({&layer});
    Frames frames(host, kScreen, ShellContext());
    const auto update = [&] {
        tiles.update();
    };
    REQUIRE(frames.settle(update));
    const U64 rebuilds = frames.rebuilds;
    const auto before = tiles.tiles;

    host.click(360.0f, 21.0f);
    CHECK_FALSE(frames.run(update).rebuilt);
    REQUIRE(frames.settle(update));
    CHECK(DirtyFrames(frames, update) == 0);
    CHECK_FALSE(tiles.b.visible());
    CHECK_FALSE(tiles.c.visible());

    host.click(757.0f, 21.0f);
    CHECK_FALSE(frames.run(update).rebuilt);
    REQUIRE(frames.settle(update));
    CHECK(DirtyFrames(frames, update) == 0);
    CHECK(tiles.b.visible());
    CHECK(tiles.c.visible());
    CHECK(tiles.tiles == before);
    CHECK(frames.rebuilds == rebuilds);
}

TEST_CASE("Changing a tile dock's tiles or panes never rebuilds and settles again",
          "[core][sakura][retention]") {
    TileDock tiles;
    Sakura::Layer layer;
    layer.update({.content = &tiles.dock});
    LayerHost host({&layer});
    Frames frames(host, kScreen, ShellContext());
    const auto update = [&] {
        tiles.update();
    };
    REQUIRE(frames.settle(update));
    const U64 rebuilds = frames.rebuilds;

    tiles.tiles = Column({0.3f, 0.7f}, {Leaf("a"), Row({0.5f, 0.5f}, {Leaf("b"), Leaf("c")})});
    CHECK(RepaintOnly(frames.run(update)));
    REQUIRE(frames.settle(update));
    CHECK(DirtyFrames(frames, update) == 0);

    tiles.withD = true;
    const auto added = frames.run(update);
    CHECK(added.repainted);
    CHECK_FALSE(added.rebuilt);
    REQUIRE(frames.settle(update));
    CHECK(DirtyFrames(frames, update) == 0);
    CHECK(tiles.d.layouts > 0);

    tiles.withD = false;
    const auto removed = frames.run(update);
    CHECK(removed.repainted);
    CHECK_FALSE(removed.rebuilt);
    REQUIRE(frames.settle(update));
    CHECK(DirtyFrames(frames, update) == 0);
    CHECK(frames.rebuilds == rebuilds);
}

TEST_CASE("A layer holding an overlay settles and stays clean under identical updates",
          "[core][sakura][retention]") {
    Probe fill;
    Probe corner({60.0f, 20.0f});
    Sakura::Overlay overlay;
    Sakura::Layer layer;
    LayerHost host({&layer});
    Frames frames(host, kScreen, ShellContext());
    const auto update = [&] {
        overlay.update({.items = {
            {.child = &fill},
            {.child = &corner, .anchor = Sakura::Anchor::BottomRight, .offset = {-5.0f, -5.0f}},
        }});
        layer.update({.content = &overlay});
    };

    REQUIRE(frames.settle(update));
    CHECK(DirtyFrames(frames, update) == 0);
    Hover(host, frames, 300.0f, 300.0f);
    CHECK(frames.run(update).clean());
    Hover(host, frames, 760.0f, 580.0f);
    CHECK(frames.run(update).clean());
}

TEST_CASE("An idle popup layer stays clean", "[core][sakura][retention]") {
    Probe below({800.0f, 600.0f});
    LayerSlot slot;
    slot.hold(below, kScreen);
    LayerHost host({&slot});
    Frames frames(host, kScreen, ShellContext());

    REQUIRE(frames.settle());
    CHECK(DirtyFrames(frames) == 0);
    Hover(host, frames, 300.0f, 300.0f);
    CHECK(frames.run().clean());
}

TEST_CASE("A popup requested every frame stays clean and closing or reopening it only repaints",
          "[core][sakura][retention]") {
    Probe content({100.0f, 50.0f});
    LayerSlot slot;
    const Sakura::PopupRequest request = {
        .content = &content,
        .anchor = {100.0f, 100.0f, 1.0f, 1.0f},
        .placement = Sakura::PopupPlacement::Point,
        .panel = true,
        .onDismiss = [] {},
    };
    slot.request({request});
    LayerHost host({&slot});
    Frames frames(host, kScreen, ShellContext());

    REQUIRE(frames.settle());
    CHECK(DirtyFrames(frames) == 0);
    Hover(host, frames, 150.0f, 130.0f);
    CHECK(frames.run().clean());

    slot.request({});
    CHECK(RepaintOnly(frames.run()));
    CHECK(DirtyFrames(frames) == 0);

    slot.request({request});
    CHECK(RepaintOnly(frames.run()));
    CHECK(DirtyFrames(frames) == 0);
}

TEST_CASE("An open popup component updated every frame stays clean and dismissing it only repaints",
          "[core][sakura][retention]") {
    Probe content({100.0f, 50.0f});
    Sakura::Popup popup;
    bool open = true;
    LayerSlot slot;
    slot.hold(popup, {100.0f, 100.0f, 80.0f, 20.0f});
    LayerHost host({&slot});
    Frames frames(host, kScreen, ShellContext());
    const auto update = [&] {
        popup.update({.content = &content, .open = open, .panel = true, .onDismiss = [&] { open = false; }});
    };

    REQUIRE(frames.settle(update));
    CHECK(DirtyFrames(frames, update) == 0);
    Hover(host, frames, 150.0f, 150.0f);
    CHECK(frames.run(update).clean());

    host.click(600.0f, 500.0f);
    CHECK_FALSE(open);
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(DirtyFrames(frames, update) == 0);

    open = true;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(DirtyFrames(frames, update) == 0);
}

TEST_CASE("A closed menu updated every frame stays clean", "[core][sakura][retention]") {
    Sakura::Menu menu;
    LayerSlot slot;
    slot.hold(menu, {100.0f, 100.0f, 1.0f, 1.0f});
    LayerHost host({&slot});
    Frames frames(host, kScreen, ShellContext());
    const auto update = [&] {
        menu.update({
            .entries = {{.label = "Open", .children = {{.label = "Recent"}}}, {.label = "Save"}, {.separator = true},
                        {.label = "Quit", .shortcut = "Q"}},
            .open = false,
            .atCursor = false,
        });
    };

    REQUIRE(frames.settle(update));
    CHECK(DirtyFrames(frames, update) == 0);
}

TEST_CASE("A closed dropdown updated every frame stays clean", "[core][sakura][retention]") {
    Sakura::Dropdown dropdown;
    LayerSlot slot;
    slot.onLayout = [&](const Sakura::Context& ctx) {
        dropdown.request(ctx, {100.0f, 100.0f, 200.0f, 25.0f});
    };
    LayerHost host({&slot});
    Frames frames(host, kScreen, ShellContext());
    const auto update = [&] {
        dropdown.update({.options = {"alpha", "beta", "gamma"}, .selected = 1, .onSelect = [](U64) {}});
    };

    REQUIRE(frames.settle(update));
    CHECK(DirtyFrames(frames, update) == 0);
}

TEST_CASE("An open menu updated every frame settles and repaints once per row hover",
          "[core][sakura][retention]") {
    bool open = true;
    Sakura::Menu menu;
    LayerSlot slot;
    slot.hold(menu, {100.0f, 100.0f, 1.0f, 1.0f});
    LayerHost host({&slot});
    Frames frames(host, kScreen, ShellContext());
    const auto update = [&] {
        menu.update({
            .entries = {{.label = "Open", .children = {{.label = "Recent"}}}, {.label = "Save"}, {.separator = true},
                        {.label = "Quit", .shortcut = "Q"}},
            .open = open,
            .atCursor = false,
            .onClose = [&] { open = false; },
        });
    };

    REQUIRE(frames.settle(update));
    CHECK(DirtyFrames(frames, update) == 0);
    const U64 rebuilds = frames.rebuilds;

    Hover(host, frames, 120.0f, 142.0f);
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(DirtyFrames(frames, update) == 0);
    Hover(host, frames, 122.0f, 144.0f);
    CHECK(frames.run(update).clean());

    Hover(host, frames, 120.0f, 175.0f);
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(DirtyFrames(frames, update) == 0);

    Hover(host, frames, 400.0f, 400.0f);
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(DirtyFrames(frames, update) == 0);
    Hover(host, frames, 420.0f, 420.0f);
    CHECK(frames.run(update).clean());
    CHECK(frames.rebuilds == rebuilds);
}

TEST_CASE("A menu closed under a hovered row reopens without that row highlighted", "[core][sakura][retention]") {
    bool open = true;
    Sakura::Menu menu;
    LayerSlot slot;
    slot.hold(menu, {100.0f, 100.0f, 1.0f, 1.0f});
    LayerHost host({&slot});
    Frames frames(host, kScreen, ShellContext());
    frames.verify = false;
    const auto update = [&] {
        menu.update({
            .entries = {{.label = "Open"}, {.label = "Save"}, {.label = "Quit", .shortcut = "Q"}},
            .open = open,
            .atCursor = false,
            .onClose = [&] { open = false; },
        });
    };
    REQUIRE(frames.settle(update));
    const auto idle = Drawn(host);

    Hover(host, frames, 120.0f, 142.0f);
    REQUIRE(frames.settle(update));
    REQUIRE(Drawn(host) != idle);
    open = false;
    REQUIRE(frames.settle(update));
    open = true;
    REQUIRE(frames.settle(update));
    CHECK(Drawn(host) == idle);
}

TEST_CASE("Two popups requested every frame and a popup nested in the second one settle and stay clean",
          "[core][sakura][retention]") {
    Probe first({100.0f, 50.0f});
    Probe leaf({80.0f, 40.0f});
    Nest second({100.0f, 50.0f}, leaf);
    LayerSlot slot;
    const Sakura::PopupRequest one = {
        .content = &first,
        .anchor = {100.0f, 100.0f, 1.0f, 1.0f},
        .placement = Sakura::PopupPlacement::Point,
        .panel = true,
    };
    const Sakura::PopupRequest two = {
        .content = &second,
        .anchor = {400.0f, 100.0f, 1.0f, 1.0f},
        .placement = Sakura::PopupPlacement::Point,
        .panel = true,
        .onDismiss = [] {},
    };
    slot.request({one, two});
    LayerHost host({&slot});
    Frames frames(host, kScreen, ShellContext());

    REQUIRE(frames.settle());
    CHECK(DirtyFrames(frames) == 0);
    CHECK(first.layouts > 0);
    CHECK(leaf.layouts > 0);
    CHECK(host.popups.interactiveAt({550.0f, 120.0f}));
    Hover(host, frames, 150.0f, 130.0f);
    CHECK(frames.run().clean());
    Hover(host, frames, 550.0f, 120.0f);
    CHECK(frames.run().clean());

    const U64 restructures = frames.restructures;
    slot.request({two});
    CHECK(RepaintOnly(frames.run()));
    CHECK(DirtyFrames(frames) == 0);
    CHECK(frames.restructures == restructures);

    slot.request({one, two});
    const auto reopened = frames.run();
    CHECK(reopened.repainted);
    CHECK_FALSE(reopened.rebuilt);
    REQUIRE(frames.settle());
    CHECK(DirtyFrames(frames) == 0);

    slot.request({one});
    CHECK(RepaintOnly(frames.run()));
    CHECK(DirtyFrames(frames) == 0);
    CHECK_FALSE(host.popups.interactiveAt({550.0f, 120.0f}));
}

TEST_CASE("An open submenu settles, repaints once per submenu row hover, and closes with a repaint only",
          "[core][sakura][retention]") {
    bool open = true;
    U64 recent = 0;
    Sakura::Menu menu;
    LayerSlot slot;
    slot.hold(menu, {100.0f, 100.0f, 1.0f, 1.0f});
    LayerHost host({&slot});
    Frames frames(host, kScreen, ShellContext());
    const auto update = [&] {
        menu.update({
            .entries = {{.label = "Open", .children = {{.label = "Recent", .onClick = [&] { ++recent; }},
                                                       {.label = "Browse"}}},
                        {.label = "Save"}, {.separator = true}, {.label = "Quit", .shortcut = "Q"}},
            .open = open,
            .atCursor = false,
            .onClose = [&] { open = false; },
        });
    };
    REQUIRE(frames.settle(update));
    const U64 rebuilds = frames.rebuilds;

    Hover(host, frames, 120.0f, 119.0f);
    const auto opened = frames.run(update);
    CHECK(opened.repainted);
    CHECK_FALSE(opened.rebuilt);
    REQUIRE(frames.settle(update));
    CHECK(DirtyFrames(frames, update) == 0);
    REQUIRE(host.popups.interactiveAt({160.0f, 119.0f}));

    Hover(host, frames, 160.0f, 119.0f);
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(DirtyFrames(frames, update) == 0);
    Hover(host, frames, 161.0f, 120.0f);
    CHECK(frames.run(update).clean());

    Hover(host, frames, 160.0f, 142.0f);
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(DirtyFrames(frames, update) == 0);

    const U64 restructures = frames.restructures;
    Hover(host, frames, 120.0f, 175.0f);
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(DirtyFrames(frames, update) == 0);
    CHECK(frames.restructures == restructures);
    CHECK_FALSE(host.popups.interactiveAt({160.0f, 119.0f}));

    Hover(host, frames, 120.0f, 119.0f);
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(DirtyFrames(frames, update) == 0);
    REQUIRE(host.popups.interactiveAt({160.0f, 119.0f}));

    host.click(160.0f, 119.0f);
    CHECK(recent == 1);
    CHECK_FALSE(open);
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(DirtyFrames(frames, update) == 0);
    update();
    CHECK_FALSE(Frames::Access::paintDirty(host));
    CHECK(frames.rebuilds == rebuilds);
}

TEST_CASE("A menu that was opened once and then closed stays clean under identical updates",
          "[core][sakura][retention]") {
    bool open = true;
    Sakura::Menu menu;
    LayerSlot slot;
    slot.hold(menu, {100.0f, 100.0f, 1.0f, 1.0f});
    LayerHost host({&slot});
    Frames frames(host, kScreen, ShellContext());
    const auto update = [&] {
        menu.update({
            .entries = {{.label = "Open", .children = {{.label = "Recent"}}}, {.label = "Save"}},
            .open = open,
            .atCursor = false,
            .onClose = [&] { open = false; },
        });
    };
    REQUIRE(frames.settle(update));
    Hover(host, frames, 120.0f, 142.0f);
    REQUIRE(frames.settle(update));

    host.click(600.0f, 500.0f);
    REQUIRE_FALSE(open);
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(DirtyFrames(frames, update) == 0);
    update();
    CHECK_FALSE(Frames::Access::paintDirty(host));

    open = true;
    const auto reopened = frames.run(update);
    CHECK(reopened.repainted);
    CHECK_FALSE(reopened.rebuilt);
    REQUIRE(frames.settle(update));
    CHECK(DirtyFrames(frames, update) == 0);
}

TEST_CASE("An open dropdown settles, repaints once per row hover, and stays clean after a selection closes it",
          "[core][sakura][retention]") {
    std::optional<U64> selected = 1;
    Sakura::Dropdown dropdown;
    LayerSlot slot;
    slot.onLayout = [&](const Sakura::Context& ctx) {
        dropdown.request(ctx, {100.0f, 100.0f, 200.0f, 25.0f});
    };
    LayerHost host({&slot});
    Frames frames(host, kScreen, ShellContext());
    const auto update = [&] {
        dropdown.update({.options = {"alpha", "beta", "gamma"}, .selected = selected,
                         .onSelect = [&](U64 row) { selected = row; }});
    };
    REQUIRE(frames.settle(update));
    const U64 rebuilds = frames.rebuilds;

    dropdown.open();
    const auto opened = frames.run(update);
    CHECK(opened.repainted);
    CHECK_FALSE(opened.rebuilt);
    REQUIRE(frames.settle(update));
    CHECK(DirtyFrames(frames, update) == 0);

    Hover(host, frames, 150.0f, 150.0f);
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(DirtyFrames(frames, update) == 0);
    Hover(host, frames, 152.0f, 152.0f);
    CHECK(frames.run(update).clean());

    host.click(150.0f, 199.0f);
    CHECK(selected == std::optional<U64>{2});
    CHECK_FALSE(dropdown.isOpen());
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(DirtyFrames(frames, update) == 0);
    update();
    CHECK_FALSE(Frames::Access::paintDirty(host));

    dropdown.open();
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(DirtyFrames(frames, update) == 0);

    host.click(600.0f, 500.0f);
    CHECK_FALSE(dropdown.isOpen());
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(DirtyFrames(frames, update) == 0);
    CHECK(frames.rebuilds == rebuilds);
}

TEST_CASE("An open node combo settles and stays clean after it closes", "[core][sakura][retention]") {
    SakuraTest::FakeWindow window;
    std::string value = "AM";
    Sakura::NodeCombo combo;
    LayerSlot slot;
    slot.hold(combo, {100.0f, 100.0f, 200.0f, 25.0f});
    LayerHost host({&slot});
    Frames frames(host, kScreen, ShellContext(&window));
    const auto update = [&] {
        combo.update({.id = "mode", .options = {"FM", "AM", "USB"}, .value = value,
                      .onChange = [&](const std::string& next) { value = next; }});
    };
    REQUIRE(frames.settle(update));
    const U64 rebuilds = frames.rebuilds;

    host.click(150.0f, 110.0f);
    REQUIRE_FALSE(host.popups.covers({600.0f, 500.0f}));
    const auto opened = frames.run(update);
    CHECK(opened.repainted);
    CHECK_FALSE(opened.rebuilt);
    REQUIRE(frames.settle(update));
    CHECK(DirtyFrames(frames, update) == 0);
    REQUIRE(host.popups.covers({600.0f, 500.0f}));

    host.click(600.0f, 500.0f);
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(DirtyFrames(frames, update) == 0);
    CHECK_FALSE(host.popups.covers({600.0f, 500.0f}));
    CHECK(frames.rebuilds == rebuilds);
}

TEST_CASE("Paint changes inside a hidden subtree stay off the frame until the subtree is shown again",
          "[core][sakura][retention]") {
    F32 offset = 0.0f;
    Sakura::Box first;
    Sakura::Box inside;
    Sakura::Box other;
    Sakura::Stack inner;
    Sakura::Stack outer;
    Sakura::Panel panel;
    Host host(panel);
    Frames frames(host, kScreen, ShellContext());
    const auto update = [&] {
        first.update({.instances = {{.rect = {0.0f, 0.0f, 10.0f, 10.0f}}}});
        inside.update({.instances = {{.rect = {offset, 20.0f, 10.0f, 10.0f}}}});
        other.update({.instances = {{.rect = {0.0f, 40.0f, 10.0f, 10.0f}}}});
        inner.update({.items = {{.child = &inside}, {.child = &other}}});
        outer.update({.items = {{.child = &first}, {.child = &inner}}});
        panel.update({.id = "panel", .content = &outer});
    };
    REQUIRE(frames.settle(update));
    CHECK(DirtyFrames(frames, update) == 0);
    const U64 rebuilds = frames.rebuilds;
    const U64 restructures = frames.restructures;

    inner.setVisible(false);
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(DirtyFrames(frames, update) == 0);

    offset = 4.0f;
    update();
    CHECK(Frames::Access::paintDirty(inside));
    CHECK_FALSE(Frames::Access::paintDirty(host));
    CHECK(frames.run().clean());
    offset = 6.0f;
    CHECK(frames.run(update).clean());
    CHECK(DirtyFrames(frames, update) == 0);
    CHECK(Frames::Access::paintDirty(inside));

    inner.setVisible(true);
    CHECK(Frames::Access::paintDirty(host));
    CHECK(RepaintOnly(frames.run(update)));
    CHECK_FALSE(Frames::Access::paintDirty(inside));
    CHECK(DirtyFrames(frames, update) == 0);

    offset = 8.0f;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(DirtyFrames(frames, update) == 0);
    CHECK(frames.rebuilds == rebuilds);
    CHECK(frames.restructures == restructures);
}

TEST_CASE("A surface image inside a hidden subtree hints not visible until the subtree is shown again",
          "[core][sakura][retention]") {
    SakuraTest::FakeWindow window;
    const auto texture = std::make_shared<SurfaceTexture>();
    Sakura::SurfaceImage image;
    Sakura::Stack inner;
    Sakura::Stack outer;
    Sakura::Panel panel;
    Host host(panel);
    Frames frames(host, kScreen, ShellContext(&window));
    const auto update = [&] {
        image.update({.texture = texture});
        inner.update({.items = {{.child = &image, .size = 100.0f}}});
        outer.update({.items = {{.child = &inner, .size = 120.0f}}});
        panel.update({.id = "panel", .content = &outer});
    };
    const auto presented = [&] {
        frames.run(update);
        return texture->visibleForPresentation(frames.ctx.frame);
    };

    CHECK(presented());
    CHECK(presented());

    inner.setVisible(false);
    for (int frame = 0; frame < 3; ++frame) {
        CHECK_FALSE(presented());
    }

    inner.setVisible(true);
    CHECK(presented());
    CHECK(presented());
}

TEST_CASE("A HUD layer updated every frame stays clean and ignores moves inside its HUDs",
          "[core][sakura][retention]") {
    Probe left({100.0f, 40.0f});
    Probe right({100.0f, 40.0f});
    Sakura::HudLayer layer;
    Sakura::Layer base;
    LayerHost host({&base, &layer});
    Frames frames(host, kScreen, ShellContext());
    const auto update = [&] {
        layer.update({.huds = {
            {.content = &left, .anchor = Sakura::Anchor::TopLeft, .clickable = true, .onClick = [] {}},
            {.content = &right, .anchor = Sakura::Anchor::BottomRight},
        }});
    };

    REQUIRE(frames.settle(update));
    CHECK(DirtyFrames(frames, update) == 0);
    Hover(host, frames, 50.0f, 40.0f);
    CHECK(frames.run(update).clean());
    Hover(host, frames, 700.0f, 560.0f);
    CHECK(frames.run(update).clean());
}

TEST_CASE("A modal taller than the window fits inside it, scrolls its content, and keeps the close button",
          "[core][sakura][retention]") {
    SakuraTest::FakeWindow window;
    Probe content({100.0f, 2000.0f});
    content.consume = false;
    U64 closed = 0;
    Sakura::ModalLayer modal;
    Sakura::Layer base;
    LayerHost host({&base, &modal});
    Frames frames(host, kScreen, ShellContext(&window));
    const auto update = [&] {
        modal.update({.content = &content, .onClose = [&] { ++closed; }});
    };
    REQUIRE(frames.settle(update));
    CHECK(DirtyFrames(frames, update) == 0);
    const U64 rebuilds = frames.rebuilds;

    const Rect first = Peek::Frame(content);
    CHECK(first.height == 2000.0f);
    CHECK(first.y > 19.0f);
    CHECK(host.click(400.0f, 600.0f - 19.0f - 12.0f - 20.0f));
    CHECK(closed == 1);

    host.send(Wheel(400.0f, 300.0f, 0.0f, -3.0f));
    CHECK_FALSE(frames.run(update).rebuilt);
    REQUIRE(frames.settle(update));
    CHECK(Peek::Frame(content).y < first.y);
    CHECK(DirtyFrames(frames, update) == 0);
    CHECK(frames.rebuilds == rebuilds);
}

TEST_CASE("A modal layer stays clean while empty and while open, and toggling it never rebuilds",
          "[core][sakura][retention]") {
    SakuraTest::FakeWindow window;
    Probe content({100.0f, 50.0f});
    bool open = false;
    U64 closed = 0;
    Sakura::ModalLayer modal;
    Sakura::Layer base;
    LayerHost host({&base, &modal});
    Frames frames(host, kScreen, ShellContext(&window));
    const auto update = [&] {
        modal.update({.content = open ? &content : nullptr, .onClose = [&] { ++closed; }});
    };

    REQUIRE(frames.settle(update));
    CHECK(DirtyFrames(frames, update) == 0);
    const U64 rebuilds = frames.rebuilds;

    open = true;
    const auto opened = frames.run(update);
    CHECK(opened.repainted);
    CHECK_FALSE(opened.rebuilt);
    REQUIRE(frames.settle(update));
    CHECK(DirtyFrames(frames, update) == 0);

    const F32 bottom = 600.0f - (content.laidFrame.y - 12.0f);
    Hover(host, frames, 400.0f, bottom - 32.0f);
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(DirtyFrames(frames, update) == 0);
    Hover(host, frames, 402.0f, bottom - 30.0f);
    CHECK(frames.run(update).clean());
    Hover(host, frames, 20.0f, 20.0f);
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(DirtyFrames(frames, update) == 0);

    open = false;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(DirtyFrames(frames, update) == 0);

    open = true;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(DirtyFrames(frames, update) == 0);
    CHECK(frames.rebuilds == rebuilds);
    CHECK(closed == 0);
}

TEST_CASE("A composed shell settles and stays clean under identical updates", "[core][sakura][retention]") {
    SakuraTest::FakeWindow window;
    (void)Sakura::Private::TakeToasts();
    TileDock workspace;
    Probe notes;
    Probe hud({120.0f, 40.0f});
    Probe inspector;
    Dock tabs;
    Sakura::Layer content;
    content.update({.content = &tabs});
    Sakura::HudLayer huds;
    Sakura::FloatingLayer floating;
    Sakura::ToastLayer toasts;
    LayerHost host({&content, &huds, &floating, &toasts});
    Frames frames(host, kScreen, ShellContext(&window));
    const auto update = [&] {
        workspace.update();
        tabs.update({.tabs = true, .accepts = "workspace", .panes = {
            {.id = "workspace", .title = "Workspace", .group = "workspace", .content = &workspace.dock,
             .closable = false},
            {.id = "notes", .title = "Notes", .group = "workspace", .content = &notes},
        }});
        huds.update({.huds = {{.content = &hud, .anchor = Sakura::Anchor::BottomLeft}}});
        floating.update({.windows = {{.id = "inspector", .title = "Inspector", .content = &inspector,
                                      .size = {300.0f, 200.0f}}}});
        toasts.update();
    };

    REQUIRE(frames.settle(update));
    CHECK(DirtyFrames(frames, update) == 0);
    CHECK(workspace.a.layouts == 0);

    host.click(10.0f, 19.0f);
    frames.run(update);
    REQUIRE(frames.settle(update));
    CHECK(DirtyFrames(frames, update) == 0);
    CHECK(workspace.a.layouts > 0);

    Hover(host, frames, 100.0f, 400.0f);
    frames.run(update);
    REQUIRE(frames.settle(update));
    CHECK(DirtyFrames(frames, update) == 0);
    Hover(host, frames, 104.0f, 404.0f);
    CHECK(frames.run(update).clean());
}

TEST_CASE("A floating layer with two windows stays clean and repaints once per close hover change",
          "[core][sakura][retention]") {
    Probe first;
    Probe second;
    Sakura::FloatingLayer layer;
    Sakura::Layer base;
    LayerHost host({&base, &layer});
    Frames frames(host, kScreen, ShellContext());
    const auto update = [&] {
        layer.update({.windows = {
            {.id = "a", .title = "A", .content = &first, .size = {300.0f, 200.0f}},
            {.id = "b", .title = "B", .content = &second, .size = {300.0f, 200.0f}},
        }});
    };

    REQUIRE(frames.settle(update));
    CHECK(DirtyFrames(frames, update) == 0);

    Hover(host, frames, 100.0f, 100.0f);
    CHECK(frames.run(update).clean());

    Hover(host, frames, 260.0f, 300.0f);
    CHECK(frames.run(update).clean());
    Hover(host, frames, 262.0f, 310.0f);
    CHECK(frames.run(update).clean());

    Hover(host, frames, 537.0f, 212.0f);
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(DirtyFrames(frames, update) == 0);

    Hover(host, frames, 400.0f, 350.0f);
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(DirtyFrames(frames, update) == 0);
}

TEST_CASE("Moving a floating window repaints while dragging and settles after release",
          "[core][sakura][retention]") {
    Probe first;
    Probe second;
    Sakura::FloatingLayer layer;
    Sakura::Layer base;
    LayerHost host({&base, &layer});
    Frames frames(host, kScreen, ShellContext());
    const auto update = [&] {
        layer.update({.windows = {
            {.id = "a", .title = "A", .content = &first, .size = {300.0f, 200.0f}},
            {.id = "b", .title = "B", .content = &second, .size = {300.0f, 200.0f}},
        }});
    };
    REQUIRE(frames.settle(update));
    const U64 rebuilds = frames.rebuilds;
    const U64 restructures = frames.restructures;
    const Rect before = second.laidFrame;

    Press(host, frames, 400.0f, 235.0f);
    CHECK(frames.run(update).clean());
    Hover(host, frames, 450.0f, 285.0f);
    CHECK(RepaintOnly(frames.run(update)));
    Hover(host, frames, 460.0f, 295.0f);
    CHECK(RepaintOnly(frames.run(update)));
    Release(host, frames, 460.0f, 295.0f);
    frames.run(update);

    REQUIRE(frames.settle(update));
    CHECK(DirtyFrames(frames, update) == 0);
    CHECK(second.laidFrame.x == before.x + 60.0f);
    CHECK(second.laidFrame.y == before.y + 60.0f);
    CHECK(frames.rebuilds == rebuilds);
    CHECK(frames.restructures == restructures);

    host.click(260.0f, 300.0f);
    const auto raised = frames.run(update);
    CHECK(raised.repainted);
    CHECK_FALSE(raised.rebuilt);
    REQUIRE(frames.settle(update));
    CHECK(DirtyFrames(frames, update) == 0);
}

TEST_CASE("A floating window title drag held still stays clean and repaints only when the window moves",
          "[core][sakura][retention]") {
    Probe first;
    Probe second;
    Sakura::FloatingLayer layer;
    Sakura::Layer base;
    LayerHost host({&base, &layer});
    Frames frames(host, kScreen, ShellContext());
    const auto update = [&] {
        layer.update({.windows = {
            {.id = "a", .title = "A", .content = &first, .size = {300.0f, 200.0f}},
            {.id = "b", .title = "B", .content = &second, .size = {300.0f, 200.0f}},
        }});
    };
    REQUIRE(frames.settle(update));
    const Rect before = second.laidFrame;

    Press(host, frames, 400.0f, 235.0f);
    CHECK(frames.run(update).clean());
    for (int frame = 0; frame < 3; ++frame) {
        Hover(host, frames, 400.0f, 235.0f);
        CHECK(frames.run(update).clean());
    }

    Hover(host, frames, 450.0f, 285.0f);
    CHECK(RepaintOnly(frames.run(update)));
    for (int frame = 0; frame < 3; ++frame) {
        Hover(host, frames, 450.0f, 285.0f);
        CHECK(frames.run(update).clean());
    }
    Release(host, frames, 450.0f, 285.0f);
    frames.run(update);
    CHECK(second.laidFrame.x == before.x + 50.0f);
    CHECK(second.laidFrame.y == before.y + 50.0f);
}

TEST_CASE("A toast layer stays clean when empty and repaints only while a toast fades",
          "[core][sakura][retention]") {
    SakuraTest::FakeWindow window;
    (void)Sakura::Private::TakeToasts();
    Sakura::ToastLayer layer;
    Sakura::Layer base;
    LayerHost host({&base, &layer});
    Frames frames(host, kScreen, ShellContext(&window));
    frames.ctx.time = 9.0;
    const auto update = [&] {
        layer.update();
    };
    const auto at = [&](F64 time) {
        frames.ctx.time = time;
        return frames.run(update);
    };

    REQUIRE(frames.settle(update));
    CHECK(DirtyFrames(frames, update) == 0);
    const U64 rebuilds = frames.rebuilds;

    Sakura::PushToast(Sakura::ToastType::Info, 1000, "Saved the flowgraph");
    const auto shown = at(10.0);
    CHECK(shown.repainted);
    CHECK_FALSE(shown.rebuilt);
    CHECK(RepaintOnly(at(10.05)));
    CHECK(RepaintOnly(at(10.10)));
    CHECK(RepaintOnly(at(10.20)));

    for (const F64 time : {10.30, 10.40, 10.60, 10.80, 11.00, 11.10}) {
        CAPTURE(time);
        CHECK(at(time).clean());
    }

    CHECK(RepaintOnly(at(11.20)));
    CHECK(RepaintOnly(at(11.25)));
    CHECK(RepaintOnly(at(11.40)));

    const auto removed = at(11.45);
    CHECK_FALSE(removed.rebuilt);
    frames.ctx.time = 11.50;
    REQUIRE(frames.settle(update));
    CHECK(DirtyFrames(frames, update) == 0);
    CHECK(frames.rebuilds == rebuilds);
}

TEST_CASE("A background layer without particles stays clean as time advances",
          "[core][sakura][retention]") {
    Sakura::BackgroundLayer background;
    Sakura::Layer base;
    LayerHost host({&background, &base});
    Frames frames(host, kScreen, ShellContext());
    frames.ctx.deltaTime = 0.016f;
    const auto tick = [&] {
        frames.ctx.time += 0.016;
        background.update({.particles = false});
    };

    REQUIRE(frames.settle(tick));
    CHECK(DirtyFrames(frames, tick) == 0);
}

TEST_CASE("A background layer with particles repaints every frame without rebuilding and settles when stopped",
          "[core][sakura][retention]") {
    bool particles = true;
    Sakura::BackgroundLayer background;
    LayerHost host({&background});
    Frames frames(host, kScreen, ShellContext());
    frames.ctx.deltaTime = 0.016f;
    const auto tick = [&] {
        frames.ctx.time += 0.016;
        background.update({.particles = particles, .particleCount = 16});
    };
    frames.run(tick);
    const U64 rebuilds = frames.rebuilds;
    const U64 restructures = frames.restructures;

    for (U64 frame = 0; frame < 10; ++frame) {
        CAPTURE(frame);
        CHECK(RepaintOnly(frames.run(tick)));
    }
    CHECK(frames.rebuilds == rebuilds);
    CHECK(frames.restructures == restructures);

    particles = false;
    REQUIRE(frames.settle(tick));
    CHECK(DirtyFrames(frames, tick) == 0);
}

TEST_CASE("A node field panel with node inputs updated every frame stays clean",
          "[core][sakura][retention]") {
    SakuraTest::FakeWindow window;
    F32 gainValue = 1.5f;
    Sakura::NodeFloatInput gain;
    Sakura::NodeFloatInput step;
    Sakura::NodeBoolInput enabled;
    Sakura::NodeCombo mode;
    Sakura::NodeRangeInput level;
    Sakura::NodeIntInput taps;
    Sakura::NodeUIntInput size;
    Sakura::NodeTextInput name;
    Sakura::NodePathInput path;
    Sakura::NodeInputRow shape;
    Sakura::NodeLabel status;
    Sakura::NodeProgressBar progress;
    Sakura::NodeTable table;
    Sakura::NodeField gainField;
    Sakura::NodeField stepField;
    Sakura::NodeField enabledField;
    Sakura::NodeField modeField;
    Sakura::NodeField levelField;
    Sakura::NodeField tapsField;
    Sakura::NodeField sizeField;
    Sakura::NodeField nameField;
    Sakura::NodeField pathField;
    Sakura::NodeField shapeField;
    Sakura::NodeFieldGrid grid;
    Sakura::NodeContentStack stack;
    Sakura::NodeFieldPanel panel;
    Sakura::Layer layer;
    layer.update({.content = &panel});
    LayerHost host({&layer});
    Frames frames(host, {0.0f, 0.0f, 400.0f, 600.0f}, ShellContext(&window));
    const auto update = [&] {
        gain.update({.id = "gain", .value = gainValue, .unit = "dB", .onChange = [](F32) {}});
        step.update({.id = "step", .value = 0.25f, .step = 0.05f, .onChange = [](F32) {}});
        enabled.update({.id = "enabled", .value = true, .onChange = [](bool) {}});
        mode.update({.id = "mode", .options = {"FM", "AM", "USB"}, .value = "AM",
                     .onChange = [](const std::string&) {}});
        level.update({.id = "level", .min = 0.0f, .max = 10.0f, .value = 4.0f, .onChange = [](F32) {}});
        taps.update({.id = "taps", .value = 64, .onChange = [](I64) {}});
        size.update({.id = "size", .value = 2048, .onChange = [](U64) {}});
        name.update({.id = "name", .value = "", .hint = "Name", .onChange = [](const std::string&) {}});
        path.update({.id = "path", .value = "/tmp/capture.bin", .onChange = [](const std::string&) {}});
        shape.update({.id = "shape", .shape = "[8, 1024]", .axes = "[0]",
                      .combos = {{.options = {"CF32", "F32"}, .value = "CF32", .width = 64.0f}}});
        status.update({.id = "status", .str = "Running", .boxed = true});
        progress.update({.id = "progress", .value = 0.4f, .overlay = "40%"});
        table.update({.id = "table", .columns = {"Key", "Value"}, .rows = {{"rate", "2 MHz"}, {"gain", "10 dB"}}});
        gainField.update({.id = "gainField", .label = "Gain", .content = &gain});
        stepField.update({.id = "stepField", .label = "Step", .content = &step});
        enabledField.update({.id = "enabledField", .label = "Enabled", .content = &enabled});
        modeField.update({.id = "modeField", .label = "Mode", .content = &mode});
        levelField.update({.id = "levelField", .label = "Level", .content = &level});
        tapsField.update({.id = "tapsField", .label = "Taps", .content = &taps});
        sizeField.update({.id = "sizeField", .label = "Size", .content = &size});
        nameField.update({.id = "nameField", .label = "Name", .content = &name});
        pathField.update({.id = "pathField", .label = "Path", .content = &path});
        shapeField.update({.id = "shapeField", .label = "Shape", .content = &shape});
        grid.update({.id = "grid", .items = {{.child = &tapsField}, {.child = &sizeField}}});
        stack.update({.id = "stack", .items = {
            {.child = &status},
            {.child = &progress},
            {.child = &gainField},
            {.child = &stepField},
            {.child = &enabledField},
            {.child = &modeField},
            {.child = &levelField},
            {.child = &grid},
            {.child = &nameField},
            {.child = &pathField},
            {.child = &shapeField},
            {.child = &table},
        }});
        panel.update({.id = "panel", .content = &stack, .height = Sakura::Fill});
    };

    REQUIRE(frames.settle(update));
    CHECK(DirtyFrames(frames, update) == 0);

    Hover(host, frames, 300.0f, 590.0f);
    frames.run(update);
    REQUIRE(frames.settle(update));
    CHECK(DirtyFrames(frames, update) == 0);

    const U64 rebuilds = frames.rebuilds;
    gainValue = 3.0f;
    const auto changed = frames.run(update);
    CHECK(changed.repainted);
    CHECK_FALSE(changed.rebuilt);
    REQUIRE(frames.settle(update));
    CHECK(DirtyFrames(frames, update) == 0);
    CHECK(frames.rebuilds == rebuilds);
}

TEST_CASE("A node editor with two nodes and a wire stays clean under identical updates",
          "[core][sakura][retention]") {
    SakuraTest::FakeWindow window;
    GraphNodes nodes;
    Sakura::NodeLabel status;
    Sakura::NodeContentStack sourceContent;
    Sakura::NodeFloatInput gain;
    Sakura::NodeField gainField;
    Sakura::NodeContentStack sinkContent;
    NodeEditor editor;
    Sakura::Layer layer;
    layer.update({.content = &editor});
    LayerHost host({&layer});
    Frames frames(host, kScreen, ShellContext(&window));
    const auto update = [&] {
        status.update({.id = "status", .str = "Running"});
        sourceContent.update({.id = "source:content", .items = {{.child = &status}}});
        gain.update({.id = "gain", .value = 1.5f, .onChange = [](F32) {}});
        gainField.update({.id = "gainField", .label = "Gain", .content = &gain});
        sinkContent.update({.id = "sink:content", .items = {{.child = &gainField}}});
        editor.update({
            .id = "editor",
            .nodes = {
                nodes.node("source", {40.0f, 40.0f}, &sourceContent, {{.id = "out", .label = "out"}}),
                nodes.node("sink", {360.0f, 80.0f}, &sinkContent, {{.id = "in", .isInput = true, .label = "in"}}),
            },
            .links = {{.id = "link",
                       .start = {.nodeId = "source", .pinId = "out", .isInput = false},
                       .end = {.nodeId = "sink", .pinId = "in", .isInput = true}}},
            .onLinkCreated = [](NodeEditor::PinRef, NodeEditor::PinRef) {},
            .onLinkDestroyed = [](const std::string&) {},
        });
    };

    REQUIRE(frames.settle(update));
    CHECK(DirtyFrames(frames, update) == 0);
    REQUIRE(nodes.geometry.size() == 2);

    Hover(host, frames, 700.0f, 500.0f);
    frames.run(update);
    REQUIRE(frames.settle(update));
    CHECK(DirtyFrames(frames, update) == 0);
    Hover(host, frames, 702.0f, 503.0f);
    CHECK(frames.run(update).clean());
    Hover(host, frames, 690.0f, 480.0f);
    CHECK(frames.run(update).clean());
}

TEST_CASE("Dragging a node repaints while moving and settles once its geometry is fed back",
          "[core][sakura][retention]") {
    SakuraTest::FakeWindow window;
    GraphNodes nodes;
    Sakura::NodeLabel status;
    Sakura::NodeContentStack sourceContent;
    Sakura::NodeLabel other;
    Sakura::NodeContentStack sinkContent;
    NodeEditor editor;
    Sakura::Layer layer;
    layer.update({.content = &editor});
    LayerHost host({&layer});
    Frames frames(host, kScreen, ShellContext(&window));
    const auto update = [&] {
        status.update({.id = "status", .str = "Running"});
        sourceContent.update({.id = "source:content", .items = {{.child = &status}}});
        other.update({.id = "other", .str = "Idle"});
        sinkContent.update({.id = "sink:content", .items = {{.child = &other}}});
        editor.update({
            .id = "editor",
            .nodes = {
                nodes.node("source", {40.0f, 40.0f}, &sourceContent, {{.id = "out", .label = "out"}}),
                nodes.node("sink", {360.0f, 80.0f}, &sinkContent, {{.id = "in", .isInput = true, .label = "in"}}),
            },
            .links = {{.id = "link",
                       .start = {.nodeId = "source", .pinId = "out", .isInput = false},
                       .end = {.nodeId = "sink", .pinId = "in", .isInput = true}}},
        });
    };
    Hover(host, frames, 700.0f, 500.0f);
    REQUIRE(frames.settle(update));
    const U64 rebuilds = frames.rebuilds;

    Press(host, frames, 100.0f, 50.0f);
    frames.run(update);
    Hover(host, frames, 150.0f, 80.0f);
    CHECK(RepaintOnly(frames.run(update)));
    Hover(host, frames, 160.0f, 90.0f);
    CHECK(RepaintOnly(frames.run(update)));
    Release(host, frames, 160.0f, 90.0f);
    frames.run(update);

    REQUIRE(frames.settle(update));
    CHECK(DirtyFrames(frames, update) == 0);
    CHECK(nodes.geometry.at("source").gridPosition.x == 100.0f);
    CHECK(nodes.geometry.at("source").gridPosition.y == 80.0f);
    CHECK(frames.rebuilds == rebuilds);
}

TEST_CASE("A column dock settles, and column resizes and new panes settle again without rebuilds",
          "[core][sakura][retention]") {
    Probe a;
    Probe b;
    Probe c;
    Probe d;
    bool withD = false;
    std::optional<DockTiles> tiles = Row({0.0f, 0.0f, 0.0f}, {Leaf("a"), Leaf("b"), Leaf("c")});
    Dock dock;
    Sakura::Layer layer;
    layer.update({.content = &dock});
    LayerHost host({&layer});
    Frames frames(host, kScreen, ShellContext());
    const auto update = [&] {
        frames.ctx.time += 0.1;
        std::vector<Dock::Pane> panes = {MakePane("a", &a), MakePane("b", &b), MakePane("c", &c)};
        if (withD) {
            panes.push_back(MakePane("d", &d));
        }
        dock.update({
            .accepts = "panel",
            .panes = std::move(panes),
            .tiles = tiles,
            .onTiles = [&](std::optional<DockTiles> next) {
                tiles = std::move(next);
            },
            .columns = true,
            .flexPane = "a",
        });
    };

    REQUIRE(frames.settle(update));
    CHECK(DirtyFrames(frames, update) == 0);
    const U64 rebuilds = frames.rebuilds;
    CHECK(c.visible());

    const auto before = tiles;
    host.send(Mouse(MouseEventType::Click, 291.0f, 300.0f));
    host.send(Mouse(MouseEventType::Move, 241.0f, 300.0f));
    host.send(Mouse(MouseEventType::Release, 241.0f, 300.0f));
    CHECK(tiles != before);
    CHECK_FALSE(frames.run(update).rebuilt);
    REQUIRE(frames.settle(update, 30));
    CHECK(DirtyFrames(frames, update) == 0);

    withD = true;
    CHECK_FALSE(frames.run(update).rebuilt);
    REQUIRE(frames.settle(update, 30));
    CHECK(DirtyFrames(frames, update) == 0);
    CHECK(d.visible());
    CHECK(frames.rebuilds == rebuilds);
}

TEST_CASE("Hovering a toast repaints once for its close badge and stays clean while held",
          "[core][sakura][retention]") {
    SakuraTest::FakeWindow window;
    (void)Sakura::Private::TakeToasts();
    Sakura::ToastLayer layer;
    Sakura::Layer base;
    LayerHost host({&base, &layer});
    Frames frames(host, kScreen, ShellContext(&window));
    const auto update = [&] {
        layer.update();
    };
    const auto at = [&](F64 time) {
        frames.ctx.time = time;
        return frames.run(update);
    };

    Sakura::PushToast(Sakura::ToastType::Info, 1000, "Saved the flowgraph");
    at(10.0);
    at(10.2);
    CHECK(at(10.3).clean());

    REQUIRE(layer.covers({40.0f, 570.0f}));
    host.send(Mouse(MouseEventType::Move, 40.0f, 570.0f));
    CHECK(RepaintOnly(at(10.4)));
    for (const F64 time : {10.5, 11.0, 12.0, 15.0}) {
        CAPTURE(time);
        CHECK(at(time).clean());
    }
    CHECK(layer.covers({40.0f, 570.0f}));
}

TEST_CASE("A column dock at a pixel ratio of two settles and stays clean, and a wheel over it stays clean",
          "[core][sakura][retention]") {
    ColumnDock columns;
    Sakura::Layer layer;
    layer.update({.content = &columns.dock});
    LayerHost host({&layer});
    auto ctx = ShellContext();
    ctx.pixelRatio = 2.0f;
    ctx.framebufferSize = {1600, 1200};
    Frames frames(host, {0.0f, 0.0f, 1600.0f, 1200.0f}, ctx);
    const auto update = [&] {
        frames.ctx.time += 0.1;
        columns.update();
    };
    REQUIRE(frames.settle(update, 30));
    CHECK(DirtyFrames(frames, update) == 0);
    const U64 rebuilds = frames.rebuilds;

    Hover(host, frames, 400.0f, 600.0f);
    CHECK(frames.run(update).clean());
    Hover(host, frames, 420.0f, 640.0f);
    CHECK(frames.run(update).clean());

    host.send(Wheel(200.0f, 32.0f, -1.0f, 0.0f));
    CHECK(frames.run(update).clean());
    CHECK(frames.rebuilds == rebuilds);
}

TEST_CASE("Dragging and double clicking the divider between stacked panes repaints only and settles",
          "[core][sakura][retention]") {
    ColumnDock columns;
    columns.tiles = Row({0.0f, 0.0f}, {Leaf("a"), Column({0.5f, 0.5f}, {Leaf("b"), Leaf("c")})});
    Sakura::Layer layer;
    layer.update({.content = &columns.dock});
    LayerHost host({&layer});
    Frames frames(host, kScreen, ShellContext());
    const auto update = [&] {
        frames.ctx.time += 0.1;
        columns.update();
    };
    REQUIRE(frames.settle(update, 30));
    CHECK(DirtyFrames(frames, update) == 0);
    const U64 rebuilds = frames.rebuilds;
    const U64 restructures = frames.restructures;
    const auto before = columns.tiles;

    Hover(host, frames, 600.0f, 300.0f);
    CHECK(RepaintOnly(frames.run(update)));
    Hover(host, frames, 620.0f, 300.0f);
    CHECK(frames.run(update).clean());

    Press(host, frames, 620.0f, 300.0f);
    CHECK_FALSE(frames.run(update).rebuilt);
    Hover(host, frames, 620.0f, 200.0f);
    CHECK(RepaintOnly(frames.run(update)));
    Hover(host, frames, 620.0f, 200.0f);
    CHECK(frames.run(update).clean());
    Release(host, frames, 620.0f, 200.0f);
    CHECK(columns.tiles != before);
    REQUIRE(frames.settle(update, 30));
    CHECK(DirtyFrames(frames, update) == 0);

    const auto dragged = columns.tiles;
    host.send(Mouse(MouseEventType::Click, 620.0f, 201.0f, MouseButton::Left, 2));
    CHECK(RepaintOnly(frames.run(update)));
    Release(host, frames, 620.0f, 201.0f);
    CHECK(columns.tiles != dragged);
    REQUIRE(frames.settle(update, 30));
    CHECK(DirtyFrames(frames, update) == 0);
    CHECK(frames.rebuilds == rebuilds);
    CHECK(frames.restructures == restructures);
}

TEST_CASE("A tab dock with end chrome around a column dock settles and idle hovers over its chrome stay clean",
          "[core][sakura][retention]") {
    ColumnDock columns;
    Probe end({30.0f, 20.0f});
    Dock dock;
    Sakura::Layer layer;
    layer.update({.content = &dock});
    LayerHost host({&layer});
    Frames frames(host, kScreen, ShellContext());
    const auto update = [&] {
        frames.ctx.time += 0.1;
        columns.update();
        dock.update({.tabs = true, .accepts = "main", .panes = {MakePane("main", &columns.dock)}, .end = &end});
    };
    REQUIRE(frames.settle(update, 30));
    CHECK(DirtyFrames(frames, update) == 0);
    const U64 rebuilds = frames.rebuilds;
    CHECK(end.visible());

    for (const auto& point : {Extent2D<F32>{779.0f, 20.0f}, Extent2D<F32>{770.0f, 18.0f},
                              Extent2D<F32>{700.0f, 19.0f}, Extent2D<F32>{300.0f, 300.0f}}) {
        CAPTURE(point.x, point.y);
        Hover(host, frames, point.x, point.y);
        CHECK(frames.run(update).clean());
    }
    CHECK(frames.rebuilds == rebuilds);
}

TEST_CASE("A column dock cancelled or left in the middle of a press settles clean",
          "[core][sakura][retention]") {
    ColumnDock columns;
    Sakura::Layer layer;
    layer.update({.content = &columns.dock});
    LayerHost host({&layer});
    Frames frames(host, kScreen, ShellContext());
    const auto update = [&] {
        frames.ctx.time += 0.1;
        columns.update();
    };
    REQUIRE(frames.settle(update, 30));
    const U64 rebuilds = frames.rebuilds;
    const auto before = columns.tiles;

    Press(host, frames, 291.0f, 300.0f);
    frames.run(update);
    Hover(host, frames, 241.0f, 300.0f);
    CHECK(RepaintOnly(frames.run(update)));
    host.send(Mouse(MouseEventType::Cancel, 241.0f, 300.0f));
    CHECK(columns.tiles != before);
    CHECK_FALSE(frames.run(update).rebuilt);
    REQUIRE(frames.settle(update, 30));
    CHECK(DirtyFrames(frames, update) == 0);

    Press(host, frames, 495.0f, 300.0f);
    frames.run(update);
    Hover(host, frames, 525.0f, 300.0f);
    CHECK(RepaintOnly(frames.run(update)));
    host.send(Mouse(MouseEventType::Leave, 525.0f, 300.0f));
    CHECK_FALSE(frames.run(update).rebuilt);
    REQUIRE(frames.settle(update, 30));
    CHECK(DirtyFrames(frames, update) == 0);
    host.send(Mouse(MouseEventType::Leave, 525.0f, 300.0f));
    CHECK(frames.run(update).clean());
    CHECK(frames.rebuilds == rebuilds);
}

TEST_CASE("Focusing a floating window restructures and repaints in a single frame",
          "[core][sakura][retention]") {
    Probe first;
    Probe second;
    Sakura::FloatingLayer layer;
    Sakura::Layer base;
    LayerHost host({&base, &layer});
    Frames frames(host, kScreen, ShellContext());
    const auto update = [&] {
        layer.update({.windows = {
            {.id = "a", .title = "A", .content = &first, .size = {300.0f, 200.0f}},
            {.id = "b", .title = "B", .content = &second, .size = {300.0f, 200.0f}},
        }});
    };
    REQUIRE(frames.settle(update));

    host.click(260.0f, 300.0f);
    const auto focus = frames.run(update);
    CHECK_FALSE(focus.rebuilt);
    CHECK(focus.restructured);
    CHECK(focus.repainted);
    CHECK(frames.run(update).clean());
    CHECK(DirtyFrames(frames, update) == 0);
}

TEST_CASE("Adding and removing a floating window settles in a single frame", "[core][sakura][retention]") {
    Probe first;
    Probe second;
    bool both = false;
    Sakura::FloatingLayer layer;
    Sakura::Layer base;
    LayerHost host({&base, &layer});
    Frames frames(host, kScreen, ShellContext());
    const auto update = [&] {
        std::vector<Sakura::FloatingLayer::Window> windows = {
            {.id = "a", .title = "A", .content = &first, .size = {300.0f, 200.0f}},
        };
        if (both) {
            windows.push_back({.id = "b", .title = "B", .content = &second, .size = {300.0f, 200.0f}});
        }
        layer.update({.windows = std::move(windows)});
    };
    REQUIRE(frames.settle(update));

    both = true;
    CHECK_FALSE(frames.run(update).clean());
    CHECK(frames.run(update).clean());

    both = false;
    CHECK_FALSE(frames.run(update).clean());
    CHECK(frames.run(update).clean());
}

TEST_CASE("Adding and removing a floating window restructures exactly once each and then settles",
          "[core][sakura][retention]") {
    Probe first;
    Probe second;
    bool both = false;
    Sakura::FloatingLayer layer;
    Sakura::Layer base;
    LayerHost host({&base, &layer});
    Frames frames(host, kScreen, ShellContext());
    const auto update = [&] {
        std::vector<Sakura::FloatingLayer::Window> windows = {
            {.id = "a", .title = "A", .content = &first, .size = {300.0f, 200.0f}},
        };
        if (both) {
            windows.push_back({.id = "b", .title = "B", .content = &second, .size = {300.0f, 200.0f}});
        }
        layer.update({.windows = std::move(windows)});
    };
    REQUIRE(frames.settle(update));
    const U64 rebuilds = frames.rebuilds;
    const U64 restructures = frames.restructures;

    both = true;
    const auto added = frames.run(update);
    CHECK(added.restructured);
    CHECK_FALSE(added.rebuilt);
    CHECK(added.repainted);
    CHECK(frames.run(update).clean());
    CHECK(DirtyFrames(frames, update) == 0);
    CHECK(frames.restructures == restructures + 1);
    CHECK(second.visible());

    both = false;
    const auto removed = frames.run(update);
    CHECK(removed.restructured);
    CHECK_FALSE(removed.rebuilt);
    CHECK(removed.repainted);
    CHECK(frames.run(update).clean());
    CHECK(DirtyFrames(frames, update) == 0);
    CHECK(frames.restructures == restructures + 2);
    CHECK_FALSE(second.visible());

    both = true;
    const auto readded = frames.run(update);
    CHECK(readded.restructured);
    CHECK_FALSE(readded.rebuilt);
    CHECK(frames.run(update).clean());
    CHECK(DirtyFrames(frames, update) == 0);
    CHECK(frames.restructures == restructures + 3);
    CHECK(frames.rebuilds == rebuilds);
}

TEST_CASE("Removing a tile dock pane together with its leaf restructures exactly once and then settles",
          "[core][sakura][retention]") {
    TileDock tiles;
    Sakura::Layer layer;
    layer.update({.content = &tiles.dock});
    LayerHost host({&layer});
    Frames frames(host, kScreen, ShellContext());
    const auto update = [&] {
        tiles.update();
    };
    const auto original = tiles.tiles;
    tiles.withD = true;
    REQUIRE(frames.settle(update));
    REQUIRE(tiles.d.visible());
    const U64 rebuilds = frames.rebuilds;
    const U64 restructures = frames.restructures;

    tiles.withD = false;
    tiles.tiles = original;
    const auto removed = frames.run(update);
    CHECK(removed.restructured);
    CHECK_FALSE(removed.rebuilt);
    CHECK(removed.repainted);
    REQUIRE(frames.settle(update));
    CHECK(DirtyFrames(frames, update) == 0);
    CHECK(frames.restructures == restructures + 1);
    CHECK_FALSE(tiles.d.visible());
    CHECK(tiles.tiles == original);
    CHECK(frames.rebuilds == rebuilds);
}

TEST_CASE("Editing the entries of an open menu only repaints and settles again", "[core][sakura][retention]") {
    std::vector<Sakura::MenuEntry> entries = {
        {.label = "Open"},
        {.label = "Save", .shortcut = "S"},
        {.separator = true},
        {.label = "Quit"},
    };
    Sakura::Menu menu;
    LayerSlot slot;
    slot.hold(menu, {100.0f, 100.0f, 1.0f, 1.0f});
    LayerHost host({&slot});
    Frames frames(host, kScreen, ShellContext());
    const auto update = [&] {
        menu.update({.entries = entries, .open = true, .atCursor = false});
    };
    REQUIRE(frames.settle(update));
    CHECK(DirtyFrames(frames, update) == 0);
    const U64 rebuilds = frames.rebuilds;
    const U64 restructures = frames.restructures;

    const std::vector<std::function<void()>> edits = {
        [&] { entries[1].label = "Save As"; },
        [&] { entries[1].shortcut = "Shift+S"; },
        [&] { entries[0].selected = true; },
        [&] { entries[3].enabled = false; },
        [&] { entries[1].colorKey = "warning_yellow"; },
        [&] { entries[0].label = "Open Recent"; },
    };
    for (U64 index = 0; index < edits.size(); ++index) {
        CAPTURE(index);
        edits[index]();
        CHECK(RepaintOnly(frames.run(update)));
        CHECK(DirtyFrames(frames, update) == 0);
    }
    CHECK(frames.rebuilds == rebuilds);
    CHECK(frames.restructures == restructures);
}

TEST_CASE("HUD config changes only repaint, a new HUD restructures once, and removing one only repaints",
          "[core][sakura][retention]") {
    Probe left({100.0f, 40.0f});
    Probe right({100.0f, 40.0f});
    bool both = false;
    F32 margin = 12.0f;
    std::string background = "background";
    Sakura::HudLayer layer;
    Sakura::Layer base;
    LayerHost host({&base, &layer});
    Frames frames(host, kScreen, ShellContext());
    const auto update = [&] {
        std::vector<Sakura::HudLayer::Hud> huds = {
            {.content = &left, .anchor = Sakura::Anchor::TopLeft, .margin = margin, .backgroundColorKey = background},
        };
        if (both) {
            huds.push_back({.content = &right, .anchor = Sakura::Anchor::BottomRight});
        }
        layer.update({.huds = std::move(huds)});
    };
    REQUIRE(frames.settle(update));
    CHECK(DirtyFrames(frames, update) == 0);
    const U64 rebuilds = frames.rebuilds;
    const U64 restructures = frames.restructures;

    background = "card";
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(DirtyFrames(frames, update) == 0);

    margin = 20.0f;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(DirtyFrames(frames, update) == 0);
    CHECK(left.laidFrame == Rect{32.0f, 32.0f, 100.0f, 40.0f});

    both = true;
    const auto added = frames.run(update);
    CHECK(added.restructured);
    CHECK_FALSE(added.rebuilt);
    CHECK(added.repainted);
    CHECK(frames.run(update).clean());
    CHECK(DirtyFrames(frames, update) == 0);
    CHECK(frames.restructures == restructures + 1);
    CHECK(right.laidFrame == Rect{676.0f, 536.0f, 100.0f, 40.0f});

    both = false;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(DirtyFrames(frames, update) == 0);
    CHECK_FALSE(right.visible());
    CHECK_FALSE(layer.covers({700.0f, 560.0f}));

    both = true;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(DirtyFrames(frames, update) == 0);
    CHECK(right.visible());
    CHECK(frames.restructures == restructures + 1);
    CHECK(frames.rebuilds == rebuilds);

    for (const auto type : {MouseEventType::Cancel, MouseEventType::Leave}) {
        host.send(Mouse(type, 50.0f, 40.0f));
        CHECK(frames.run(update).clean());
    }
}

TEST_CASE("Swapping modal content restructures once for new content and only repaints for content shown before",
          "[core][sakura][retention]") {
    SakuraTest::FakeWindow window;
    Probe first({100.0f, 50.0f});
    Probe second({200.0f, 80.0f});
    Probe* content = &first;
    Sakura::ModalLayer modal;
    Sakura::Layer base;
    LayerHost host({&base, &modal});
    Frames frames(host, kScreen, ShellContext(&window));
    const auto update = [&] {
        modal.update({.content = content});
    };
    REQUIRE(frames.settle(update));
    CHECK(DirtyFrames(frames, update) == 0);
    const U64 rebuilds = frames.rebuilds;
    const U64 restructures = frames.restructures;

    content = &second;
    const auto swapped = frames.run(update);
    CHECK(swapped.restructured);
    CHECK_FALSE(swapped.rebuilt);
    CHECK(frames.run(update).clean());
    CHECK(DirtyFrames(frames, update) == 0);
    CHECK(second.laidFrame.width == 526.0f);
    CHECK(second.laidFrame.height == 80.0f);
    CHECK_FALSE(first.visible());

    content = &first;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(DirtyFrames(frames, update) == 0);
    CHECK(first.visible());
    CHECK_FALSE(second.visible());

    content = &second;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(DirtyFrames(frames, update) == 0);
    CHECK(frames.restructures == restructures + 1);
    CHECK(frames.rebuilds == rebuilds);
}

TEST_CASE("Adding a tile dock pane restructures exactly once and then settles", "[core][sakura][retention]") {
    TileDock tiles;
    Sakura::Layer layer;
    layer.update({.content = &tiles.dock});
    LayerHost host({&layer});
    Frames frames(host, kScreen, ShellContext());
    const auto update = [&] {
        tiles.update();
    };
    REQUIRE(frames.settle(update));
    const U64 restructures = frames.restructures;

    tiles.withD = true;
    const auto added = frames.run(update);
    CHECK(added.restructured);
    CHECK_FALSE(added.rebuilt);
    REQUIRE(frames.settle(update));
    CHECK(DirtyFrames(frames, update) == 0);
    CHECK(frames.restructures == restructures + 1);
}

TEST_CASE("A popup chain deeper than one arrangement pass places every level without further input",
          "[core][sakura][retention]") {
    constexpr U64 kDepth = 40;
    Probe leaf({40.0f, 20.0f});
    std::vector<std::unique_ptr<Nest>> levels(kDepth);
    for (U64 index = kDepth; index-- > 0;) {
        Sakura::Component& inner = index + 1 < kDepth ? static_cast<Sakura::Component&>(*levels[index + 1])
                                                      : static_cast<Sakura::Component&>(leaf);
        levels[index] = std::make_unique<Nest>(Extent2D<F32>{40.0f, 20.0f}, inner);
    }
    LayerSlot slot;
    slot.request({{.content = levels.front().get(), .anchor = {10.0f, 10.0f, 1.0f, 1.0f},
                   .placement = Sakura::PopupPlacement::Point}});
    LayerHost host({&slot});
    Frames frames(host);
    frames.verify = false;
    REQUIRE(frames.settle([] {}, 16));

    CHECK(Find<Nest>(host).size() == kDepth);
    const auto probes = Find<Probe>(host);
    CHECK(std::find(probes.begin(), probes.end(), &leaf) != probes.end());
}
