#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <jetstream/render/sakura/components/button.hh>
#include <jetstream/render/sakura/components/label.hh>
#include <jetstream/render/sakura/components/node/content_stack.hh>
#include <jetstream/render/sakura/components/node/editor.hh>
#include <jetstream/render/sakura/components/node/field.hh>
#include <jetstream/render/sakura/components/node/text_input.hh>
#include <jetstream/render/sakura/typography.hh>

#include "harness.hh"
#include "component_harness.hh"

#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace Jetstream;
using namespace SakuraTest;

namespace {


using NodeEditor = Sakura::NodeEditor;
using PinRef = NodeEditor::PinRef;
using ResizeAxes = NodeEditor::ResizeAxes;

struct Content : Probe {
    using Probe::Probe;

    Extent2D<F32> shift() const {
        return translation();
    }
};

struct Block {
    std::string id;
    Content content{{0.0f, 80.0f}};
    Sakura::Component* field = nullptr;
    Sakura::NodeContentStack stack;
    NodeEditor::Node spec;
    bool present = true;
    bool withContent = true;
    std::vector<NodeEditor::Geometry> reports;
    U64 menus = 0;
    U64 toggles = 0;
    bool detachable = false;
    U64 detaches = 0;

    const NodeEditor::Geometry& last() const {
        return reports.back();
    }
};

struct Graph {
    SakuraTest::FakeWindow window;
    NodeEditor editor;
    Host host{editor};
    Frames frames{host};
    std::vector<std::unique_ptr<Block>> blocks;
    std::vector<NodeEditor::Link> links;
    std::vector<std::pair<PinRef, PinRef>> created;
    std::vector<std::string> destroyed;
    std::vector<Extent2D<F32>> menus;
    std::vector<Extent2D<F32>> doubleClicks;
    std::vector<std::vector<std::string>> copies;
    std::optional<Rect> viewport;
    std::optional<Extent2D<F32>> center;
    U64 centerReports = 0;
    bool frameOnOpen = false;
    std::optional<NodeEditor::View> savedView;
    std::vector<NodeEditor::View> viewReports;
    F32 scale = 1.0f;

    explicit Graph(F32 pixelRatio = 1.0f) {
        frames.ctx.render = &window;
        frames.ctx.pixelRatio = pixelRatio;
    }

    Block& add(const std::string& id,
               Extent2D<F32> position,
               Extent2D<F32> dimensions = {120.0f, 0.0f},
               ResizeAxes resize = ResizeAxes::X,
               std::vector<NodeEditor::Pin> pins = {}) {
        auto block = std::make_unique<Block>();
        block->id = id;
        block->spec = {
            .id = id,
            .header = false,
            .title = id,
            .pins = std::move(pins),
            .resize = resize,
            .dimensions = dimensions,
            .gridPosition = position,
        };
        blocks.push_back(std::move(block));
        return *blocks.back();
    }

    void update() {
        std::vector<NodeEditor::Node> nodes;
        for (auto& owned : blocks) {
            Block& block = *owned;
            if (!block.present) {
                continue;
            }
            std::vector<Sakura::NodeContentStack::Item> items = {{.child = &block.content}};
            if (block.field) {
                items.push_back({.child = block.field});
            }
            block.stack.update({.id = block.id + ":content", .items = std::move(items)});
            NodeEditor::Node node = block.spec;
            node.content = block.withContent ? &block.stack : nullptr;
            node.onContextMenu = [&block] {
                ++block.menus;
            };
            if (node.collapsible) {
                node.onToggleCollapse = [&block] {
                    ++block.toggles;
                };
            }
            if (block.detachable) {
                node.onDetach = [&block] {
                    ++block.detaches;
                };
            }
            node.onGeometryChange = [&block](const NodeEditor::Geometry& next) {
                block.reports.push_back(next);
                block.spec.gridPosition = next.gridPosition;
                block.spec.dimensions.x = next.contentDimensions.x;
                if ((static_cast<U8>(block.spec.resize) & static_cast<U8>(ResizeAxes::Y)) != 0) {
                    block.spec.dimensions.y = next.contentDimensions.y;
                }
            };
            nodes.push_back(std::move(node));
        }
        editor.update({
            .id = "editor",
            .frameOnOpen = frameOnOpen,
            .scale = scale,
            .nodes = std::move(nodes),
            .links = links,
            .onEditorContextMenu = [this](Extent2D<F32> point) {
                menus.push_back(point);
            },
            .onEditorDoubleClick = [this](Extent2D<F32> point) {
                doubleClicks.push_back(point);
            },
            .onLinkCreated = [this](PinRef start, PinRef end) {
                created.emplace_back(std::move(start), std::move(end));
            },
            .onLinkDestroyed = [this](const std::string& id) {
                destroyed.push_back(id);
            },
            .onCopyShortcut = [this](const std::vector<std::string>& ids) {
                copies.push_back(ids);
            },
            .onViewportGridChange = [this](Rect next) {
                viewport = next;
                center = Extent2D<F32>{next.x + next.width * 0.5f, next.y + next.height * 0.5f};
                ++centerReports;
            },
            .view = savedView,
            .onViewChange = [this](NodeEditor::View view) { viewReports.push_back(view); },
        });
    }

    FrameStats step() {
        return frames.run([this] {
            update();
        });
    }

    bool send(MouseEventType type, F32 x, F32 y, MouseButton button = MouseButton::Left, U8 clicks = 1,
              KeyModifiers modifiers = {}) {
        frames.ctx.mouse = Extent2D<F32>{x, y};
        MouseEvent event = Mouse(type, x, y, button, clicks);
        event.modifiers = modifiers;
        return host.send(event);
    }

    void press(F32 x, F32 y, MouseButton button = MouseButton::Left, U8 clicks = 1, KeyModifiers modifiers = {}) {
        const U64 generation = Sakura::Access::focusGeneration(host);
        send(MouseEventType::Click, x, y, button, clicks, modifiers);
        Sakura::Access::clickFocus(host, Extent2D<F32>{x, y}, true, generation);
        step();
    }

    void move(F32 x, F32 y) {
        send(MouseEventType::Move, x, y);
        step();
    }

    void release(F32 x, F32 y, MouseButton button = MouseButton::Left) {
        send(MouseEventType::Release, x, y, button);
        step();
    }

    void drag(F32 fromX, F32 fromY, F32 toX, F32 toY) {
        press(fromX, fromY);
        move(toX, toY);
        release(toX, toY);
    }

    void click(F32 x, F32 y, U8 clicks = 1) {
        press(x, y, MouseButton::Left, clicks);
        release(x, y);
    }

    std::vector<std::string> copy() {
        const U64 before = copies.size();
        CHECK(Tap(host, KeyCode::C, {.control = true}));
        step();
        REQUIRE(copies.size() == before + 1);
        return copies.back();
    }
};

bool Contains(const Rect& outer, const Rect& inner) {
    return inner.x >= outer.x && inner.y >= outer.y && inner.right() <= outer.right() &&
           inner.bottom() <= outer.bottom();
}

NodeEditor::Pin Output(const std::string& id) {
    return {.id = id, .isInput = false, .label = id};
}

NodeEditor::Pin Input(const std::string& id, bool detach = false) {
    return {.id = id, .isInput = true, .label = id, .enableDetach = detach};
}

PinRef Ref(const std::string& node, const std::string& pin, bool input) {
    return {.nodeId = node, .pinId = pin, .isInput = input};
}

NodeEditor::Link Wire(const std::string& id, PinRef start, PinRef end) {
    return {.id = id, .start = std::move(start), .end = std::move(end)};
}

void CheckPoint(Extent2D<F32> actual, F32 x, F32 y) {
    CHECK(actual.x == Catch::Approx(x));
    CHECK(actual.y == Catch::Approx(y));
}

void CheckGeometry(const NodeEditor::Geometry& geometry, Extent2D<F32> position, Extent2D<F32> dimensions) {
    CheckPoint(geometry.gridPosition, position.x, position.y);
    CheckPoint(geometry.contentDimensions, dimensions.x, dimensions.y);
}

void CheckNear(Extent2D<F32> actual, F32 x, F32 y, F32 margin) {
    CHECK(actual.x == Catch::Approx(x).margin(margin));
    CHECK(actual.y == Catch::Approx(y).margin(margin));
}

Extent2D<F32> CanvasPoint(Graph& graph, F32 x, F32 y) {
    const U64 before = graph.menus.size();
    graph.press(x, y, MouseButton::Right);
    graph.release(x, y, MouseButton::Right);
    REQUIRE(graph.menus.size() == before + 1);
    return graph.menus.back();
}

bool Settle(Graph& graph) {
    return graph.frames.settle([&] {
        graph.update();
    });
}

U64 Dirty(Graph& graph) {
    return DirtyFrames(graph.frames, [&] {
        graph.update();
    });
}

}  // namespace

TEST_CASE("Saved editor navigation survives delayed nodes and rejects nonfinite input",
          "[core][sakura][node-editor][recovery]") {
    Graph graph;
    graph.frameOnOpen = true;
    const NodeEditor::View saved{.pan = {-145.5f, 78.25f}, .zoom = 1.75f};
    graph.savedView = saved;
    graph.step();
    REQUIRE(graph.viewReports.size() == 1);
    CHECK(graph.viewReports.back() == saved);
    const auto viewport = graph.viewport;
    graph.add("late", {1500.0f, 800.0f});
    graph.step();
    CHECK(graph.viewport == viewport);
    CHECK(graph.editor.zoom() == saved.zoom);
    CHECK(graph.viewReports.size() == 1);

    SECTION("invalid pan") { graph.savedView->pan.x = std::numeric_limits<F32>::infinity(); }
    SECTION("invalid zoom") { graph.savedView->zoom = std::numeric_limits<F32>::quiet_NaN(); }
    SECTION("nonpositive zoom") { graph.savedView->zoom = 0.0f; }
    graph.step();
    CHECK(graph.viewport == viewport);
    CHECK(graph.editor.zoom() == saved.zoom);
}

TEST_CASE("An idle editor reports the visible viewport in grid coordinates",
          "[core][sakura][node-editor]") {
    SECTION("At a pixel ratio of one") {
        Graph graph;
        graph.step();
        REQUIRE(graph.center.has_value());
        CheckPoint(*graph.center, 400.0f, 300.0f);
        REQUIRE(graph.viewport.has_value());
        CheckPoint({graph.viewport->x, graph.viewport->y}, 0.0f, 0.0f);
        CheckPoint({graph.viewport->width, graph.viewport->height}, 800.0f, 600.0f);
    }
    SECTION("At a pixel ratio of two") {
        Graph graph(2.0f);
        graph.step();
        REQUIRE(graph.center.has_value());
        CheckPoint(*graph.center, 200.0f, 150.0f);
        REQUIRE(graph.viewport.has_value());
        CheckPoint({graph.viewport->width, graph.viewport->height}, 400.0f, 300.0f);
    }
}

TEST_CASE("Opening a graph centers content that fits and pins wider content to the margin",
          "[core][sakura][node-editor]") {
    SECTION("A small graph far from the origin is centered") {
        Graph graph;
        graph.frameOnOpen = true;
        graph.add("a", {500.0f, 400.0f});
        graph.step();
        REQUIRE(graph.center.has_value());
        CHECK(graph.center->x > 560.0f);
        CHECK(graph.center->x < 640.0f);
        CHECK(graph.center->y > 400.0f);
        CHECK(graph.viewport->x < 500.0f);
        CHECK(graph.viewport->y < 400.0f);
    }
    SECTION("A graph wider than the view starts at the margin") {
        Graph graph;
        graph.frameOnOpen = true;
        graph.add("a", {100.0f, 50.0f});
        graph.add("b", {1200.0f, 50.0f});
        graph.step();
        REQUIRE(graph.viewport.has_value());
        CHECK(graph.viewport->x == Catch::Approx(100.0f - 24.0f));
    }
    SECTION("Without the option the view stays at the grid origin") {
        Graph graph;
        graph.add("a", {500.0f, 400.0f});
        graph.step();
        REQUIRE(graph.viewport.has_value());
        CheckPoint({graph.viewport->x, graph.viewport->y}, 0.0f, 0.0f);
    }
}

TEST_CASE("Opening a graph follows nodes as they load until the first click", "[core][sakura][node-editor]") {
    Graph graph;
    graph.frameOnOpen = true;
    graph.add("a", {100.0f, 100.0f});
    graph.step();
    const Extent2D<F32> first = *graph.center;

    graph.add("b", {400.0f, 100.0f});
    graph.step();
    CHECK(graph.center->x > first.x);

    graph.click(780.0f, 580.0f);
    const Extent2D<F32> settled = *graph.center;
    graph.add("c", {900.0f, 500.0f});
    graph.step();
    CheckPoint(*graph.center, settled.x, settled.y);
}

TEST_CASE("Opening a graph stops following nodes after a scroll, a zoom or an explicit stop",
          "[core][sakura][node-editor]") {
    Graph graph;
    graph.frameOnOpen = true;
    graph.add("a", {100.0f, 100.0f});
    graph.step();

    SECTION("A scroll over the canvas") {
        graph.host.send(Wheel(780.0f, 580.0f, -1.0f));
        graph.step();
    }
    SECTION("A zoom step") {
        graph.editor.zoomStep(1);
        graph.step();
    }
    SECTION("An explicit stop") {
        graph.editor.stopFraming();
        graph.step();
    }

    const Extent2D<F32> settled = *graph.center;
    graph.add("b", {900.0f, 500.0f});
    graph.step();
    CheckPoint(*graph.center, settled.x, settled.y);
}

TEST_CASE("Opening a graph frames again when the view resizes before the first click",
          "[core][sakura][node-editor]") {
    Graph graph;
    graph.frameOnOpen = true;
    graph.add("a", {500.0f, 400.0f});
    graph.step();
    REQUIRE(graph.viewport.has_value());
    const Rect first = *graph.viewport;

    graph.frames.bounds = {0.0f, 0.0f, 600.0f, 600.0f};
    graph.step();
    CHECK(graph.viewport->x == Catch::Approx(first.x + 100.0f).margin(1.0f));
    CHECK(graph.viewport->y == Catch::Approx(first.y).margin(1.0f));

    graph.click(10.0f, 10.0f);
    const Rect clicked = *graph.viewport;
    graph.frames.bounds = kScreen;
    graph.step();
    CHECK(graph.viewport->x == Catch::Approx(clicked.x));
    CHECK(graph.viewport->y == Catch::Approx(clicked.y));
}

TEST_CASE("Opening a graph frames again when the editor scale changes before the first click",
          "[core][sakura][node-editor]") {
    Graph graph;
    graph.frameOnOpen = true;
    Block& a = graph.add("a", {500.0f, 400.0f});
    a.withContent = false;
    graph.step();
    REQUIRE(graph.center.has_value());
    const Extent2D<F32> first = *graph.center;

    graph.scale = 0.5f;
    graph.step();
    CheckNear(*graph.center, first.x, first.y, 1.0f);
}

TEST_CASE("The editor scale multiplies the zoom ratio and keeps the zoom anchor",
          "[core][sakura][node-editor]") {
    Graph graph;
    graph.scale = 0.5f;
    graph.step();
    REQUIRE(graph.viewport.has_value());
    CheckPoint({graph.viewport->width, graph.viewport->height}, 1600.0f, 1200.0f);
    CheckPoint(*graph.center, 800.0f, 600.0f);
    CHECK(graph.editor.zoom() == 1.0f);

    graph.editor.zoomStep(1);
    graph.step();
    CHECK(graph.viewport->width == Catch::Approx(800.0f / 0.525f).margin(0.5f));
    CheckNear(*graph.center, 800.0f, 600.0f, 1.0f);
}

TEST_CASE("The viewport center is reported only when it moves", "[core][sakura][node-editor]") {
    Graph graph;
    graph.add("a", {40.0f, 40.0f});
    graph.step();
    graph.step();
    graph.step();
    CHECK(graph.centerReports == 1);

    graph.press(600.0f, 500.0f);
    graph.move(650.0f, 520.0f);
    graph.release(650.0f, 520.0f);
    const U64 panned = graph.centerReports;
    CHECK(panned > 1);
    graph.step();
    graph.step();
    CHECK(graph.centerReports == panned);
}

TEST_CASE("Dragging the empty canvas pans the grid by the pointer delta", "[core][sakura][node-editor]") {
    Graph graph;
    graph.add("a", {40.0f, 40.0f});
    graph.step();

    graph.press(600.0f, 500.0f);
    graph.move(650.0f, 520.0f);
    CheckPoint(*graph.center, 350.0f, 280.0f);
    graph.move(560.0f, 470.0f);
    CheckPoint(*graph.center, 440.0f, 330.0f);
    graph.release(560.0f, 470.0f);

    graph.move(700.0f, 100.0f);
    CheckPoint(*graph.center, 440.0f, 330.0f);
    CheckGeometry(graph.blocks[0]->last(), {40.0f, 40.0f}, {120.0f, 80.0f});
}

TEST_CASE("Stationary hover samples do not redraw an active graph interaction",
          "[core][sakura][node-editor][retention][stationary]") {
    const F32 ratio = GENERATE(1.0f, 2.0f);
    Graph graph(ratio);
    auto& node = graph.add("a", {40.0f, 40.0f});
    node.content.consume = false;
    REQUIRE(Settle(graph));
    Extent2D<F32> point;
    bool draggingNode = false;
    SECTION("background pan") {
        point = {600.0f, 400.0f};
    }
    SECTION("node drag") {
        point = {45.0f * ratio, 45.0f * ratio};
        draggingNode = true;
    }
    graph.press(point.x, point.y);
    REQUIRE(Settle(graph));
    const auto stationary = [&] {
        for (U64 frame = 0; frame < 100; ++frame) {
            graph.send(MouseEventType::Move, point.x, point.y);
            INFO("idle frame " << frame);
            CHECK(graph.step().clean());
        }
    };
    stationary();
    const auto center = *graph.center;
    point.x += 20.0f * ratio;
    point.y += 20.0f * ratio;
    graph.move(point.x, point.y);
    if (draggingNode) {
        CheckPoint(node.last().gridPosition, 60.0f, 60.0f);
    } else {
        CheckPoint(*graph.center, center.x - 20.0f, center.y - 20.0f);
    }
    REQUIRE(Settle(graph));
    stationary();
    graph.release(point.x, point.y);
    REQUIRE(Settle(graph));
    stationary();
}

TEST_CASE("Panning at a pixel ratio of two moves the grid by half the pointer delta",
          "[core][sakura][node-editor]") {
    Graph graph(2.0f);
    graph.step();

    graph.press(600.0f, 500.0f);
    graph.move(650.0f, 520.0f);
    graph.release(650.0f, 520.0f);

    CheckPoint(*graph.center, 175.0f, 140.0f);
}

TEST_CASE("Panning keeps node content frames in place and moves them through the layer translation",
          "[core][sakura][node-editor]") {
    Graph graph;
    Block& a = graph.add("a", {40.0f, 40.0f});
    graph.step();
    REQUIRE(a.content.laidFrame == Rect{48.0f, 48.0f, 120.0f, 80.0f});
    CheckPoint(a.content.shift(), 0.0f, 0.0f);

    graph.drag(600.0f, 500.0f, 650.0f, 520.0f);

    CHECK(a.content.laidFrame == Rect{48.0f, 48.0f, 120.0f, 80.0f});
    CheckPoint(a.content.shift(), 50.0f, 20.0f);
    CHECK(a.content.laidClip == Rect{40.0f, 40.0f, 136.0f, 96.0f});
    CheckGeometry(a.last(), {40.0f, 40.0f}, {120.0f, 80.0f});

    a.content.events.clear();
    graph.press(110.0f, 80.0f);
    REQUIRE(a.content.count(MouseEventType::Click) == 1);
    CheckPoint(a.content.events.back().position, 60.0f, 60.0f);
    graph.release(110.0f, 80.0f);
}

TEST_CASE("Scrolling never pans the canvas and reaches the content of the node under the pointer",
          "[core][sakura][node-editor]") {
    Graph graph;
    Block& a = graph.add("a", {40.0f, 40.0f});
    graph.step();

    CHECK_FALSE(graph.send(MouseEventType::Scroll, 600.0f, 500.0f));
    graph.step();
    CheckPoint(*graph.center, 400.0f, 300.0f);
    CHECK(a.content.count(MouseEventType::Scroll) == 0);

    CHECK(graph.send(MouseEventType::Scroll, 100.0f, 100.0f));
    graph.step();
    CHECK(a.content.count(MouseEventType::Scroll) == 1);
    CheckPoint(*graph.center, 400.0f, 300.0f);
}

TEST_CASE("A node panned out of the viewport stops laying out its content until it returns",
          "[core][sakura][node-editor]") {
    Graph graph;
    Block& a = graph.add("a", {40.0f, 40.0f});
    graph.step();
    REQUIRE(a.content.layouts > 0);

    graph.press(600.0f, 500.0f);
    graph.move(300.0f, 500.0f);
    const U64 hidden = a.content.layouts;
    graph.step();
    graph.step();
    CHECK(a.content.layouts == hidden);

    graph.move(610.0f, 500.0f);
    graph.release(610.0f, 500.0f);
    CHECK(a.content.layouts > hidden);
    CHECK(a.content.laidFrame == Rect{48.0f, 48.0f, 120.0f, 80.0f});
}

TEST_CASE("A culled node whose content grows into the viewport shows again without other input",
          "[core][sakura][node-editor]") {
    Graph graph;
    graph.frames.verify = false;
    Block& a = graph.add("a", {40.0f, 40.0f});
    const auto shown = [&] {
        const auto found = Find<Content>(graph.host);
        return std::find(found.begin(), found.end(), &a.content) != found.end();
    };
    graph.step();
    REQUIRE(shown());

    graph.press(600.0f, 500.0f);
    graph.move(600.0f, 300.0f);
    graph.release(600.0f, 300.0f);
    graph.step();
    REQUIRE_FALSE(shown());

    a.content.resize({0.0f, 400.0f});
    graph.step();
    graph.step();
    CHECK(shown());
}

TEST_CASE("Dragging a node moves it by the exact pointer delta without grid snapping",
          "[core][sakura][node-editor]") {
    Graph graph;
    Block& a = graph.add("a", {40.0f, 40.0f});
    a.content.consume = false;
    graph.step();
    REQUIRE(a.reports.size() == 1);
    CheckGeometry(a.last(), {40.0f, 40.0f}, {120.0f, 80.0f});

    graph.press(100.0f, 100.0f);
    graph.move(113.0f, 107.0f);
    CheckGeometry(a.last(), {53.0f, 47.0f}, {120.0f, 80.0f});
    graph.release(113.0f, 107.0f);

    graph.step();
    CheckGeometry(a.last(), {53.0f, 47.0f}, {120.0f, 80.0f});
    CheckPoint(*graph.center, 400.0f, 300.0f);
}

TEST_CASE("Dragging a node at a pixel ratio of two moves it by half the pointer delta",
          "[core][sakura][node-editor]") {
    Graph graph(2.0f);
    Block& a = graph.add("a", {40.0f, 40.0f});
    a.content.consume = false;
    graph.step();

    graph.drag(150.0f, 150.0f, 176.0f, 164.0f);

    CheckPoint(a.last().gridPosition, 53.0f, 47.0f);
}

TEST_CASE("Small pointer jitter while dragging reports geometry only past half a grid unit",
          "[core][sakura][node-editor]") {
    Graph graph;
    Block& a = graph.add("a", {40.0f, 40.0f});
    a.content.consume = false;
    graph.step();
    const U64 reports = a.reports.size();

    graph.press(100.0f, 100.0f);
    graph.move(100.4f, 100.3f);
    CHECK(a.reports.size() == reports);
    graph.move(101.0f, 100.0f);
    CHECK(a.reports.size() == reports + 1);
    graph.release(101.0f, 100.0f);
}

TEST_CASE("Clicking a node raises it above the nodes it overlaps", "[core][sakura][node-editor]") {
    Graph graph;
    Block& a = graph.add("a", {40.0f, 40.0f});
    Block& b = graph.add("b", {100.0f, 80.0f});
    a.content.consume = false;
    b.content.consume = false;
    graph.step();

    graph.click(150.0f, 110.0f);
    CHECK(a.content.count(MouseEventType::Click) == 0);
    CHECK(b.content.count(MouseEventType::Click) == 1);

    graph.click(60.0f, 60.0f);
    CHECK(a.content.count(MouseEventType::Click) == 1);

    graph.click(150.0f, 110.0f);
    CHECK(a.content.count(MouseEventType::Click) == 2);
    CHECK(b.content.count(MouseEventType::Click) == 1);
    CheckPoint(a.last().gridPosition, 40.0f, 40.0f);
    CheckPoint(b.last().gridPosition, 100.0f, 80.0f);
}

TEST_CASE("A foreground node blocks resize presses aimed at the grip of the node underneath",
          "[core][sakura][node-editor]") {
    for (const ResizeAxes front : {ResizeAxes::None, ResizeAxes::XY}) {
        CAPTURE(static_cast<int>(front));
        Graph graph;
        Block& back = graph.add("back", {0.0f, 0.0f}, {300.0f, 300.0f}, ResizeAxes::XY);
        Block& top = graph.add("front", {250.0f, 250.0f}, {120.0f, 100.0f}, front);
        back.content.consume = false;
        top.content.consume = false;
        graph.step();

        graph.drag(310.0f, 310.0f, 370.0f, 370.0f);

        CheckGeometry(back.last(), {0.0f, 0.0f}, {300.0f, 300.0f});
        CheckGeometry(top.last(), {310.0f, 310.0f}, {120.0f, 100.0f});
    }
}

TEST_CASE("Pressing a consuming content widget routes the whole gesture to it and leaves the node in place",
          "[core][sakura][node-editor]") {
    Graph graph;
    Block& a = graph.add("a", {40.0f, 40.0f});
    graph.step();

    graph.press(100.0f, 100.0f);
    graph.move(160.0f, 150.0f);
    graph.release(160.0f, 150.0f);

    CHECK(a.content.count(MouseEventType::Click) == 1);
    CHECK(a.content.count(MouseEventType::Move) == 1);
    CHECK(a.content.count(MouseEventType::Release) == 1);
    CheckPoint(a.last().gridPosition, 40.0f, 40.0f);
    CheckPoint(*graph.center, 400.0f, 300.0f);
}

TEST_CASE("Resizing an X axis node changes only its width", "[core][sakura][node-editor]") {
    Graph graph;
    Block& a = graph.add("a", {40.0f, 40.0f}, {120.0f, 120.0f}, ResizeAxes::X);
    graph.step();
    CheckGeometry(a.last(), {40.0f, 40.0f}, {120.0f, 120.0f});

    graph.drag(172.0f, 172.0f, 232.0f, 212.0f);

    CheckGeometry(a.last(), {40.0f, 40.0f}, {180.0f, 120.0f});
}

TEST_CASE("Resizing a Y axis node changes only its height", "[core][sakura][node-editor]") {
    Graph graph;
    Block& a = graph.add("a", {40.0f, 40.0f}, {120.0f, 120.0f}, ResizeAxes::Y);
    graph.step();
    CheckGeometry(a.last(), {40.0f, 40.0f}, {120.0f, 120.0f});

    graph.drag(172.0f, 172.0f, 232.0f, 212.0f);

    CheckGeometry(a.last(), {40.0f, 40.0f}, {120.0f, 160.0f});
}

TEST_CASE("Resizing an XY node changes both dimensions and survives the geometry round trip",
          "[core][sakura][node-editor]") {
    Graph graph;
    Block& a = graph.add("a", {40.0f, 40.0f}, {120.0f, 120.0f}, ResizeAxes::XY);
    graph.step();

    graph.drag(172.0f, 172.0f, 232.0f, 212.0f);
    CheckGeometry(a.last(), {40.0f, 40.0f}, {180.0f, 160.0f});

    graph.step();
    graph.step();
    CheckGeometry(a.last(), {40.0f, 40.0f}, {180.0f, 160.0f});
    CheckPoint(a.spec.dimensions, 180.0f, 160.0f);
}

TEST_CASE("A node without resize axes treats its grip corner as a drag handle",
          "[core][sakura][node-editor]") {
    Graph graph;
    Block& a = graph.add("a", {40.0f, 40.0f}, {120.0f, 120.0f}, ResizeAxes::None);
    a.content.consume = false;
    graph.step();

    graph.drag(172.0f, 172.0f, 232.0f, 212.0f);

    CheckGeometry(a.last(), {100.0f, 80.0f}, {120.0f, 120.0f});
}

TEST_CASE("Resizing clamps at the configured minimum dimensions", "[core][sakura][node-editor]") {
    Graph graph;
    Block& a = graph.add("a", {40.0f, 40.0f}, {200.0f, 200.0f}, ResizeAxes::XY);
    a.spec.minimumDimensions = {150.0f, 140.0f};
    graph.step();
    CheckGeometry(a.last(), {40.0f, 40.0f}, {200.0f, 200.0f});

    graph.drag(252.0f, 252.0f, -248.0f, -248.0f);

    CheckGeometry(a.last(), {40.0f, 40.0f}, {150.0f, 140.0f});
}

TEST_CASE("Resizing clamps the height at the content minimum when it exceeds the configured one",
          "[core][sakura][node-editor]") {
    Graph graph;
    Block& a = graph.add("a", {40.0f, 40.0f}, {120.0f, 120.0f}, ResizeAxes::XY);
    a.spec.minimumDimensions = {120.0f, 40.0f};
    graph.step();

    graph.drag(172.0f, 172.0f, 172.0f, -400.0f);

    CheckGeometry(a.last(), {40.0f, 40.0f}, {120.0f, 80.0f});
}

TEST_CASE("Resizing at a pixel ratio of two converts the drag to logical units and keeps logical minimums",
          "[core][sakura][node-editor]") {
    Graph graph(2.0f);
    Block& a = graph.add("a", {40.0f, 40.0f}, {200.0f, 200.0f}, ResizeAxes::XY);
    a.spec.minimumDimensions = {150.0f, 140.0f};
    graph.step();

    graph.press(508.0f, 508.0f);
    graph.move(548.0f, 528.0f);
    CheckGeometry(a.last(), {40.0f, 40.0f}, {220.0f, 210.0f});
    graph.move(-492.0f, -492.0f);
    CheckGeometry(a.last(), {40.0f, 40.0f}, {150.0f, 140.0f});
    graph.release(-492.0f, -492.0f);
}

TEST_CASE("An auto height node follows its content and reports the measured height",
          "[core][sakura][node-editor]") {
    Graph graph;
    Block& a = graph.add("a", {40.0f, 40.0f});
    graph.step();
    CheckGeometry(a.last(), {40.0f, 40.0f}, {120.0f, 80.0f});

    a.content.resize({0.0f, 120.0f});
    graph.step();
    CheckGeometry(a.last(), {40.0f, 40.0f}, {120.0f, 120.0f});
    CHECK(a.content.laidFrame == Rect{48.0f, 48.0f, 120.0f, 120.0f});

    a.content.resize({0.0f, 60.0f});
    graph.step();
    CheckGeometry(a.last(), {40.0f, 40.0f}, {120.0f, 60.0f});
    CHECK(a.spec.dimensions.y == 0.0f);
}

TEST_CASE("A loading node floors its height and repaints its pulse as time advances",
          "[core][sakura][node-editor]") {
    Graph graph;
    Block& a = graph.add("a", {40.0f, 40.0f});
    a.withContent = false;
    a.spec.state = NodeEditor::NodeState::Loading;
    REQUIRE(graph.frames.settle([&] {
        graph.update();
    }));
    CheckGeometry(a.last(), {40.0f, 40.0f}, {120.0f, 96.0f});

    graph.frames.ctx.time += 0.25;
    CHECK(RepaintOnly(graph.step()));
    graph.frames.ctx.time += 0.25;
    CHECK(RepaintOnly(graph.step()));

    a.spec.state = NodeEditor::NodeState::Normal;
    a.spec.dimensions.y = 0.0f;
    REQUIRE(graph.frames.settle([&] {
        graph.update();
    }));
    graph.frames.ctx.time += 0.25;
    CHECK(graph.step().clean());
}

TEST_CASE("A node toggling between loading and normal keeps its loading decoration mounted",
          "[core][sakura][node-editor]") {
    Graph graph;
    Block& a = graph.add("a", {40.0f, 40.0f});
    a.withContent = false;
    REQUIRE(graph.frames.settle([&] {
        graph.update();
    }));

    const auto toggle = [&](const NodeEditor::NodeState state) {
        a.spec.state = state;
        a.spec.dimensions.y = 0.0f;
        REQUIRE(graph.frames.settle([&] {
            graph.update();
        }));
    };
    toggle(NodeEditor::NodeState::Loading);
    toggle(NodeEditor::NodeState::Normal);
    const U64 restructures = graph.frames.restructures;
    const U64 rebuilds = graph.frames.rebuilds;

    for (U64 cycle = 0; cycle < 3; ++cycle) {
        toggle(NodeEditor::NodeState::Loading);
        toggle(NodeEditor::NodeState::Normal);
    }
    CHECK(graph.frames.restructures == restructures);
    CHECK(graph.frames.rebuilds == rebuilds);
}

TEST_CASE("Dragging from an output pin to an input pin reports the link endpoints",
          "[core][sakura][node-editor]") {
    Graph graph;
    graph.add("src", {40.0f, 40.0f}, {120.0f, 0.0f}, ResizeAxes::X, {Output("out")});
    graph.add("dst", {400.0f, 40.0f}, {120.0f, 0.0f}, ResizeAxes::X, {Input("in")});
    graph.step();

    graph.press(176.0f, 55.0f);
    graph.move(300.0f, 60.0f);
    CHECK(graph.created.empty());
    graph.move(401.0f, 56.0f);
    graph.release(401.0f, 56.0f);

    REQUIRE(graph.created.size() == 1);
    CHECK(graph.created[0].first == Ref("src", "out", false));
    CHECK(graph.created[0].second == Ref("dst", "in", true));
    CHECK(graph.destroyed.empty());
    CheckPoint(*graph.center, 400.0f, 300.0f);
    CheckPoint(graph.blocks[0]->last().gridPosition, 40.0f, 40.0f);
}

TEST_CASE("Dragging from an input pin to an output pin reports the input as the start",
          "[core][sakura][node-editor]") {
    Graph graph;
    graph.add("src", {40.0f, 40.0f}, {120.0f, 0.0f}, ResizeAxes::X, {Output("out")});
    graph.add("dst", {400.0f, 40.0f}, {120.0f, 0.0f}, ResizeAxes::X, {Input("in")});
    graph.step();

    graph.drag(400.0f, 55.0f, 177.0f, 54.0f);

    REQUIRE(graph.created.size() == 1);
    CHECK(graph.created[0].first == Ref("dst", "in", true));
    CHECK(graph.created[0].second == Ref("src", "out", false));
}

TEST_CASE("Releasing a new wire on an invalid target creates no link", "[core][sakura][node-editor]") {
    Graph graph;
    graph.add("src", {40.0f, 40.0f}, {120.0f, 0.0f}, ResizeAxes::X, {Output("out")});
    graph.add("dst", {400.0f, 40.0f}, {120.0f, 0.0f}, ResizeAxes::X, {Input("in")});
    graph.add("alt", {400.0f, 300.0f}, {120.0f, 0.0f}, ResizeAxes::X, {Output("out")});
    graph.add("relay", {40.0f, 300.0f}, {120.0f, 0.0f}, ResizeAxes::X, {Input("in"), Output("out")});
    graph.step();

    SECTION("Empty canvas") {
        graph.drag(176.0f, 55.0f, 700.0f, 560.0f);
    }
    SECTION("A pin of the same direction") {
        graph.drag(176.0f, 55.0f, 536.0f, 315.0f);
    }
    SECTION("A pin on the same node") {
        graph.drag(176.0f, 336.0f, 40.0f, 315.0f);
    }
    SECTION("A pair that is already linked") {
        graph.links = {Wire("link", Ref("src", "out", false), Ref("dst", "in", true))};
        graph.step();
        graph.drag(176.0f, 55.0f, 400.0f, 55.0f);
    }

    CHECK(graph.created.empty());
    CHECK(graph.destroyed.empty());
}

TEST_CASE("Pressing a connected input with detach enabled destroys its link without starting a wire",
          "[core][sakura][node-editor]") {
    Graph graph;
    graph.add("src", {40.0f, 40.0f}, {120.0f, 0.0f}, ResizeAxes::X, {Output("out")});
    graph.add("dst", {400.0f, 40.0f}, {120.0f, 0.0f}, ResizeAxes::X, {Input("in", true)});
    graph.add("alt", {400.0f, 300.0f}, {120.0f, 0.0f}, ResizeAxes::X, {Output("out")});
    graph.links = {Wire("link", Ref("src", "out", false), Ref("dst", "in", true))};
    graph.step();

    graph.drag(400.0f, 55.0f, 536.0f, 315.0f);

    REQUIRE(graph.destroyed.size() == 1);
    CHECK(graph.destroyed[0] == "link");
    CHECK(graph.created.empty());
}

TEST_CASE("Pressing a connected input without detach starts a rewire from that input",
          "[core][sakura][node-editor]") {
    Graph graph;
    graph.add("src", {40.0f, 40.0f}, {120.0f, 0.0f}, ResizeAxes::X, {Output("out")});
    graph.add("dst", {400.0f, 40.0f}, {120.0f, 0.0f}, ResizeAxes::X, {Input("in")});
    graph.add("alt", {400.0f, 300.0f}, {120.0f, 0.0f}, ResizeAxes::X, {Output("out")});
    graph.links = {Wire("link", Ref("src", "out", false), Ref("dst", "in", true))};
    graph.step();

    graph.drag(400.0f, 55.0f, 536.0f, 315.0f);

    CHECK(graph.destroyed.empty());
    REQUIRE(graph.created.size() == 1);
    CHECK(graph.created[0].first == Ref("dst", "in", true));
    CHECK(graph.created[0].second == Ref("alt", "out", false));
}

TEST_CASE("A pin covered by a foreground node cannot start a wire", "[core][sakura][node-editor]") {
    Graph graph;
    graph.add("src", {40.0f, 40.0f}, {120.0f, 0.0f}, ResizeAxes::X, {Output("out")});
    graph.add("dst", {400.0f, 40.0f}, {120.0f, 0.0f}, ResizeAxes::X, {Input("in")});
    Block& lid = graph.add("lid", {150.0f, 30.0f});
    lid.content.consume = false;
    graph.step();

    graph.drag(176.0f, 55.0f, 400.0f, 55.0f);

    CHECK(graph.created.empty());
    CheckPoint(lid.last().gridPosition, 374.0f, 30.0f);
}

TEST_CASE("Hovering a wire requests its tooltip and the hover follows a moved node",
          "[core][sakura][node-editor]") {
    Graph graph;
    graph.frames.verify = false;
    graph.add("src", {40.0f, 40.0f}, {120.0f, 0.0f}, ResizeAxes::X, {Output("out")});
    Block& dst = graph.add("dst", {400.0f, 40.0f}, {120.0f, 0.0f}, ResizeAxes::X, {Input("in")});
    dst.content.consume = false;
    U64 tooltips = 0;
    auto link = Wire("link", Ref("src", "out", false), Ref("dst", "in", true));
    link.tooltip.content = [&tooltips]() -> Sakura::Component* {
        ++tooltips;
        return nullptr;
    };
    graph.links = {link};
    graph.step();

    graph.move(288.0f, 55.0f);
    CHECK(tooltips == 1);
    graph.move(288.0f, 300.0f);
    CHECK(tooltips == 1);

    graph.drag(450.0f, 100.0f, 450.0f, 300.0f);
    CheckPoint(dst.last().gridPosition, 400.0f, 240.0f);
    const U64 moved = tooltips;
    graph.move(288.0f, 55.0f);
    CHECK(tooltips == moved);
    graph.move(288.0f, 155.0f);
    CHECK(tooltips == moved + 1);

    graph.links[0].unresolved = true;
    graph.move(288.0f, 156.0f);
    CHECK(tooltips == moved + 1);
}

TEST_CASE("Lazily prepared wire tooltips appear after the delay and follow the pointer along the wire",
          "[core][sakura][node-editor]") {
    SakuraTest::FakeWindow window;
    NodeEditor editor;
    LayerSlot slot;
    slot.hold(editor, kScreen);
    LayerHost host({&slot});
    Frames frames(host);
    frames.verify = false;
    frames.timeStep = 0.1;
    frames.ctx.render = &window;
    frames.ctx.hoverDelay = 0.25f;
    Probe tip({80.0f, 20.0f});
    bool asked = false;
    bool ready = false;
    const auto node = [](const std::string& id, Extent2D<F32> position, NodeEditor::Pin pin) {
        return NodeEditor::Node{.id = id, .header = false, .title = id, .pins = {std::move(pin)},
                                .resize = ResizeAxes::X, .dimensions = {120.0f, 0.0f}, .gridPosition = position};
    };
    auto link = Wire("link", Ref("src", "out", false), Ref("dst", "in", true));
    link.tooltip.content = [&]() -> Sakura::Component* {
        asked = true;
        return ready ? &tip : nullptr;
    };
    const auto update = [&] {
        ready = ready || std::exchange(asked, false);
        editor.update({
            .id = "editor",
            .nodes = {node("src", {40.0f, 40.0f}, Output("out")), node("dst", {400.0f, 40.0f}, Input("in"))},
            .links = {link},
        });
    };
    const auto hover = [&](F32 x, F32 y) {
        frames.ctx.mouse = Extent2D<F32>{x, y};
        host.send(Mouse(MouseEventType::Move, x, y));
    };
    REQUIRE(frames.settle(update));

    hover(288.0f, 55.0f);
    for (U64 frame = 0; frame < 8; ++frame) {
        frames.run(update);
    }
    REQUIRE(tip.visible());
    REQUIRE(tip.layouts > 0);
    const Rect entered = tip.currentFrame();
    hover(288.0f, 55.0f);
    CHECK(frames.run(update).clean());

    hover(296.0f, 55.0f);
    frames.run(update);
    CHECK(tip.visible());
    CHECK(tip.currentFrame().x == entered.x + 8.0f);
    CHECK(DirtyFrames(frames, update) == 0);
}

TEST_CASE("Right clicking a node opens its menu and right clicking the canvas reports the grid point",
          "[core][sakura][node-editor]") {
    Graph graph;
    Block& a = graph.add("a", {40.0f, 40.0f});
    graph.step();

    graph.press(100.0f, 100.0f, MouseButton::Right);
    graph.release(100.0f, 100.0f, MouseButton::Right);
    CHECK(a.menus == 0);
    CHECK(a.content.count(MouseEventType::Click) == 1);
    a.content.consume = false;

    graph.press(100.0f, 100.0f, MouseButton::Right);
    graph.release(100.0f, 100.0f, MouseButton::Right);
    CHECK(a.menus == 1);
    CHECK(graph.menus.empty());

    graph.drag(600.0f, 500.0f, 650.0f, 520.0f);
    graph.press(600.0f, 500.0f, MouseButton::Right);
    graph.move(700.0f, 560.0f);
    graph.release(700.0f, 560.0f, MouseButton::Right);
    REQUIRE(graph.menus.size() == 1);
    CheckPoint(graph.menus[0], 550.0f, 480.0f);
    CHECK(a.menus == 1);
    CheckPoint(*graph.center, 350.0f, 280.0f);
}

TEST_CASE("Content that takes a right press keeps the drag and its release outside the node",
          "[core][sakura][node-editor]") {
    Graph graph;
    Block& a = graph.add("a", {40.0f, 40.0f});
    Block& b = graph.add("b", {400.0f, 40.0f});
    graph.step();

    graph.press(100.0f, 100.0f, MouseButton::Right);
    REQUIRE(a.content.count(MouseEventType::Click) == 1);
    const U64 moves = a.content.count(MouseEventType::Move);
    graph.move(460.0f, 100.0f);
    graph.move(700.0f, 500.0f);
    CHECK(a.content.count(MouseEventType::Move) == moves + 2);
    CHECK(a.content.count(MouseEventType::Leave) == 0);
    CHECK(b.content.count(MouseEventType::Move) == 0);

    graph.release(700.0f, 500.0f, MouseButton::Right);
    CHECK(a.content.count(MouseEventType::Release) == 1);
    CHECK(b.content.count(MouseEventType::Release) == 0);
    CHECK(a.menus == 0);
    CHECK(graph.menus.empty());

    graph.move(460.0f, 100.0f);
    CHECK(b.content.count(MouseEventType::Move) == 1);
}

TEST_CASE("Content keeps its capture until every button it captured is released",
          "[core][sakura][node-editor]") {
    Graph graph;
    Block& a = graph.add("a", {40.0f, 40.0f});
    Block& b = graph.add("b", {400.0f, 40.0f});
    graph.step();

    graph.press(100.0f, 100.0f);
    graph.press(100.0f, 100.0f, MouseButton::Right);
    CHECK(a.content.count(MouseEventType::Click) == 2);
    graph.release(100.0f, 100.0f);
    CHECK(a.content.count(MouseEventType::Release) == 1);

    const U64 moves = a.content.count(MouseEventType::Move);
    graph.move(460.0f, 100.0f);
    CHECK(a.content.count(MouseEventType::Move) == moves + 1);
    CHECK(b.content.count(MouseEventType::Move) == 0);

    graph.release(460.0f, 100.0f, MouseButton::Right);
    CHECK(a.content.count(MouseEventType::Release) == 2);
    CHECK(b.content.count(MouseEventType::Release) == 0);
    CHECK(a.menus == 0);
    CHECK(graph.menus.empty());

    graph.move(461.0f, 100.0f);
    CHECK(b.content.count(MouseEventType::Move) == 1);
}

TEST_CASE("Double clicking the canvas or a wire requests the block picker but a node does not",
          "[core][sakura][node-editor]") {
    Graph graph;
    graph.add("src", {40.0f, 40.0f}, {120.0f, 0.0f}, ResizeAxes::X, {Output("out")});
    graph.add("dst", {400.0f, 40.0f}, {120.0f, 0.0f}, ResizeAxes::X, {Input("in")});
    graph.links = {Wire("link", Ref("src", "out", false), Ref("dst", "in", true))};
    graph.step();

    graph.click(100.0f, 100.0f, 2);
    CHECK(graph.doubleClicks.empty());

    graph.click(600.0f, 500.0f);
    CHECK(graph.doubleClicks.empty());

    graph.click(600.0f, 500.0f, 2);
    REQUIRE(graph.doubleClicks.size() == 1);
    CheckPoint(graph.doubleClicks[0], 600.0f, 500.0f);

    graph.click(288.0f, 55.0f, 2);
    REQUIRE(graph.doubleClicks.size() == 2);
    CheckPoint(graph.doubleClicks[1], 288.0f, 55.0f);
}

TEST_CASE("Pressing a wire body neither pans the canvas nor moves nodes", "[core][sakura][node-editor]") {
    Graph graph;
    graph.add("src", {40.0f, 40.0f}, {120.0f, 0.0f}, ResizeAxes::X, {Output("out")});
    graph.add("dst", {400.0f, 40.0f}, {120.0f, 0.0f}, ResizeAxes::X, {Input("in")});
    graph.links = {Wire("link", Ref("src", "out", false), Ref("dst", "in", true))};
    graph.step();

    graph.drag(288.0f, 55.0f, 338.0f, 105.0f);

    CheckPoint(*graph.center, 400.0f, 300.0f);
    CHECK(graph.created.empty());
    CHECK(graph.destroyed.empty());
}

TEST_CASE("Clicking the collapse chevron toggles the node but dragging from it does not",
          "[core][sakura][node-editor]") {
    Graph graph;
    Block& a = graph.add("a", {40.0f, 40.0f});
    a.spec.header = true;
    a.spec.collapsible = true;
    a.content.consume = false;
    graph.step();

    graph.click(160.0f, 56.0f);
    CHECK(a.toggles == 1);
    CheckPoint(a.last().gridPosition, 40.0f, 40.0f);

    graph.drag(160.0f, 56.0f, 170.0f, 66.0f);
    CHECK(a.toggles == 1);
    CheckPoint(a.last().gridPosition, 50.0f, 50.0f);
}

TEST_CASE("Header buttons keep an inset from the right edge of the node", "[core][sakura][node-editor]") {
    Graph graph;
    Block& a = graph.add("a", {40.0f, 40.0f});
    a.spec.header = true;
    a.spec.collapsible = true;
    a.content.consume = false;
    graph.step();

    graph.click(166.5f, 56.0f);
    CHECK(a.toggles == 0);

    graph.click(163.0f, 56.0f);
    CHECK(a.toggles == 1);
}

TEST_CASE("The header detach button sits left of the chevron and fires only on a click",
          "[core][sakura][node-editor]") {
    Graph graph;
    Block& a = graph.add("a", {40.0f, 40.0f});
    a.spec.header = true;
    a.spec.collapsible = true;
    a.detachable = true;
    a.content.consume = false;
    graph.step();

    graph.click(160.0f, 56.0f);
    CHECK(a.toggles == 1);
    CHECK(a.detaches == 0);

    graph.click(137.0f, 56.0f);
    CHECK(a.detaches == 1);
    CHECK(a.toggles == 1);
    CheckPoint(a.last().gridPosition, 40.0f, 40.0f);

    graph.drag(137.0f, 56.0f, 147.0f, 66.0f);
    CHECK(a.detaches == 1);
    CheckPoint(a.last().gridPosition, 50.0f, 50.0f);

    a.detachable = false;
    graph.step();
    graph.click(147.0f, 66.0f);
    CHECK(a.detaches == 1);
}

TEST_CASE("A header button press that drags away and returns before the release does not fire",
          "[core][sakura][node-editor]") {
    Graph graph;
    Block& a = graph.add("a", {40.0f, 40.0f});
    a.spec.header = true;
    a.spec.collapsible = true;
    a.detachable = true;
    a.content.consume = false;
    graph.step();

    graph.press(137.0f, 56.0f);
    graph.move(187.0f, 106.0f);
    graph.move(137.0f, 56.0f);
    graph.release(137.0f, 56.0f);
    CHECK(a.detaches == 0);
    CheckPoint(a.last().gridPosition, 40.0f, 40.0f);

    graph.press(137.0f, 56.0f);
    graph.move(138.0f, 57.0f);
    graph.release(138.0f, 57.0f);
    CHECK(a.detaches == 1);

    graph.press(137.0f, 56.0f);
    graph.send(MouseEventType::Leave, 137.0f, 56.0f);
    graph.step();
    graph.move(137.0f, 56.0f);
    graph.release(137.0f, 56.0f);
    CHECK(a.detaches == 1);
}

TEST_CASE("Dragging a node past the viewport edge auto pans by the frame delta time",
          "[core][sakura][node-editor]") {
    Graph graph;
    Block& a = graph.add("a", {40.0f, 40.0f});
    a.content.consume = false;
    graph.step();

    graph.press(100.0f, 100.0f);
    graph.frames.ctx.deltaTime = 0.1f;
    graph.move(900.0f, 300.0f);
    CheckPoint(*graph.center, 500.0f, 300.0f);
    CheckPoint(a.last().gridPosition, 940.0f, 240.0f);
    graph.step();
    CheckPoint(*graph.center, 600.0f, 300.0f);
    CheckPoint(a.last().gridPosition, 1040.0f, 240.0f);

    graph.move(400.0f, 300.0f);
    CheckPoint(*graph.center, 600.0f, 300.0f);
    CheckPoint(a.last().gridPosition, 540.0f, 240.0f);
    graph.release(400.0f, 300.0f);
    graph.step();
    CheckPoint(*graph.center, 600.0f, 300.0f);
}

TEST_CASE("Stationary moves past the viewport edge keep auto panning once per frame",
          "[core][sakura][node-editor]") {
    Graph graph;
    Block& a = graph.add("a", {40.0f, 40.0f});
    a.content.consume = false;
    graph.step();

    graph.press(100.0f, 100.0f);
    graph.frames.ctx.deltaTime = 0.1f;
    graph.move(900.0f, 300.0f);
    CheckPoint(*graph.center, 500.0f, 300.0f);
    for (U64 frame = 1; frame <= 3; ++frame) {
        graph.move(900.0f, 300.0f);
        CheckPoint(*graph.center, 500.0f + 100.0f * static_cast<F32>(frame), 300.0f);
        CheckPoint(a.last().gridPosition, 940.0f + 100.0f * static_cast<F32>(frame), 240.0f);
    }
    graph.release(900.0f, 300.0f);
}

TEST_CASE("Panning or resizing past the viewport edge never auto pans", "[core][sakura][node-editor]") {
    Graph graph;
    Block& a = graph.add("a", {40.0f, 40.0f}, {120.0f, 120.0f}, ResizeAxes::XY);
    graph.frames.ctx.deltaTime = 0.1f;
    graph.step();

    graph.press(172.0f, 172.0f);
    graph.move(900.0f, 172.0f);
    graph.step();
    CheckPoint(*graph.center, 400.0f, 300.0f);
    CheckGeometry(a.last(), {40.0f, 40.0f}, {848.0f, 120.0f});
    graph.release(900.0f, 172.0f);

    graph.press(600.0f, 500.0f);
    graph.move(900.0f, 500.0f);
    graph.step();
    CheckPoint(*graph.center, 100.0f, 300.0f);
    graph.release(900.0f, 500.0f);
}

TEST_CASE("A cancelled pointer ends a node drag where it stopped", "[core][sakura][node-editor]") {
    Graph graph;
    Block& a = graph.add("a", {40.0f, 40.0f});
    a.content.consume = false;
    graph.step();

    graph.press(100.0f, 100.0f);
    graph.move(120.0f, 110.0f);
    CheckPoint(a.last().gridPosition, 60.0f, 50.0f);
    graph.send(MouseEventType::Cancel, 120.0f, 110.0f);
    graph.step();
    graph.move(200.0f, 200.0f);
    graph.release(200.0f, 200.0f);

    CheckPoint(a.last().gridPosition, 60.0f, 50.0f);
}

TEST_CASE("Removing a node during its drag ends the interaction safely", "[core][sakura][node-editor]") {
    Graph graph;
    Block& a = graph.add("a", {40.0f, 40.0f});
    Block& b = graph.add("b", {400.0f, 300.0f});
    a.content.consume = false;
    graph.step();

    graph.press(100.0f, 100.0f);
    graph.move(150.0f, 120.0f);
    CheckPoint(a.last().gridPosition, 90.0f, 60.0f);
    const U64 reports = a.reports.size();

    a.present = false;
    graph.step();
    graph.move(250.0f, 220.0f);
    graph.release(250.0f, 220.0f);
    CHECK(a.reports.size() == reports);
    CheckGeometry(b.last(), {400.0f, 300.0f}, {120.0f, 80.0f});

    graph.drag(600.0f, 100.0f, 640.0f, 120.0f);
    CheckPoint(*graph.center, 360.0f, 280.0f);
}

TEST_CASE("Removing a node during its resize ends the interaction safely", "[core][sakura][node-editor]") {
    Graph graph;
    Block& a = graph.add("a", {40.0f, 40.0f}, {120.0f, 120.0f}, ResizeAxes::XY);
    Block& b = graph.add("b", {400.0f, 300.0f}, {120.0f, 120.0f}, ResizeAxes::XY);
    graph.step();

    graph.press(172.0f, 172.0f);
    graph.move(212.0f, 202.0f);
    CheckGeometry(a.last(), {40.0f, 40.0f}, {160.0f, 150.0f});

    a.present = false;
    graph.step();
    graph.move(300.0f, 300.0f);
    graph.release(300.0f, 300.0f);
    CheckGeometry(b.last(), {400.0f, 300.0f}, {120.0f, 120.0f});

    graph.drag(600.0f, 100.0f, 640.0f, 120.0f);
    CheckPoint(*graph.center, 360.0f, 280.0f);

    a.present = true;
    graph.step();
    CheckGeometry(a.last(), {40.0f, 40.0f}, {160.0f, 150.0f});
}

TEST_CASE("Removing the start node of a pending wire cancels the wire", "[core][sakura][node-editor]") {
    Graph graph;
    Block& src = graph.add("src", {40.0f, 40.0f}, {120.0f, 0.0f}, ResizeAxes::X, {Output("out")});
    graph.add("dst", {400.0f, 40.0f}, {120.0f, 0.0f}, ResizeAxes::X, {Input("in")});
    graph.step();

    graph.press(176.0f, 55.0f);
    graph.move(300.0f, 55.0f);
    src.present = false;
    graph.step();
    graph.move(401.0f, 56.0f);
    graph.release(401.0f, 56.0f);
    CHECK(graph.created.empty());

    graph.drag(600.0f, 500.0f, 640.0f, 520.0f);
    CheckPoint(*graph.center, 360.0f, 280.0f);
}

TEST_CASE("A press consumed by the content of a node behind another raises and selects that node",
          "[core][sakura][node-editor]") {
    Graph graph;
    Block& a = graph.add("a", {40.0f, 40.0f});
    Block& b = graph.add("b", {100.0f, 80.0f});
    graph.step();

    graph.click(150.0f, 110.0f);
    CHECK(a.content.count(MouseEventType::Click) == 0);
    CHECK(b.content.count(MouseEventType::Click) == 1);
    CHECK(graph.copy() == std::vector<std::string>{"b"});

    graph.press(60.0f, 60.0f);
    graph.move(90.0f, 95.0f);
    graph.release(90.0f, 95.0f);
    CHECK(a.content.count(MouseEventType::Click) == 1);
    CHECK(a.content.count(MouseEventType::Move) == 1);
    CHECK(a.content.count(MouseEventType::Release) == 1);
    CheckPoint(a.last().gridPosition, 40.0f, 40.0f);
    CheckPoint(*graph.center, 400.0f, 300.0f);
    CHECK(graph.copy() == std::vector<std::string>{"a"});

    graph.click(150.0f, 110.0f);
    CHECK(a.content.count(MouseEventType::Click) == 2);
    CHECK(b.content.count(MouseEventType::Click) == 1);
    CheckPoint(b.last().gridPosition, 100.0f, 80.0f);
}

TEST_CASE("Node content receives no presses outside its node and its clip stays inside the node",
          "[core][sakura][node-editor]") {
    Graph graph;
    Block& a = graph.add("a", {40.0f, 40.0f});
    Block& b = graph.add("b", {100.0f, 80.0f});
    graph.step();

    const Rect body = {40.0f, 40.0f, 136.0f, 96.0f};
    CHECK(Contains(body, a.content.laidClip));
    CHECK(Contains(a.content.laidClip, a.content.laidFrame));

    graph.click(600.0f, 500.0f);
    graph.click(150.0f, 110.0f);
    graph.click(220.0f, 170.0f);
    CHECK(a.content.count(MouseEventType::Click) == 0);
    CHECK(b.content.count(MouseEventType::Click) == 2);

    graph.click(60.0f, 60.0f);
    graph.click(170.0f, 50.0f);
    CHECK(a.content.count(MouseEventType::Click) == 2);
    CheckPoint(a.last().gridPosition, 40.0f, 40.0f);

    graph.drag(600.0f, 500.0f, 500.0f, 500.0f);
    CHECK(a.content.laidClip == Rect{100.0f, 40.0f, 76.0f, 96.0f});
    CHECK(Contains(body, a.content.laidClip));
}

TEST_CASE("Pressing or panning the empty canvas clears the node selection", "[core][sakura][node-editor]") {
    Graph graph;
    Block& a = graph.add("a", {40.0f, 40.0f});
    a.content.consume = false;
    graph.step();

    graph.click(60.0f, 60.0f);
    CHECK(graph.copy() == std::vector<std::string>{"a"});

    graph.click(600.0f, 500.0f);
    CHECK(graph.copy().empty());

    graph.click(60.0f, 60.0f);
    CHECK(graph.copy() == std::vector<std::string>{"a"});

    graph.drag(600.0f, 500.0f, 650.0f, 520.0f);
    CheckPoint(*graph.center, 350.0f, 280.0f);
    CHECK(graph.copy().empty());

    graph.click(110.0f, 80.0f);
    CHECK(graph.copy() == std::vector<std::string>{"a"});
    CheckPoint(a.last().gridPosition, 40.0f, 40.0f);
}

TEST_CASE("Node titles and identifiers render in Space Grotesk while pin labels stay monospace",
          "[core][sakura][node-editor]") {
    Graph graph;
    auto& block = graph.add("fft", {100.0f, 100.0f});
    block.spec.header = true;
    block.spec.title = "Spectrum";
    block.spec.subtitle = "fft";
    REQUIRE(graph.frames.settle([&] { graph.update(); }));

    std::vector<Sakura::Label*> labels;
    std::function<void(Sakura::Component&)> collect = [&](Sakura::Component& node) {
        if (auto* label = dynamic_cast<Sakura::Label*>(&node)) {
            labels.push_back(label);
        }
        for (auto* child : Peek::Children(node)) {
            if (child) {
                collect(*child);
            }
        }
    };
    collect(graph.editor);

    std::vector<std::string> header;
    std::vector<std::string> mono;
    for (auto* label : labels) {
        for (const auto& instance : label->config().instances) {
            if (label->config().fontName == "default_display_semibold") {
                header.push_back(instance.str);
            } else if (label->config().fontName == Sakura::Typography::MonoFont) {
                mono.push_back(instance.str);
            }
        }
    }
    CHECK(header == std::vector<std::string>{"Spectrum", "fft"});
    CHECK(std::find(mono.begin(), mono.end(), "Spectrum") == mono.end());
    CHECK(std::find(mono.begin(), mono.end(), "fft") == mono.end());
}

TEST_CASE("Node timing lines all fit, including the total after many module lines",
          "[core][sakura][node-editor]") {
    Graph graph;
    auto& block = graph.add("analyzer", {100.0f, 100.0f});
    for (U64 index = 0; index < 8; ++index) {
        block.spec.timing.push_back("module " + std::to_string(index) + ": 0.100 ms");
    }
    block.spec.timing.push_back("= 0.800 ms (12)");
    REQUIRE(graph.frames.settle([&] { graph.update(); }));

    const Sakura::Label* timing = nullptr;
    std::function<void(Sakura::Component&)> visit = [&](Sakura::Component& node) {
        if (auto* label = dynamic_cast<Sakura::Label*>(&node)) {
            const auto& instances = label->config().instances;
            if (!instances.empty() && instances.back().str == "= 0.800 ms (12)") {
                timing = label;
            }
        }
        for (auto* child : Peek::Children(node)) {
            if (child) {
                visit(*child);
            }
        }
    };
    visit(graph.editor);
    REQUIRE(timing);
    CHECK(timing->config().instances.size() == 9);
    CHECK(timing->config().capacity >= 9);

    const U64 capacity = timing->config().capacity;
    const U64 rebuilds = graph.frames.rebuilds;
    const auto lines = block.spec.timing;
    block.spec.timing.clear();
    REQUIRE(graph.frames.settle([&] { graph.update(); }));
    block.spec.timing = lines;
    REQUIRE(graph.frames.settle([&] { graph.update(); }));
    CHECK(timing->config().capacity == capacity);
    CHECK(graph.frames.rebuilds == rebuilds);
}

TEST_CASE("Covered parts of a node button ignore presses until its node rises above the cover",
          "[core][sakura][node-editor]") {
    Graph graph;
    U64 clicks = 0;
    Sakura::Button button;
    button.update({.str = "Covered", .size = {Sakura::Fill, 40.0f}, .onClick = [&] { ++clicks; }});
    Block& back = graph.add("back", {20.0f, 20.0f}, {220.0f, 0.0f});
    back.content.consume = false;
    back.field = &button;
    Block& front = graph.add("front", {140.0f, 20.0f}, {200.0f, 200.0f}, ResizeAxes::XY);
    graph.step();
    graph.step();

    const Rect frame = Peek::Frame(button);
    REQUIRE_FALSE(frame.empty());
    const F32 y = frame.y + frame.height * 0.5f;
    const F32 exposed = front.last().gridPosition.x - 10.0f;
    const F32 covered = front.last().gridPosition.x + 10.0f;
    REQUIRE(exposed > frame.x);
    REQUIRE(covered < frame.right());

    graph.move(covered, y);
    graph.click(covered, y);
    CHECK(clicks == 0);

    graph.move(exposed, y);
    graph.press(exposed, y);
    graph.move(covered, y);
    graph.release(covered, y);
    CHECK(clicks == 1);

    graph.move(covered, y);
    graph.click(covered, y);
    CHECK(clicks == 2);
}

TEST_CASE("Node editors report which nodes are on screen and whether they are shown",
          "[core][sakura][node-editor]") {
    Graph graph;
    Block& near = graph.add("near", {20.0f, 20.0f});
    Block& far = graph.add("far", {5000.0f, 5000.0f});
    graph.step();
    graph.step();
    REQUIRE_FALSE(near.reports.empty());
    REQUIRE_FALSE(far.reports.empty());
    CHECK(near.last().onscreen);
    CHECK_FALSE(far.last().onscreen);
    CHECK(graph.editor.shown());

    far.spec.gridPosition = {60.0f, 200.0f};
    graph.step();
    graph.step();
    CHECK(far.last().onscreen);

    graph.editor.setVisible(false);
    graph.step();
    CHECK_FALSE(graph.editor.shown());
    graph.editor.setVisible(true);
    graph.step();
    CHECK(graph.editor.shown());
}

TEST_CASE("Clicking inside a focused node field moves the caret without blurring or committing it",
          "[core][sakura][node-editor]") {
    Graph graph;
    std::vector<std::string> changes;
    Sakura::NodeTextInput input;
    input.update({.id = "caret", .value = "12", .submit = Sakura::NodeTextInput::Submit::OnEnter,
                  .onChange = [&](const std::string& value) { changes.push_back(value); }});
    Block& a = graph.add("a", {40.0f, 40.0f}, {200.0f, 0.0f});
    a.content.consume = false;
    a.field = &input;
    graph.step();
    graph.step();

    const Rect frame = Peek::Frame(input);
    REQUIRE_FALSE(frame.empty());
    const F32 y = frame.y + frame.height * 0.5f;
    graph.click(frame.right() - 4.0f, y);
    REQUIRE(input.focused());
    Type(graph.host, "3");
    graph.step();

    graph.click(frame.x + 4.0f, y);
    CHECK(input.focused());
    CHECK(changes.empty());

    CHECK(Tap(graph.host, KeyCode::Enter));
    graph.step();
    REQUIRE(changes.size() == 1);
    CHECK(changes[0].size() == 3);
    CHECK_FALSE(input.focused());
}

TEST_CASE("Modifier clicks build a node selection that drags together and Shift drags select live nodes by box",
          "[core][sakura][node-editor]") {
    Graph graph;
    Block& a = graph.add("a", {40.0f, 40.0f});
    Block& b = graph.add("b", {300.0f, 40.0f});
    Block& c = graph.add("c", {40.0f, 300.0f});
    for (Block* block : {&a, &b, &c}) {
        block->content.consume = false;
    }
    graph.step();
    graph.step();

    graph.click(60.0f, 60.0f);
    graph.press(320.0f, 60.0f, MouseButton::Left, 1, {.shift = true});
    graph.release(320.0f, 60.0f);
    CHECK(graph.copy() == std::vector<std::string>{"a", "b"});

    graph.drag(60.0f, 60.0f, 110.0f, 90.0f);
    CheckPoint(a.last().gridPosition, 90.0f, 70.0f);
    CheckPoint(b.last().gridPosition, 350.0f, 70.0f);
    CheckPoint(c.last().gridPosition, 40.0f, 300.0f);
    CHECK(graph.copy() == std::vector<std::string>{"a", "b"});

    graph.press(370.0f, 90.0f, MouseButton::Left, 1, {.super = true});
    graph.release(370.0f, 90.0f);
    CHECK(graph.copy() == std::vector<std::string>{"a"});

    graph.press(20.0f, 20.0f, MouseButton::Left, 1, {.shift = true});
    graph.move(30.0f, 280.0f);
    graph.send(MouseEventType::Move, 100.0f, 340.0f);
    graph.release(100.0f, 340.0f);
    CHECK(graph.copy() == std::vector<std::string>{"a", "c"});

    graph.click(700.0f, 550.0f);
    CHECK(graph.copy().empty());

    graph.click(120.0f, 100.0f);
    REQUIRE(graph.copy() == std::vector<std::string>{"a"});
    graph.press(700.0f, 550.0f, MouseButton::Left, 1, {.shift = true});
    a.present = false;
    graph.step();
    graph.send(MouseEventType::Move, 30.0f, 280.0f);
    graph.release(30.0f, 280.0f);
    CHECK(graph.copy() == std::vector<std::string>{"c"});
}

TEST_CASE("Adding or removing a node restructures the editor once without a rebuild and then settles",
          "[core][sakura][node-editor][retention]") {
    Graph graph;
    graph.frames.ctx.palette = DistinctPalette();
    graph.add("a", {40.0f, 40.0f});
    graph.add("b", {300.0f, 40.0f});
    REQUIRE(Settle(graph));
    CHECK(Dirty(graph) == 0);
    const U64 rebuilds = graph.frames.rebuilds;
    const U64 restructures = graph.frames.restructures;

    Block& c = graph.add("c", {300.0f, 300.0f});
    const auto added = graph.step();
    CHECK(added.restructured);
    CHECK_FALSE(added.rebuilt);
    REQUIRE(Settle(graph));
    CHECK(Dirty(graph) == 0);
    CHECK(graph.frames.restructures == restructures + 1);
    CHECK(c.content.layouts > 0);

    c.present = false;
    const auto removed = graph.step();
    CHECK(removed.restructured);
    CHECK_FALSE(removed.rebuilt);
    REQUIRE(Settle(graph));
    CHECK(Dirty(graph) == 0);
    CHECK(graph.frames.restructures == restructures + 2);
    CHECK(graph.frames.rebuilds == rebuilds);
}

TEST_CASE("Panning a node out of view and back only repaints and the panned editor settles",
          "[core][sakura][node-editor][retention]") {
    Graph graph;
    graph.frames.ctx.palette = DistinctPalette();
    Block& a = graph.add("a", {40.0f, 40.0f});
    REQUIRE(Settle(graph));
    const U64 rebuilds = graph.frames.rebuilds;
    const U64 restructures = graph.frames.restructures;

    const auto panTo = [&](F32 x, F32 y) {
        graph.send(MouseEventType::Move, x, y);
        return graph.step();
    };
    graph.send(MouseEventType::Click, 600.0f, 500.0f);
    graph.step();
    CHECK(RepaintOnly(panTo(300.0f, 500.0f)));
    const U64 hidden = a.content.layouts;
    CHECK(Dirty(graph) == 0);
    CHECK(a.content.layouts == hidden);

    CHECK(RepaintOnly(panTo(610.0f, 500.0f)));
    graph.send(MouseEventType::Release, 610.0f, 500.0f);
    REQUIRE(Settle(graph));
    CHECK(Dirty(graph) == 0);
    CHECK(a.content.layouts > hidden);
    CHECK(a.content.laidFrame == Rect{48.0f, 48.0f, 120.0f, 80.0f});
    CHECK(graph.frames.rebuilds == rebuilds);
    CHECK(graph.frames.restructures == restructures);
}

TEST_CASE("Zooming keeps the grid point under the viewport center fixed and scales distances around it",
          "[core][sakura][node-editor]") {
    Graph graph;
    Block& a = graph.add("a", {40.0f, 40.0f});
    graph.step();
    graph.drag(600.0f, 500.0f, 650.0f, 520.0f);
    CheckPoint(*graph.center, 350.0f, 280.0f);
    CheckNear(CanvasPoint(graph, 400.0f, 300.0f), 350.0f, 280.0f, 0.5f);

    for (int step = 0; step < 2; ++step) {
        graph.editor.zoomStep(1);
        graph.step();
    }
    REQUIRE(graph.editor.zoom() == Catch::Approx(1.1f));
    CheckNear(*graph.center, 350.0f, 280.0f, 0.5f);
    CheckNear(CanvasPoint(graph, 400.0f, 300.0f), 350.0f, 280.0f, 0.5f);
    CheckNear(CanvasPoint(graph, 510.0f, 300.0f), 450.0f, 280.0f, 0.5f);
    CheckNear(CanvasPoint(graph, 400.0f, 410.0f), 350.0f, 380.0f, 0.5f);
    CHECK(a.content.laidFrame.width == Catch::Approx(132.0f));

    for (int step = 0; step < 4; ++step) {
        graph.editor.zoomStep(-1);
        graph.step();
    }
    REQUIRE(graph.editor.zoom() == Catch::Approx(0.9f));
    CheckNear(*graph.center, 350.0f, 280.0f, 0.5f);
    CheckNear(CanvasPoint(graph, 400.0f, 300.0f), 350.0f, 280.0f, 0.5f);
    CheckNear(CanvasPoint(graph, 490.0f, 300.0f), 450.0f, 280.0f, 0.5f);
    CHECK(a.content.laidFrame.width == Catch::Approx(108.0f));

    graph.editor.zoomReset();
    graph.step();
    CHECK(graph.editor.zoom() == 1.0f);
    CheckNear(CanvasPoint(graph, 500.0f, 300.0f), 450.0f, 280.0f, 0.5f);
}

TEST_CASE("Zoom steps move in even five percent increments and stop at the smallest and largest zoom",
          "[core][sakura][node-editor]") {
    Graph graph;
    graph.step();
    CHECK(graph.editor.zoom() == 1.0f);
    CHECK(graph.editor.canZoom(1));
    CHECK(graph.editor.canZoom(-1));

    std::vector<F32> zooms;
    while (graph.editor.canZoom(1)) {
        graph.editor.zoomStep(1);
        zooms.push_back(graph.editor.zoom());
    }
    REQUIRE(zooms.size() == 20);
    for (U64 index = 0; index < zooms.size(); ++index) {
        CHECK(zooms[index] == Catch::Approx(1.0f + 0.05f * static_cast<F32>(index + 1)));
    }
    graph.editor.zoomStep(1);
    CHECK(graph.editor.zoom() == 2.0f);
    CHECK(graph.editor.canZoom(-1));

    graph.editor.zoomReset();
    CHECK(graph.editor.zoom() == 1.0f);
    zooms.clear();
    while (graph.editor.canZoom(-1)) {
        graph.editor.zoomStep(-1);
        zooms.push_back(graph.editor.zoom());
    }
    REQUIRE(zooms.size() == 10);
    for (U64 index = 0; index < zooms.size(); ++index) {
        CHECK(zooms[index] == Catch::Approx(1.0f - 0.05f * static_cast<F32>(index + 1)));
    }
    graph.editor.zoomStep(-1);
    CHECK(graph.editor.zoom() == 0.5f);
    CHECK_FALSE(graph.editor.canZoom(-1));
    CHECK(graph.editor.canZoom(1));
}

TEST_CASE("Control or Command with Plus, Minus and Zero zoom the focused editor", "[core][sakura][node-editor]") {
    Graph graph;
    graph.step();
    graph.click(600.0f, 500.0f);
    REQUIRE(FocusOwner(graph.host) == &graph.editor);

    CHECK(Tap(graph.host, KeyCode::Equal, {.control = true}));
    CHECK(graph.editor.zoom() == Catch::Approx(1.05f));
    CHECK(Tap(graph.host, KeyCode::KeypadAdd, {.super = true}));
    CHECK(graph.editor.zoom() == Catch::Approx(1.1f));
    CHECK(Sakura::Access::key(graph.host, {KeyEventType::Press, KeyCode::Equal, {.control = true}, true}));
    CHECK(graph.editor.zoom() == Catch::Approx(1.15f));
    CHECK(Tap(graph.host, KeyCode::Minus, {.control = true}));
    CHECK(graph.editor.zoom() == Catch::Approx(1.1f));
    CHECK(Tap(graph.host, KeyCode::KeypadSubtract, {.super = true}));
    CHECK(graph.editor.zoom() == Catch::Approx(1.05f));

    CHECK_FALSE(Sakura::Access::key(graph.host, {KeyEventType::Press, KeyCode::Digit0, {.control = true}, true}));
    CHECK(graph.editor.zoom() == Catch::Approx(1.05f));
    CHECK(Tap(graph.host, KeyCode::Digit0, {.control = true}));
    CHECK(graph.editor.zoom() == 1.0f);
    CHECK(Tap(graph.host, KeyCode::Keypad0, {.super = true}));
    CHECK(graph.editor.zoom() == 1.0f);

    CHECK_FALSE(Tap(graph.host, KeyCode::Equal));
    CHECK_FALSE(Tap(graph.host, KeyCode::Minus, {.shift = true}));
    CHECK(graph.editor.zoom() == 1.0f);
}

TEST_CASE("Zoom shortcuts leave the zoom alone while a node text field has focus", "[core][sakura][node-editor]") {
    Graph graph;
    Sakura::NodeTextInput input;
    input.update({.id = "name", .value = "12"});
    Block& a = graph.add("a", {40.0f, 40.0f}, {200.0f, 0.0f});
    a.content.consume = false;
    a.field = &input;
    graph.step();
    graph.step();

    const Rect frame = Peek::Frame(input);
    REQUIRE_FALSE(frame.empty());
    graph.click(frame.x + 4.0f, frame.y + frame.height * 0.5f);
    REQUIRE(input.focused());

    Tap(graph.host, KeyCode::Equal, {.control = true});
    Tap(graph.host, KeyCode::Minus, {.control = true});
    graph.step();
    CHECK(graph.editor.zoom() == 1.0f);
    CHECK(input.focused());
}

TEST_CASE("A zoom change repaints the editor without a rebuild or restructure and then settles",
          "[core][sakura][node-editor][retention]") {
    Graph graph;
    graph.frames.ctx.palette = DistinctPalette();
    graph.add("a", {40.0f, 40.0f});
    graph.add("b", {900.0f, 700.0f});
    REQUIRE(Settle(graph));
    const U64 rebuilds = graph.frames.rebuilds;
    const U64 restructures = graph.frames.restructures;

    graph.editor.zoomStep(1);
    CHECK(RepaintOnly(graph.step()));
    REQUIRE(Settle(graph));
    CHECK(Dirty(graph) == 0);

    while (graph.editor.canZoom(-1)) {
        graph.editor.zoomStep(-1);
        CAPTURE(graph.editor.zoom());
        CHECK(RepaintOnly(graph.step()));
        REQUIRE(Settle(graph));
        CHECK(Dirty(graph) == 0);
    }

    graph.editor.zoomStep(-1);
    CHECK(graph.step().clean());

    graph.editor.zoomReset();
    CHECK(RepaintOnly(graph.step()));
    REQUIRE(Settle(graph));
    CHECK(Dirty(graph) == 0);
    CHECK(graph.frames.rebuilds == rebuilds);
    CHECK(graph.frames.restructures == restructures);
}

TEST_CASE("Two zoom shortcuts in one frame keep the grid point under the viewport center fixed",
          "[core][sakura][node-editor]") {
    Graph graph;
    graph.step();
    graph.drag(600.0f, 500.0f, 650.0f, 520.0f);
    graph.click(600.0f, 500.0f);
    REQUIRE(FocusOwner(graph.host) == &graph.editor);

    CHECK(Tap(graph.host, KeyCode::Equal, {.control = true}));
    CHECK(Tap(graph.host, KeyCode::Equal, {.control = true}));
    graph.step();
    REQUIRE(graph.editor.zoom() == Catch::Approx(1.1f));
    CheckNear(CanvasPoint(graph, 400.0f, 300.0f), 350.0f, 280.0f, 0.5f);
}

TEST_CASE("Pasting while another layer covers the pointer places the block at the viewport center",
          "[core][sakura][node-editor]") {
    SakuraTest::FakeWindow window;
    NodeEditor editor;
    Probe cover({200.0f, 150.0f});
    Probe body({0.0f, 80.0f});
    Sakura::NodeContentStack stack;
    LayerSlot slot;
    slot.hold(editor, kScreen);
    LayerSlot top;
    top.hold(cover, {500.0f, 400.0f, 200.0f, 150.0f});
    LayerHost host({&slot, &top});
    Frames frames(host);
    frames.verify = false;
    frames.ctx.render = &window;
    std::optional<Extent2D<F32>> center;
    std::vector<Extent2D<F32>> pastes;
    const auto update = [&] {
        stack.update({.id = "a:content", .items = {{.child = &body}}});
        editor.update({
            .id = "editor",
            .pasteEnabled = true,
            .nodes = {{.id = "a", .header = false, .title = "a", .content = &stack, .resize = ResizeAxes::X,
                       .dimensions = {120.0f, 0.0f}, .gridPosition = Extent2D<F32>{40.0f, 40.0f}}},
            .onPasteShortcut = [&](Extent2D<F32> point) { pastes.push_back(point); },
            .onViewportGridChange = [&](Rect next) {
                center = Extent2D<F32>{next.x + next.width * 0.5f, next.y + next.height * 0.5f};
            },
        });
    };
    REQUIRE(frames.settle(update));
    REQUIRE(center.has_value());

    const U64 generation = Sakura::Access::focusGeneration(host);
    host.send(Mouse(MouseEventType::Click, 100.0f, 100.0f));
    Sakura::Access::clickFocus(host, Extent2D<F32>{100.0f, 100.0f}, true, generation);
    host.send(Mouse(MouseEventType::Release, 100.0f, 100.0f));
    frames.run(update);
    REQUIRE(FocusOwner(host) == &editor);
    host.send(Mouse(MouseEventType::Move, 600.0f, 450.0f));
    frames.run(update);

    CHECK(Tap(host, KeyCode::V, {.control = true}));
    REQUIRE(pastes.size() == 1);
    CHECK(pastes[0].x == Catch::Approx(center->x));
    CHECK(pastes[0].y == Catch::Approx(center->y));
}

TEST_CASE("The editor only consumes a middle release after its own middle press", "[core][sakura][node-editor]") {
    Graph graph;
    graph.step();
    CHECK_FALSE(graph.send(MouseEventType::Release, 300.0f, 300.0f, MouseButton::Middle));
    CHECK(graph.send(MouseEventType::Click, 300.0f, 300.0f, MouseButton::Middle));
    CHECK(graph.send(MouseEventType::Release, 300.0f, 300.0f, MouseButton::Middle));
    CHECK_FALSE(graph.send(MouseEventType::Release, 300.0f, 300.0f, MouseButton::Middle));
}
