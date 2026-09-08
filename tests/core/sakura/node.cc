#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <jetstream/render/sakura/components/node/node.hh>
#include <jetstream/render/tools/imnodes.h>
#include <jetstream/render/tools/imnodes_internal.h>

#include "compositor/default/views/flowgraph/editor/node.hh"
#include "compositor/default/model/meta.hh"

#include "harness.hh"

#include <render/sakura/components/node/base.hh>

#include <string>
#include <tuple>
#include <vector>

using namespace Jetstream;

namespace {

// Fixed content so the node group gets a real, stable rect.
void NodeContent(const Sakura::Context&) {
    ImGui::Dummy(ImVec2(160.0f, 120.0f));
}

int objectId(const std::string& id) {
    return Sakura::Private::NodeEditorObjectId(id);
}

const ImNodeData* componentNodeData(const std::string& id) {
    auto& editor = ImNodes::EditorContextGet();
    const int idx = ImNodes::ObjectPoolFind(editor.Nodes, objectId(id));
    return idx >= 0 ? &editor.Nodes.Pool[idx] : nullptr;
}

ImVec2 componentNodeDimensions(const std::string& id) {
    return ImNodes::GetNodeDimensions(objectId(id));
}

// Press the grip, drag by delta while held, then release.
void dragFromCorner(SakuraTest::HeadlessUi& ui,
                    const std::string& id,
                    ImVec2 delta,
                    Sakura::Node& node,
                    const Sakura::Context& ctx) {
    const ImVec2 origin = ImNodes::GetNodeScreenSpacePos(objectId(id));
    const ImVec2 size = componentNodeDimensions(id);
    const ImVec2 grip = ImVec2(origin.x + size.x - 2.0f, origin.y + size.y - 2.0f);

    ui.setMouse(grip, true);
    ui.editorFrame([&] {
        node.render(ctx, NodeContent);
    });

    ui.setMouse(ImVec2(grip.x + delta.x, grip.y + delta.y), true);
    ui.editorFrame([&] {
        node.render(ctx, NodeContent);
    });

    ui.setMouse(ImVec2(grip.x + delta.x, grip.y + delta.y), false);
    ui.editorFrame([&] {
        node.render(ctx, NodeContent);
    });
    ui.setMouse(ImVec2(-100.0f, -100.0f), false);
}

Sakura::Node::Config baseConfig(const std::string& id) {
    Sakura::Node::Config config;
    config.id = id;
    config.dimensions = {200.0f, 160.0f};
    config.minimumDimensions = {120.0f, 100.0f};
    return config;
}

FlowgraphNode::BlockData baseBlock(const std::string& name, const std::string& module) {
    FlowgraphNode::BlockData block;
    block.name = name;
    block.module = module;
    block.title = "Title";
    block.state = Block::State::Created;
    block.nodeSize = Block::NodeSize::M;
    return block;
}

FlowgraphNode::Config baseConfigFor(FlowgraphNode::BlockData block) {
    FlowgraphNode::Config config;
    config.id = block.name;
    config.block = std::move(block);
    return config;
}

FlowgraphNode::Surface attachedSurface(const std::string& id) {
    FlowgraphNode::Surface surface;
    surface.id = id;
    surface.height = 512.0f;
    surface.detached = false;
    return surface;
}

int imnodesId(const std::string& configId) {
    return Sakura::Private::NodeEditorObjectId(FlowgraphNodeId(configId));
}

const ImNodeData* flowgraphNodeData(const std::string& configId) {
    auto& editor = ImNodes::EditorContextGet();
    const int idx = ImNodes::ObjectPoolFind(editor.Nodes, imnodesId(configId));
    return idx >= 0 ? &editor.Nodes.Pool[idx] : nullptr;
}

ImVec2 flowgraphNodeDimensions(const std::string& configId) {
    return ImNodes::GetNodeDimensions(imnodesId(configId));
}

struct LayoutLog {
    std::vector<std::tuple<F32, F32, F32, F32>> entries;

    void record(F32 x, F32 y, F32 width, F32 height) {
        entries.emplace_back(x, y, width, height);
    }

    const std::tuple<F32, F32, F32, F32>& last() const {
        return entries.back();
    }
};

// Renders the node for one editor frame.
void renderFrame(SakuraTest::HeadlessUi& ui,
                 const Sakura::Context& ctx,
                 FlowgraphNode& node) {
    ui.editorFrame([&] {
        node.render(ctx);
    });
}

// Update + render until the layout settles, then enforce the node
// invariant: a Y-resizable node never rests below its minimum height.
void settle(SakuraTest::HeadlessUi& ui,
            const Sakura::Context& ctx,
            FlowgraphNode& node,
            FlowgraphNode::Config& config,
            U64 frames) {
    for (U64 i = 0; i < frames; ++i) {
        node.update(config);
        renderFrame(ui, ctx, node);
    }

    const auto* data = flowgraphNodeData(config.id);
    REQUIRE(data != nullptr);
    if (!(data->ResizeFlags & ImNodesNodeResizeFlags_Y)) {
        return;
    }
    if (config.block.state == Block::State::Creating) {
        return;
    }
    const auto padding = ImNodes::GetStyle().NodePadding;
    REQUIRE(flowgraphNodeDimensions(config.id).y - 2.0f * padding.y >=
            data->ResizeMinimumSize.y - 1.0f);
}

// Press the node grip, drag by delta, release.
void dragCorner(SakuraTest::HeadlessUi& ui,
                const Sakura::Context& ctx,
                FlowgraphNode& node,
                const std::string& configId,
                ImVec2 delta) {
    const ImVec2 origin = ImNodes::GetNodeScreenSpacePos(imnodesId(configId));
    const ImVec2 size = flowgraphNodeDimensions(configId);
    const ImVec2 grip = ImVec2(origin.x + size.x - 2.0f, origin.y + size.y - 2.0f);

    ui.setMouse(grip, true);
    renderFrame(ui, ctx, node);

    ui.setMouse(ImVec2(grip.x + delta.x, grip.y + delta.y), true);
    renderFrame(ui, ctx, node);

    ui.setMouse(ImVec2(grip.x + delta.x, grip.y + delta.y), false);
    renderFrame(ui, ctx, node);
    ui.setMouse(ImVec2(-100.0f, -100.0f), false);
}

// Drive updates during gestures and feed callbacks back one frame later,
// like the presenter's poll -> update -> render cycle. Keep the metadata
// separate from the view so persistence assertions cannot pass on UI state alone.
struct NodeSession {
    SakuraTest::HeadlessUi ui;
    Sakura::Context ctx;
    FlowgraphNode node;
    FlowgraphNode::Config config;
    NodeMeta meta;
    std::optional<FlowgraphNode::Layout> layoutMail;
    std::optional<bool> collapseMail;
    LayoutLog layouts;
    std::vector<bool> toggles;

    NodeSession(FlowgraphNode::Config config, F32 scale = 1.0f, NodeMeta meta = {}) :
        ui(scale, ImVec2(1400.0f, 1400.0f)), ctx(ui.sakura()),
        config(std::move(config)), meta(meta) {}

    void tick() {
        if (layoutMail.has_value()) {
            meta.x = layoutMail->x;
            meta.y = layoutMail->y;
            meta.width = layoutMail->width;
            meta.height = layoutMail->height;
            layoutMail.reset();
        }
        if (collapseMail.has_value()) {
            meta.configCollapsed = *collapseMail;
            collapseMail.reset();
        }
        config.block.layout = FlowgraphNode::Layout{meta.x, meta.y, meta.width, meta.height};
        config.block.configCollapsed = meta.configCollapsed;
        config.onLayout = [this](F32 x, F32 y, F32 width, F32 height) {
            layouts.record(x, y, width, height);
            layoutMail = FlowgraphNode::Layout{x, y, width, height};
        };
        config.onConfigCollapse = [this](bool collapsed) {
            toggles.push_back(collapsed);
            collapseMail = collapsed;
        };
        node.update(config);
        renderFrame(ui, ctx, node);
        REQUIRE(flowgraphNodeData(config.id) != nullptr);
    }

    void frames(U64 count = 6) {
        for (U64 i = 0; i < count; ++i) {
            tick();
        }
    }

    void gesture(ImVec2 start, ImVec2 delta = ImVec2(0.0f, 0.0f)) {
        ui.setMouse(start, false);
        tick();
        ui.setMouse(start, true);
        tick();
        ui.setMouse(ImVec2(start.x + delta.x, start.y + delta.y), true);
        tick();
        ui.setMouse(ImVec2(start.x + delta.x, start.y + delta.y), false);
        tick();
        ui.setMouse(ImVec2(-100.0f, -100.0f), false);
        frames();
    }

    void resize(ImVec2 delta) {
        const auto* data = flowgraphNodeData(config.id);
        REQUIRE(data != nullptr);
        gesture(ImVec2(data->Rect.Max.x - 2.0f, data->Rect.Max.y - 2.0f), delta);
    }

    ImVec2 chevron() const {
        const auto* data = flowgraphNodeData(config.id);
        REQUIRE(data != nullptr);
        // The icon sits at the right of the title content, inside node padding.
        return ImVec2(data->Rect.Max.x - data->LayoutStyle.Padding.x - 4.0f,
                      data->TitleBarContentRect.GetCenter().y);
    }
};
}  // namespace

TEST_CASE("Errored python block still floors and resizes",
          "[core][sakura][flowgraph_node][errored]") {
    SakuraTest::HeadlessUi ui(1.0f, ImVec2(600.0f, 900.0f));
    const auto ctx = ui.sakura();
    const auto padding = ImNodes::GetStyle().NodePadding;

    // Chrome probe: same module, same state and diagnostic, no fields.
    FlowgraphNode chromeProbe;
    auto chromeConfig = baseConfigFor(baseBlock("err-chrome", "radio"));
    chromeConfig.block.state = Block::State::Errored;
    chromeConfig.block.diagnostic = "Python runtime raised: NameError.";
    settle(ui, ctx, chromeProbe, chromeConfig, 3);
    const F32 chrome = flowgraphNodeDimensions("err-chrome").y - 2.0f * padding.y;
    REQUIRE(chrome > 0.0f);

    // A block that errored at runtime must keep its floor and its
    // resize handles: the user still needs to reposition and resize it.
    FlowgraphNode node;
    auto config = baseConfigFor(baseBlock("err-block", "radio"));
    config.block.configFields.push_back(
        {.id = "err-block:code", .format = "python", .encoded = "pass"});
    config.block.state = Block::State::Errored;
    config.block.diagnostic = "Python runtime raised: NameError.";
    settle(ui, ctx, node, config, 5);

    const auto* data = flowgraphNodeData("err-block");
    REQUIRE(data != nullptr);
    REQUIRE((data->ResizeFlags & ImNodesNodeResizeFlags_Y) != 0);
    const F32 floor = data->ResizeMinimumSize.y;
    REQUIRE(floor >= chrome + 120.0f - 0.5f);

    const auto before = flowgraphNodeDimensions("err-block");
    REQUIRE(before.y - 2.0f * padding.y >= floor - 1.0f);
    dragCorner(ui, ctx, node, "err-block", ImVec2(0.0f, 300.0f));
    settle(ui, ctx, node, config, 4);
    REQUIRE(flowgraphNodeDimensions("err-block").y ==
            Catch::Approx(before.y + 300.0f).margin(2.0f));
}

TEST_CASE("Node adopts content-driven width only when not horizontally resizable",
          "[core][sakura][node]") {
    SakuraTest::HeadlessUi ui;
    const auto ctx = ui.sakura();

    // The child renders content wider than the requested node size: the
    // same divergence that made resizable nodes ratchet wider forever.
    // The persisted request (contentDimensions) reveals whether the node
    // adopted the wider content on the following frame.
    const auto wideChild = [](const Sakura::Context&) {
        ImGui::Dummy(ImVec2(300.0f, 40.0f));
    };

    const auto measured = [&](Sakura::Node& node, Sakura::Node::ResizeAxes axes) {
        Extent2D<F32> content{};
        auto config = baseConfig("grow-probe");
        config.resize = axes;
        config.dimensions = {200.0f, 100.0f};
        config.onGeometryChange = [&content](Extent2D<F32>,
                                             Extent2D<F32>,
                                             Extent2D<F32>,
                                             Extent2D<F32> contentDimensions) {
            content = contentDimensions;
        };
        REQUIRE(node.update(config));
        ui.editorFrame([&] {
            node.render(ctx, wideChild);
        });
        REQUIRE(node.update(config));
        ui.editorFrame([&] {
            node.render(ctx, wideChild);
        });
        return content;
    };

    Sakura::Node resizable;
    const auto resizableContent = measured(resizable, Sakura::Node::ResizeAxes::XY);
    REQUIRE(resizableContent.x == Catch::Approx(200.0f).margin(1.0f));

    // A node without X resize adopts the wider content once.
    Sakura::Node fixed;
    const auto fixedContent = measured(fixed, Sakura::Node::ResizeAxes::None);
    REQUIRE(fixedContent.x >= 300.0f);
}

TEST_CASE("Foreground node blocks resize clicks from reaching overlapped nodes",
          "[core][sakura][node]") {
    SakuraTest::HeadlessUi ui(1.0f, ImVec2(600.0f, 600.0f));
    const auto ctx = ui.sakura();

    // A big resizable node behind, a smaller node in front covering the
    // back node's grip corner. Each scenario uses a fresh pair: the
    // blocked click selects/drag-moves the front node, which must not
    // leak geometry into the next scenario.
    const auto runScenario = [&](const std::string& backId,
                                 const std::string& frontId,
                                 const Sakura::Node::ResizeAxes frontAxes,
                                 const F32 gripInset) {
        Sakura::Node back;
        auto backConfig = baseConfig(backId);
        backConfig.resize = Sakura::Node::ResizeAxes::XY;
        backConfig.gridPosition = Extent2D<F32>{0.0f, 0.0f};
        backConfig.dimensions = {300.0f, 300.0f};
        REQUIRE(back.update(backConfig));

        Sakura::Node front;
        auto frontConfig = baseConfig(frontId);
        frontConfig.resize = frontAxes;
        frontConfig.gridPosition = Extent2D<F32>{250.0f, 250.0f};
        frontConfig.dimensions = {100.0f, 100.0f};
        REQUIRE(front.update(frontConfig));

        const auto frame = [&] {
            ui.editorFrame([&] {
                back.render(ctx, NodeContent);
                front.render(ctx, NodeContent);
            });
        };
        frame();

        const auto* backData = componentNodeData(backId);
        REQUIRE(backData != nullptr);
        const ImVec2 grip{backData->Rect.Max.x - gripInset,
                          backData->Rect.Max.y - gripInset};
        ui.setMouse(grip, true);
        frame();
        ui.setMouse(ImVec2(grip.x + 60.0f, grip.y + 60.0f), true);
        frame();
        ui.setMouse(ImVec2(grip.x + 60.0f, grip.y + 60.0f), false);
        frame();
        ui.setMouse(ImVec2(-100.0f, -100.0f), false);

        const auto* after = componentNodeData(backId);
        REQUIRE(after != nullptr);
        REQUIRE(after->ContentSize.x == Catch::Approx(300.0f).margin(0.5f));
        REQUIRE(after->ContentSize.y == Catch::Approx(300.0f).margin(0.5f));
    };

    // The front node is not resizable at all.
    runScenario("overlap-back", "overlap-front", Sakura::Node::ResizeAxes::None, 3.0f);

    // The front node is resizable, but the click lands on its body —
    // not on its own grip — so the node behind must not react either.
    runScenario("overlap-back-2", "overlap-front-2", Sakura::Node::ResizeAxes::XY, 8.0f);
}

TEST_CASE("Deleting a node during an active resize ends the interaction safely",
          "[core][sakura][node]") {
    SakuraTest::HeadlessUi ui(1.0f, ImVec2(600.0f, 600.0f));
    const auto ctx = ui.sakura();

    {
        Sakura::Node victim;
        auto config = baseConfig("resize-victim");
        config.resize = Sakura::Node::ResizeAxes::XY;
        config.dimensions = {200.0f, 160.0f};
        REQUIRE(victim.update(config));
        const auto render = [&] {
            ui.editorFrame([&] {
                victim.render(ctx, NodeContent);
            });
        };
        render();

        // Grab the grip and start dragging.
        const auto* data = componentNodeData("resize-victim");
        REQUIRE(data != nullptr);
        const ImVec2 grip{data->Rect.Max.x - 3.0f, data->Rect.Max.y - 3.0f};
        ui.setMouse(grip, true);
        render();
        ui.setMouse(ImVec2(grip.x + 40.0f, grip.y + 30.0f), true);
        render();

        auto& editor = ImNodes::EditorContextGet();
        REQUIRE(editor.ClickInteraction.Type == ImNodesClickInteractionType_NodeResize);
    }
    // The victim (and its ContentSizeRef target) is destroyed here, while
    // the resize interaction is still active and the mouse is held.

    Sakura::Node survivor;
    auto survivorConfig = baseConfig("resize-survivor");
    survivorConfig.dimensions = {120.0f, 80.0f};
    REQUIRE(survivor.update(survivorConfig));
    ui.editorFrame([&] {
        survivor.render(ctx, NodeContent);
    });

    // The stale interaction must have been dropped, not written through
    // the freed ContentSizeRef.
    auto& editor = ImNodes::EditorContextGet();
    REQUIRE(editor.ClickInteraction.Type != ImNodesClickInteractionType_NodeResize);
    REQUIRE(!ImNodes::GetCurrentContext()->NodeResizeIdx.HasValue());

    ui.setMouse(ImVec2(-100.0f, -100.0f), false);
    ui.editorFrame([&] {
        survivor.render(ctx, NodeContent);
    });
}

TEST_CASE("Auto-sized resizable axes retain drags and report them",
          "[core][sakura][node]") {
    SakuraTest::HeadlessUi ui(1.0f, ImVec2(600.0f, 600.0f));
    const auto ctx = ui.sakura();

    Sakura::Node node;
    auto config = baseConfig("auto-resize");
    config.resize = Sakura::Node::ResizeAxes::XY;
    config.dimensions = {0.0f, 0.0f};
    Extent2D<F32> reported{};
    config.onGeometryChange = [&](Extent2D<F32>,
                                  Extent2D<F32>,
                                  Extent2D<F32>,
                                  Extent2D<F32> contentDimensions) {
        reported = contentDimensions;
    };
    REQUIRE(node.update(config));
    ui.editorFrame([&] {
        node.render(ctx, NodeContent);
    });

    const auto* data = componentNodeData("auto-resize");
    REQUIRE(data != nullptr);
    const ImVec2 grip{data->Rect.Max.x - 3.0f, data->Rect.Max.y - 3.0f};
    ui.setMouse(grip, true);
    ui.editorFrame([&] {
        node.render(ctx, NodeContent);
    });
    ui.setMouse(ImVec2(grip.x + 60.0f, grip.y + 40.0f), true);
    ui.editorFrame([&] {
        node.render(ctx, NodeContent);
    });
    ui.setMouse(ImVec2(grip.x + 60.0f, grip.y + 40.0f), false);
    ui.editorFrame([&] {
        node.render(ctx, NodeContent);
    });
    ui.setMouse(ImVec2(-100.0f, -100.0f), false);

    // The drag writes back into the component state; subsequent frames with
    // a still-zero request must not erase it.
    ui.editorFrame([&] {
        node.render(ctx, NodeContent);
    });

    const auto* after = componentNodeData("auto-resize");
    REQUIRE(after != nullptr);
    REQUIRE(after->ContentSize.x == Catch::Approx(160.0f + 60.0f).margin(1.0f));
    REQUIRE(after->ContentSize.y == Catch::Approx(120.0f + 40.0f).margin(1.0f));
    REQUIRE(reported.x == Catch::Approx(160.0f + 60.0f).margin(1.0f));
    REQUIRE(reported.y == Catch::Approx(120.0f + 40.0f).margin(1.0f));
}

TEST_CASE("Node maps resize axes to imnodes resize flags", "[core][sakura][node]") {
    SakuraTest::HeadlessUi ui;
    const auto ctx = ui.sakura();

    for (const auto& [axes, expected] :
         {std::pair{Sakura::Node::ResizeAxes::None, ImNodesNodeResizeFlags_None},
          std::pair{Sakura::Node::ResizeAxes::X, ImNodesNodeResizeFlags_X},
          std::pair{Sakura::Node::ResizeAxes::Y, ImNodesNodeResizeFlags_Y},
          std::pair{Sakura::Node::ResizeAxes::XY, ImNodesNodeResizeFlags_XY}}) {
        const std::string id = "axes" + std::to_string(static_cast<int>(expected));
        Sakura::Node node;
        auto config = baseConfig(id);
        config.resize = axes;
        REQUIRE(node.update(config));
        ui.editorFrame([&] {
            node.render(ctx, NodeContent);
        });

        const auto* data = componentNodeData(id);
        REQUIRE(data != nullptr);
        REQUIRE(data->ResizeFlags == expected);
    }
}

TEST_CASE("Node plumbs scaled minimum dimensions into imnodes", "[core][sakura][node]") {
    for (const F32 scale : {1.0f, 1.5f, 2.0f}) {
        CAPTURE(scale);
        SakuraTest::HeadlessUi ui(scale, ImVec2(900.0f, 900.0f));
        const auto ctx = ui.sakura();

        Sakura::Node node;
        auto config = baseConfig("minimums");
        config.minimumDimensions = {150.0f, 90.0f};
        REQUIRE(node.update(config));
        ui.editorFrame([&] {
            node.render(ctx, NodeContent);
        });

        const auto* data = componentNodeData("minimums");
        REQUIRE(data != nullptr);
        REQUIRE(data->ResizeMinimumSize.x == Catch::Approx(150.0f * scale));
        REQUIRE(data->ResizeMinimumSize.y == Catch::Approx(90.0f * scale));
    }
}

TEST_CASE("Node resizes width only when restricted to the X axis",
          "[core][sakura][node][drag]") {
    SakuraTest::HeadlessUi ui;
    const auto ctx = ui.sakura();

    Sakura::Node node;
    auto config = baseConfig("x-only");
    config.resize = Sakura::Node::ResizeAxes::X;
    REQUIRE(node.update(config));

    ui.editorFrame([&] {
        node.render(ctx, NodeContent);
    });
    const auto before = componentNodeDimensions("x-only");

    dragFromCorner(ui, "x-only", ImVec2(60.0f, 40.0f), node, ctx);

    const auto after = componentNodeDimensions("x-only");
    REQUIRE(after.x == Catch::Approx(before.x + 60.0f).margin(1.0f));
    REQUIRE(after.y == Catch::Approx(before.y).margin(1.0f));
}

TEST_CASE("Node resizes height only when restricted to the Y axis",
          "[core][sakura][node][drag]") {
    SakuraTest::HeadlessUi ui;
    const auto ctx = ui.sakura();

    Sakura::Node node;
    auto config = baseConfig("y-only");
    config.resize = Sakura::Node::ResizeAxes::Y;
    REQUIRE(node.update(config));

    ui.editorFrame([&] {
        node.render(ctx, NodeContent);
    });
    const auto before = componentNodeDimensions("y-only");

    dragFromCorner(ui, "y-only", ImVec2(60.0f, 40.0f), node, ctx);

    const auto after = componentNodeDimensions("y-only");
    REQUIRE(after.x == Catch::Approx(before.x).margin(1.0f));
    REQUIRE(after.y == Catch::Approx(before.y + 40.0f).margin(1.0f));
}

TEST_CASE("Node resizes both axes with the XY axes mask",
          "[core][sakura][node][drag]") {
    SakuraTest::HeadlessUi ui;
    const auto ctx = ui.sakura();

    Sakura::Node node;
    auto config = baseConfig("xy");
    config.resize = Sakura::Node::ResizeAxes::XY;
    REQUIRE(node.update(config));

    ui.editorFrame([&] {
        node.render(ctx, NodeContent);
    });
    const auto before = componentNodeDimensions("xy");

    dragFromCorner(ui, "xy", ImVec2(60.0f, 40.0f), node, ctx);

    const auto after = componentNodeDimensions("xy");
    REQUIRE(after.x == Catch::Approx(before.x + 60.0f).margin(1.0f));
    REQUIRE(after.y == Catch::Approx(before.y + 40.0f).margin(1.0f));
}

TEST_CASE("Node enforces configured minimums while resizing",
          "[core][sakura][node][drag]") {
    SakuraTest::HeadlessUi ui;
    const auto ctx = ui.sakura();

    Sakura::Node node;
    auto config = baseConfig("clamped");
    config.resize = Sakura::Node::ResizeAxes::XY;
    config.minimumDimensions = {180.0f, 140.0f};
    REQUIRE(node.update(config));

    ui.editorFrame([&] {
        node.render(ctx, NodeContent);
    });
    const auto before = componentNodeDimensions("clamped");

    // A far negative drag must clamp at the configured minimums.
    dragFromCorner(ui, "clamped", ImVec2(-500.0f, -500.0f), node, ctx);

    const auto after = componentNodeDimensions("clamped");
    // Outer rect = content + 2x NodePadding (8px per side by default).
    const auto padding = ImNodes::GetStyle().NodePadding;
    REQUIRE(after.x == Catch::Approx(180.0f + 2.0f * padding.x).margin(1.5f));
    REQUIRE(after.y == Catch::Approx(140.0f + 2.0f * padding.y).margin(1.5f));
    REQUIRE(after.x < before.x);
    REQUIRE(after.y < before.y);
}

TEST_CASE("Node without resize axes ignores grip interaction",
          "[core][sakura][node][drag]") {
    SakuraTest::HeadlessUi ui;
    const auto ctx = ui.sakura();

    Sakura::Node node;
    auto config = baseConfig("inert");
    config.resize = Sakura::Node::ResizeAxes::None;
    REQUIRE(node.update(config));

    ui.editorFrame([&] {
        node.render(ctx, NodeContent);
    });
    const auto before = componentNodeDimensions("inert");

    dragFromCorner(ui, "inert", ImVec2(60.0f, 40.0f), node, ctx);

    const auto after = componentNodeDimensions("inert");
    REQUIRE(after.x == Catch::Approx(before.x).margin(1.0f));
    REQUIRE(after.y == Catch::Approx(before.y).margin(1.0f));
}

TEST_CASE("Surface and flexible-field nodes resize on both axes; plain nodes stay width-only",
          "[core][sakura][flowgraph_node]") {
    SakuraTest::HeadlessUi ui(1.0f, ImVec2(600.0f, 900.0f));
    const auto ctx = ui.sakura();

    // Attached surface -> XY.
    FlowgraphNode surfaceNode;
    auto surfaceConfig = baseConfigFor(baseBlock("surface-block", "radio"));
    surfaceConfig.block.surfaces.push_back(attachedSurface("surface-block:surface:out"));
    surfaceNode.update(surfaceConfig);
    renderFrame(ui, ctx, surfaceNode);
    const auto* surfaceData = flowgraphNodeData("surface-block");
    REQUIRE(surfaceData != nullptr);
    REQUIRE(surfaceData->ResizeFlags == ImNodesNodeResizeFlags_XY);

    // Flexible markdown field -> XY.
    FlowgraphNode markdownNode;
    auto markdownConfig = baseConfigFor(baseBlock("markdown-block", "note"));
    markdownConfig.block.configFields.push_back(
        {.id = "markdown-block:text", .format = "markdown", .encoded = "hello"});
    markdownNode.update(markdownConfig);
    renderFrame(ui, ctx, markdownNode);
    const auto* markdownData = flowgraphNodeData("markdown-block");
    REQUIRE(markdownData != nullptr);
    REQUIRE(markdownData->ResizeFlags == ImNodesNodeResizeFlags_XY);

    // Plain node -> X only.
    FlowgraphNode plainNode;
    auto plainConfig = baseConfigFor(baseBlock("plain-block", "note"));
    plainNode.update(plainConfig);
    renderFrame(ui, ctx, plainNode);
    const auto* plainData = flowgraphNodeData("plain-block");
    REQUIRE(plainData != nullptr);
    REQUIRE(plainData->ResizeFlags == ImNodesNodeResizeFlags_X);
}

TEST_CASE("Auto-height nodes report zero height and ignore saved layout heights",
          "[core][sakura][flowgraph_node][auto_height]") {
    SakuraTest::HeadlessUi ui(1.0f, ImVec2(600.0f, 900.0f));
    const auto ctx = ui.sakura();

    LayoutLog log;
    FlowgraphNode node;
    auto config = baseConfigFor(baseBlock("auto-block", "note"));
    config.onLayout = [&log](F32 x, F32 y, F32 width, F32 height) {
        log.record(x, y, width, height);
    };

    node.update(config);
    renderFrame(ui, ctx, node);
    node.update(config);
    renderFrame(ui, ctx, node);

    // The node renders with real content, but reports no height.
    REQUIRE(std::get<2>(log.last()) > 0.0f);
    REQUIRE(std::get<3>(log.last()) == Catch::Approx(0.0f));
    REQUIRE(flowgraphNodeDimensions("auto-block").y > 0.0f);

    // A saved layout height does not stick to auto-height nodes.
    config.block.layout = FlowgraphNode::Layout{.x = 10.0f, .y = 10.0f, .width = 0.0f, .height = 80.0f};
    node.update(config);
    renderFrame(ui, ctx, node);
    REQUIRE(std::get<3>(log.last()) == Catch::Approx(0.0f));
}

TEST_CASE("Detached surfaces are excluded from node dynamics until attached",
          "[core][sakura][flowgraph_node][detachment]") {
    SakuraTest::HeadlessUi ui(1.0f, ImVec2(600.0f, 900.0f));
    const auto ctx = ui.sakura();

    SakuraTest::ResizeLog resizeLog;
    LayoutLog layoutLog;
    FlowgraphNode node;
    auto config = baseConfigFor(baseBlock("detach-block", "radio"));
    auto surface = attachedSurface("detach-block:surface:out");
    surface.detached = true;
    surface.onAttachedSize = [&resizeLog](const Sakura::SurfaceResize& resize) {
        resizeLog.record(resize);
    };
    config.block.surfaces.push_back(surface);
    config.onLayout = [&layoutLog](F32 x, F32 y, F32 width, F32 height) {
        layoutLog.record(x, y, width, height);
    };

    node.update(config);
    renderFrame(ui, ctx, node);
    node.update(config);
    renderFrame(ui, ctx, node);

    // Detached: node is width-only, auto-height, and silent.
    REQUIRE(flowgraphNodeData("detach-block")->ResizeFlags == ImNodesNodeResizeFlags_X);
    REQUIRE(std::get<3>(layoutLog.last()) == Catch::Approx(0.0f));
    REQUIRE(resizeLog.count() == 0);

    // Attaching flips the node to XY and seeds from the persisted size.
    config.block.surfaces[0].detached = false;
    node.update(config);
    renderFrame(ui, ctx, node);
    REQUIRE(flowgraphNodeData("detach-block")->ResizeFlags == ImNodesNodeResizeFlags_XY);

    // The surface receives its full persisted height; the node carries it
    // plus the fixed chrome.
    node.update(config);
    renderFrame(ui, ctx, node);
    REQUIRE(resizeLog.count() == 1);
    REQUIRE(resizeLog.entries[0].logicalSize.y == Catch::Approx(512.0f).margin(1.0f));
    REQUIRE(resizeLog.entries[0].logicalSize.x > 0);
    REQUIRE(std::get<3>(layoutLog.last()) >= 512.0f);
}

TEST_CASE("User resize grows the node and reallocates surface height",
          "[core][sakura][flowgraph_node][drag]") {
    SakuraTest::HeadlessUi ui(1.0f, ImVec2(600.0f, 900.0f));
    const auto ctx = ui.sakura();

    LayoutLog layoutLog;
    SakuraTest::ResizeLog resizeLog;
    FlowgraphNode node;
    auto config = baseConfigFor(baseBlock("e2e-block", "radio"));
    auto surface = attachedSurface("e2e-block:surface:out");
    surface.onAttachedSize = [&resizeLog](const Sakura::SurfaceResize& resize) {
        resizeLog.record(resize);
    };
    config.block.surfaces.push_back(surface);
    config.onLayout = [&layoutLog](F32 x, F32 y, F32 width, F32 height) {
        layoutLog.record(x, y, width, height);
    };

    settle(ui, ctx, node, config, 4);
    REQUIRE(resizeLog.count() >= 1);
    const F32 heightBefore = std::get<3>(layoutLog.last());
    const F32 surfaceBefore = static_cast<F32>(resizeLog.entries.back().logicalSize.y);

    // Drag the corner down by 80 device pixels.
    dragCorner(ui, ctx, node, "e2e-block", ImVec2(0.0f, 80.0f));

    // Geometry propagates through the editor and the layout reallocates.
    settle(ui, ctx, node, config, 4);

    const F32 heightAfter = std::get<3>(layoutLog.last());
    const F32 surfaceAfter = static_cast<F32>(resizeLog.entries.back().logicalSize.y);

    REQUIRE(resizeLog.count() >= 2);
    REQUIRE(heightAfter == Catch::Approx(heightBefore + 80.0f).margin(2.0f));
    // All growth goes to the flexible surface, none to the fixed chrome.
    REQUIRE(surfaceAfter == Catch::Approx(surfaceBefore + 80.0f).margin(2.0f));
}

TEST_CASE("Dragging above the measured minimum clamps at the flexible minimum",
          "[core][sakura][flowgraph_node][drag]") {
    SakuraTest::HeadlessUi ui(1.0f, ImVec2(600.0f, 900.0f));
    const auto ctx = ui.sakura();

    LayoutLog layoutLog;
    FlowgraphNode node;
    auto config = baseConfigFor(baseBlock("clamp-block", "note"));
    config.block.configFields.push_back(
        {.id = "clamp-block:text", .format = "markdown", .encoded = "hello"});
    // Start above the flexible minimum so the memo reflects the floor.
    config.block.layout = FlowgraphNode::Layout{.x = 10.0f, .y = 10.0f, .width = 220.0f, .height = 300.0f};
    config.onLayout = [&layoutLog](F32 x, F32 y, F32 width, F32 height) {
        layoutLog.record(x, y, width, height);
    };

    settle(ui, ctx, node, config, 4);

    const auto* data = flowgraphNodeData("clamp-block");
    REQUIRE(data != nullptr);
    // Chrome-less note module: the floor is exactly the markdown minimum.
    REQUIRE(data->ResizeMinimumSize.y == Catch::Approx(120.0f).margin(0.5f));

    const auto before = flowgraphNodeDimensions("clamp-block");
    dragCorner(ui, ctx, node, "clamp-block", ImVec2(0.0f, -1000.0f));
    // The clamped height is reported through onLayout despite the stale layout.
    settle(ui, ctx, node, config, 3);
    REQUIRE(layoutLog.entries.size() >= 2);
    const F32 reportedHeight = std::get<3>(layoutLog.last());
    REQUIRE(reportedHeight == Catch::Approx(120.0f).margin(1.0f));
    REQUIRE(reportedHeight < std::get<3>(layoutLog.entries.front()));

    // Persist the reported height; the node must settle at the clamp.
    config.block.layout->height = reportedHeight;
    settle(ui, ctx, node, config, 4);
    const auto padding = ImNodes::GetStyle().NodePadding;
    REQUIRE(flowgraphNodeDimensions("clamp-block").y ==
            Catch::Approx(120.0f + 2.0f * padding.y).margin(1.5f));
    REQUIRE(flowgraphNodeDimensions("clamp-block").y < before.y);
}

TEST_CASE("Incomplete python block still seeds, floors, and resizes",
          "[core][sakura][flowgraph_node][incomplete]") {
    SakuraTest::HeadlessUi ui(1.0f, ImVec2(600.0f, 900.0f));
    const auto ctx = ui.sakura();
    const auto padding = ImNodes::GetStyle().NodePadding;

    // Chrome probe: same module, same state and diagnostic, no fields.
    FlowgraphNode chromeProbe;
    auto chromeConfig = baseConfigFor(baseBlock("inc-chrome", "radio"));
    chromeConfig.block.state = Block::State::Incomplete;
    chromeConfig.block.diagnostic = "Block 'inc-chrome' has unconnected input 'input0'.";
    settle(ui, ctx, chromeProbe, chromeConfig, 3);
    const F32 chrome = flowgraphNodeDimensions("inc-chrome").y - 2.0f * padding.y;
    REQUIRE(chrome > 0.0f);

    // A freshly created python block is Incomplete until its inputs are
    // wired: it must still seed, enforce its floor, and resize.
    FlowgraphNode node;
    auto config = baseConfigFor(baseBlock("inc-block", "radio"));
    config.block.configFields.push_back(
        {.id = "inc-block:code", .format = "python", .encoded = "pass"});
    config.block.state = Block::State::Incomplete;
    config.block.diagnostic = "Block 'inc-block' has unconnected input 'input0'.";
    settle(ui, ctx, node, config, 5);

    const auto* data = flowgraphNodeData("inc-block");
    REQUIRE(data != nullptr);
    REQUIRE((data->ResizeFlags & ImNodesNodeResizeFlags_Y) != 0);
    const F32 floor = data->ResizeMinimumSize.y;
    REQUIRE(floor >= chrome + 120.0f - 0.5f);
    REQUIRE(flowgraphNodeDimensions("inc-block").y - 2.0f * padding.y >= floor - 1.0f);

    // Height drags work while the block is unconnected.
    const auto before = flowgraphNodeDimensions("inc-block");
    dragCorner(ui, ctx, node, "inc-block", ImVec2(0.0f, 300.0f));
    settle(ui, ctx, node, config, 4);
    REQUIRE(flowgraphNodeDimensions("inc-block").y ==
            Catch::Approx(before.y + 300.0f).margin(2.0f));
    dragCorner(ui, ctx, node, "inc-block", ImVec2(0.0f, -600.0f));
    settle(ui, ctx, node, config, 4);
    REQUIRE(flowgraphNodeDimensions("inc-block").y == Catch::Approx(before.y).margin(2.0f));
    REQUIRE(flowgraphNodeDimensions("inc-block").y - 2.0f * padding.y >= floor - 1.0f);
}

TEST_CASE("Node height drag survives the async meta layout round-trip",
          "[core][sakura][flowgraph_node][layout]") {
    SakuraTest::HeadlessUi ui(1.0f, ImVec2(600.0f, 900.0f));
    const auto ctx = ui.sakura();

    FlowgraphNode node;
    auto config = baseConfigFor(baseBlock("meta-block", "radio"));
    config.block.configFields.push_back(
        {.id = "meta-block:code", .format = "python", .encoded = "pass"});

    // Saved meta from a previous session: height set, width 0 (older saves).
    FlowgraphNode::Layout meta{20.0f, 20.0f, 0.0f, 216.0f};

    // The presenter loop: mail drains into meta first (poll order), then
    // the config is rebuilt with the saved layout, then the node renders.
    // onLayout writes meta through a one-tick-deferred mail, exactly like
    // MailSetNodeMeta.
    std::optional<FlowgraphNode::Layout> mail;
    auto tick = [&]() {
        if (mail.has_value()) {
            meta = *mail;
            mail.reset();
        }
        auto frameConfig = config;
        frameConfig.block.layout = meta;
        frameConfig.onLayout = [&mail](F32 x, F32 y, F32 w, F32 h) {
            mail = FlowgraphNode::Layout{x, y, w, h};
        };
        node.update(frameConfig);
        renderFrame(ui, ctx, node);
    };

    for (int i = 0; i < 6; ++i) {
        tick();
    }
    const auto before = flowgraphNodeDimensions("meta-block");

    // Drag the grip +300 through the real interaction path while the
    // presenter keeps re-applying the saved meta every tick.
    const ImVec2 origin = ImNodes::GetNodeScreenSpacePos(imnodesId("meta-block"));
    const ImVec2 size = flowgraphNodeDimensions("meta-block");
    const ImVec2 grip{origin.x + size.x - 2.0f, origin.y + size.y - 2.0f};
    ui.setMouse(grip, true);
    tick();
    ui.setMouse(ImVec2(grip.x, grip.y + 300.0f), true);
    tick();
    ui.setMouse(ImVec2(grip.x, grip.y + 300.0f), false);
    tick();
    ui.setMouse(ImVec2(-100.0f, -100.0f), false);
    for (int i = 0; i < 6; ++i) {
        tick();
    }

    const auto after = flowgraphNodeDimensions("meta-block");
    REQUIRE(after.y == Catch::Approx(before.y + 300.0f).margin(2.0f));

    // And a shrink back down survives the same round-trip.
    const ImVec2 origin2 = ImNodes::GetNodeScreenSpacePos(imnodesId("meta-block"));
    const ImVec2 size2 = flowgraphNodeDimensions("meta-block");
    const ImVec2 grip2{origin2.x + size2.x - 2.0f, origin2.y + size2.y - 2.0f};
    ui.setMouse(grip2, true);
    tick();
    ui.setMouse(ImVec2(grip2.x, grip2.y - 300.0f), true);
    tick();
    ui.setMouse(ImVec2(grip2.x, grip2.y - 300.0f), false);
    tick();
    ui.setMouse(ImVec2(-100.0f, -100.0f), false);
    for (int i = 0; i < 6; ++i) {
        tick();
    }
    REQUIRE(flowgraphNodeDimensions("meta-block").y ==
            Catch::Approx(before.y).margin(2.0f));
}

TEST_CASE("Connecting a block keeps its NodeSize width when the surface first appears",
          "[core][sakura][flowgraph_node]") {
    SakuraTest::HeadlessUi ui(1.0f, ImVec2(600.0f, 900.0f));
    const auto ctx = ui.sakura();

    LayoutLog layoutLog;
    FlowgraphNode node;
    auto config = baseConfigFor(baseBlock("xl-block", "radio"));
    config.block.state = Block::State::Incomplete;
    config.block.nodeSize = Block::NodeSize::XL;
    config.onLayout = [&layoutLog](F32 x, F32 y, F32 width, F32 height) {
        layoutLog.record(x, y, width, height);
    };

    // Created but not connected: XL width, auto height.
    settle(ui, ctx, node, config, 3);
    REQUIRE(layoutLog.entries.size() >= 1);
    REQUIRE(std::get<2>(layoutLog.last()) == Catch::Approx(460.0f));
    REQUIRE(std::get<3>(layoutLog.last()) == Catch::Approx(0.0f));

    // Connected: the surface manifests with the default 256x256 meta.
    SakuraTest::ResizeLog resizeLog;
    FlowgraphNode::Surface surface = attachedSurface("xl-block:surface:out");
    surface.height = 256.0f;
    surface.onAttachedSize = [&resizeLog](const Sakura::SurfaceResize& resize) {
        resizeLog.record(resize);
    };
    config.block.state = Block::State::Created;
    config.block.surfaces.push_back(surface);
    settle(ui, ctx, node, config, 4);

    // The surface receives its full meta height on top of the chrome;
    // the NodeSize width survives.
    REQUIRE(resizeLog.entries.back().logicalSize.y == Catch::Approx(256.0f).margin(1.0f));
    REQUIRE(std::get<3>(layoutLog.last()) >= 256.0f);
    REQUIRE(std::get<2>(layoutLog.last()) == Catch::Approx(460.0f));
}

TEST_CASE("Resize floor converges to the flexible minimum after the node grows",
          "[core][sakura][flowgraph_node]") {
    SakuraTest::HeadlessUi ui(1.0f, ImVec2(600.0f, 900.0f));
    const auto ctx = ui.sakura();

    FlowgraphNode node;
    auto config = baseConfigFor(baseBlock("floor-block", "note"));
    config.block.configFields.push_back(
        {.id = "floor-block:text", .format = "markdown", .encoded = "hello"});

    // Warm-up caps the floor at the current height.
    settle(ui, ctx, node, config, 4);
    const auto* data = flowgraphNodeData("floor-block");
    REQUIRE(data != nullptr);
    REQUIRE(data->ResizeMinimumSize.y <= 120.0f);

    // Growth raises the floor to the markdown minimum (120).
    config.block.layout = FlowgraphNode::Layout{.x = 10.0f, .y = 10.0f, .width = 220.0f, .height = 600.0f};
    settle(ui, ctx, node, config, 5);
    data = flowgraphNodeData("floor-block");
    REQUIRE(data != nullptr);
    REQUIRE(data->ResizeMinimumSize.y == Catch::Approx(120.0f).margin(0.5f));
}

TEST_CASE("Detaching all surfaces reports auto height and re-attach honors the surface meta",
          "[core][sakura][flowgraph_node][detachment]") {
    SakuraTest::HeadlessUi ui(1.0f, ImVec2(600.0f, 900.0f));
    const auto ctx = ui.sakura();

    LayoutLog layoutLog;
    FlowgraphNode node;
    auto config = baseConfigFor(baseBlock("cycle-block", "radio"));
    FlowgraphNode::Surface surface = attachedSurface("cycle-block:surface:out");
    config.block.surfaces.push_back(surface);
    config.onLayout = [&layoutLog](F32 x, F32 y, F32 width, F32 height) {
        layoutLog.record(x, y, width, height);
    };

    settle(ui, ctx, node, config, 4);
    REQUIRE(std::get<3>(layoutLog.last()) >= 512.0f);

    // Detaching collapses to auto height; the report follows.
    config.block.surfaces[0].detached = true;
    settle(ui, ctx, node, config, 4);
    REQUIRE(std::get<3>(layoutLog.last()) == Catch::Approx(0.0f));
    const F32 cycleChrome = flowgraphNodeDimensions("cycle-block").y -
                            2.0f * ImNodes::GetStyle().NodePadding.y;
    REQUIRE(flowgraphNodeDimensions("cycle-block").y < 100.0f);

    // Re-attaching restores the node as chrome plus the surface meta, so
    // the surface receives its full persisted height.
    config.block.surfaces[0].detached = false;
    config.block.surfaces[0].height = 256.0f;
    settle(ui, ctx, node, config, 5);
    REQUIRE(std::get<3>(layoutLog.last()) ==
            Catch::Approx(256.0f + cycleChrome).margin(1.0f));
}

TEST_CASE("Detach/reattach cycles preserve the persisted surface height",
          "[core][sakura][flowgraph_node][detachment]") {
    SakuraTest::HeadlessUi ui(1.0f, ImVec2(600.0f, 900.0f));
    const auto ctx = ui.sakura();

    SakuraTest::ResizeLog resizeLog;
    FlowgraphNode node;
    auto config = baseConfigFor(baseBlock("persist-block", "radio"));
    FlowgraphNode::Surface surface = attachedSurface("persist-block:surface:out");
    surface.height = 400.0f;
    surface.onAttachedSize = [&resizeLog](const Sakura::SurfaceResize& resize) {
        resizeLog.record(resize);
    };
    config.block.surfaces.push_back(surface);

    // First attach: whatever the surface emits is what MailResizeSurface
    // persists into the meta.
    settle(ui, ctx, node, config, 5);
    REQUIRE(resizeLog.count() >= 1);
    const F32 persisted = static_cast<F32>(resizeLog.entries.back().logicalSize.y);
    REQUIRE(persisted > 150.0f);

    // Each detach/reattach cycle must hand the surface exactly the
    // persisted height back: the meta stores the surface's own height,
    // not the node's total height.
    for (int cycle = 0; cycle < 2; ++cycle) {
        config.block.surfaces[0].detached = true;
        settle(ui, ctx, node, config, 4);
        config.block.surfaces[0].height = persisted;
        config.block.surfaces[0].detached = false;
        resizeLog.entries.clear();
        settle(ui, ctx, node, config, 5);
        REQUIRE(resizeLog.count() >= 1);
        REQUIRE(static_cast<F32>(resizeLog.entries.back().logicalSize.y) ==
                Catch::Approx(persisted).margin(1.0f));
    }
}

TEST_CASE("Below-minimum allocations never reach surface callbacks",
          "[core][sakura][flowgraph_node][allocation]") {
    SakuraTest::HeadlessUi ui(1.0f, ImVec2(600.0f, 900.0f));
    const auto ctx = ui.sakura();

    SakuraTest::ResizeLog resizeLog;
    FlowgraphNode node;
    auto config = baseConfigFor(baseBlock("transient-block", "radio"));
    auto surface = attachedSurface("transient-block:surface:out");
    surface.height = 400.0f;
    surface.onAttachedSize = [&resizeLog](const Sakura::SurfaceResize& resize) {
        resizeLog.record(resize);
    };
    config.block.surfaces.push_back(surface);
    // A stale saved layout below the node floor: the correction must fire,
    // but no frame may allocate — or emit — a below-minimum surface size.
    config.block.layout = FlowgraphNode::Layout{.x = 10.0f,
                                                .y = 10.0f,
                                                .width = 357.0f,
                                                .height = 100.0f};

    for (int i = 0; i < 6; ++i) {
        node.update(config);
        renderFrame(ui, ctx, node);
        for (const auto& entry : resizeLog.entries) {
            REQUIRE(entry.logicalSize.y >= FlowgraphNode::SurfaceHeightSpec().minimum - 0.5f);
        }
    }

    // The node still settles at its floor.
    const auto padding = ImNodes::GetStyle().NodePadding;
    const auto* data = flowgraphNodeData("transient-block");
    REQUIRE(data != nullptr);
    REQUIRE(flowgraphNodeDimensions("transient-block").y - 2.0f * padding.y >=
            data->ResizeMinimumSize.y - 1.0f);
    REQUIRE(resizeLog.entries.back().logicalSize.y >= FlowgraphNode::SurfaceHeightSpec().minimum);
}

TEST_CASE("Cold surface seed restores the full persisted height once chrome is known",
          "[core][sakura][flowgraph_node][detachment]") {
    SakuraTest::HeadlessUi ui(1.0f, ImVec2(600.0f, 900.0f));
    const auto ctx = ui.sakura();

    // No node meta height (absent, or explicitly zero): the surface meta
    // alone must survive the seed. Every emitted height — not just the
    // final one — must equal the persisted height: intermediate values
    // reach MailResizeSurface and would persist a shrunk surface.
    const auto runCold = [&](const std::string& blockId, const bool zeroHeight) {
        SakuraTest::ResizeLog resizeLog;
        LayoutLog layoutLog;
        FlowgraphNode node;
        auto config = baseConfigFor(baseBlock(blockId, "radio"));
        auto surface = attachedSurface(blockId + ":surface:out");
        surface.height = 400.0f;
        surface.onAttachedSize = [&resizeLog](const Sakura::SurfaceResize& resize) {
            resizeLog.record(resize);
        };
        config.block.surfaces.push_back(surface);
        if (zeroHeight) {
            config.block.layout = FlowgraphNode::Layout{.x = 10.0f,
                                                        .y = 10.0f,
                                                        .width = 357.0f,
                                                        .height = 0.0f};
        }
        config.onLayout = [&layoutLog](F32 x, F32 y, F32 width, F32 height) {
            layoutLog.record(x, y, width, height);
        };

        for (int i = 0; i < 6; ++i) {
            node.update(config);
            renderFrame(ui, ctx, node);
            for (const auto& entry : resizeLog.entries) {
                REQUIRE(entry.logicalSize.y == Catch::Approx(400.0f).margin(1.0f));
            }
        }
        REQUIRE(std::get<3>(layoutLog.last()) >= 400.0f);
    };

    runCold("cold-block", false);
    runCold("cold-block-zero", true);
}

TEST_CASE("Cold restore preserves every surface alongside flexible fields",
          "[core][sakura][flowgraph_node][persistence]") {
    SakuraTest::HeadlessUi ui(1.0f, ImVec2(700.0f, 1400.0f));
    const auto ctx = ui.sakura();

    SakuraTest::ResizeLog logA;
    SakuraTest::ResizeLog logB;
    LayoutLog layoutLog;
    FlowgraphNode node;
    auto config = baseConfigFor(baseBlock("multi-flex-block", "radio"));
    config.block.configFields.push_back(
        {.id = "multi-flex-block:text", .format = "markdown", .encoded = "hello"});

    auto surfaceA = attachedSurface("multi-flex-block:surface:a");
    surfaceA.height = 400.0f;
    surfaceA.onAttachedSize = [&logA](const Sakura::SurfaceResize& resize) {
        logA.record(resize);
    };
    auto surfaceB = attachedSurface("multi-flex-block:surface:b");
    surfaceB.height = 260.0f;
    surfaceB.onAttachedSize = [&logB](const Sakura::SurfaceResize& resize) {
        logB.record(resize);
    };
    config.block.surfaces = {surfaceA, surfaceB};
    config.onLayout = [&layoutLog](F32 x, F32 y, F32 width, F32 height) {
        layoutLog.record(x, y, width, height);
    };

    settle(ui, ctx, node, config, 7);

    REQUIRE(logA.count() >= 1);
    REQUIRE(logB.count() >= 1);
    for (const auto& entry : logA.entries) {
        REQUIRE(entry.logicalSize.y == 400);
    }
    for (const auto& entry : logB.entries) {
        REQUIRE(entry.logicalSize.y == 260);
    }
    REQUIRE(std::get<3>(layoutLog.last()) >= 400.0f + 260.0f + 120.0f);
}

TEST_CASE("Note blocks create at their flexible minimum and clicking the knob does not snap",
          "[core][sakura][flowgraph_node][creation]") {
    SakuraTest::HeadlessUi ui(1.0f, ImVec2(600.0f, 900.0f));
    const auto ctx = ui.sakura();

    LayoutLog layoutLog;
    FlowgraphNode node;
    auto config = baseConfigFor(baseBlock("note-block", "note"));
    config.block.configFields.push_back(
        {.id = "note-block:text", .format = "markdown", .encoded = "hello"});
    config.block.state = Block::State::Creating;
    config.onLayout = [&layoutLog](F32 x, F32 y, F32 width, F32 height) {
        layoutLog.record(x, y, width, height);
    };

    // Loading phase: creation floor applies.
    settle(ui, ctx, node, config, 2);
    REQUIRE(std::get<3>(layoutLog.last()) <= 100.0f);

    // Completed: the note settles at its markdown minimum (120).
    config.block.state = Block::State::Created;
    settle(ui, ctx, node, config, 4);
    REQUIRE(std::get<3>(layoutLog.last()) == Catch::Approx(120.0f).margin(1.0f));

    const auto* data = flowgraphNodeData("note-block");
    REQUIRE(data != nullptr);
    REQUIRE(data->ResizeMinimumSize.y == Catch::Approx(120.0f).margin(0.5f));

    // Clicking the knob without dragging must not change the size.
    const auto before = flowgraphNodeDimensions("note-block");
    ui.setMouse(ImVec2(ImNodes::GetNodeScreenSpacePos(imnodesId("note-block")).x +
                           before.x - 2.0f,
                       ImNodes::GetNodeScreenSpacePos(imnodesId("note-block")).y +
                           before.y - 2.0f),
                true);
    renderFrame(ui, ctx, node);
    ui.setMouse(ImVec2(-100.0f, -100.0f), false);
    renderFrame(ui, ctx, node);
    const auto after = flowgraphNodeDimensions("note-block");
    REQUIRE(after.x == Catch::Approx(before.x).margin(1.0f));
    REQUIRE(after.y == Catch::Approx(before.y).margin(1.0f));
}

TEST_CASE("Detaching every surface keeps the node height while flexible fields remain",
          "[core][sakura][flowgraph_node][detachment]") {
    SakuraTest::HeadlessUi ui(1.0f, ImVec2(600.0f, 900.0f));
    const auto ctx = ui.sakura();

    LayoutLog layoutLog;
    SakuraTest::ResizeLog resizeLog;
    FlowgraphNode node;
    auto config = baseConfigFor(baseBlock("keep-block", "radio"));
    config.block.configFields.push_back(
        {.id = "keep-block:text", .format = "markdown", .encoded = "hello"});
    auto surface = attachedSurface("keep-block:surface:out");
    surface.onAttachedSize = [&resizeLog](const Sakura::SurfaceResize& resize) {
        resizeLog.record(resize);
    };
    config.block.surfaces.push_back(surface);
    config.onLayout = [&layoutLog](F32 x, F32 y, F32 width, F32 height) {
        layoutLog.record(x, y, width, height);
    };

    settle(ui, ctx, node, config, 4);
    REQUIRE(std::get<3>(layoutLog.last()) >= 512.0f);
    REQUIRE(resizeLog.count() >= 1);
    const F32 keepHeight = std::get<3>(layoutLog.last());

    // The editor absorbs the space the surface vacated.
    config.block.surfaces[0].detached = true;
    settle(ui, ctx, node, config, 4);
    REQUIRE(std::get<3>(layoutLog.last()) == Catch::Approx(keepHeight).margin(1.0f));
    REQUIRE(resizeLog.count() >= 1);
    REQUIRE(flowgraphNodeData("keep-block")->ResizeFlags == ImNodesNodeResizeFlags_XY);
}

TEST_CASE("Mixed attached and detached surfaces track only the attached one",
          "[core][sakura][flowgraph_node][detachment]") {
    SakuraTest::HeadlessUi ui(1.0f, ImVec2(600.0f, 900.0f));
    const auto ctx = ui.sakura();

    SakuraTest::ResizeLog logA;
    SakuraTest::ResizeLog logB;
    LayoutLog layoutLog;
    FlowgraphNode node;
    auto config = baseConfigFor(baseBlock("mixed-block", "radio"));
    auto surfaceA = attachedSurface("mixed-block:surface:a");
    surfaceA.onAttachedSize = [&logA](const Sakura::SurfaceResize& resize) {
        logA.record(resize);
    };
    auto surfaceB = attachedSurface("mixed-block:surface:b");
    surfaceB.height = 256.0f;
    surfaceB.detached = true;
    surfaceB.onAttachedSize = [&logB](const Sakura::SurfaceResize& resize) {
        logB.record(resize);
    };
    config.block.surfaces.push_back(surfaceA);
    config.block.surfaces.push_back(surfaceB);
    config.onLayout = [&layoutLog](F32 x, F32 y, F32 width, F32 height) {
        layoutLog.record(x, y, width, height);
    };

    settle(ui, ctx, node, config, 4);
    REQUIRE(flowgraphNodeData("mixed-block")->ResizeFlags == ImNodesNodeResizeFlags_XY);
    REQUIRE(std::get<3>(layoutLog.last()) >= 512.0f);
    REQUIRE(logA.count() >= 1);
    REQUIRE(logB.count() == 0);

    // Height persists across the swap; B fills the node.
    config.block.surfaces[0].detached = true;
    config.block.surfaces[1].detached = false;
    settle(ui, ctx, node, config, 5);
    REQUIRE(std::get<3>(layoutLog.last()) >= 512.0f);
    REQUIRE(logA.count() >= 1);
    REQUIRE(logB.count() >= 1);
    REQUIRE(logB.entries.back().logicalSize.y == Catch::Approx(512.0f).margin(1.0f));
}

TEST_CASE("A saved layout wins over the surface meta when both are present",
          "[core][sakura][flowgraph_node][persistence]") {
    SakuraTest::HeadlessUi ui(1.0f, ImVec2(600.0f, 900.0f));
    const auto ctx = ui.sakura();

    LayoutLog layoutLog;
    SakuraTest::ResizeLog resizeLog;
    FlowgraphNode node;
    auto config = baseConfigFor(baseBlock("saved-block", "radio"));
    auto surface = attachedSurface("saved-block:surface:out");
    surface.onAttachedSize = [&resizeLog](const Sakura::SurfaceResize& resize) {
        resizeLog.record(resize);
    };
    config.block.surfaces.push_back(surface);
    config.block.layout = FlowgraphNode::Layout{.x = 10.0f, .y = 10.0f, .width = 220.0f, .height = 400.0f};
    config.onLayout = [&layoutLog](F32 x, F32 y, F32 width, F32 height) {
        layoutLog.record(x, y, width, height);
    };

    settle(ui, ctx, node, config, 4);

    // Loads at the saved size; the surface fills the rest.
    REQUIRE(std::get<3>(layoutLog.last()) == Catch::Approx(400.0f).margin(1.0f));
    REQUIRE(resizeLog.count() == 1);
    REQUIRE(resizeLog.entries[0].logicalSize.y > 300);
    REQUIRE(resizeLog.entries[0].logicalSize.y < 400);
}

TEST_CASE("Metrics consume fixed height and the surface allocation follows",
          "[core][sakura][flowgraph_node][spacing]") {
    SakuraTest::HeadlessUi ui(1.0f, ImVec2(600.0f, 900.0f));
    const auto ctx = ui.sakura();

    SakuraTest::ResizeLog resizeLog;
    FlowgraphNode node;
    auto config = baseConfigFor(baseBlock("metric-block", "radio"));
    auto surface = attachedSurface("metric-block:surface:out");
    surface.onAttachedSize = [&resizeLog](const Sakura::SurfaceResize& resize) {
        resizeLog.record(resize);
    };
    config.block.surfaces.push_back(surface);

    settle(ui, ctx, node, config, 4);
    REQUIRE(resizeLog.count() >= 1);
    const F32 withoutMetrics = static_cast<F32>(resizeLog.entries.back().logicalSize.y);

    // One metric row (plus its spacing row) shifts into the fixed chrome.
    config.block.metrics.push_back({.id = "metric-block:m1",
                                    .label = "Rate",
                                    .format = "label"});
    settle(ui, ctx, node, config, 5);
    REQUIRE(resizeLog.count() >= 2);
    const F32 withOneMetric = static_cast<F32>(resizeLog.entries.back().logicalSize.y);
    REQUIRE(withOneMetric < withoutMetrics - 5.0f);

    // Removing the metrics restores the original allocation.
    config.block.metrics.clear();
    settle(ui, ctx, node, config, 5);
    REQUIRE(resizeLog.count() >= 3);
    REQUIRE(static_cast<F32>(resizeLog.entries.back().logicalSize.y) ==
            Catch::Approx(withoutMetrics).margin(1.0f));
}

TEST_CASE("Incomplete blocks seed their minimum height on completion",
          "[core][sakura][flowgraph_node][creation]") {
    SakuraTest::HeadlessUi ui(1.0f, ImVec2(600.0f, 900.0f));
    const auto ctx = ui.sakura();

    LayoutLog layoutLog;
    FlowgraphNode node;
    auto config = baseConfigFor(baseBlock("incomplete-block", "note"));
    config.block.configFields.push_back(
        {.id = "incomplete-block:text", .format = "markdown", .encoded = "hello"});
    config.block.state = Block::State::Incomplete;
    config.onLayout = [&layoutLog](F32 x, F32 y, F32 width, F32 height) {
        layoutLog.record(x, y, width, height);
    };

    // Incomplete is live: the node already reports its flexible minimum
    // so it can be positioned and resized while unconnected.
    settle(ui, ctx, node, config, 3);
    REQUIRE(std::get<3>(layoutLog.last()) == Catch::Approx(120.0f).margin(1.0f));

    // The input resolves: the block completes at its flexible minimum.
    config.block.state = Block::State::Created;
    settle(ui, ctx, node, config, 4);
    REQUIRE(std::get<3>(layoutLog.last()) == Catch::Approx(120.0f).margin(1.0f));
    REQUIRE(flowgraphNodeData("incomplete-block")->ResizeMinimumSize.y == Catch::Approx(120.0f).margin(0.5f));
}

TEST_CASE("Reconfiguring flexible fields reflows the node height",
          "[core][sakura][flowgraph_node]") {
    SakuraTest::HeadlessUi ui(1.0f, ImVec2(600.0f, 900.0f));
    const auto ctx = ui.sakura();

    LayoutLog layoutLog;
    FlowgraphNode node;
    auto config = baseConfigFor(baseBlock("reconfig-block", "radio"));
    config.onLayout = [&layoutLog](F32 x, F32 y, F32 width, F32 height) {
        layoutLog.record(x, y, width, height);
    };

    settle(ui, ctx, node, config, 3);
    REQUIRE(std::get<3>(layoutLog.last()) == Catch::Approx(0.0f));
    const F32 reconfigChrome = flowgraphNodeDimensions("reconfig-block").y -
                               2.0f * ImNodes::GetStyle().NodePadding.y;

    // A markdown field appears: the node seeds its flexible minimum.
    config.block.configFields.push_back(
        {.id = "reconfig-block:text", .format = "markdown", .encoded = "hello"});
    settle(ui, ctx, node, config, 4);
    REQUIRE(std::get<3>(layoutLog.last()) >= reconfigChrome + 120.0f - 1.0f);

    // The field disappears: the node returns to auto height.
    config.block.configFields.clear();
    settle(ui, ctx, node, config, 4);
    REQUIRE(std::get<3>(layoutLog.last()) == Catch::Approx(0.0f));
}

TEST_CASE("Shallow attached surfaces seed the node at the surface minimum",
          "[core][sakura][flowgraph_node][creation]") {
    SakuraTest::HeadlessUi ui(1.0f, ImVec2(600.0f, 900.0f));
    const auto ctx = ui.sakura();

    LayoutLog layoutLog;
    SakuraTest::ResizeLog resizeLog;
    FlowgraphNode node;
    auto config = baseConfigFor(baseBlock("shallow-block", "radio"));
    auto surface = attachedSurface("shallow-block:surface:out");
    surface.height = 100.0f;
    surface.onAttachedSize = [&resizeLog](const Sakura::SurfaceResize& resize) {
        resizeLog.record(resize);
    };
    config.block.surfaces.push_back(surface);
    config.onLayout = [&layoutLog](F32 x, F32 y, F32 width, F32 height) {
        layoutLog.record(x, y, width, height);
    };

    // A surface shorter than the 150-unit minimum must not drag the node
    // below it: the node grows so the surface receives its full minimum.
    settle(ui, ctx, node, config, 4);
    REQUIRE(std::get<3>(layoutLog.last()) >= 150.0f);
    REQUIRE(flowgraphNodeData("shallow-block")->ResizeMinimumSize.y >= 150.0f);
    // The surface may report transient allocations during warm-up, but its
    // settled size honors the 150 minimum.
    REQUIRE(resizeLog.count() >= 1);
    REQUIRE(resizeLog.entries.back().logicalSize.y >= 150);
}

TEST_CASE("Python editor receives its declared minimum height",
          "[core][sakura][flowgraph_node][creation]") {
    SakuraTest::HeadlessUi ui(1.0f, ImVec2(600.0f, 900.0f));
    const auto ctx = ui.sakura();

    // Measure the fixed chrome: the same module without fields renders
    // title + subtitle only (auto-height), so its content IS the chrome.
    FlowgraphNode chromeProbe;
    auto chromeConfig = baseConfigFor(baseBlock("py-chrome", "radio"));
    settle(ui, ctx, chromeProbe, chromeConfig, 3);
    const auto padding = ImNodes::GetStyle().NodePadding;
    const F32 chrome = flowgraphNodeDimensions("py-chrome").y - 2.0f * padding.y;
    REQUIRE(chrome > 0.0f);

    // The python editor declares a 120-unit minimum: the node floor must
    // cover chrome + 120, and the node must rest at that floor so the
    // editor fills its space.
    FlowgraphNode node;
    auto config = baseConfigFor(baseBlock("py-block", "radio"));
    config.block.configFields.push_back(
        {.id = "py-block:code", .format = "python", .encoded = "pass"});
    settle(ui, ctx, node, config, 5);

    const F32 floor = flowgraphNodeData("py-block")->ResizeMinimumSize.y;
    REQUIRE(floor >= chrome + 120.0f - 0.5f);
    REQUIRE(flowgraphNodeDimensions("py-block").y - 2.0f * padding.y >= floor - 1.0f);

    // Height drags work in both directions down to the honest floor.
    const auto before = flowgraphNodeDimensions("py-block");
    dragCorner(ui, ctx, node, "py-block", ImVec2(0.0f, -600.0f));
    settle(ui, ctx, node, config, 4);
    const auto shrunk = flowgraphNodeDimensions("py-block");
    REQUIRE(shrunk.y == Catch::Approx(before.y).margin(1.0f));
    dragCorner(ui, ctx, node, "py-block", ImVec2(0.0f, 300.0f));
    settle(ui, ctx, node, config, 4);
    REQUIRE(flowgraphNodeDimensions("py-block").y == Catch::Approx(before.y + 300.0f).margin(2.0f));
}

TEST_CASE("Creating nodes render at the loading height floor",
          "[core][sakura][flowgraph_node][lifecycle]") {
    SakuraTest::HeadlessUi ui(1.0f, ImVec2(600.0f, 900.0f));
    const auto ctx = ui.sakura();

    FlowgraphNode node;
    auto config = baseConfigFor(baseBlock("creating-block", "radio"));
    config.block.state = Block::State::Creating;
    config.block.layout = FlowgraphNode::Layout{.x = 20.0f, .y = 20.0f, .width = 200.0f, .height = 0.0f};

    node.update(config);
    renderFrame(ui, ctx, node);

    // Loading nodes never collapse below the 96-unit floor.
    REQUIRE(flowgraphNodeDimensions("creating-block").y >= 96.0f);
}

TEST_CASE("Simple config grids reflow during node resizing and collapse without reserving height",
          "[core][sakura][flowgraph_node][field-grid][collapse]") {
    SakuraTest::HeadlessUi ui(1.0f, ImVec2(900.0f, 900.0f));
    const auto ctx = ui.sakura();

    FlowgraphNode node;
    auto config = baseConfigFor(baseBlock("grid-block", "radio"));
    for (U64 i = 0; i < 4; ++i) {
        config.block.configFields.push_back({
            .id = "grid-block:field:" + std::to_string(i),
            .label = "Value",
            .format = "range:0:1",
            .encoded = "0.5",
        });
    }
    settle(ui, ctx, node, config, 4);
    const auto narrow = flowgraphNodeDimensions(config.id);
    REQUIRE(flowgraphNodeData(config.id)->ResizeFlags == ImNodesNodeResizeFlags_X);

    dragCorner(ui, ctx, node, config.id, ImVec2(100.0f, 0.0f));
    settle(ui, ctx, node, config, 5);
    const auto wide = flowgraphNodeDimensions(config.id);
    REQUIRE(wide.x == Catch::Approx(narrow.x + 100.0f).margin(1.0f));
    REQUIRE(wide.y < narrow.y - 20.0f);

    config.block.configCollapsed = true;
    settle(ui, ctx, node, config, 4);
    const auto collapsed = flowgraphNodeDimensions(config.id);
    REQUIRE(collapsed.x == Catch::Approx(wide.x).margin(1.0f));
    REQUIRE(collapsed.y < wide.y - 20.0f);

    config.block.configCollapsed = false;
    settle(ui, ctx, node, config, 4);
    REQUIRE(flowgraphNodeDimensions(config.id).y == Catch::Approx(wide.y).margin(1.0f));
    dragCorner(ui, ctx, node, config.id, ImVec2(-100.0f, 0.0f));
    settle(ui, ctx, node, config, 5);
    REQUIRE(flowgraphNodeDimensions(config.id).x == Catch::Approx(narrow.x).margin(1.0f));
    REQUIRE(flowgraphNodeDimensions(config.id).y == Catch::Approx(narrow.y).margin(1.0f));
}

TEST_CASE("Collapsed flexible config fields stop reserving node height and resize axes",
          "[core][sakura][flowgraph_node][collapse][allocation]") {
    SakuraTest::HeadlessUi ui(1.0f, ImVec2(600.0f, 900.0f));
    const auto ctx = ui.sakura();

    LayoutLog layoutLog;
    FlowgraphNode node;
    auto config = baseConfigFor(baseBlock("collapse-editor", "radio"));
    config.block.configFields.push_back(
        {.id = "collapse-editor:code", .format = "python", .encoded = "pass"});
    config.onLayout = [&layoutLog](F32 x, F32 y, F32 width, F32 height) {
        layoutLog.record(x, y, width, height);
    };
    settle(ui, ctx, node, config, 5);

    SECTION("Default editor height") {}
    SECTION("User-resized editor height") {
        dragCorner(ui, ctx, node, config.id, ImVec2(0.0f, 180.0f));
        settle(ui, ctx, node, config, 4);
    }
    SECTION("User-resized editor with a detached surface") {
        auto surface = attachedSurface("collapse-editor:surface");
        surface.detached = true;
        config.block.surfaces.push_back(surface);
        settle(ui, ctx, node, config, 4);
        dragCorner(ui, ctx, node, config.id, ImVec2(0.0f, 180.0f));
        settle(ui, ctx, node, config, 4);
    }

    const auto expanded = flowgraphNodeDimensions(config.id);
    const F32 expandedHeight = std::get<3>(layoutLog.last());
    REQUIRE(flowgraphNodeData(config.id)->ResizeFlags == ImNodesNodeResizeFlags_XY);

    config.block.configCollapsed = true;
    settle(ui, ctx, node, config, 4);
    REQUIRE(flowgraphNodeData(config.id)->ResizeFlags == ImNodesNodeResizeFlags_X);
    REQUIRE(std::get<3>(layoutLog.last()) == Catch::Approx(expandedHeight).margin(1.0f));
    REQUIRE(flowgraphNodeDimensions(config.id).y < expanded.y - 100.0f);

    config.block.configCollapsed = false;
    settle(ui, ctx, node, config, 5);
    REQUIRE(flowgraphNodeData(config.id)->ResizeFlags == ImNodesNodeResizeFlags_XY);
    REQUIRE(flowgraphNodeDimensions(config.id).y == Catch::Approx(expanded.y).margin(1.0f));
    dragCorner(ui, ctx, node, config.id, ImVec2(0.0f, 100.0f));
    settle(ui, ctx, node, config, 4);
    REQUIRE(flowgraphNodeDimensions(config.id).y == Catch::Approx(expanded.y + 100.0f).margin(2.0f));
}

TEST_CASE("Collapsed editor height survives layout feedback and view recreation",
          "[core][sakura][flowgraph_node][collapse][persistence]") {
    SakuraTest::HeadlessUi ui(1.0f, ImVec2(900.0f, 900.0f));
    const auto ctx = ui.sakura();

    auto config = baseConfigFor(baseBlock("collapse-meta", "radio"));
    config.block.configFields.push_back(
        {.id = "collapse-meta:code", .format = "python", .encoded = "pass"});
    FlowgraphNode::Layout meta{20.0f, 20.0f, 220.0f, 400.0f};
    std::optional<FlowgraphNode::Layout> mail;
    auto tick = [&](FlowgraphNode& node, U64 frames) {
        for (U64 i = 0; i < frames; ++i) {
            if (mail.has_value()) {
                meta = *mail;
                mail.reset();
            }
            auto frameConfig = config;
            frameConfig.block.layout = meta;
            frameConfig.onLayout = [&mail](F32 x, F32 y, F32 width, F32 height) {
                mail = FlowgraphNode::Layout{x, y, width, height};
            };
            node.update(frameConfig);
            renderFrame(ui, ctx, node);
        }
    };

    F32 expandedHeight = 0.0f;
    {
        FlowgraphNode node;
        tick(node, 5);
        expandedHeight = flowgraphNodeDimensions(config.id).y;

        for (U64 cycle = 0; cycle < 3; ++cycle) {
            config.block.configCollapsed = true;
            tick(node, 4);
            const auto collapsed = flowgraphNodeDimensions(config.id);
            REQUIRE(collapsed.y < expandedHeight - 100.0f);
            REQUIRE(flowgraphNodeData(config.id)->ResizeFlags == ImNodesNodeResizeFlags_X);
            REQUIRE(meta.height == Catch::Approx(400.0f).margin(1.0f));

            // Width changes must persist without replacing the hidden height.
            dragCorner(ui, ctx, node, config.id, ImVec2(60.0f, 0.0f));
            tick(node, 4);
            REQUIRE(flowgraphNodeDimensions(config.id).x ==
                    Catch::Approx(collapsed.x + 60.0f).margin(1.0f));
            REQUIRE(meta.width == Catch::Approx(220.0f + (cycle + 1) * 60.0f).margin(1.0f));
            REQUIRE(meta.height == Catch::Approx(400.0f).margin(1.0f));

            config.block.configCollapsed = false;
            tick(node, 5);
            REQUIRE(flowgraphNodeDimensions(config.id).y ==
                    Catch::Approx(expandedHeight).margin(1.0f));
        }

        config.block.configCollapsed = true;
        tick(node, 4);
    }

    // A new view must retain the expanded height even when opened collapsed.
    FlowgraphNode restored;
    tick(restored, 5);
    REQUIRE(flowgraphNodeDimensions(config.id).y < expandedHeight - 100.0f);
    REQUIRE(meta.height == Catch::Approx(400.0f).margin(1.0f));
    config.block.configCollapsed = false;
    tick(restored, 5);
    REQUIRE(flowgraphNodeDimensions(config.id).y == Catch::Approx(expandedHeight).margin(1.0f));
    REQUIRE(flowgraphNodeData(config.id)->ResizeFlags == ImNodesNodeResizeFlags_XY);
}

TEST_CASE("Removing a collapsed editor releases its saved height",
          "[core][sakura][flowgraph_node][collapse][persistence]") {
    SakuraTest::HeadlessUi ui(1.0f, ImVec2(600.0f, 900.0f));
    const auto ctx = ui.sakura();

    LayoutLog layoutLog;
    FlowgraphNode node;
    auto config = baseConfigFor(baseBlock("removed-editor", "radio"));
    config.block.configFields.push_back(
        {.id = "removed-editor:code", .format = "python", .encoded = "pass"});
    config.block.layout = FlowgraphNode::Layout{20.0f, 20.0f, 220.0f, 400.0f};
    config.block.configCollapsed = true;
    config.onLayout = [&layoutLog, &config](F32 x, F32 y, F32 width, F32 height) {
        layoutLog.record(x, y, width, height);
        config.block.layout = FlowgraphNode::Layout{x, y, width, height};
    };
    settle(ui, ctx, node, config, 4);
    REQUIRE(std::get<3>(layoutLog.last()) == Catch::Approx(400.0f).margin(1.0f));

    config.block.configFields.clear();
    SECTION("Field removed") {}
    SECTION("Field replaced with a fixed-height control") {
        config.block.configFields.push_back(
            {.id = "removed-editor:value", .format = "range:0:1", .encoded = "0.5"});
    }
    settle(ui, ctx, node, config, 4);
    REQUIRE(flowgraphNodeData(config.id)->ResizeFlags == ImNodesNodeResizeFlags_X);
    REQUIRE(flowgraphNodeDimensions(config.id).y < 100.0f);
    REQUIRE(std::get<3>(layoutLog.last()) == 0.0f);

    // A later editor must seed its minimum, not inherit the removed one.
    config.block.configFields = {
        {.id = "removed-editor:new", .format = "python", .encoded = "pass"}};
    config.block.configCollapsed = false;
    settle(ui, ctx, node, config, 5);
    REQUIRE(std::get<3>(layoutLog.last()) < 400.0f - 100.0f);
    REQUIRE(flowgraphNodeData(config.id)->ResizeFlags == ImNodesNodeResizeFlags_XY);
}

TEST_CASE("Collapsed editors preserve height across lifecycle changes and display scales",
          "[core][sakura][flowgraph_node][collapse][lifecycle]") {
    for (const F32 scale : {1.0f, 2.0f}) {
        for (const std::string format : {"python", "markdown"}) {
            CAPTURE(scale, format);
            SakuraTest::HeadlessUi ui(scale, ImVec2(1200.0f, 1200.0f));
            const auto ctx = ui.sakura();

            FlowgraphNode node;
            auto config = baseConfigFor(baseBlock("lifecycle-editor", "radio"));
            const FlowgraphConfigFieldConfig field{
                .id = "lifecycle-editor:code", .format = format, .encoded = "pass"};
            config.block.configFields.push_back(field);
            config.block.layout = FlowgraphNode::Layout{20.0f, 20.0f, 220.0f, 400.0f};
            config.onLayout = [&config](F32 x, F32 y, F32 width, F32 height) {
                config.block.layout = FlowgraphNode::Layout{x, y, width, height};
            };
            settle(ui, ctx, node, config, 5);
            const auto expanded = flowgraphNodeDimensions(config.id);
            const F32 floor = flowgraphNodeData(config.id)->ResizeMinimumSize.y;

            config.block.configCollapsed = true;
            settle(ui, ctx, node, config, 4);
            for (const auto state : {Block::State::Incomplete, Block::State::Errored,
                                     Block::State::Creating, Block::State::Created}) {
                CAPTURE(state);
                config.block.state = state;
                // The presenter omits fields while the block is creating.
                config.block.configFields = state == Block::State::Creating
                    ? std::vector<FlowgraphConfigFieldConfig>{}
                    : std::vector<FlowgraphConfigFieldConfig>{field};
                settle(ui, ctx, node, config, 4);
                REQUIRE(config.block.layout->height == Catch::Approx(400.0f).margin(1.0f));
                REQUIRE(flowgraphNodeData(config.id)->ResizeFlags == ImNodesNodeResizeFlags_X);
            }

            config.block.configCollapsed = false;
            settle(ui, ctx, node, config, 5);
            REQUIRE(flowgraphNodeDimensions(config.id).y == Catch::Approx(expanded.y).margin(1.0f));
            REQUIRE(flowgraphNodeData(config.id)->ResizeMinimumSize.y == Catch::Approx(floor).margin(1.0f));
            REQUIRE(flowgraphNodeData(config.id)->ResizeFlags == ImNodesNodeResizeFlags_XY);
        }
    }
}

TEST_CASE("Cold collapsed editors do not reserve height from attached surfaces",
          "[core][sakura][flowgraph_node][collapse][detachment]") {
    SakuraTest::HeadlessUi ui(1.0f, ImVec2(600.0f, 900.0f));
    const auto ctx = ui.sakura();

    SakuraTest::ResizeLog resizeLog;
    FlowgraphNode node;
    auto config = baseConfigFor(baseBlock("collapsed-surface", "radio"));
    config.block.configFields.push_back(
        {.id = "collapsed-surface:code", .format = "python", .encoded = "pass"});
    config.block.configCollapsed = true;
    auto surface = attachedSurface("collapsed-surface:surface");
    surface.height = 400.0f;
    surface.onAttachedSize = [&resizeLog](const Sakura::SurfaceResize& resize) {
        resizeLog.record(resize);
    };
    config.block.surfaces.push_back(surface);
    config.onLayout = [&config](F32 x, F32 y, F32 width, F32 height) {
        config.block.layout = FlowgraphNode::Layout{x, y, width, height};
    };
    settle(ui, ctx, node, config, 6);
    REQUIRE(resizeLog.count() >= 1);
    for (const auto& resize : resizeLog.entries) {
        REQUIRE(resize.logicalSize.y == 400);
    }
    const auto attached = flowgraphNodeDimensions(config.id);
    REQUIRE(flowgraphNodeData(config.id)->ResizeFlags == ImNodesNodeResizeFlags_XY);

    config.block.surfaces[0].detached = true;
    const auto resizeCount = resizeLog.count();
    settle(ui, ctx, node, config, 4);
    REQUIRE(flowgraphNodeDimensions(config.id).y < attached.y - 100.0f);
    REQUIRE(flowgraphNodeData(config.id)->ResizeFlags == ImNodesNodeResizeFlags_X);
    REQUIRE(resizeLog.count() == resizeCount);

    config.block.surfaces[0].detached = false;
    settle(ui, ctx, node, config, 6);
    REQUIRE(flowgraphNodeDimensions(config.id).y == Catch::Approx(attached.y).margin(1.0f));
    REQUIRE(flowgraphNodeData(config.id)->ResizeFlags == ImNodesNodeResizeFlags_XY);
    REQUIRE(resizeLog.entries.back().logicalSize.y == 400);
}

TEST_CASE("Chevron clicks preserve resized layout through movement and serialized restoration",
          "[core][sakura][flowgraph_node][collapse][interaction][persistence]") {
    for (const F32 scale : {1.0f, 2.0f}) {
        for (const std::string format : {"python", "markdown"}) {
            CAPTURE(scale, format);
            auto config = baseConfigFor(baseBlock("gesture-editor", "radio"));
            config.block.configFields.push_back(
                {.id = "gesture-editor:code", .format = format, .encoded = "pass"});
            Parser::Map saved;
            NodeMeta expected;
            {
                NodeSession session(config, scale);
                session.frames();
                const auto initial = flowgraphNodeDimensions(config.id);
                session.resize(ImVec2(80.0f * scale, 180.0f * scale));
                const auto expanded = flowgraphNodeDimensions(config.id);
                REQUIRE(expanded.y == Catch::Approx(initial.y + 180.0f * scale).margin(2.0f));
                REQUIRE(expanded.x == Catch::Approx(initial.x + 80.0f * scale).margin(2.0f));
                const F32 height = session.meta.height;

                // Resizing must really have succeeded before testing restoration.
                REQUIRE(height > 250.0f);
                for (U64 cycle = 0; cycle < 2; ++cycle) {
                    const auto eventStart = session.layouts.entries.size();
                    session.gesture(session.chevron());
                    REQUIRE(session.meta.configCollapsed);
                    REQUIRE(session.toggles.size() == cycle * 2 + 1);
                    REQUIRE(session.toggles.back());
                    const auto collapsed = flowgraphNodeDimensions(config.id);
                    REQUIRE(collapsed.y < expanded.y - 100.0f * scale);

                    // The collapsed node accepts X drags but must ignore Y drags.
                    session.resize(ImVec2(40.0f * scale, 60.0f * scale));
                    REQUIRE(flowgraphNodeDimensions(config.id).x ==
                            Catch::Approx(collapsed.x + 40.0f * scale).margin(2.0f));
                    REQUIRE(flowgraphNodeDimensions(config.id).y == Catch::Approx(collapsed.y).margin(1.0f));
                    const F32 x = session.meta.x;
                    const F32 y = session.meta.y;
                    const auto* data = flowgraphNodeData(config.id);
                    session.gesture(data->TitleBarContentRect.GetCenter(),
                                    ImVec2(30.0f * scale, 20.0f * scale));
                    REQUIRE(session.meta.x == Catch::Approx(x + 30.0f).margin(1.0f));
                    REQUIRE(session.meta.y == Catch::Approx(y + 20.0f).margin(1.0f));

                    // Check every persisted event, not just the settled frame.
                    for (U64 i = eventStart; i < session.layouts.entries.size(); ++i) {
                        REQUIRE(std::get<3>(session.layouts.entries[i]) == Catch::Approx(height).margin(1.0f));
                    }
                    session.gesture(session.chevron());
                    REQUIRE_FALSE(session.meta.configCollapsed);
                    REQUIRE(session.toggles.size() == (cycle + 1) * 2);
                    REQUIRE_FALSE(session.toggles.back());
                    REQUIRE(flowgraphNodeDimensions(config.id).y == Catch::Approx(expanded.y).margin(1.0f));
                }

                session.gesture(session.chevron());
                const auto eventCount = session.layouts.entries.size();
                session.frames(12);
                REQUIRE(session.layouts.entries.size() == eventCount);
                expected = session.meta;
                REQUIRE(expected.serialize(saved) == Result::SUCCESS);
            }

            // Fresh UI and view: no retained node or ImGui state can rescue a bad save.
            NodeMeta restored;
            REQUIRE(restored.deserialize(saved) == Result::SUCCESS);
            NodeSession session(config, scale, restored);
            session.frames();
            REQUIRE(session.meta.configCollapsed);
            REQUIRE(session.meta.height == Catch::Approx(expected.height).margin(1.0f));
            session.gesture(session.chevron());
            REQUIRE_FALSE(session.meta.configCollapsed);
            REQUIRE(session.meta.x == Catch::Approx(expected.x).margin(1.0f));
            REQUIRE(session.meta.y == Catch::Approx(expected.y).margin(1.0f));
            REQUIRE(session.meta.width == Catch::Approx(expected.width).margin(1.0f));
            const F32 padding = 2.0f * ImNodes::GetStyle().NodePadding.y;
            REQUIRE(flowgraphNodeDimensions(config.id).y ==
                    Catch::Approx(expected.height * scale + padding).margin(1.0f));
        }
    }
}

TEST_CASE("Chevron gestures do not collapse on cancelled clicks or node drags",
          "[core][sakura][flowgraph_node][collapse][interaction]") {
    auto config = baseConfigFor(baseBlock("cancel-chevron", "radio"));
    config.block.configFields.push_back(
        {.id = "cancel-chevron:code", .format = "python", .encoded = "pass"});
    NodeSession session(config);
    session.frames();

    const auto chevron = session.chevron();
    session.ui.setMouse(chevron, false);
    session.tick();
    session.ui.setMouse(chevron, true);
    session.tick();
    REQUIRE(session.toggles.empty());
    session.ui.setMouse(ImVec2(-100.0f, -100.0f), false);
    session.frames();
    REQUIRE(session.toggles.empty());
    REQUIRE_FALSE(session.meta.configCollapsed);

    // Starting on the chevron is also a valid node-drag gesture.
    const F32 x = session.meta.x;
    session.gesture(session.chevron(), ImVec2(60.0f, 0.0f));
    REQUIRE(session.meta.x == Catch::Approx(x + 60.0f).margin(1.0f));
    REQUIRE(session.toggles.empty());
    REQUIRE_FALSE(session.meta.configCollapsed);

    // A cancelled gesture must not prevent the next genuine click.
    session.gesture(session.chevron());
    REQUIRE(session.toggles == std::vector<bool>{true});
    REQUIRE(session.meta.configCollapsed);

    session.config.block.configFields.clear();
    session.frames();
    session.gesture(session.chevron());
    REQUIRE(session.toggles == std::vector<bool>{true});
}

TEST_CASE("Collapsing config leaves ports and metrics in the node layout",
          "[core][sakura][flowgraph_node][collapse][ports][allocation]") {
    auto config = baseConfigFor(baseBlock("port-editor", "radio"));
    config.block.inputs = {
        {.port = {.id = "port-editor:in-a", .label = "Input A"}},
        {.port = {.id = "port-editor:in-b", .label = "Input B"}},
    };
    Tensor tensor(DeviceType::CPU, DataType::F32, {16});
    REQUIRE(tensor.setAttribute("sampleRate", F32{48000.0f}) == Result::SUCCESS);
    config.block.outputs = {
        {.port = {.id = "port-editor:out", .label = "Output"}, .tensor = tensor},
    };
    config.block.metrics.push_back({.id = "port-editor:rate", .label = "Rate", .format = "label"});
    config.block.configFields = {
        {.id = "port-editor:code", .format = "python", .encoded = "pass"},
        {.id = "port-editor:text", .format = "multiline", .encoded = "fixed-height text"},
    };
    NodeSession session(config);
    session.frames();
    session.resize(ImVec2(0.0f, 180.0f));
    const auto expanded = flowgraphNodeDimensions(config.id);
    const F32 height = session.meta.height;

    const auto checkPorts = [&]() {
        const auto* data = flowgraphNodeData(config.id);
        auto& editor = ImNodes::EditorContextGet();
        std::vector<F32> positions;
        const std::vector<std::string> ids = {"port-editor:in-a", "port-editor:in-b", "port-editor:out"};
        for (U64 i = 0; i < ids.size(); ++i) {
            const Sakura::NodeEditor::PinRef ref{
                .nodeId = FlowgraphNodeId(config.id),
                .pinId = FlowgraphPinId(ids[i]),
                .isInput = i < 2,
            };
            const int index = ImNodes::ObjectPoolFind(editor.Pins, Sakura::Private::NodeEditorPinObjectId(ref));
            REQUIRE(index >= 0);
            REQUIRE(editor.Pins.InUse[index]);
            const auto& pin = editor.Pins.Pool[index];
            REQUIRE(pin.ParentNodeIdx == ImNodes::ObjectPoolFind(editor.Nodes, imnodesId(config.id)));
            REQUIRE(pin.Type == (i < 2 ? ImNodesAttributeType_Input : ImNodesAttributeType_Output));
            REQUIRE(pin.AttributeRect.Min.y >= data->Rect.Min.y);
            REQUIRE(pin.AttributeRect.Max.y <= data->Rect.Max.y);
            positions.push_back(pin.Pos.y);
        }
        return positions;
    };
    const auto pins = checkPorts();
    session.gesture(session.chevron());
    REQUIRE(session.meta.configCollapsed);
    REQUIRE(checkPorts() == pins);
    const auto collapsed = flowgraphNodeDimensions(config.id);
    REQUIRE(collapsed.y < expanded.y - 100.0f);
    REQUIRE(session.meta.height == Catch::Approx(height).margin(1.0f));

    // Metrics are outside config: removing one must still change collapsed chrome.
    session.config.block.metrics.clear();
    session.frames();
    REQUIRE(flowgraphNodeDimensions(config.id).y < collapsed.y);
    REQUIRE(checkPorts() == pins);
    session.config.block.metrics = config.block.metrics;
    session.frames();
    REQUIRE(flowgraphNodeDimensions(config.id).y == Catch::Approx(collapsed.y).margin(1.0f));

    session.gesture(session.chevron());
    REQUIRE_FALSE(session.meta.configCollapsed);
    REQUIRE(checkPorts() == pins);
    REQUIRE(flowgraphNodeDimensions(config.id).y == Catch::Approx(expanded.y).margin(1.0f));
}

TEST_CASE("Every node size seeds its width and honors an explicit saved width",
          "[core][sakura][flowgraph_node][creation][persistence]") {
    for (const auto& [size, width] :
         {std::pair{Block::NodeSize::XS, 120.0f},
          std::pair{Block::NodeSize::S, 140.0f},
          std::pair{Block::NodeSize::M, 220.0f},
          std::pair{Block::NodeSize::L, 320.0f},
          std::pair{Block::NodeSize::XL, 460.0f}}) {
        CAPTURE(size, width);
        auto config = baseConfigFor(baseBlock("default-width", "radio"));
        config.block.nodeSize = size;
        NodeSession session(config);
        session.frames();
        REQUIRE(session.meta.width == Catch::Approx(width).margin(1.0f));
        REQUIRE(session.meta.height == 0.0f);
        session.meta.width = 360.0f;
        session.frames();
        REQUIRE(session.meta.width == Catch::Approx(360.0f).margin(1.0f));
        const F32 padding = 2.0f * ImNodes::GetStyle().NodePadding.x;
        REQUIRE(flowgraphNodeDimensions(config.id).x == Catch::Approx(360.0f + padding).margin(1.0f));
    }
}

TEST_CASE("Config grids coexist with flexible editors and attached surface allocations",
          "[core][sakura][flowgraph_node][field-grid][allocation]") {
    SakuraTest::HeadlessUi ui(1.0f, ImVec2(900.0f, 1200.0f));
    const auto ctx = ui.sakura();

    SakuraTest::ResizeLog resizeLog;
    FlowgraphNode node;
    auto config = baseConfigFor(baseBlock("grid-surface", "radio"));
    for (U64 i = 0; i < 4; ++i) {
        config.block.configFields.push_back({
            .id = "grid-surface:field:" + std::to_string(i),
            .label = "Value",
            .format = "range:0:1",
            .encoded = "0.5",
        });
        if (i == 1) {
            config.block.configFields.push_back(
                {.id = "grid-surface:text", .format = "markdown", .encoded = "hello"});
        }
    }
    auto surface = attachedSurface("grid-surface:surface");
    surface.height = 400.0f;
    surface.onAttachedSize = [&resizeLog](const Sakura::SurfaceResize& resize) {
        resizeLog.record(resize);
    };
    config.block.surfaces.push_back(surface);
    settle(ui, ctx, node, config, 7);
    REQUIRE(resizeLog.count() >= 1);
    for (const auto& resize : resizeLog.entries) {
        REQUIRE(resize.logicalSize.y == 400);
    }
    const auto narrow = flowgraphNodeDimensions(config.id);
    const F32 narrowFloor = flowgraphNodeData(config.id)->ResizeMinimumSize.y;

    dragCorner(ui, ctx, node, config.id, ImVec2(100.0f, 0.0f));
    settle(ui, ctx, node, config, 6);
    REQUIRE(flowgraphNodeDimensions(config.id).x == Catch::Approx(narrow.x + 100.0f).margin(1.0f));
    REQUIRE(flowgraphNodeDimensions(config.id).y == Catch::Approx(narrow.y).margin(1.0f));
    REQUIRE(flowgraphNodeData(config.id)->ResizeMinimumSize.y < narrowFloor - 20.0f);
    REQUIRE(resizeLog.entries.back().logicalSize.y > 400);
    const F32 expandedFloor = flowgraphNodeData(config.id)->ResizeMinimumSize.y;
    const U64 expandedSurfaceHeight = resizeLog.entries.back().logicalSize.y;

    config.block.configCollapsed = true;
    settle(ui, ctx, node, config, 6);
    REQUIRE(flowgraphNodeData(config.id)->ResizeFlags == ImNodesNodeResizeFlags_XY);
    REQUIRE(flowgraphNodeData(config.id)->ResizeMinimumSize.y < expandedFloor - 120.0f);
    REQUIRE(resizeLog.entries.back().logicalSize.y > expandedSurfaceHeight + 120);
    REQUIRE(flowgraphNodeDimensions(config.id).y == Catch::Approx(narrow.y).margin(1.0f));

    config.block.configCollapsed = false;
    settle(ui, ctx, node, config, 6);
    REQUIRE(flowgraphNodeData(config.id)->ResizeMinimumSize.y == Catch::Approx(expandedFloor).margin(1.0f));
    REQUIRE(resizeLog.entries.back().logicalSize.y == Catch::Approx(expandedSurfaceHeight).margin(1.0f));
    for (const auto& resize : resizeLog.entries) {
        REQUIRE(resize.logicalSize.y >= FlowgraphNode::SurfaceHeightSpec().minimum);
    }
}
