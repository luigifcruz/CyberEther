#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "compositor/default/model/meta.hh"
#include "compositor/default/views/flowgraph/editor/node.hh"

#include <jetstream/render/sakura/components/node/editor.hh>
#include <jetstream/render/sakura/typography.hh>

#include "fonts.hh"
#include "component_harness.hh"

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace Jetstream;
using namespace SakuraTest;

namespace {

using NodeEditor = Sakura::NodeEditor;
using Typography = Sakura::Typography;

constexpr F32 kPadding = 8.0f;
constexpr F32 kNodeFont = Typography::FontSize * Typography::ChromeScale;
constexpr F32 kTitleFont = Typography::FontSize * 1.15f;
constexpr Rect kViewport = {0.0f, 0.0f, 1600.0f, 1600.0f};
constexpr F32 kGap = Sakura::NodeContentStack::Spacing;
constexpr F32 kFieldGap = Sakura::NodeField::Gap;
constexpr F32 kSurfaceMinimum = 150.0f;
constexpr F32 kEditorMinimum = 120.0f;

struct ScaledFontWindow : MonoFontWindow {
    explicit ScaledFontWindow(F32 scale) {
        rescale(scale);
    }
};

class SurfaceTexture final : public Render::Texture {
 public:
    SurfaceTexture() : Texture(Config{.size = {256, 256}}) {}

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

FlowgraphNode::BlockData MakeBlock(const std::string& name, const std::string& module = "radio") {
    FlowgraphNode::BlockData block;
    block.name = name;
    block.module = module;
    block.title = "Title";
    block.state = Block::State::Created;
    block.nodeSize = Block::NodeSize::M;
    return block;
}

FlowgraphNode::Config MakeConfig(FlowgraphNode::BlockData block) {
    FlowgraphNode::Config config;
    config.id = block.name;
    config.block = std::move(block);
    return config;
}

FlowgraphConfigFieldConfig EditorField(const std::string& id, const std::string& format = "python") {
    return {
        .id = id,
        .name = "code",
        .format = {{"type", format}},
        .values = {{"code", std::string("pass")}},
    };
}

FlowgraphConfigFieldConfig RangeField(const std::string& id) {
    return {
        .id = id,
        .name = "value",
        .label = "Value",
        .format = {{"type", "range"}, {"min", 0.0f}, {"max", 1.0f}},
        .values = {{"value", 0.5f}},
    };
}

FlowgraphNode::Surface AttachedSurface(const std::string& id, F32 height = 512.0f) {
    return {
        .id = id,
        .texture = std::make_shared<SurfaceTexture>(),
        .height = height,
    };
}

FlowgraphNode::Config VisualizationConfig(const std::string& id) {
    auto config = MakeConfig(MakeBlock(id));
    config.block.nodeSize = Block::NodeSize::L;
    config.block.state = Block::State::Incomplete;
    config.block.inputs.push_back({.port = {.id = "buffer", .label = "Input"}});
    config.block.configFields = {RangeField(id + ":rangeMin"), RangeField(id + ":rangeMax")};
    return config;
}

struct SurfaceState {
    SurfaceMeta meta;
    std::optional<Sakura::SurfaceResize> mail;
    std::vector<Sakura::SurfaceResize> resizes;

    U64 last() const {
        return resizes.back().logicalSize.y;
    }
};

struct NodeSession {
    ScaledFontWindow window;
    F32 scale;
    FlowgraphNode node;
    NodeEditor editor;
    Host host;
    Frames frames;
    FlowgraphNode::Config config;
    NodeMeta meta;
    ConfigMeta configMeta;
    bool feedLayout = true;
    std::optional<NodeMeta> layoutMail;
    std::optional<bool> collapseMail;
    std::vector<NodeMeta> layouts;
    std::vector<bool> toggles;
    std::unordered_map<std::string, SurfaceState> surfaces;
    std::optional<NodeEditor::Geometry> geometry;
    NodeEditor::ResizeAxes axes = NodeEditor::ResizeAxes::None;
    bool collapsible = false;
    U64 pins = 0;

    explicit NodeSession(FlowgraphNode::Config config,
                         F32 scale = 1.0f,
                         NodeMeta meta = {},
                         ConfigMeta configMeta = {}) :
        window(scale),
        scale(scale),
        host(editor),
        frames(host, kViewport, SessionContext(scale, window)),
        config(std::move(config)),
        meta(meta),
        configMeta(configMeta) {}

    static Sakura::Context SessionContext(F32 scale, Render::Window& window) {
        auto ctx = MakeContext({kViewport.width, kViewport.height});
        ctx.pixelRatio = scale;
        ctx.render = &window;
        return ctx;
    }

    SurfaceState& track(const std::string& id, U64 height = 256) {
        auto& state = surfaces[id];
        state.meta.attachedHeight = height;
        return state;
    }

    void tick() {
        if (layoutMail.has_value()) {
            meta = *layoutMail;
            layoutMail.reset();
        }
        if (collapseMail.has_value()) {
            configMeta.collapsed = *collapseMail;
            collapseMail.reset();
        }
        for (auto& [_, state] : surfaces) {
            if (state.mail.has_value()) {
                state.meta.attachedHeight = state.mail->logicalSize.y;
                state.mail.reset();
            }
        }
        for (auto& surface : config.block.surfaces) {
            const auto it = surfaces.find(surface.id);
            if (it == surfaces.end()) {
                continue;
            }
            surface.height = static_cast<F32>(it->second.meta.attachedHeight);
            surface.detached = it->second.meta.detached;
            surface.onAttachedSize = [this, id = surface.id](const Sakura::SurfaceResize& resize) {
                auto& state = surfaces.at(id);
                state.resizes.push_back(resize);
                state.mail = resize;
            };
        }
        if (feedLayout) {
            config.block.layout = FlowgraphNode::Layout{meta.x, meta.y, meta.width, meta.height};
        }
        config.block.configCollapsed = configMeta.collapsed;
        config.onLayout = [this](F32 x, F32 y, F32 width, F32 height) {
            const NodeMeta next{.x = x, .y = y, .width = width, .height = height};
            layouts.push_back(next);
            layoutMail = next;
        };
        config.onConfigCollapse = [this](bool collapsed) {
            toggles.push_back(collapsed);
            collapseMail = collapsed;
        };
        node.update(config);
        auto spec = node.takeNode();
        axes = spec.resize;
        collapsible = spec.collapsible;
        pins = spec.pins.size();
        spec.onGeometryChange = [this, forward = std::move(spec.onGeometryChange)](const NodeEditor::Geometry& next) {
            geometry = next;
            if (forward) {
                forward(next);
            }
        };
        editor.update({.id = "editor", .nodes = {std::move(spec)}});
        window.frame([this] {
            frames.run();
        });
    }

    void run(U64 count = 6) {
        for (U64 frame = 0; frame < count; ++frame) {
            tick();
        }
    }

    Extent2D<F32> size() const {
        REQUIRE(geometry.has_value());
        return geometry->contentDimensions;
    }

    Extent2D<F32> origin() const {
        REQUIRE(geometry.has_value());
        return geometry->gridPosition;
    }

    Extent2D<F32> px(F32 x, F32 y) const {
        return {x * scale, y * scale};
    }

    Extent2D<F32> grip() const {
        return px(origin().x + size().x + 2.0f * kPadding - 3.0f, origin().y + size().y + 2.0f * kPadding - 3.0f);
    }

    Extent2D<F32> chevron() const {
        return px(origin().x + kPadding + size().x - kNodeFont * 0.5f, origin().y + kPadding + kTitleFont * 0.5f);
    }

    Extent2D<F32> header() const {
        return px(origin().x + kPadding + 12.0f, origin().y + kPadding + 4.0f);
    }

    Extent2D<F32> away() const {
        return {kViewport.width - 4.0f, kViewport.height - 4.0f};
    }

    void send(MouseEventType type, Extent2D<F32> point) {
        frames.ctx.mouse = point;
        host.send(Mouse(type, point.x, point.y));
    }

    void gesture(Extent2D<F32> start, Extent2D<F32> delta = {0.0f, 0.0f}) {
        const Extent2D<F32> end = {start.x + delta.x, start.y + delta.y};
        send(MouseEventType::Move, start);
        tick();
        send(MouseEventType::Click, start);
        tick();
        send(MouseEventType::Move, end);
        tick();
        send(MouseEventType::Release, end);
        tick();
        send(MouseEventType::Move, away());
        run();
    }

    void resize(Extent2D<F32> delta) {
        gesture(grip(), delta);
    }

    SurfaceState& surface(const std::string& id) {
        return surfaces.at(id);
    }
};

struct VisualizationSession : NodeSession {
    std::string surfaceId = "plot:surface:default";
    std::vector<FlowgraphConfigFieldConfig> fields;

    explicit VisualizationSession(F32 scale, ConfigMeta configMeta = {}) :
        NodeSession(VisualizationConfig("plot"), scale, {}, configMeta),
        fields(config.block.configFields) {
        track(surfaceId);
    }

    SurfaceState& saved() {
        return surface(surfaceId);
    }

    void connect() {
        config.block.state = Block::State::Created;
        config.block.configFields = fields;
        config.block.surfaces = {AttachedSurface(surfaceId)};
    }

    void disconnect() {
        config.block.state = Block::State::Incomplete;
        config.block.surfaces.clear();
    }

    void startResized() {
        run();
        connect();
        run();
        REQUIRE_FALSE(saved().resizes.empty());
        const U64 initial = saved().meta.attachedHeight;
        resize({0.0f, 300.0f * scale});
        REQUIRE(saved().meta.attachedHeight == initial + 300);
        REQUIRE_FALSE(saved().mail.has_value());
    }

    void checkRestored(U64 height, Extent2D<F32> dimensions) {
        for (U64 frame = 0; frame < 8; ++frame) {
            CAPTURE(frame);
            tick();
            if (saved().mail.has_value()) {
                CHECK(saved().mail->logicalSize.y == height);
            }
            CHECK(saved().meta.attachedHeight == height);
        }
        CHECK(size().x == Catch::Approx(dimensions.x).margin(1.0f));
        CHECK(size().y == Catch::Approx(dimensions.y).margin(1.0f));
    }
};

F32 ChromeHeight(FlowgraphNode::Config config) {
    config.block.configFields.clear();
    config.block.surfaces.clear();
    NodeSession probe(std::move(config));
    probe.run(3);
    return probe.size().y;
}

}  // namespace

TEST_CASE("Plain nodes resize only horizontally while surfaces and flexible fields enable both axes",
          "[core][sakura][flowgraph][node-view]") {
    SECTION("Attached surface") {
        auto config = MakeConfig(MakeBlock("surface-block"));
        config.block.surfaces.push_back(AttachedSurface("surface-block:surface:out"));
        NodeSession session(config);
        session.run(2);
        CHECK(session.axes == NodeEditor::ResizeAxes::XY);
    }
    SECTION("Detached surface") {
        auto config = MakeConfig(MakeBlock("detached-block"));
        config.block.surfaces.push_back(AttachedSurface("detached-block:surface:out"));
        config.block.surfaces.back().detached = true;
        NodeSession session(config);
        session.run(2);
        CHECK(session.axes == NodeEditor::ResizeAxes::X);
    }
    SECTION("Markdown field") {
        auto config = MakeConfig(MakeBlock("markdown-block", "note"));
        config.block.configFields.push_back(EditorField("markdown-block:text", "markdown"));
        NodeSession session(config);
        session.run(2);
        CHECK(session.axes == NodeEditor::ResizeAxes::XY);
    }
    SECTION("Fixed-height fields only") {
        auto config = MakeConfig(MakeBlock("range-block"));
        config.block.configFields.push_back(RangeField("range-block:value"));
        NodeSession session(config);
        session.run(2);
        CHECK(session.axes == NodeEditor::ResizeAxes::X);
    }
    SECTION("No content") {
        NodeSession session(MakeConfig(MakeBlock("plain-block", "note")));
        session.run(2);
        CHECK(session.axes == NodeEditor::ResizeAxes::X);
    }
}

TEST_CASE("Every node size seeds its default width and honors an explicit saved width",
          "[core][sakura][flowgraph][node-view]") {
    const auto [size, width] = GENERATE(std::pair{Block::NodeSize::XS, 120.0f},
                                        std::pair{Block::NodeSize::S, 140.0f},
                                        std::pair{Block::NodeSize::M, 220.0f},
                                        std::pair{Block::NodeSize::L, 320.0f},
                                        std::pair{Block::NodeSize::XL, 460.0f});
    CAPTURE(size, width);
    auto config = MakeConfig(MakeBlock("default-width"));
    config.block.nodeSize = size;
    NodeSession session(config);
    session.run();
    CHECK(session.meta.width == Catch::Approx(width).margin(1.0f));
    CHECK(session.meta.height == 0.0f);
    CHECK(session.size().x == Catch::Approx(width).margin(1.0f));

    session.meta.width = 360.0f;
    session.run();
    CHECK(session.meta.width == Catch::Approx(360.0f).margin(1.0f));
    CHECK(session.size().x == Catch::Approx(360.0f).margin(1.0f));
}

TEST_CASE("Auto-height nodes report zero height and ignore saved layout heights",
          "[core][sakura][flowgraph][node-view]") {
    NodeSession session(MakeConfig(MakeBlock("auto-block")));
    session.run(3);
    REQUIRE_FALSE(session.layouts.empty());
    CHECK(session.layouts.back().width > 0.0f);
    CHECK(session.layouts.back().height == 0.0f);
    CHECK(session.size().y > 0.0f);

    session.meta = {.x = 10.0f, .y = 10.0f, .width = 0.0f, .height = 80.0f};
    session.run(3);
    CHECK(session.meta.height == 0.0f);
    CHECK(session.layouts.back().height == 0.0f);
}

TEST_CASE("User resize grows the node and hands all growth to the attached surface",
          "[core][sakura][flowgraph][node-view]") {
    const F32 scale = GENERATE(1.0f, 2.0f);
    CAPTURE(scale);
    auto config = MakeConfig(MakeBlock("grow-block"));
    config.block.surfaces.push_back(AttachedSurface("grow-block:surface:out"));
    NodeSession session(config, scale);
    auto& surface = session.track("grow-block:surface:out", 512);
    session.run();
    REQUIRE_FALSE(surface.resizes.empty());
    CHECK(surface.last() == 512);
    const F32 before = session.meta.height;
    REQUIRE(before > 512.0f);

    session.resize({0.0f, 80.0f * scale});
    CHECK(session.meta.height == Catch::Approx(before + 80.0f).margin(1.0f));
    CHECK(surface.last() == Catch::Approx(512 + 80).margin(1.0));
    CHECK(surface.meta.attachedHeight == Catch::Approx(512 + 80).margin(1.0));
}

TEST_CASE("Dragging past the floor clamps the width and the flexible height at every display scale",
          "[core][sakura][flowgraph][node-view]") {
    const F32 scale = GENERATE(1.0f, 1.5f, 2.0f);
    CAPTURE(scale);
    auto config = MakeConfig(MakeBlock("clamp-note", "note"));
    config.block.configFields.push_back(EditorField("clamp-note:text", "markdown"));
    NodeSession session(config, scale, {.x = 10.0f, .y = 10.0f, .width = 300.0f, .height = 600.0f});
    session.run();
    REQUIRE(session.size().y == Catch::Approx(600.0f).margin(1.0f));

    session.resize({-1000.0f * scale, -1000.0f * scale});
    CHECK(session.meta.width == Catch::Approx(FlowgraphNode::MinimumNodeWidth).margin(1.0f));
    CHECK(session.meta.height == Catch::Approx(kEditorMinimum).margin(1.0f));
    CHECK(session.size().x == Catch::Approx(FlowgraphNode::MinimumNodeWidth).margin(1.0f));
    CHECK(session.size().y == Catch::Approx(kEditorMinimum).margin(1.0f));

    session.resize({0.0f, 200.0f * scale});
    CHECK(session.meta.height == Catch::Approx(kEditorMinimum + 200.0f).margin(1.0f));
}

TEST_CASE("Python editor nodes seed at their declared minimum, floor there, and resize in every live state",
          "[core][sakura][flowgraph][node-view]") {
    const auto state = GENERATE(Block::State::Created, Block::State::Incomplete, Block::State::Errored);
    CAPTURE(state);
    auto config = MakeConfig(MakeBlock("python-block"));
    config.block.state = state;
    if (state != Block::State::Created) {
        config.block.diagnostic = "Block 'python-block' has unconnected input 'input0'.";
    }
    const F32 chrome = ChromeHeight(config);
    REQUIRE(chrome > 0.0f);

    config.block.configFields.push_back(EditorField("python-block:code"));
    NodeSession session(config);
    session.run();
    CHECK(session.axes == NodeEditor::ResizeAxes::XY);
    const F32 floor = chrome + kFieldGap + kEditorMinimum;
    CHECK(session.size().y == Catch::Approx(floor).margin(1.0f));
    CHECK(session.meta.height == Catch::Approx(floor).margin(1.0f));

    session.resize({0.0f, -600.0f});
    CHECK(session.size().y == Catch::Approx(floor).margin(1.0f));
    session.resize({0.0f, 300.0f});
    CHECK(session.size().y == Catch::Approx(floor + 300.0f).margin(1.0f));
    session.resize({0.0f, -600.0f});
    CHECK(session.size().y == Catch::Approx(floor).margin(1.0f));
}

TEST_CASE("Node height drags survive the deferred metadata round-trip in both directions",
          "[core][sakura][flowgraph][node-view]") {
    auto config = MakeConfig(MakeBlock("meta-block"));
    config.block.configFields.push_back(EditorField("meta-block:code"));
    NodeSession session(config, 1.0f, {.x = 20.0f, .y = 20.0f, .width = 0.0f, .height = 216.0f});
    session.run();
    REQUIRE(session.size().y == Catch::Approx(216.0f).margin(1.0f));

    session.resize({0.0f, 300.0f});
    CHECK(session.size().y == Catch::Approx(516.0f).margin(1.0f));
    CHECK(session.meta.height == Catch::Approx(516.0f).margin(1.0f));

    session.resize({0.0f, -300.0f});
    CHECK(session.size().y == Catch::Approx(216.0f).margin(1.0f));
    CHECK(session.meta.height == Catch::Approx(216.0f).margin(1.0f));
}

TEST_CASE("Connecting a block keeps its node size width when its surface first appears",
          "[core][sakura][flowgraph][node-view]") {
    auto config = MakeConfig(MakeBlock("xl-block"));
    config.block.state = Block::State::Incomplete;
    config.block.nodeSize = Block::NodeSize::XL;
    NodeSession session(config);
    session.run(3);
    CHECK(session.meta.width == Catch::Approx(460.0f));
    CHECK(session.meta.height == 0.0f);

    auto& surface = session.track("xl-block:surface:out", 256);
    session.config.block.state = Block::State::Created;
    session.config.block.surfaces.push_back(AttachedSurface("xl-block:surface:out"));
    session.run();
    REQUIRE_FALSE(surface.resizes.empty());
    CHECK(surface.last() == 256);
    CHECK(surface.meta.attachedHeight == 256);
    CHECK(session.meta.width == Catch::Approx(460.0f));
    CHECK(session.meta.height > 256.0f);
}

TEST_CASE("Detaching every surface reports auto height and reattaching restores the surface meta",
          "[core][sakura][flowgraph][node-view]") {
    auto config = MakeConfig(MakeBlock("cycle-block"));
    config.block.surfaces.push_back(AttachedSurface("cycle-block:surface:out"));
    NodeSession session(config);
    auto& surface = session.track("cycle-block:surface:out", 512);
    session.run();
    CHECK(session.meta.height > 512.0f);

    surface.meta.detached = true;
    session.run();
    CHECK(session.axes == NodeEditor::ResizeAxes::X);
    CHECK(session.meta.height == 0.0f);
    const F32 chrome = session.size().y;
    CHECK(chrome < 100.0f);

    const U64 resizes = surface.resizes.size();
    surface.meta.detached = false;
    surface.meta.attachedHeight = 256;
    session.run();
    CHECK(session.axes == NodeEditor::ResizeAxes::XY);
    REQUIRE(surface.resizes.size() > resizes);
    for (U64 index = resizes; index < surface.resizes.size(); ++index) {
        CHECK(surface.resizes[index].logicalSize.y == 256);
    }
    CHECK(session.meta.height == Catch::Approx(chrome + kGap + 256.0f).margin(1.0f));
}

TEST_CASE("Detach and reattach cycles hand the surface its persisted height back",
          "[core][sakura][flowgraph][node-view]") {
    auto config = MakeConfig(MakeBlock("persist-block"));
    config.block.surfaces.push_back(AttachedSurface("persist-block:surface:out"));
    NodeSession session(config);
    auto& surface = session.track("persist-block:surface:out", 400);
    session.run();
    REQUIRE_FALSE(surface.resizes.empty());
    CHECK(surface.last() == 400);

    for (U64 cycle = 0; cycle < 3; ++cycle) {
        CAPTURE(cycle);
        surface.meta.detached = true;
        session.run();
        CHECK(surface.meta.attachedHeight == 400);
        const U64 resizes = surface.resizes.size();
        surface.meta.detached = false;
        session.run();
        REQUIRE(surface.resizes.size() > resizes);
        for (U64 index = resizes; index < surface.resizes.size(); ++index) {
            CHECK(surface.resizes[index].logicalSize.y == 400);
        }
        CHECK(surface.meta.attachedHeight == 400);
    }
}

TEST_CASE("Resized visualization surfaces preserve their height through input reconnection",
          "[core][sakura][flowgraph][node-view]") {
    const F32 scale = GENERATE(1.0f, 2.0f);
    const U64 loadingFrames = GENERATE(0, 1, 3);
    const bool collapsed = GENERATE(false, true);
    CAPTURE(scale, loadingFrames, collapsed);

    VisualizationSession session(scale, {.collapsed = collapsed});
    session.run();
    session.connect();
    session.run();
    REQUIRE_FALSE(session.saved().resizes.empty());
    const U64 initialHeight = session.saved().meta.attachedHeight;
    const F32 initialNodeHeight = session.meta.height;
    session.resize({0.0f, 300.0f * scale});
    REQUIRE(session.saved().meta.attachedHeight == initialHeight + 300);
    REQUIRE(session.meta.height == Catch::Approx(initialNodeHeight + 300.0f).margin(1.0f));
    const U64 resizedHeight = session.saved().meta.attachedHeight;
    const NodeMeta resizedNode = session.meta;

    for (U64 cycle = 0; cycle < 3; ++cycle) {
        CAPTURE(cycle);
        const U64 resizes = session.saved().resizes.size();
        session.disconnect();
        session.run();
        REQUIRE(session.saved().resizes.size() == resizes);

        if (loadingFrames > 0) {
            session.config.block.state = Block::State::Creating;
            session.config.block.configFields.clear();
            session.run(loadingFrames);
        }
        REQUIRE(session.saved().resizes.size() == resizes);
        CHECK(session.saved().meta.attachedHeight == resizedHeight);

        session.connect();
        for (U64 frame = 0; frame < 8; ++frame) {
            CAPTURE(frame);
            session.tick();
            if (session.saved().mail.has_value()) {
                CHECK(session.saved().mail->logicalSize.y == resizedHeight);
            }
        }
        REQUIRE(session.saved().resizes.size() > resizes);
        CHECK(session.saved().meta.attachedHeight == resizedHeight);
        CHECK(session.meta.height == Catch::Approx(resizedNode.height).margin(1.0f));
        CHECK(session.meta.width == Catch::Approx(resizedNode.width).margin(1.0f));
    }
}

TEST_CASE("A surface replaced between frames receives the persisted size again",
          "[core][sakura][flowgraph][node-view]") {
    const F32 scale = GENERATE(1.0f, 2.0f);
    CAPTURE(scale);
    VisualizationSession session(scale);
    session.startResized();
    const U64 height = session.saved().meta.attachedHeight;
    const auto dimensions = session.size();
    const auto previous = session.config.block.surfaces.front().texture;
    const U64 resizes = session.saved().resizes.size();

    session.connect();
    REQUIRE(session.config.block.surfaces.front().texture != previous);
    session.checkRestored(height, dimensions);
    CHECK(session.saved().resizes.size() == resizes + 1);
}

TEST_CASE("A delayed texture restores the surface height after its block is created",
          "[core][sakura][flowgraph][node-view]") {
    const F32 scale = GENERATE(1.0f, 2.0f);
    const U64 delay = GENERATE(1, 4);
    CAPTURE(scale, delay);
    VisualizationSession session(scale);
    session.startResized();
    const U64 height = session.saved().meta.attachedHeight;
    const auto dimensions = session.size();
    const U64 resizes = session.saved().resizes.size();

    session.disconnect();
    session.run();
    session.config.block.state = Block::State::Created;
    session.config.block.configFields = session.fields;
    for (U64 frame = 0; frame < delay; ++frame) {
        CAPTURE(frame);
        session.tick();
        CHECK(session.saved().resizes.size() == resizes);
        CHECK(session.saved().meta.attachedHeight == height);
    }

    session.config.block.surfaces = {AttachedSurface(session.surfaceId)};
    session.checkRestored(height, dimensions);
    CHECK(session.saved().resizes.size() > resizes);
}

TEST_CASE("Disconnecting with a queued surface resize preserves the latest user height",
          "[core][sakura][flowgraph][node-view]") {
    const F32 scale = GENERATE(1.0f, 2.0f);
    const U64 disconnectedFrames = GENERATE(1, 4);
    CAPTURE(scale, disconnectedFrames);
    VisualizationSession session(scale);
    session.startResized();
    const U64 previous = session.saved().meta.attachedHeight;
    const auto grip = session.grip();
    const Extent2D<F32> target = {grip.x, grip.y + 80.0f * scale};

    session.send(MouseEventType::Move, grip);
    session.tick();
    session.send(MouseEventType::Click, grip);
    session.tick();
    session.send(MouseEventType::Move, target);
    session.tick();
    REQUIRE(session.saved().mail.has_value());
    const U64 latest = session.saved().mail->logicalSize.y;
    REQUIRE(latest == previous + 80);
    REQUIRE(session.saved().meta.attachedHeight == previous);
    session.send(MouseEventType::Release, target);
    session.send(MouseEventType::Move, session.away());
    const U64 resizes = session.saved().resizes.size();
    const auto dimensions = session.size();

    session.disconnect();
    session.run(disconnectedFrames);
    CHECK(session.saved().meta.attachedHeight == latest);
    CHECK_FALSE(session.saved().mail.has_value());
    CHECK(session.saved().resizes.size() == resizes);

    session.connect();
    session.checkRestored(latest, dimensions);
    CHECK(session.saved().resizes.size() > resizes);

    session.resize({0.0f, -40.0f * scale});
    session.run();
    CHECK(session.saved().meta.attachedHeight == latest - 40);
    CHECK_FALSE(session.saved().mail.has_value());
}

TEST_CASE("A stale saved height below the floor never allocates a below-minimum surface",
          "[core][sakura][flowgraph][node-view]") {
    auto config = MakeConfig(MakeBlock("stale-block"));
    config.block.surfaces.push_back(AttachedSurface("stale-block:surface:out"));
    NodeSession session(config, 1.0f, {.x = 10.0f, .y = 10.0f, .width = 357.0f, .height = 100.0f});
    auto& surface = session.track("stale-block:surface:out", 400);
    for (U64 frame = 0; frame < 6; ++frame) {
        session.tick();
        for (const auto& resize : surface.resizes) {
            CHECK(resize.logicalSize.y >= kSurfaceMinimum);
        }
    }
    REQUIRE_FALSE(surface.resizes.empty());
    CHECK(surface.last() >= kSurfaceMinimum);
    CHECK(session.meta.height >= kSurfaceMinimum);
}

TEST_CASE("A cold surface seed restores the full persisted height with or without a saved node height",
          "[core][sakura][flowgraph][node-view]") {
    const bool savedLayout = GENERATE(false, true);
    CAPTURE(savedLayout);
    auto config = MakeConfig(MakeBlock("cold-block"));
    config.block.surfaces.push_back(AttachedSurface("cold-block:surface:out"));
    NodeSession session(config, 1.0f, {.x = 10.0f, .y = 10.0f, .width = 357.0f, .height = 0.0f});
    session.feedLayout = savedLayout;
    auto& surface = session.track("cold-block:surface:out", 400);
    for (U64 frame = 0; frame < 6; ++frame) {
        session.tick();
        for (const auto& resize : surface.resizes) {
            CHECK(resize.logicalSize.y == 400);
        }
    }
    REQUIRE_FALSE(surface.resizes.empty());
    CHECK(session.layouts.back().height > 400.0f);
}

TEST_CASE("A cold restore preserves every surface alongside flexible fields",
          "[core][sakura][flowgraph][node-view]") {
    auto config = MakeConfig(MakeBlock("multi-block"));
    config.block.configFields.push_back(EditorField("multi-block:text", "markdown"));
    config.block.surfaces = {AttachedSurface("multi-block:surface:a"), AttachedSurface("multi-block:surface:b")};
    NodeSession session(config);
    auto& a = session.track("multi-block:surface:a", 400);
    auto& b = session.track("multi-block:surface:b", 260);
    session.run(7);
    REQUIRE_FALSE(a.resizes.empty());
    REQUIRE_FALSE(b.resizes.empty());
    for (const auto& resize : a.resizes) {
        CHECK(resize.logicalSize.y == 400);
    }
    for (const auto& resize : b.resizes) {
        CHECK(resize.logicalSize.y == 260);
    }
    CHECK(session.meta.height >= 400.0f + 260.0f + kEditorMinimum);
}

TEST_CASE("Note blocks finish loading at their flexible minimum and a grip click does not snap",
          "[core][sakura][flowgraph][node-view]") {
    auto config = MakeConfig(MakeBlock("note-block", "note"));
    config.block.configFields.push_back(EditorField("note-block:text", "markdown"));
    config.block.state = Block::State::Creating;
    NodeSession session(config);
    session.run(2);
    CHECK(session.meta.height <= 100.0f);

    session.config.block.state = Block::State::Created;
    session.run();
    CHECK(session.meta.height == Catch::Approx(kEditorMinimum).margin(1.0f));
    CHECK(session.size().y == Catch::Approx(kEditorMinimum).margin(1.0f));

    const auto before = session.size();
    const auto layouts = session.layouts.size();
    session.gesture(session.grip());
    CHECK(session.size().x == Catch::Approx(before.x).margin(1.0f));
    CHECK(session.size().y == Catch::Approx(before.y).margin(1.0f));
    CHECK(session.layouts.size() == layouts);
}

TEST_CASE("Detaching every surface keeps the node height while flexible fields remain",
          "[core][sakura][flowgraph][node-view]") {
    auto config = MakeConfig(MakeBlock("keep-block"));
    config.block.configFields.push_back(EditorField("keep-block:text", "markdown"));
    config.block.surfaces.push_back(AttachedSurface("keep-block:surface:out"));
    NodeSession session(config);
    auto& surface = session.track("keep-block:surface:out", 512);
    session.run();
    REQUIRE_FALSE(surface.resizes.empty());
    const F32 height = session.meta.height;
    CHECK(height > 512.0f + kEditorMinimum);

    surface.meta.detached = true;
    session.run();
    CHECK(session.meta.height == Catch::Approx(height).margin(1.0f));
    CHECK(session.size().y == Catch::Approx(height).margin(1.0f));
    CHECK(session.axes == NodeEditor::ResizeAxes::XY);
}

TEST_CASE("Mixed attached and detached surfaces track only the attached one",
          "[core][sakura][flowgraph][node-view]") {
    auto config = MakeConfig(MakeBlock("mixed-block"));
    config.block.surfaces = {AttachedSurface("mixed-block:surface:a"), AttachedSurface("mixed-block:surface:b")};
    NodeSession session(config);
    auto& a = session.track("mixed-block:surface:a", 512);
    auto& b = session.track("mixed-block:surface:b", 256);
    b.meta.detached = true;
    session.run();
    CHECK(session.axes == NodeEditor::ResizeAxes::XY);
    REQUIRE_FALSE(a.resizes.empty());
    CHECK(a.last() == 512);
    CHECK(b.resizes.empty());
    const F32 height = session.meta.height;

    a.meta.detached = true;
    b.meta.detached = false;
    session.run();
    CHECK(session.meta.height == Catch::Approx(height).margin(1.0f));
    REQUIRE_FALSE(b.resizes.empty());
    CHECK(b.last() == 512);
    CHECK(a.meta.attachedHeight == 512);
}

TEST_CASE("A saved node height wins over the surface meta when both are present",
          "[core][sakura][flowgraph][node-view]") {
    auto config = MakeConfig(MakeBlock("saved-block"));
    config.block.surfaces.push_back(AttachedSurface("saved-block:surface:out"));
    NodeSession session(config, 1.0f, {.x = 10.0f, .y = 10.0f, .width = 220.0f, .height = 400.0f});
    auto& surface = session.track("saved-block:surface:out", 512);
    session.run();
    CHECK(session.meta.height == Catch::Approx(400.0f).margin(1.0f));
    REQUIRE(surface.resizes.size() == 1);
    CHECK(surface.last() > 300);
    CHECK(surface.last() < 400);
}

TEST_CASE("Metrics consume fixed height and the surface allocation follows",
          "[core][sakura][flowgraph][node-view]") {
    auto config = MakeConfig(MakeBlock("metric-block"));
    config.block.surfaces.push_back(AttachedSurface("metric-block:surface:out"));
    NodeSession session(config);
    auto& surface = session.track("metric-block:surface:out", 512);
    session.run();
    REQUIRE_FALSE(surface.resizes.empty());
    const U64 withoutMetrics = surface.last();
    const F32 height = session.meta.height;

    session.config.block.metrics.push_back({
        .id = "metric-block:rate",
        .label = "Rate",
        .format = {{"type", "label"}},
        .value = std::string("2 MHz"),
    });
    session.run();
    CHECK(surface.last() + 5 < withoutMetrics);
    CHECK(session.meta.height == Catch::Approx(height).margin(1.0f));

    session.config.block.metrics.clear();
    session.run();
    CHECK(surface.last() == withoutMetrics);
    CHECK(session.meta.height == Catch::Approx(height).margin(1.0f));
}

TEST_CASE("Incomplete note blocks already rest at their flexible minimum and keep it on completion",
          "[core][sakura][flowgraph][node-view]") {
    auto config = MakeConfig(MakeBlock("incomplete-note", "note"));
    config.block.configFields.push_back(EditorField("incomplete-note:text", "markdown"));
    config.block.state = Block::State::Incomplete;
    NodeSession session(config);
    session.run(3);
    CHECK(session.axes == NodeEditor::ResizeAxes::XY);
    CHECK(session.meta.height == Catch::Approx(kEditorMinimum).margin(1.0f));

    session.config.block.state = Block::State::Created;
    session.run();
    CHECK(session.meta.height == Catch::Approx(kEditorMinimum).margin(1.0f));
}

TEST_CASE("Adding and removing a flexible field reflows the node between seeded and auto height",
          "[core][sakura][flowgraph][node-view]") {
    auto config = MakeConfig(MakeBlock("reconfig-block"));
    NodeSession session(config);
    session.run(3);
    CHECK(session.meta.height == 0.0f);
    const F32 chrome = session.size().y;

    session.config.block.configFields.push_back(EditorField("reconfig-block:text", "markdown"));
    session.run();
    CHECK(session.axes == NodeEditor::ResizeAxes::XY);
    CHECK(session.meta.height == Catch::Approx(chrome + kFieldGap + kEditorMinimum).margin(1.0f));

    session.config.block.configFields.clear();
    session.run();
    CHECK(session.axes == NodeEditor::ResizeAxes::X);
    CHECK(session.meta.height == 0.0f);
    CHECK(session.size().y == Catch::Approx(chrome).margin(1.0f));
}

TEST_CASE("Shallow surface meta seeds the node at the surface minimum",
          "[core][sakura][flowgraph][node-view]") {
    auto config = MakeConfig(MakeBlock("shallow-block"));
    config.block.surfaces.push_back(AttachedSurface("shallow-block:surface:out"));
    NodeSession session(config);
    auto& surface = session.track("shallow-block:surface:out", 100);
    session.run();
    REQUIRE_FALSE(surface.resizes.empty());
    for (const auto& resize : surface.resizes) {
        CHECK(resize.logicalSize.y >= kSurfaceMinimum);
    }
    CHECK(surface.last() == kSurfaceMinimum);
    CHECK(session.meta.height > kSurfaceMinimum);
}

TEST_CASE("Creating nodes render at the loading height floor",
          "[core][sakura][flowgraph][node-view]") {
    auto config = MakeConfig(MakeBlock("creating-block"));
    config.block.state = Block::State::Creating;
    NodeSession session(config, 1.0f, {.x = 20.0f, .y = 20.0f, .width = 200.0f, .height = 0.0f});
    session.run(2);
    CHECK(session.size().y >= 96.0f);
    CHECK(session.meta.height == 0.0f);
}

TEST_CASE("Simple config grids reflow while resizing and collapse without reserving height",
          "[core][sakura][flowgraph][node-view]") {
    auto config = MakeConfig(MakeBlock("grid-block"));
    for (U64 index = 0; index < 4; ++index) {
        config.block.configFields.push_back(RangeField("grid-block:field:" + std::to_string(index)));
    }
    NodeSession session(config);
    session.run();
    CHECK(session.axes == NodeEditor::ResizeAxes::X);
    const auto narrow = session.size();

    session.resize({100.0f, 0.0f});
    const auto wide = session.size();
    CHECK(wide.x == Catch::Approx(narrow.x + 100.0f).margin(1.0f));
    CHECK(wide.y < narrow.y - 20.0f);
    CHECK(session.meta.height == 0.0f);

    session.configMeta.collapsed = true;
    session.run();
    CHECK(session.size().x == Catch::Approx(wide.x).margin(1.0f));
    CHECK(session.size().y < wide.y - 20.0f);
    CHECK(session.meta.height == 0.0f);

    session.configMeta.collapsed = false;
    session.run();
    CHECK(session.size().y == Catch::Approx(wide.y).margin(1.0f));

    session.resize({-100.0f, 0.0f});
    CHECK(session.size().x == Catch::Approx(narrow.x).margin(1.0f));
    CHECK(session.size().y == Catch::Approx(narrow.y).margin(1.0f));
}

TEST_CASE("Collapsed flexible config fields stop reserving node height and the vertical resize axis",
          "[core][sakura][flowgraph][node-view]") {
    auto config = MakeConfig(MakeBlock("collapse-editor"));
    config.block.configFields.push_back(EditorField("collapse-editor:code"));
    NodeSession session(config);
    session.run();

    SECTION("Default editor height") {}
    SECTION("User-resized editor height") {
        session.resize({0.0f, 180.0f});
    }
    SECTION("User-resized editor with a detached surface") {
        session.config.block.surfaces.push_back(AttachedSurface("collapse-editor:surface"));
        session.track("collapse-editor:surface").meta.detached = true;
        session.run();
        session.resize({0.0f, 180.0f});
    }

    const auto expanded = session.size();
    const F32 height = session.meta.height;
    CHECK(session.axes == NodeEditor::ResizeAxes::XY);
    CHECK(height == Catch::Approx(expanded.y).margin(1.0f));

    session.configMeta.collapsed = true;
    session.run();
    CHECK(session.axes == NodeEditor::ResizeAxes::X);
    CHECK(session.meta.height == Catch::Approx(height).margin(1.0f));
    CHECK(session.size().y < expanded.y - 100.0f);

    session.configMeta.collapsed = false;
    session.run();
    CHECK(session.axes == NodeEditor::ResizeAxes::XY);
    CHECK(session.size().y == Catch::Approx(expanded.y).margin(1.0f));
    session.resize({0.0f, 100.0f});
    CHECK(session.size().y == Catch::Approx(expanded.y + 100.0f).margin(1.0f));
}

TEST_CASE("Collapsed editor height survives width drags, layout feedback, and view recreation",
          "[core][sakura][flowgraph][node-view]") {
    auto config = MakeConfig(MakeBlock("collapse-meta"));
    config.block.configFields.push_back(EditorField("collapse-meta:code"));
    NodeMeta saved;
    ConfigMeta savedConfig;
    {
        NodeSession session(config, 1.0f, {.x = 20.0f, .y = 20.0f, .width = 220.0f, .height = 400.0f});
        session.run();
        REQUIRE(session.size().y == Catch::Approx(400.0f).margin(1.0f));

        for (U64 cycle = 0; cycle < 3; ++cycle) {
            CAPTURE(cycle);
            session.configMeta.collapsed = true;
            session.run();
            const auto collapsed = session.size();
            CHECK(collapsed.y < 300.0f);
            CHECK(session.axes == NodeEditor::ResizeAxes::X);
            CHECK(session.meta.height == Catch::Approx(400.0f).margin(1.0f));

            session.resize({60.0f, 0.0f});
            CHECK(session.size().x == Catch::Approx(collapsed.x + 60.0f).margin(1.0f));
            CHECK(session.meta.width == Catch::Approx(220.0f + 60.0f * static_cast<F32>(cycle + 1)).margin(1.0f));
            CHECK(session.meta.height == Catch::Approx(400.0f).margin(1.0f));

            session.configMeta.collapsed = false;
            session.run();
            CHECK(session.size().y == Catch::Approx(400.0f).margin(1.0f));
        }

        session.configMeta.collapsed = true;
        session.run();
        saved = session.meta;
        savedConfig = session.configMeta;
    }

    NodeSession restored(config, 1.0f, saved, savedConfig);
    restored.run();
    CHECK(restored.size().y < 300.0f);
    CHECK(restored.meta.height == Catch::Approx(400.0f).margin(1.0f));
    restored.configMeta.collapsed = false;
    restored.run();
    CHECK(restored.size().y == Catch::Approx(400.0f).margin(1.0f));
    CHECK(restored.axes == NodeEditor::ResizeAxes::XY);
}

TEST_CASE("Removing a collapsed editor releases its saved height",
          "[core][sakura][flowgraph][node-view]") {
    auto config = MakeConfig(MakeBlock("removed-editor"));
    config.block.configFields.push_back(EditorField("removed-editor:code"));
    NodeSession session(config, 1.0f, {.x = 20.0f, .y = 20.0f, .width = 220.0f, .height = 400.0f}, {.collapsed = true});
    session.run();
    CHECK(session.meta.height == Catch::Approx(400.0f).margin(1.0f));

    session.config.block.configFields.clear();
    SECTION("Field removed") {}
    SECTION("Field replaced with a fixed-height control") {
        session.config.block.configFields.push_back(RangeField("removed-editor:value"));
    }
    session.run();
    CHECK(session.axes == NodeEditor::ResizeAxes::X);
    CHECK(session.size().y < 100.0f);
    CHECK(session.meta.height == 0.0f);

    session.config.block.configFields = {EditorField("removed-editor:new")};
    session.configMeta.collapsed = false;
    session.run();
    CHECK(session.axes == NodeEditor::ResizeAxes::XY);
    CHECK(session.meta.height < 300.0f);
}

TEST_CASE("Collapsed editors keep their saved height across lifecycle states and display scales",
          "[core][sakura][flowgraph][node-view]") {
    const F32 scale = GENERATE(1.0f, 2.0f);
    const std::string format = GENERATE(std::string("python"), std::string("markdown"));
    CAPTURE(scale, format);
    auto config = MakeConfig(MakeBlock("lifecycle-editor"));
    const auto field = EditorField("lifecycle-editor:code", format);
    config.block.configFields.push_back(field);
    NodeSession session(config, scale, {.x = 20.0f, .y = 20.0f, .width = 220.0f, .height = 400.0f});
    session.run();
    const auto expanded = session.size();
    REQUIRE(expanded.y == Catch::Approx(400.0f).margin(1.0f));

    session.configMeta.collapsed = true;
    session.run();
    for (const auto state : {Block::State::Incomplete, Block::State::Errored, Block::State::Creating,
                             Block::State::Created}) {
        CAPTURE(state);
        session.config.block.state = state;
        session.config.block.configFields = state == Block::State::Creating
            ? std::vector<FlowgraphConfigFieldConfig>{}
            : std::vector<FlowgraphConfigFieldConfig>{field};
        session.run(4);
        CHECK(session.meta.height == Catch::Approx(400.0f).margin(1.0f));
        CHECK(session.axes == NodeEditor::ResizeAxes::X);
    }

    session.configMeta.collapsed = false;
    session.run();
    CHECK(session.size().y == Catch::Approx(expanded.y).margin(1.0f));
    CHECK(session.axes == NodeEditor::ResizeAxes::XY);
}

TEST_CASE("Cold collapsed editors do not reserve height from attached surfaces",
          "[core][sakura][flowgraph][node-view]") {
    auto config = MakeConfig(MakeBlock("collapsed-surface"));
    config.block.configFields.push_back(EditorField("collapsed-surface:code"));
    config.block.surfaces.push_back(AttachedSurface("collapsed-surface:surface"));
    NodeSession session(config, 1.0f, {}, {.collapsed = true});
    auto& surface = session.track("collapsed-surface:surface", 400);
    session.run();
    REQUIRE_FALSE(surface.resizes.empty());
    for (const auto& resize : surface.resizes) {
        CHECK(resize.logicalSize.y == 400);
    }
    const auto attached = session.size();
    CHECK(session.axes == NodeEditor::ResizeAxes::XY);

    const U64 resizes = surface.resizes.size();
    surface.meta.detached = true;
    session.run();
    CHECK(session.size().y < attached.y - 100.0f);
    CHECK(session.axes == NodeEditor::ResizeAxes::X);
    CHECK(surface.resizes.size() == resizes);

    surface.meta.detached = false;
    session.run();
    CHECK(session.size().y == Catch::Approx(attached.y).margin(1.0f));
    CHECK(session.axes == NodeEditor::ResizeAxes::XY);
    CHECK(surface.last() == 400);
}

TEST_CASE("Chevron clicks keep the resized layout through movement and serialized restoration",
          "[core][sakura][flowgraph][node-view]") {
    const F32 scale = GENERATE(1.0f, 2.0f);
    const std::string format = GENERATE(std::string("python"), std::string("markdown"));
    CAPTURE(scale, format);
    auto config = MakeConfig(MakeBlock("gesture-editor"));
    config.block.configFields.push_back(EditorField("gesture-editor:code", format));
    Parser::Map saved;
    Parser::Map savedConfig;
    NodeMeta expected;
    {
        NodeSession session(config, scale);
        session.run();
        CHECK(session.collapsible);
        const auto initial = session.size();
        session.resize({80.0f * scale, 180.0f * scale});
        const auto expanded = session.size();
        REQUIRE(expanded.x == Catch::Approx(initial.x + 80.0f).margin(1.0f));
        REQUIRE(expanded.y == Catch::Approx(initial.y + 180.0f).margin(1.0f));
        const F32 height = session.meta.height;
        REQUIRE(height > 250.0f);

        for (U64 cycle = 0; cycle < 2; ++cycle) {
            CAPTURE(cycle);
            const U64 start = session.layouts.size();
            session.gesture(session.chevron());
            REQUIRE(session.configMeta.collapsed);
            REQUIRE(session.toggles.size() == cycle * 2 + 1);
            CHECK(session.toggles.back());
            const auto collapsed = session.size();
            CHECK(collapsed.y < expanded.y - 100.0f);

            session.resize({40.0f * scale, 60.0f * scale});
            CHECK(session.size().x == Catch::Approx(collapsed.x + 40.0f).margin(1.0f));
            CHECK(session.size().y == Catch::Approx(collapsed.y).margin(1.0f));

            const F32 x = session.meta.x;
            const F32 y = session.meta.y;
            session.gesture(session.header(), {30.0f * scale, 20.0f * scale});
            CHECK(session.meta.x == Catch::Approx(x + 30.0f).margin(1.0f));
            CHECK(session.meta.y == Catch::Approx(y + 20.0f).margin(1.0f));

            for (U64 index = start; index < session.layouts.size(); ++index) {
                CHECK(session.layouts[index].height == Catch::Approx(height).margin(1.0f));
            }
            session.gesture(session.chevron());
            REQUIRE_FALSE(session.configMeta.collapsed);
            REQUIRE(session.toggles.size() == (cycle + 1) * 2);
            CHECK_FALSE(session.toggles.back());
            CHECK(session.size().y == Catch::Approx(expanded.y).margin(1.0f));
        }

        session.gesture(session.chevron());
        const U64 events = session.layouts.size();
        session.run(12);
        CHECK(session.layouts.size() == events);
        expected = session.meta;
        REQUIRE(expected.serialize(saved) == Result::SUCCESS);
        REQUIRE(session.configMeta.serialize(savedConfig) == Result::SUCCESS);
    }

    NodeMeta restoredMeta;
    REQUIRE(restoredMeta.deserialize(saved) == Result::SUCCESS);
    ConfigMeta restoredConfig;
    REQUIRE(restoredConfig.deserialize(savedConfig) == Result::SUCCESS);
    NodeSession session(config, scale, restoredMeta, restoredConfig);
    session.run();
    REQUIRE(session.configMeta.collapsed);
    CHECK(session.meta.height == Catch::Approx(expected.height).margin(1.0f));
    session.gesture(session.chevron());
    REQUIRE_FALSE(session.configMeta.collapsed);
    CHECK(session.meta.x == Catch::Approx(expected.x).margin(1.0f));
    CHECK(session.meta.y == Catch::Approx(expected.y).margin(1.0f));
    CHECK(session.meta.width == Catch::Approx(expected.width).margin(1.0f));
    CHECK(session.size().y == Catch::Approx(expected.height).margin(1.0f));
}

TEST_CASE("Chevron gestures do not toggle on cancelled clicks or node drags",
          "[core][sakura][flowgraph][node-view]") {
    auto config = MakeConfig(MakeBlock("cancel-chevron"));
    config.block.configFields.push_back(EditorField("cancel-chevron:code"));
    NodeSession session(config);
    session.run();

    const auto chevron = session.chevron();
    session.send(MouseEventType::Move, chevron);
    session.tick();
    session.send(MouseEventType::Click, chevron);
    session.tick();
    session.send(MouseEventType::Cancel, chevron);
    session.tick();
    session.send(MouseEventType::Release, chevron);
    session.send(MouseEventType::Move, session.away());
    session.run();
    CHECK(session.toggles.empty());
    CHECK_FALSE(session.configMeta.collapsed);

    const F32 x = session.meta.x;
    session.gesture(session.chevron(), {60.0f, 0.0f});
    CHECK(session.meta.x == Catch::Approx(x + 60.0f).margin(1.0f));
    CHECK(session.toggles.empty());
    CHECK_FALSE(session.configMeta.collapsed);

    session.gesture(session.chevron());
    CHECK(session.toggles == std::vector<bool>{true});
    CHECK(session.configMeta.collapsed);

    session.config.block.configFields.clear();
    session.run();
    CHECK_FALSE(session.collapsible);
    session.gesture(session.chevron());
    CHECK(session.toggles == std::vector<bool>{true});
}

TEST_CASE("Collapsing config keeps ports and metrics in the node layout",
          "[core][sakura][flowgraph][node-view]") {
    auto config = MakeConfig(MakeBlock("port-editor"));
    config.block.inputs = {
        {.port = {.id = "port-editor:in-a", .label = "Input A"}},
        {.port = {.id = "port-editor:in-b", .label = "Input B"}},
    };
    config.block.outputs = {{.port = {.id = "port-editor:out", .label = "Output"}}};
    config.block.metrics.push_back({
        .id = "port-editor:rate",
        .label = "Rate",
        .format = {{"type", "label"}},
        .value = std::string("48 kHz"),
    });
    config.block.configFields = {
        EditorField("port-editor:code"),
        {.id = "port-editor:text", .name = "text", .format = {{"type", "multiline"}},
         .values = {{"text", std::string("fixed-height text")}}},
    };
    auto portless = config;
    portless.block.inputs.clear();
    portless.block.outputs.clear();
    portless.block.metrics.clear();
    const F32 chrome = ChromeHeight(portless);

    NodeSession session(config);
    session.run();
    session.resize({0.0f, 180.0f});
    const auto expanded = session.size();
    const F32 height = session.meta.height;
    CHECK(session.pins == 3);

    session.gesture(session.chevron());
    REQUIRE(session.configMeta.collapsed);
    CHECK(session.pins == 3);
    const auto collapsed = session.size();
    CHECK(collapsed.y < expanded.y - 100.0f);
    CHECK(collapsed.y > chrome + 3.0f * kNodeFont);
    CHECK(session.meta.height == Catch::Approx(height).margin(1.0f));

    session.config.block.metrics.clear();
    session.run();
    CHECK(session.size().y < collapsed.y);
    CHECK(session.size().y > chrome + 3.0f * kNodeFont);
    session.config.block.metrics = config.block.metrics;
    session.run();
    CHECK(session.size().y == Catch::Approx(collapsed.y).margin(1.0f));

    session.gesture(session.chevron());
    REQUIRE_FALSE(session.configMeta.collapsed);
    CHECK(session.pins == 3);
    CHECK(session.size().y == Catch::Approx(expanded.y).margin(1.0f));
}

TEST_CASE("Config grids reflow beside an attached surface and collapsing hands their height to it",
          "[core][sakura][flowgraph][node-view]") {
    auto config = MakeConfig(MakeBlock("grid-surface"));
    for (U64 index = 0; index < 4; ++index) {
        config.block.configFields.push_back(RangeField("grid-surface:field:" + std::to_string(index)));
    }
    config.block.surfaces.push_back(AttachedSurface("grid-surface:surface"));
    NodeSession session(config);
    auto& surface = session.track("grid-surface:surface", 400);
    session.run(7);
    REQUIRE_FALSE(surface.resizes.empty());
    for (const auto& resize : surface.resizes) {
        CHECK(resize.logicalSize.y == 400);
    }
    const auto narrow = session.size();

    session.resize({100.0f, 0.0f});
    CHECK(session.size().x == Catch::Approx(narrow.x + 100.0f).margin(1.0f));
    CHECK(session.size().y == Catch::Approx(narrow.y).margin(1.0f));
    CHECK(surface.last() > 400);
    const U64 expandedSurface = surface.last();
    const U64 resizes = surface.resizes.size();
    session.run();
    CHECK(surface.resizes.size() == resizes);

    session.configMeta.collapsed = true;
    session.run();
    CHECK(session.axes == NodeEditor::ResizeAxes::XY);
    CHECK(surface.last() > expandedSurface + 20);
    CHECK(session.size().y == Catch::Approx(narrow.y).margin(1.0f));

    session.configMeta.collapsed = false;
    session.run();
    CHECK(surface.last() == expandedSurface);
    CHECK(session.size().y == Catch::Approx(narrow.y).margin(1.0f));
    for (const auto& resize : surface.resizes) {
        CHECK(resize.logicalSize.y >= kSurfaceMinimum);
    }
}

TEST_CASE("Detaching the config hides its fields and chevron while keeping the editor height",
          "[core][sakura][flowgraph][node-view]") {
    auto config = MakeConfig(MakeBlock("detached-config"));
    config.block.configFields.push_back(EditorField("detached-config:code"));
    NodeSession session(config, 1.0f, {.x = 20.0f, .y = 20.0f, .width = 220.0f, .height = 400.0f});
    session.run();
    CHECK(session.collapsible);
    CHECK(session.axes == NodeEditor::ResizeAxes::XY);

    session.config.block.configDetached = true;
    session.run();
    CHECK_FALSE(session.collapsible);
    CHECK(session.axes == NodeEditor::ResizeAxes::X);
    CHECK(session.size().y < 300.0f);
    CHECK(session.meta.height == Catch::Approx(400.0f).margin(1.0f));

    session.config.block.configDetached = false;
    session.run();
    CHECK(session.collapsible);
    CHECK(session.axes == NodeEditor::ResizeAxes::XY);
    CHECK(session.size().y == Catch::Approx(400.0f).margin(1.0f));
}

TEST_CASE("A height drag beside a flexible editor hands all growth to the surface in a single emission",
          "[core][sakura][flowgraph][node-view]") {
    const F32 scale = GENERATE(1.0f, 2.0f);
    const std::string format = GENERATE(std::string("python"), std::string("markdown"));
    CAPTURE(scale, format);
    auto config = MakeConfig(MakeBlock("flex-surface"));
    config.block.configFields.push_back(EditorField("flex-surface:code", format));
    config.block.surfaces.push_back(AttachedSurface("flex-surface:surface"));
    NodeSession session(config, scale);
    auto& surface = session.track("flex-surface:surface", 400);
    session.run();
    REQUIRE(surface.resizes.size() == 1);
    REQUIRE(surface.last() == 400);
    const F32 height = session.size().y;

    session.resize({0.0f, 100.0f * scale});
    CHECK(surface.resizes.size() == 2);
    CHECK(surface.last() == 500);
    CHECK(surface.meta.attachedHeight == 500);
    CHECK(session.size().y == Catch::Approx(height + 100.0f).margin(1.0f));
    CHECK(session.meta.height == Catch::Approx(height + 100.0f).margin(1.0f));

    session.run(12);
    CHECK(surface.resizes.size() == 2);
    CHECK(surface.last() == 500);
    CHECK(session.size().y == Catch::Approx(height + 100.0f).margin(1.0f));

    session.resize({0.0f, -100.0f * scale});
    CHECK(surface.resizes.size() == 3);
    CHECK(surface.last() == 400);
    session.run(12);
    CHECK(surface.resizes.size() == 3);
}

TEST_CASE("Config grids beside a flexible editor and a surface reach one path-independent allocation",
          "[core][sakura][flowgraph][node-view]") {
    auto config = MakeConfig(MakeBlock("grid-flex-surface"));
    for (U64 index = 0; index < 4; ++index) {
        config.block.configFields.push_back(RangeField("grid-flex-surface:field:" + std::to_string(index)));
        if (index == 1) {
            config.block.configFields.push_back(EditorField("grid-flex-surface:text", "markdown"));
        }
    }
    config.block.surfaces.push_back(AttachedSurface("grid-flex-surface:surface"));
    NodeSession session(config);
    auto& surface = session.track("grid-flex-surface:surface", 400);
    session.run(7);
    REQUIRE(surface.resizes.size() == 1);
    REQUIRE(surface.last() == 400);
    const auto narrow = session.size();

    session.resize({100.0f, 0.0f});
    CHECK(session.size().x == Catch::Approx(narrow.x + 100.0f).margin(1.0f));
    CHECK(session.size().y == Catch::Approx(narrow.y).margin(1.0f));
    CHECK(surface.resizes.size() == 2);
    CHECK(surface.last() > 400);
    const U64 expandedSurface = surface.last();
    session.run(12);
    CHECK(surface.resizes.size() == 2);

    session.configMeta.collapsed = true;
    session.run();
    CHECK(session.axes == NodeEditor::ResizeAxes::XY);
    CHECK(surface.resizes.size() == 3);
    CHECK(surface.last() > expandedSurface + 120);
    CHECK(session.size().y == Catch::Approx(narrow.y).margin(1.0f));

    session.configMeta.collapsed = false;
    session.run();
    CHECK(surface.resizes.size() == 4);
    CHECK(surface.last() == expandedSurface);
    CHECK(session.size().y == Catch::Approx(narrow.y).margin(1.0f));

    session.resize({-100.0f, 0.0f});
    CHECK(surface.resizes.size() == 5);
    CHECK(surface.last() == 400);
    for (const auto& resize : surface.resizes) {
        CHECK(resize.logicalSize.y >= kSurfaceMinimum);
    }
}

TEST_CASE("A collapsed node being recreated stays at its collapsed height and keeps the hidden height",
          "[core][sakura][flowgraph][node-view]") {
    const F32 scale = GENERATE(1.0f, 2.0f);
    const std::string format = GENERATE(std::string("python"), std::string("markdown"));
    CAPTURE(scale, format);
    auto config = MakeConfig(MakeBlock("recreated-editor"));
    const auto field = EditorField("recreated-editor:code", format);
    config.block.configFields.push_back(field);
    NodeSession session(config, scale, {.x = 20.0f, .y = 20.0f, .width = 220.0f, .height = 400.0f},
                        {.collapsed = true});
    session.run();
    const F32 collapsed = session.size().y;
    REQUIRE(collapsed < 100.0f);
    CHECK(session.meta.height == Catch::Approx(400.0f).margin(1.0f));

    session.config.block.state = Block::State::Creating;
    session.config.block.configFields.clear();
    for (U64 frame = 0; frame < 4; ++frame) {
        CAPTURE(frame);
        session.tick();
        CHECK(session.size().y < 300.0f);
        CHECK(session.meta.height == Catch::Approx(400.0f).margin(1.0f));
    }
    CHECK(session.size().y >= 96.0f);

    session.config.block.state = Block::State::Created;
    session.config.block.configFields = {field};
    session.run();
    CHECK(session.size().y == Catch::Approx(collapsed).margin(1.0f));
    CHECK(session.meta.height == Catch::Approx(400.0f).margin(1.0f));

    session.configMeta.collapsed = false;
    session.run();
    CHECK(session.size().y == Catch::Approx(400.0f).margin(1.0f));
    CHECK(session.axes == NodeEditor::ResizeAxes::XY);
}
