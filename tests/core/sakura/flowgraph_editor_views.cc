#include <catch2/catch_test_macros.hpp>

#include "fonts.hh"
#include "component_harness.hh"
#include "typist.hh"

#include "compositor/default/presenters/flowgraph/editor/surface.hh"
#include "compositor/default/presenters/flowgraph/surface.hh"
#include "compositor/default/views/components/modal_header.hh"
#include "compositor/default/views/flowgraph/detached_config.hh"
#include "compositor/default/views/flowgraph/editor/base.hh"
#include "compositor/default/views/flowgraph/editor/inspect.hh"
#include "compositor/default/views/flowgraph/editor/menu.hh"
#include "compositor/default/views/flowgraph/editor/link.hh"
#include "compositor/default/views/flowgraph/editor/node.hh"
#include "compositor/default/views/flowgraph/surface.hh"
#include "flowgraph_fixture.hh"
#include "scene.hh"

#include <jetstream/render/sakura/components/node/content_stack.hh>
#include <jetstream/render/sakura/components/text_grid.hh>

#include <any>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

using namespace Jetstream;
using namespace SakuraTest;

namespace {


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

F32 RowY(U64 entries, U64 separators) {
    return 119.5f + 23.0f * static_cast<F32>(entries) + 9.0f * static_cast<F32>(separators);
}

constexpr F32 kMenuX = 120.0f;
const F32 kCopyY = RowY(0, 0);
const F32 kPasteY = RowY(1, 0);
const F32 kRenameY = RowY(2, 0);
const F32 kInspectY = RowY(3, 0);
const F32 kDetachY = RowY(4, 0);
const F32 kPlotY = RowY(5, 0);
const F32 kDeviceY = RowY(6, 1);
const F32 kReloadY = RowY(7, 1);
const F32 kDocumentationY = RowY(8, 2);
const F32 kDeleteY = RowY(9, 3);

template<typename Update>
bool Settles(Frames& frames, Update update) {
    if (!frames.settle(update)) {
        return false;
    }
    for (U64 frame = 0; frame < 3; ++frame) {
        if (!frames.run(update).clean()) {
            return false;
        }
    }
    return true;
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

struct MenuRig {
    FlowgraphNodeMenu menu;
    LayerSlot slot;
    LayerHost host;
    std::vector<std::string> calls;
    bool pasteEnabled = true;
    bool configDetachEnabled = true;
    bool configDetached = false;
    bool plotDetachEnabled = true;
    bool plotDetached = false;
    std::vector<FlowgraphNodeMenu::DeviceOption> devices = {{.label = "CPU", .selected = true}, {.label = "Metal"}};

    MenuRig() {
        slot.hold(menu.component(), {100.0f, 100.0f, 1.0f, 1.0f});
        host.setLayers({&slot});
    }

    std::function<void()> record(const std::string& name) {
        return [this, name] {
            calls.push_back(name);
        };
    }

    void show() {
        menu.update({
            .id = "fft:context-menu",
            .open = true,
            .pasteEnabled = pasteEnabled,
            .configDetachEnabled = configDetachEnabled,
            .configDetached = configDetached,
            .plotDetachEnabled = plotDetachEnabled,
            .plotDetached = plotDetached,
            .devices = devices,
            .onCopy = record("copy"),
            .onPaste = record("paste"),
            .onRename = record("rename"),
            .onInspect = record("inspect"),
            .onToggleConfigDetach = record("detach"),
            .onTogglePlotDetach = record("plot"),
            .onReload = record("reload"),
            .onDelete = record("delete"),
            .onDocumentation = record("documentation"),
            .onDeviceSelect = [this](U64 index) {
                calls.push_back("device:" + std::to_string(index));
            },
            .onClose = record("close"),
        });
        host.place(MakeContext(), kScreen);
    }

    bool openAt(F32 y) {
        return host.popups.interactiveAt({kMenuX, y});
    }
};

FlowgraphNode::Config NodeConfig() {
    FlowgraphNode::Config config;
    config.id = "fft";
    config.inspectorId = "graph:fft:inspector";
    config.block.name = "fft";
    config.block.module = "fft";
    config.block.title = "FFT";
    config.block.state = Block::State::Created;
    config.block.config = {{"size", std::string("1024")}};
    return config;
}

struct NodeMenuRig {
    FlowgraphNode node;
    LayerSlot slot;
    LayerHost host;
    U64 documentation = 0;
    std::vector<FlowgraphNode::Surface> surfaces;

    NodeMenuRig() {
        node.update(config());
        slot.hold(node.menuComponent(), {100.0f, 100.0f, 1.0f, 1.0f});
        host.setLayers({&slot});
    }

    FlowgraphNode::Config config() {
        auto config = NodeConfig();
        config.block.surfaces = surfaces;
        config.onDocumentation = [this] {
            ++documentation;
        };
        return config;
    }

    void choose(F32 y) {
        node.takeNode().onContextMenu();
        node.update(config());
        host.place(MakeContext(), kScreen);
        REQUIRE(host.popups.interactiveAt({kMenuX, y}));
        host.click(kMenuX, y);
        node.update(config());
        host.place(MakeContext(), kScreen);
    }

    std::vector<Sakura::FloatingLayer::Window> windows() {
        std::vector<Sakura::FloatingLayer::Window> result;
        node.appendWindows(result);
        return result;
    }
};

struct WindowStage {
    Sakura::Layer base;
    Sakura::FloatingLayer floating;
    LayerHost host;

    WindowStage() {
        host.setLayers({&base, &floating});
    }

    void show(std::vector<Sakura::FloatingLayer::Window> windows, const Sakura::Context& ctx) {
        floating.update({.windows = std::move(windows)});
        host.place(ctx, kScreen);
    }
};

struct EditorRig {
    struct Created {
        std::string type;
        Extent2D<F32> position;
    };

    Scene scene;
    FlowgraphEditor editor;
    LayerSlot slot;
    LayerHost host;
    std::vector<Created> created;
    std::vector<FlowgraphNode::BlockData> graph;
    U64 catalogs = 0;

    EditorRig() {
        slot.hold(editor.component(), kScreen);
        host.setLayers({&slot});
        frame();
    }

    void frame(std::optional<Extent2D<F32>> mouse = std::nullopt) {
        editor.update({
            .id = "graph",
            .graph = graph,
            .onBuildBlockOptions = [this]() {
                ++catalogs;
                return std::vector<FlowgraphBlockPicker::BlockOption>{
                    {.type = "fft", .title = "FFT", .category = "DSP", .devices = {{.device = DeviceType::CPU}}},
                };
            },
            .onCreateBlock = [this](const std::string& type, std::optional<Extent2D<F32>> position, DeviceType,
                                    RuntimeType, ProviderType) {
                REQUIRE(position.has_value());
                created.push_back({type, *position});
            },
        });
        auto ctx = scene.ctx;
        ctx.mouse = mouse;
        host.place(ctx, kScreen);
    }

    void doubleClick(F32 x, F32 y) {
        host.send(Mouse(MouseEventType::Click, x, y, MouseButton::Left, 2));
        host.send(Mouse(MouseEventType::Release, x, y));
    }
};

struct SurfaceModuleConfig : Module::Config {
    JST_MODULE_TYPE(flowgraph_editor_views_surface)

    Result serialize(Parser::Map&) const override { return Result::SUCCESS; }
    Result deserialize(const Parser::Map&) override { return Result::SUCCESS; }
    std::size_t hash() const override { return 0; }
};

struct SurfaceModule : Module::Impl,
                       DynamicConfig<SurfaceModuleConfig>,
                       NativeCpuRuntimeContext,
                       Scheduler::Context {
    Result define() override {
        return defineTaint(Module::Taint::SURFACE);
    }

    Result create() override {
        return surfaceCreateManifest({
            .id = "default",
            .size = {64, 64},
            .surface = std::make_shared<SurfaceTexture>(),
        });
    }

    Result computeSubmit() override {
        return Result::SUCCESS;
    }
};

struct SurfaceRig : FlowgraphFixture {
    Scene scene;
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    PresenterContext context{state, callbacks};
    std::vector<Mail> mails;
    std::shared_ptr<Module> module;

    SurfaceRig() {
        state.flowgraph.items["graph"] = std::shared_ptr<Flowgraph>(flowgraph.get(), [](Flowgraph*) {});
        callbacks.enqueueMail = [this](Mail&& mail) {
            mails.push_back(std::move(mail));
        };
        module = TestFlowgraph::Detail::BuildModule<SurfaceModule>(DeviceType::CPU, RuntimeType::NATIVE, "generic",
                                                                     nullptr, nullptr);
        REQUIRE(module->create("view", Parser::Map{}, {}, nullptr) == Result::SUCCESS);
        REQUIRE(flowgraph->blockCreate("source", TestFlowgraph::kSyntheticSourceType, {}, {}) == Result::SUCCESS);
    }

    ~SurfaceRig() {
        CHECK(module->destroy() == Result::SUCCESS);
    }

    std::vector<FlowgraphDetachedSurface::Config> build(bool collapsed) {
        const SurfaceMeta meta{.detached = true, .configCollapsed = collapsed};
        REQUIRE(flowgraph->metadata().set("surface_default", meta, "source") == Result::SUCCESS);
        auto data = ViewBlock(*flowgraph, "source");
        data.surfaces = {module->surface()};
        state.flowgraph.blocks["graph"] = {{.name = "source", .data = data}};
        return FlowgraphDetachedSurfacePresenter(context).build("graph", state.flowgraph.items.at("graph"));
    }
};

}  // namespace

TEST_CASE("Node menu entries close the menu and then run their action",
          "[core][sakura][flowgraph][editor-views]") {
    MenuRig rig;
    const std::vector<std::pair<F32, std::string>> entries = {
        {kCopyY, "copy"},
        {kPasteY, "paste"},
        {kRenameY, "rename"},
        {kInspectY, "inspect"},
        {kDetachY, "detach"},
        {kPlotY, "plot"},
        {kReloadY, "reload"},
        {kDocumentationY, "documentation"},
        {kDeleteY, "delete"},
    };
    for (const auto& [y, name] : entries) {
        CAPTURE(name);
        rig.calls.clear();
        rig.show();
        REQUIRE(rig.openAt(y));
        CHECK(rig.host.click(kMenuX, y));
        CHECK(rig.calls == std::vector<std::string>{"close", name});
    }
}

TEST_CASE("Node menu opens at the cursor and closes on outside clicks without running anything",
          "[core][sakura][flowgraph][editor-views]") {
    MenuRig rig;
    rig.show();
    CHECK(rig.openAt(kCopyY));
    CHECK(rig.openAt(kDeleteY));
    CHECK_FALSE(rig.host.popups.interactiveAt({kMenuX, 90.0f}));

    CHECK(rig.host.click(600.0f, 500.0f));
    CHECK(rig.calls == std::vector<std::string>{"close"});
}

TEST_CASE("Node menu paste, detach, and device entries follow their enabled flags",
          "[core][sakura][flowgraph][editor-views]") {
    MenuRig rig;
    rig.pasteEnabled = false;
    rig.configDetachEnabled = false;
    rig.plotDetachEnabled = false;
    rig.devices.clear();
    rig.show();

    rig.host.click(kMenuX, kPasteY);
    rig.host.click(kMenuX, kDetachY);
    rig.host.click(kMenuX, kPlotY);
    rig.host.click(kMenuX, kDeviceY);
    rig.host.place(MakeContext(), kScreen);
    CHECK(rig.calls.empty());
    CHECK(rig.openAt(kCopyY));
    CHECK_FALSE(rig.host.popups.interactiveAt({170.0f, kDeviceY}));

    rig.configDetached = true;
    rig.configDetachEnabled = true;
    rig.show();
    rig.host.click(kMenuX, kDetachY);
    CHECK(rig.calls == std::vector<std::string>{"close", "detach"});

    rig.calls.clear();
    rig.plotDetached = true;
    rig.plotDetachEnabled = true;
    rig.show();
    rig.host.click(kMenuX, kPlotY);
    CHECK(rig.calls == std::vector<std::string>{"close", "plot"});
}

TEST_CASE("Node menu device submenu reports the index of the chosen device",
          "[core][sakura][flowgraph][editor-views]") {
    MenuRig rig;
    rig.show();
    CHECK_FALSE(rig.host.popups.interactiveAt({170.0f, kDeviceY}));

    rig.host.click(kMenuX, kDeviceY);
    CHECK(rig.calls.empty());
    rig.host.place(MakeContext(), kScreen);
    REQUIRE(rig.host.popups.interactiveAt({170.0f, kDeviceY}));
    REQUIRE(rig.host.popups.interactiveAt({170.0f, kDeviceY + 23.0f}));

    rig.host.click(170.0f, kDeviceY + 23.0f);
    CHECK(rig.calls == std::vector<std::string>{"close", "device:1"});
}

TEST_CASE("Node menu settles under identical updates", "[core][sakura][flowgraph][editor-views]") {
    MenuRig rig;
    Frames frames(rig.host, kScreen, MakeContext());
    CHECK(Settles(frames, [&] { rig.show(); }));
}

TEST_CASE("Node menu documentation entry asks for the block documentation without opening a window",
          "[core][sakura][flowgraph][editor-views]") {
    Scene scene;
    NodeMenuRig rig;
    rig.choose(kDocumentationY);
    CHECK(rig.documentation == 1);
    CHECK(rig.windows().empty());

    rig.choose(kDocumentationY);
    CHECK(rig.documentation == 2);
    CHECK(rig.windows().empty());
}

TEST_CASE("Node menu inspect entry opens the inspector window that its close button removes",
          "[core][sakura][flowgraph][editor-views]") {
    Scene scene;
    NodeMenuRig rig;
    rig.choose(kInspectY);
    auto windows = rig.windows();
    REQUIRE(windows.size() == 1);
    CHECK(windows[0].id == "graph:fft:inspector");
    CHECK(windows[0].title == "Inspect & Edit (fft)");
    CHECK(windows[0].size == Extent2D<F32>{640.0f, 520.0f});
    CHECK(windows[0].padding == 0.0f);
    REQUIRE(windows[0].content != nullptr);

    WindowStage stage;
    stage.show(windows, scene.ctx);
    CHECK(Peek::Frame(*windows[0].content) == Jetstream::Rect{80.0f, 72.0f, 640.0f, 488.0f});

    CHECK(stage.host.click(707.0f, 52.0f));
    rig.node.update(rig.config());
    CHECK(rig.windows().empty());

    rig.choose(kDocumentationY);
    rig.choose(kInspectY);
    windows = rig.windows();
    REQUIRE(windows.size() == 1);
    CHECK(windows[0].id == "graph:fft:inspector");
}

TEST_CASE("Inspector applies edited YAML on submit and rejects unchanged or invalid documents",
          "[core][sakura][flowgraph][editor-views]") {
    Typist typist;
    std::vector<Parser::Map> applied;
    U64 closed = 0;
    FlowgraphNodeInspector inspector;
    const FlowgraphNodeInspector::Config config{
        .id = "graph:fft:inspector",
        .name = "fft",
        .value = {{"mode", std::string("fast")}},
        .onApply = [&](Parser::Map value) { applied.push_back(std::move(value)); },
        .onClose = [&] { ++closed; },
    };
    inspector.update(config);
    inspector.open();
    WindowStage stage;
    typist.place = [&] {
        inspector.update(config);
        stage.show({inspector.window()}, typist.ctx);
    };
    typist.root = &stage.host;
    typist.frame();
    typist.frame();

    CHECK(stage.host.click(400.0f, 300.0f));
    typist.frame();
    REQUIRE(FocusOwner(stage.host) != nullptr);

    typist.press(KeyCode::Enter, {.control = true});
    CHECK(applied.empty());

    typist.press(KeyCode::A, {.control = true});
    typist.type("mode: a: b");
    typist.press(KeyCode::Enter, {.control = true});
    CHECK(applied.empty());

    typist.press(KeyCode::A, {.control = true});
    typist.type("mode: slow");
    typist.press(KeyCode::Enter, {.control = true});
    REQUIRE(applied.size() == 1);
    REQUIRE(applied[0].contains("mode"));
    CHECK(std::any_cast<std::string>(applied[0].at("mode")) == "slow");

    typist.press(KeyCode::Enter, {.control = true});
    CHECK(applied.size() == 1);
    CHECK(closed == 0);
}

TEST_CASE("Inspector windows settle under identical updates",
          "[core][sakura][flowgraph][editor-views]") {
    Scene scene;
    FlowgraphNodeInspector inspector;
    WindowStage stage;
    const auto update = [&] {
        inspector.update({.id = "inspector", .name = "fft", .value = {{"size", std::string("1024")}}});
        stage.floating.update({.windows = {inspector.window()}});
    };
    update();
    inspector.open();
    Frames frames(stage.host, kScreen, scene.ctx);
    CHECK(Settles(frames, update));
}

TEST_CASE("Detached surfaces without config fields show only the surface image",
          "[core][sakura][flowgraph][editor-views]") {
    Scene scene;
    U64 closed = 0;
    FlowgraphDetachedSurface surface;
    surface.update({
        .id = "surface:fft:default",
        .title = "FFT (fft)",
        .texture = std::make_shared<SurfaceTexture>(),
        .onClose = [&] { ++closed; },
    });
    Host host(surface.component());
    host.place(scene.ctx, {10.0f, 20.0f, 300.0f, 200.0f});

    CHECK(surface.title() == "FFT (fft)");
    CHECK(Find<Sakura::NodeFieldPanel>(surface.component()).empty());
    const auto images = Find<Sakura::SurfaceImage>(surface.component());
    REQUIRE(images.size() == 1);
    CHECK(Peek::Frame(*images[0]) == Jetstream::Rect{10.0f, 20.0f, 300.0f, 200.0f});

    const auto close = surface.closeCallback();
    REQUIRE(close);
    close();
    CHECK(closed == 1);
}

TEST_CASE("Detached surface header action toggles its config panel and reports the open state",
          "[core][sakura][flowgraph][editor-views]") {
    Scene scene;
    std::vector<bool> toggles;
    FlowgraphDetachedSurface surface;
    FlowgraphDetachedSurface::Config config{
        .id = "surface:fft:default",
        .title = "FFT (fft)",
        .configFields = {RangeField("fft:value")},
        .texture = std::make_shared<SurfaceTexture>(),
        .onToggleConfigOpen = [&](bool open) { toggles.push_back(open); },
    };
    const Jetstream::Rect bounds = {0.0f, 0.0f, 300.0f, 400.0f};
    Host host(surface.component());
    const auto place = [&](const FlowgraphDetachedSurface::Config& next) {
        surface.update(next);
        host.place(scene.ctx, bounds);
    };
    const auto image = [&] {
        return Peek::Frame(*Find<Sakura::SurfaceImage>(surface.component()).at(0));
    };

    place(config);
    CHECK(Find<Sakura::NodeFieldPanel>(surface.component()).empty());
    auto actions = surface.actions();
    REQUIRE(actions.size() == 1);
    CHECK(actions[0].icon == ICON_LC_SLIDERS_HORIZONTAL);
    CHECK_FALSE(actions[0].active);
    const Jetstream::Rect collapsed = image();
    CHECK(collapsed.y == 0.0f);
    CHECK(collapsed.bottom() == 400.0f);

    actions[0].onClick();
    CHECK(toggles == std::vector<bool>{true});
    place(config);
    REQUIRE(Find<Sakura::NodeFieldPanel>(surface.component()).size() == 1);
    CHECK(surface.actions()[0].active);
    CHECK(image().y > collapsed.y);
    CHECK(image().bottom() == 400.0f);

    place(config);
    CHECK(Find<Sakura::NodeFieldPanel>(surface.component()).size() == 1);

    surface.actions()[0].onClick();
    CHECK(toggles == std::vector<bool>{true, false});
    place(config);
    CHECK(Find<Sakura::NodeFieldPanel>(surface.component()).empty());
    CHECK(image() == collapsed);

    auto opened = config;
    opened.id = "surface:fft:other";
    opened.configOpen = true;
    place(opened);
    CHECK(Find<Sakura::NodeFieldPanel>(surface.component()).size() == 1);
    CHECK(toggles.size() == 2);

    auto bare = config;
    bare.configFields.clear();
    place(bare);
    CHECK(surface.actions().empty());
}

TEST_CASE("Detached surfaces settle under identical updates", "[core][sakura][flowgraph][editor-views]") {
    Scene scene;
    FlowgraphDetachedSurface surface;
    const auto texture = std::make_shared<SurfaceTexture>();
    Host host(surface.component());
    Frames frames(host, {0.0f, 0.0f, 300.0f, 400.0f}, scene.ctx);
    const auto update = [&] {
        surface.update({
            .id = "surface:fft:default",
            .title = "FFT (fft)",
            .configFields = {RangeField("fft:value")},
            .configOpen = true,
            .texture = texture,
        });
    };
    CHECK(Settles(frames, update));
    CHECK(Find<Sakura::NodeFieldPanel>(surface.component()).size() == 1);
}

TEST_CASE("Detached configs fill their tile with a field panel and keep the title and close callback",
          "[core][sakura][flowgraph][editor-views]") {
    Scene scene;
    U64 closed = 0;
    FlowgraphDetachedConfig view;
    FlowgraphDetachedConfig::Config config{
        .id = "config:fft",
        .title = "FFT Config (fft)",
        .configFields = {RangeField("fft:low"), RangeField("fft:high")},
        .onClose = [&] { ++closed; },
    };
    view.update(config);
    Host host(view.component());
    const Jetstream::Rect bounds = {20.0f, 30.0f, 300.0f, 400.0f};
    host.place(scene.ctx, bounds);

    CHECK(dynamic_cast<Sakura::NodeFieldPanel*>(&view.component()) != nullptr);
    CHECK(Peek::Frame(view.component()) == bounds);
    CHECK(Find<Sakura::NodeContentStack>(view.component()).size() == 1);
    const auto fields = Find<Sakura::NodeField>(view.component());
    REQUIRE(fields.size() == 2);
    for (auto* field : fields) {
        const auto frame = Peek::Frame(*field);
        CHECK(frame.x >= bounds.x);
        CHECK(frame.right() <= bounds.right());
        CHECK(frame.y >= bounds.y);
    }
    CHECK(view.title() == "FFT Config (fft)");
    const auto close = view.closeCallback();
    REQUIRE(close);
    close();
    CHECK(closed == 1);

    config.configFields = {};
    view.update(config);
    host.place(scene.ctx, bounds);
    CHECK(Find<Sakura::NodeField>(view.component()).empty());
    CHECK(Peek::Frame(view.component()) == bounds);
}

TEST_CASE("Detached configs settle under identical updates", "[core][sakura][flowgraph][editor-views]") {
    Scene scene;
    FlowgraphDetachedConfig view;
    Host host(view.component());
    Frames frames(host, {0.0f, 0.0f, 300.0f, 400.0f}, scene.ctx);
    const auto update = [&] {
        view.update({
            .id = "config:fft",
            .title = "FFT Config (fft)",
            .configFields = {RangeField("fft:low"), RangeField("fft:high")},
        });
    };
    CHECK(Settles(frames, update));
}

TEST_CASE("Modal headers stack the title, an optional description, and a spaced divider",
          "[core][sakura][flowgraph][editor-views]") {
    Scene scene;
    ModalHeader header;
    Host host(header);
    const Jetstream::Rect bounds = {10.0f, 10.0f, 400.0f, 300.0f};

    header.update({.title = "Rename Block", .description = "Pick a new name for the block."});
    host.place(scene.ctx, bounds);
    auto texts = Direct<Sakura::Text>(header);
    auto dividers = Direct<Sakura::Divider>(header);
    REQUIRE(texts.size() == 2);
    REQUIRE(dividers.size() == 1);
    const auto title = Peek::Frame(*texts[0]);
    const auto description = Peek::Frame(*texts[1]);
    const auto divider = Peek::Frame(*dividers[0]);
    CHECK(title.y == bounds.y);
    CHECK(description.y > title.bottom());
    CHECK(divider.y > description.bottom());
    CHECK(divider.height == 9.0f);
    const F32 described = host.measureOf(scene.ctx, {400.0f, kInfinite}).y;

    header.update({.title = "Rename Block"});
    host.place(scene.ctx, bounds);
    texts = Direct<Sakura::Text>(header);
    dividers = Direct<Sakura::Divider>(header);
    REQUIRE(texts.size() == 1);
    REQUIRE(dividers.size() == 1);
    CHECK(Peek::Frame(*dividers[0]).y > Peek::Frame(*texts[0]).bottom());
    const F32 plain = host.measureOf(scene.ctx, {400.0f, kInfinite}).y;
    CHECK(plain < described);

    header.update({.title = "Rename Block", .dividerSpacing = 12.0f});
    host.place(scene.ctx, bounds);
    CHECK(Peek::Frame(*Direct<Sakura::Divider>(header).at(0)).height == 25.0f);
    CHECK(host.measureOf(scene.ctx, {400.0f, kInfinite}).y == plain + 16.0f);
}

TEST_CASE("Modal headers settle under identical updates", "[core][sakura][flowgraph][editor-views]") {
    Scene scene;
    ModalHeader header;
    Host host(header);
    Frames frames(host, {0.0f, 0.0f, 400.0f, 300.0f}, scene.ctx);
    CHECK(Settles(frames, [&] {
        header.update({.title = "Rename Block", .description = "Pick a new name for the block."});
    }));
}

TEST_CASE("Double clicking empty canvas opens the block search at that grid point and the pill at the view center",
          "[core][sakura][flowgraph][editor-views]") {
    EditorRig rig;
    rig.doubleClick(40.0f, 60.0f);
    CHECK(rig.catalogs == 1);
    rig.frame(Extent2D<F32>{40.0f, 60.0f});
    REQUIRE(rig.host.popups.interactiveAt({50.0f, 70.0f}));
    rig.doubleClick(240.0f, 188.0f);
    REQUIRE(rig.created.size() == 1);
    CHECK(rig.created[0].type == "fft");
    const auto cursor = rig.created[0].position;
    rig.frame();
    CHECK_FALSE(rig.host.popups.interactiveAt({50.0f, 70.0f}));

    rig.host.click(668.0f, 29.0f);
    CHECK(rig.catalogs == 2);
    rig.frame();
    REQUIRE(rig.host.popups.interactiveAt({650.0f, 146.0f}));
    rig.doubleClick(650.0f, 146.0f);
    REQUIRE(rig.created.size() == 2);
    CHECK(rig.created[1].type == "fft");
    CHECK(rig.created[1].position.x - cursor.x == 360.0f);
    CHECK(rig.created[1].position.y - cursor.y == 240.0f);
}

TEST_CASE("Blocks keep their node while they are destroyed for reconstruction and drop it once gone",
          "[core][sakura][flowgraph][editor-views]") {
    EditorRig rig;
    FlowgraphNode::BlockData block;
    block.name = "radio";
    block.module = "sdr";
    block.title = "Radio";
    block.state = Block::State::Created;
    rig.graph = {block};
    rig.frame();
    const auto nodes = [&] {
        return Find<Sakura::NodeContentStack>(rig.host).size();
    };
    REQUIRE(nodes() == 1);

    rig.graph.front().state = Block::State::Destroying;
    rig.frame();
    CHECK(nodes() == 1);

    rig.graph.front().state = Block::State::Creating;
    rig.frame();
    CHECK(nodes() == 1);

    rig.graph.clear();
    rig.frame();
    CHECK(nodes() == 0);
}

TEST_CASE("Double clicking the canvas while the block search is open only dismisses it",
          "[core][sakura][flowgraph][editor-views]") {
    EditorRig rig;
    rig.doubleClick(40.0f, 60.0f);
    rig.frame(Extent2D<F32>{40.0f, 60.0f});
    REQUIRE(rig.host.popups.interactiveAt({50.0f, 70.0f}));

    rig.doubleClick(600.0f, 500.0f);
    CHECK(rig.catalogs == 1);
    CHECK(rig.created.empty());
    rig.frame(Extent2D<F32>{600.0f, 500.0f});
    CHECK_FALSE(rig.host.popups.interactiveAt({50.0f, 70.0f}));
    CHECK_FALSE(rig.host.popups.interactiveAt({600.0f, 500.0f}));

    rig.doubleClick(40.0f, 60.0f);
    CHECK(rig.catalogs == 2);
    rig.frame(Extent2D<F32>{40.0f, 60.0f});
    CHECK(rig.host.popups.interactiveAt({50.0f, 70.0f}));
}

TEST_CASE("Detached surface presenter reports config panel toggles as the collapsed surface state",
          "[core][sakura][flowgraph][editor-views]") {
    SurfaceRig rig;
    FlowgraphDetachedSurface surface;
    Host host(surface.component());
    const auto toggle = [&](bool collapsed) {
        const auto configs = rig.build(collapsed);
        REQUIRE(configs.size() == 1);
        CHECK(configs[0].id == MakeSurfaceTileKey("source", "default"));
        CHECK(configs[0].configOpen == !collapsed);
        surface.update(configs[0]);
        host.place(rig.scene.ctx, {0.0f, 0.0f, 300.0f, 400.0f});
        const auto actions = surface.actions();
        REQUIRE(actions.size() == 1);
        CHECK(actions[0].active == !collapsed);
        rig.mails.clear();
        actions[0].onClick();
        REQUIRE(rig.mails.size() == 1);
        REQUIRE(std::holds_alternative<MailSetSurfaceConfigCollapsed>(rig.mails[0]));
        return std::get<MailSetSurfaceConfigCollapsed>(rig.mails[0]);
    };

    const auto opened = toggle(true);
    CHECK(opened.flowgraph == "graph");
    CHECK(opened.block == "source");
    CHECK(opened.surface == "default");
    CHECK_FALSE(opened.collapsed);

    const auto closed = toggle(false);
    CHECK(closed.block == "source");
    CHECK(closed.surface == "default");
    CHECK(closed.collapsed);
}

TEST_CASE("Inspector code editor sits on a rounded tile background with a status bar rounded only at the bottom",
          "[core][sakura][flowgraph][editor-views]") {
    Scene scene;
    FlowgraphNodeInspector inspector;
    WindowStage stage;
    const auto update = [&] {
        inspector.update({.id = "inspector", .name = "fft", .value = {{"size", std::string("1024")}}});
        stage.floating.update({.windows = {inspector.window()}});
    };
    update();
    inspector.open();

    for (const F32 ratio : {1.0f, 2.0f}) {
        CAPTURE(ratio);
        auto ctx = scene.ctx;
        ctx.pixelRatio = ratio;
        update();
        stage.host.place(ctx, kScreen);
        auto& content = *inspector.window().content;
        const auto frame = Peek::Frame(content);
        const F32 radius = Sakura::FloatingLayer::Rounding * ratio;
        std::vector<Jetstream::Rect> bars;
        std::vector<Jetstream::Rect> backgrounds;
        for (auto* box : Find<Sakura::Box>(content)) {
            const auto& config = box->config();
            if (config.instances.empty() || config.cornerRadius != radius) {
                continue;
            }
            const auto& instance = config.instances.front();
            if (instance.corners == (Sakura::Box::BottomLeft | Sakura::Box::BottomRight)) {
                bars.push_back(instance.rect);
            } else if (instance.corners == Sakura::Box::AllCorners &&
                       instance.backgroundColor == ctx.color("tile_bg")) {
                backgrounds.push_back(instance.rect);
            }
        }
        REQUIRE(bars.size() == 1);
        CHECK(bars[0].x == frame.x);
        CHECK(bars[0].width == frame.width);
        CHECK(bars[0].bottom() == frame.bottom());
        CHECK(bars[0].height > 0.0f);
        REQUIRE(backgrounds.size() == 1);
        CHECK(backgrounds[0] == frame);
    }

    Frames frames(stage.host, kScreen, scene.ctx);
    CHECK(Settles(frames, update));
    stage.host.send(Mouse(MouseEventType::Move, 790.0f, 590.0f));
    CHECK(frames.run(update).clean());
}

TEST_CASE("Pin and wire tooltips refresh tensor attributes under a resting pointer",
          "[core][sakura][flowgraph][editor-views]") {
    using NodeEditor = Sakura::NodeEditor;
    Scene scene;
    Tensor tensor;
    REQUIRE(tensor.create(DeviceType::CPU, DataType::F32, {8}) == Result::SUCCESS);
    REQUIRE(tensor.setAttribute("sampleRate", std::string("48 kHz")) == Result::SUCCESS);
    FlowgraphPinTooltip pin;
    FlowgraphLink wire;
    NodeEditor editor;
    LayerSlot slot;
    slot.hold(editor, kScreen);
    LayerHost host({&slot});
    Frames frames(host, kScreen, scene.context());
    frames.verify = false;
    frames.timeStep = 0.1;
    frames.ctx.hoverDelay = 0.25f;
    const auto node = [](const std::string& id, Extent2D<F32> position, NodeEditor::Pin port) {
        return NodeEditor::Node{.id = id, .header = false, .title = id, .pins = {std::move(port)},
                                .resize = NodeEditor::ResizeAxes::X, .dimensions = {120.0f, 0.0f},
                                .gridPosition = position};
    };
    const auto update = [&] {
        pin.update("out", "Samples", tensor);
        wire.update({
            .id = "link",
            .start = {.nodeId = "src", .pinId = "out", .isInput = false},
            .end = {.nodeId = "dst", .pinId = "in", .isInput = true},
            .connection = {.consumerName = "dst", .inputSlot = "in", .producerName = "src", .producerPort = "out",
                           .resolved = true, .tensor = tensor},
        });
        editor.update({
            .id = "editor",
            .nodes = {node("src", {40.0f, 40.0f}, {.id = "out", .label = "out", .tooltip = pin.overlay()}),
                      node("dst", {400.0f, 40.0f}, {.id = "in", .isInput = true, .label = "in"})},
            .links = {wire.takeLink()},
        });
    };
    const auto labelRect = [&](const std::string& str) -> std::optional<Rect> {
        for (const auto& leaf : Drawn(host)) {
            if (const auto* label = std::get_if<Sakura::Label::Config>(&leaf.config)) {
                for (const auto& instance : label->instances) {
                    if (instance.visible && instance.str.find(str) != std::string::npos) {
                        return instance.rect;
                    }
                }
            }
        }
        return std::nullopt;
    };
    const auto hover = [&](F32 x, F32 y) {
        frames.ctx.mouse = Extent2D<F32>{x, y};
        host.send(Mouse(MouseEventType::Move, x, y));
        for (U64 frame = 0; frame < 8; ++frame) {
            frames.run(update);
        }
    };
    const auto retag = [&](const std::string& value) {
        REQUIRE(tensor.setAttribute("sampleRate", value) == Result::SUCCESS);
        frames.run(update);
        frames.run(update);
    };
    REQUIRE(frames.settle(update));

    hover(288.0f, 55.0f);
    REQUIRE(labelRect("Tensor Metadata").has_value());
    REQUIRE(labelRect("48 kHz").has_value());
    REQUIRE(frames.run(update).clean());
    retag("96 kHz");
    CHECK(labelRect("96 kHz").has_value());
    CHECK_FALSE(labelRect("48 kHz").has_value());

    const auto out = labelRect("out");
    REQUIRE(out.has_value());
    hover(out->x + 4.0f, out->y + out->height * 0.5f);
    REQUIRE_FALSE(labelRect("Tensor Metadata").has_value());
    REQUIRE(labelRect("Samples").has_value());
    REQUIRE(labelRect("96 kHz").has_value());
    REQUIRE(frames.run(update).clean());
    retag("12 kHz");
    CHECK(labelRect("12 kHz").has_value());
    CHECK_FALSE(labelRect("96 kHz").has_value());
}

TEST_CASE("Node plot detach sends every attached surface out and the plot menu entry toggles them all",
          "[core][sakura][flowgraph][editor-views]") {
    NodeMenuRig rig;
    std::vector<std::string> calls;
    const auto plot = [&](const std::string& id, bool detached) {
        return FlowgraphNode::Surface{
            .id = id,
            .detached = detached,
            .onDetach = [&calls, id](bool value) {
                calls.push_back(id + (value ? ":detach" : ":attach"));
            },
        };
    };

    rig.surfaces = {plot("a", false), plot("b", true)};
    rig.node.update(rig.config());
    auto spec = rig.node.takeNode();
    REQUIRE(spec.onDetach);
    spec.onDetach();
    CHECK(calls == std::vector<std::string>{"a:detach"});

    calls.clear();
    rig.node.update(rig.config());
    rig.choose(kPlotY);
    CHECK(calls == std::vector<std::string>{"a:detach"});

    calls.clear();
    rig.surfaces[0].detached = true;
    rig.node.update(rig.config());
    CHECK_FALSE(rig.node.takeNode().onDetach);
    rig.node.update(rig.config());
    rig.choose(kPlotY);
    CHECK(calls == std::vector<std::string>{"a:attach", "b:attach"});

    calls.clear();
    rig.surfaces.clear();
    rig.node.update(rig.config());
    CHECK_FALSE(rig.node.takeNode().onDetach);
    rig.node.update(rig.config());
    rig.choose(kPlotY);
    CHECK(calls.empty());
}

TEST_CASE("Attached surface presenter forwards plot input and cursor and reports detach requests as surface mail",
          "[core][sakura][flowgraph][editor-views]") {
    SurfaceRig rig;
    const auto& flowgraph = rig.state.flowgraph.items.at("graph");
    REQUIRE(flowgraph->metadata().set("surface_default", SurfaceMeta{}, "source") == Result::SUCCESS);
    auto data = ViewBlock(*flowgraph, "source");
    data.surfaces = {rig.module->surface()};
    FlowgraphNode::BlockData block;
    FlowgraphSurfacePresenter(rig.context).buildSurfaces(block,
                                                         rig.state.flowgraph.items.at("graph"),
                                                         "graph",
                                                         "source",
                                                         "graph:source",
                                                         data);
    REQUIRE(block.surfaces.size() == 1);
    const auto& surface = block.surfaces.front();
    CHECK_FALSE(surface.detached);
    REQUIRE(surface.onInput);
    REQUIRE(surface.resolveCursor);
    REQUIRE(surface.onAttachedSize);
    REQUIRE(surface.onDetach);

    surface.onInput(Mouse(MouseEventType::Click, 4.0f, 8.0f));
    REQUIRE(rig.mails.size() == 1);
    const auto input = std::get<MailSurfaceInput>(rig.mails[0]);
    CHECK(input.surface == rig.module->surface());
    REQUIRE(std::holds_alternative<MouseEvent>(input.event));
    CHECK(std::get<MouseEvent>(input.event).type == MouseEventType::Click);
    CHECK(surface.resolveCursor() == rig.module->surface()->cursor());

    surface.onAttachedSize({.logicalSize = {120, 80}, .framebufferSize = {240, 160}, .scale = 2.0f});
    REQUIRE(rig.mails.size() == 2);
    const auto resize = std::get<MailResizeSurface>(rig.mails[1]);
    CHECK(resize.surface == rig.module->surface());
    CHECK(resize.placement == SurfacePlacement::Attached);
    CHECK(resize.metaKey == "surface_default");
    CHECK(resize.resize.framebufferSize == Extent2D<U64>{240, 160});

    surface.onDetach(true);
    REQUIRE(rig.mails.size() == 3);
    const auto detach = std::get<MailSetSurfaceDetached>(rig.mails[2]);
    CHECK(detach.flowgraph == "graph");
    CHECK(detach.block == "source");
    CHECK(detach.surface == "default");
    CHECK(detach.detached);
}
