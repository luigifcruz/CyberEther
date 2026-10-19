#include <catch2/catch_test_macros.hpp>

#include "harness.hh"
#include "component_harness.hh"

#include "compositor/default/actions/layout.hh"
#include "compositor/default/presenters/flowgraph/pane.hh"
#include "compositor/default/presenters/workbench.hh"
#include "compositor/default/views/flowgraph/pane.hh"
#include "compositor/default/views/titlebar.hh"
#include "compositor/default/views/workbench.hh"
#include "flowgraph_fixture.hh"

#include <jetstream/render/sakura/components/node/editor.hh>
#include <jetstream/render/sakura/components/root.hh>
#include <jetstream/render/sakura/components/shell.hh>

#include "render/sakura/runtime.hh"
#include "render/sakura/themes.hh"

#ifdef JETSTREAM_RENDER_METAL_AVAILABLE
#include "jetstream/backend/base.hh"
#include "jetstream/render/components/font.hh"
#include "jetstream/render/devices/metal/window.hh"
#include "jetstream/viewport/adapters/metal.hh"
#include "../../../resources/fonts/compressed_fa.hh"
#include "../../../resources/fonts/compressed_inter.hh"
#include "../../../resources/fonts/compressed_jbmb.hh"
#include "../../../resources/fonts/compressed_jbmm.hh"
#include "../../../resources/fonts/compressed_sg.hh"
#include "../../../resources/fonts/compressed_sgr.hh"
#include "../../../resources/fonts/compressed_sgsb.hh"
#endif

#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace Jetstream;
using namespace SakuraTest;

TEST_CASE("Workbench particles animate only while enabled",
          "[core][sakura][idle][workbench][particles]") {
    FakeWindow window(1.0f, {800.0f, 600.0f});
    Sakura::Runtime runtime;
    runtime.update({.render = &window});
    WorkbenchView workbench;
    WorkbenchView::Config config;
    const auto check = [&](bool animated) {
        workbench.update(config);
        for (U64 frame = 0; frame < 8; ++frame) {
            window.viewport.platform().clock += 1.0 / 60.0;
            REQUIRE(window.begin() == Result::SUCCESS);
            const U64 runs = Sakura::Access::layoutRuns();
            workbench.render(runtime.context());
            if (frame > 1) {
                CHECK((Sakura::Access::layoutRuns() != runs) == animated);
            }
            REQUIRE(window.end() == Result::SUCCESS);
        }
    };
    check(true);
    config.backgroundParticles = false;
    check(false);
    config.backgroundParticles = true;
    check(true);
}

#ifdef JETSTREAM_RENDER_METAL_AVAILABLE

namespace {

constexpr const char* kGraph = "graph";
constexpr Extent2D<F32> kDisplay = {1200.0f, 600.0f};
constexpr F64 kFrameWork = 0.001;
constexpr F64 kVsync = 1.0 / 120.0;
constexpr U64 kSettleFrames = 16;
constexpr U64 kIdleFrames = 100;

class MetalViewport final : public Viewport::Adapter<DeviceType::Metal> {
 public:
    MetalViewport() : Adapter(Viewport::Config{.size = {1200, 600}}) {
        auto owned = std::make_unique<RenderTest::TestPlatform>();
        platform = owned.get();
        attachPlatform(std::move(owned));
    }

    ~MetalViewport() override {
        if (layer) {
            layer->release();
        }
    }

    std::string id() const override { return "test"; }
    std::string name() const override { return "Test (Metal)"; }
    DeviceType device() const override { return DeviceType::Metal; }

    Result create() override {
        layer = CA::MetalLayer::layer()->retain();
        layer->setDevice(Backend::State<DeviceType::Metal>()->getDevice());
        layer->setPixelFormat(MTL::PixelFormatBGRA8Unorm);
        return Result::SUCCESS;
    }

    Result destroy() override { return Result::SUCCESS; }
    Result waitEvents() override { return Result::SUCCESS; }
    Result pollEvents() override { return Result::SUCCESS; }
    bool retainedPresent() const override { return true; }
    bool keepRunning() override { return true; }

    Extent2D<F32> displaySize() const override { return kDisplay; }
    Extent2D<F32> framebufferSize() const override { return kDisplay; }
    bool macOSBehaviors() const override { return true; }

    void waitIdle(F64 seconds) override {
        ++idleWaits;
        platform->clock += seconds;
    }

    void* nextDrawable() override {
        platform->clock = std::ceil(platform->clock / kVsync) * kVsync;
        layer->setDrawableSize({kDisplay.x, kDisplay.y});
        return static_cast<void*>(layer->nextDrawable());
    }

    RenderTest::TestPlatform* platform = nullptr;
    CA::MetalLayer* layer = nullptr;
    U64 idleWaits = 0;
};

struct Tick {
    bool presented = false;
    bool skipped = false;
    bool idled = false;
    bool pending = false;
    U64 layouts = 0;
    U64 draws = 0;

    bool quiet() const {
        return !presented && skipped && idled && !pending && !layouts && !draws;
    }
};

class MetalWindow : public Render::WindowImp<DeviceType::Metal> {
 public:
    using WindowImp::WindowImp;
    std::vector<std::shared_ptr<Render::Texture>> framebuffers;

 protected:
    Result bindSurface(const std::shared_ptr<Render::Surface>& surface) override {
        JST_CHECK(WindowImp::bindSurface(surface));
        framebuffers.push_back(surface->getConfig().framebuffer);
        return Result::SUCCESS;
    }

    Result unbindSurface(const std::shared_ptr<Render::Surface>& surface) override {
        JST_CHECK(WindowImp::unbindSurface(surface));
        std::erase(framebuffers, surface->getConfig().framebuffer);
        return Result::SUCCESS;
    }
};

struct Workbench : FlowgraphFixture {
    std::shared_ptr<MetalViewport> viewport;
    std::shared_ptr<MetalWindow> window;
    Sakura::Palette palette = Sakura::Themes.at(Sakura::ResolveThemeKey(""));
    Sakura::Runtime runtime;
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    PresenterContext context{state, callbacks};
    FlowgraphPanePresenter panePresenter{context};
    std::vector<Mail> mails;
    FlowgraphPane pane;
    std::vector<Sakura::Dock::Pane> panes;
    WorkspaceShellView workspace;
    Sakura::Layer content;
    Sakura::FloatingLayer floating;
    std::unique_ptr<WorkbenchView> full;
    std::unique_ptr<Sakura::Root> root = std::make_unique<Sakura::Root>();

    Workbench() {
        REQUIRE(Backend::Initialize<DeviceType::Metal>({.headless = true}) == Result::SUCCESS);
        viewport = std::make_shared<MetalViewport>();
        REQUIRE(viewport->create() == Result::SUCCESS);
        window = std::make_shared<MetalWindow>(Render::Window::Config{.scale = 1.0f}, viewport);
        REQUIRE(window->create() == Result::SUCCESS);
        REQUIRE(window->start() == Result::SUCCESS);
        font("default_mono", jbmm_compressed_data, 32.0f);
        font("default_mono_bold", jbmb_compressed_data, 32.0f);
        font("default_display", sg_compressed_data, 64.0f);
        font("default_display_semibold", sgsb_compressed_data, 32.0f);
        font("default_display_regular", sgr_compressed_data, 32.0f);
        font("default_icons", fas_compressed_data, 32.0f, true);
        font("default_body", inter_regular_compressed_data, 32.0f);
        font("default_body_italic", inter_italic_compressed_data, 32.0f);
        font("default_body_bold", inter_bold_compressed_data, 32.0f);
        font("default_body_bold_italic", inter_bold_italic_compressed_data, 32.0f);

        state.system.render = window;
        state.system.viewport = viewport;
        state.interface.focusedFlowgraph = kGraph;
        state.flowgraph.items[kGraph] = std::shared_ptr<Flowgraph>(flowgraph.get(), [](Flowgraph*) {});
        callbacks.enqueueMail = [this](Mail&& mail) {
            mails.push_back(std::move(mail));
        };
        content.update({.content = &workspace.component()});

        REQUIRE(flowgraph->blockCreate("a", TestFlowgraph::kSyntheticIsolatedType, {}, {}) == Result::SUCCESS);
        REQUIRE(flowgraph->blockCreate("b", TestFlowgraph::kSyntheticSourceType, {}, {}) == Result::SUCCESS);
        poll();
    }

    ~Workbench() {
        full.reset();
        root.reset();
        (void)window->synchronize();
        (void)window->stop();
        (void)window->destroy();
    }

    void font(const std::string& name, const unsigned int* data, F32 size, bool icons = false) {
        std::shared_ptr<Render::Components::Font> loaded;
        Render::Components::Font::Config config;
        config.data = data;
        config.size = size;
        config.icons = icons;
        REQUIRE(window->build(loaded, config) == Result::SUCCESS);
        REQUIRE(window->addFont(name, loaded) == Result::SUCCESS);
    }

    void poll() {
        auto& blocks = state.flowgraph.blocks[kGraph];
        blocks.clear();
        std::vector<std::string> names;
        REQUIRE(flowgraph->view().keys(names) == Result::SUCCESS);
        for (const auto& name : names) {
            blocks.push_back({.name = name, .data = ViewBlock(*flowgraph, name)});
        }

        pane.update(panePresenter.build(kGraph, state.flowgraph.items.at(kGraph)));
        panes.clear();
        pane.appendPanes(panes);
        std::vector<Sakura::FloatingLayer::Window> windows;
        pane.appendWindows(windows);
        workspace.updateTitlebar({.focusedFlowgraph = kGraph});
        workspace.updatePanes(panes);
        floating.update({.windows = std::move(windows)});
        root->update({.id = "workbench", .layers = {&content, &floating}});
        if (full) {
            full->update({
                .backgroundParticles = WorkbenchPresenter(context).build().backgroundParticles,
                .titlebar = {.focusedFlowgraph = kGraph},
                .flowgraphs = {panePresenter.build(kGraph, state.flowgraph.items.at(kGraph))},
            });
        }

        LayoutActions actions(state.flowgraph);
        for (const auto& mail : std::exchange(mails, {})) {
            if (const auto* layout = std::get_if<MailSetFlowgraphLayout>(&mail)) {
                CHECK(actions.handle(*layout) == Result::SUCCESS);
            }
        }
        blocks.clear();
    }

    Tick step(std::vector<Render::InputQueue::Event> events = {}) {
        for (auto& event : events) {
            viewport->input().push(std::move(event));
        }
        viewport->platform->clock += kFrameWork;

        const auto before = window->stats();
        const U64 idles = viewport->idleWaits;
        const U64 runs = Sakura::Access::layoutRuns();
        REQUIRE(window->begin() == Result::SUCCESS);
        runtime.update({.palette = &palette, .render = window.get()});
        Tick tick;
        if (full) {
            full->render(runtime.context());
        } else {
            root->render(runtime.context());
        }
        std::vector<std::pair<std::shared_ptr<Render::Texture>, U64>> revisions;
        for (const auto& texture : window->framebuffers) {
            revisions.emplace_back(texture, texture->contentRevision());
        }
        REQUIRE(window->end() == Result::SUCCESS);
        for (const auto& [texture, revision] : revisions) {
            tick.draws += revision != texture->contentRevision() ? 1 : 0;
        }
        poll();

        const auto& after = window->stats();
        tick.presented = after.presentedFrames != before.presentedFrames;
        tick.skipped = after.skippedPresents != before.skippedPresents;
        tick.idled = viewport->idleWaits != idles;
        tick.pending = window->input().hasPending();
        tick.layouts = Sakura::Access::layoutRuns() - runs;
        return tick;
    }

    bool settle() {
        for (U64 frame = 0; frame < kSettleFrames; ++frame) {
            if (step().quiet()) {
                return true;
            }
        }
        return false;
    }

    void click(Extent2D<F32> point) {
        step({Render::InputQueue::Pointer{point}});
        REQUIRE(settle());
        step({Render::InputQueue::Modifiers{}, Render::InputQueue::Button{MouseButton::Left, true}});
        step({Render::InputQueue::Modifiers{}, Render::InputQueue::Button{MouseButton::Left, false}});
        REQUIRE(settle());
    }

    void checkIdle() {
        U64 presents = 0;
        U64 layouts = 0;
        U64 pending = 0;
        U64 waits = 0;
        U64 skips = 0;
        U64 draws = 0;
        for (U64 frame = 0; frame < kIdleFrames; ++frame) {
            const auto tick = step();
            INFO("idle frame " << frame << ": layouts=" << tick.layouts << " draws=" << tick.draws
                 << " presents=" << tick.presented);
            CHECK(tick.quiet());
            presents += tick.presented ? 1 : 0;
            layouts += tick.layouts;
            draws += tick.draws;
            pending += tick.pending ? 1 : 0;
            waits += tick.idled ? 1 : 0;
            skips += tick.skipped ? 1 : 0;
        }
        CHECK(presents == 0);
        CHECK(layouts == 0);
        CHECK(draws == 0);
        CHECK(pending == 0);
        CHECK(waits == kIdleFrames);
        CHECK(skips == kIdleFrames);
        CHECK(viewport->input().drain().empty());
    }

    Jetstream::Rect editorFrame() {
        auto editors = Find<Sakura::NodeEditor>(*panes.front().content);
        REQUIRE(editors.size() == 1);
        return Peek::Frame(*editors.front());
    }

    Jetstream::Rect nodeContentFrame() {
        auto stacks = Find<Sakura::NodeContentStack>(*panes.front().content);
        std::erase_if(stacks, [](auto* stack) {
            return Peek::Frame(*stack).empty();
        });
        REQUIRE_FALSE(stacks.empty());
        return Peek::Frame(*stacks.back());
    }

    void start() {
        step({Render::InputQueue::Focus{true}});
        for (U64 frame = 0; frame < kSettleFrames; ++frame) {
            step();
        }
        const Jetstream::Rect editor = editorFrame();
        step({Render::InputQueue::Hover{true},
              Render::InputQueue::Pointer{Extent2D<F32>{editor.x + editor.width * 0.6f, editor.y + editor.height * 0.7f}}});
        REQUIRE(settle());
    }
};

}  // namespace

TEST_CASE("A click on the flowgraph editor background lets the window idle",
          "[core][sakura][idle][flowgraph]") {
    Workbench workbench;
    workbench.start();
    workbench.checkIdle();

    const Jetstream::Rect editor = workbench.editorFrame();
    workbench.click({editor.x + editor.width * 0.6f, editor.y + editor.height * 0.7f});
    workbench.checkIdle();
}

TEST_CASE("A click on a flowgraph node lets the window idle",
          "[core][sakura][idle][flowgraph]") {
    Workbench workbench;
    workbench.start();

    const Jetstream::Rect node = workbench.nodeContentFrame();
    workbench.click({node.x + 20.0f, node.y - 10.0f});
    workbench.checkIdle();
}

TEST_CASE("Flowgraph presses stop rendering while the pointer is stationary",
          "[core][sakura][idle][flowgraph][stationary]") {
    Workbench workbench;
    workbench.start();
    Extent2D<F32> point;
    SECTION("background") {
        const auto editor = workbench.editorFrame();
        point = {editor.x + editor.width * 0.6f, editor.y + editor.height * 0.7f};
    }
    SECTION("node") {
        const auto node = workbench.nodeContentFrame();
        point = {node.x + 20.0f, node.y - 10.0f};
    }
    workbench.step({Render::InputQueue::Pointer{point}});
    REQUIRE(workbench.settle());
    workbench.step({Render::InputQueue::Button{MouseButton::Left, true}});
    for (U64 frame = 0; frame < kSettleFrames; ++frame) {
        workbench.step();
    }
    workbench.checkIdle();
    workbench.step({Render::InputQueue::Button{MouseButton::Left, false}});
    REQUIRE(workbench.settle());
    workbench.checkIdle();
}

TEST_CASE("A queued flowgraph click lets the window idle",
          "[core][sakura][idle][flowgraph][queued-click]") {
    Workbench workbench;
    workbench.start();
    Extent2D<F32> point;
    SECTION("background") {
        const auto editor = workbench.editorFrame();
        point = {editor.x + editor.width * 0.6f, editor.y + editor.height * 0.7f};
    }
    SECTION("node") {
        const auto node = workbench.nodeContentFrame();
        point = {node.x + 20.0f, node.y - 10.0f};
    }
    workbench.step({Render::InputQueue::Pointer{point},
                    Render::InputQueue::Button{MouseButton::Left, true},
                    Render::InputQueue::Button{MouseButton::Left, false}});
    REQUIRE(workbench.settle());
    workbench.checkIdle();
}

TEST_CASE("The full flowgraph workbench idles with background particles enabled",
          "[core][sakura][idle][flowgraph][workbench]") {
    Workbench workbench;
    workbench.start();
    Extent2D<F32> point;
    SECTION("background") {
        const auto editor = workbench.editorFrame();
        point = {editor.x + editor.width * 0.6f, editor.y + editor.height * 0.7f};
    }
    SECTION("node") {
        const auto node = workbench.nodeContentFrame();
        point = {node.x + 20.0f, node.y - 10.0f};
    }
    workbench.full = std::make_unique<WorkbenchView>();
    workbench.poll();
    for (U64 frame = 0; frame < kSettleFrames; ++frame) {
        workbench.step();
    }
    workbench.checkIdle();
    workbench.step({Render::InputQueue::Pointer{point}, Render::InputQueue::Button{MouseButton::Left, true}});
    workbench.step({Render::InputQueue::Button{MouseButton::Left, false}});
    for (U64 frame = 0; frame < kSettleFrames; ++frame) {
        workbench.step();
    }
    workbench.checkIdle();
}

#else

TEST_CASE("A click on the flowgraph lets the window idle", "[core][sakura][idle][flowgraph]") {
    SKIP("The idle workbench test drives a real Metal window.");
}

#endif
