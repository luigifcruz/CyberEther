#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "fonts.hh"
#include "component_harness.hh"

#include "compositor/default/actions/layout.hh"
#include "compositor/default/actions/flowgraph.hh"
#include "compositor/default/presenters/flowgraph/pane.hh"
#include "compositor/default/views/flowgraph/pane.hh"
#include "flowgraph_fixture.hh"
#include "jetstream/render/tools/imgui_icons_ext.hh"

#include <algorithm>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

using namespace Jetstream;
using namespace SakuraTest;

namespace {

using DockTiles = Sakura::DockTiles;

constexpr F32 kBar = 32.0f;
constexpr const char* kGraph = "graph";
constexpr Jetstream::Rect kWideScreen = {0.0f, 0.0f, 1200.0f, 600.0f};

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

std::vector<Jetstream::Rect> HeaderLabels(Sakura::Component& dock) {
    std::vector<Jetstream::Rect> labels;
    for (auto* text : Direct<Sakura::Text>(dock)) {
        const auto& str = text->config().str;
        if (str != ICON_FA_XMARK && str != ICON_FA_UP_RIGHT_AND_DOWN_LEFT_FROM_CENTER &&
            str != ICON_FA_DOWN_LEFT_AND_UP_RIGHT_TO_CENTER) {
            labels.push_back(Peek::Frame(*text));
        }
    }
    std::sort(labels.begin(), labels.end(), [](const auto& a, const auto& b) {
        return a.y == b.y ? a.x < b.x : a.y < b.y;
    });
    return labels;
}

constexpr F32 kHeader = 20.0f;

Jetstream::Rect Body(const Jetstream::Rect& card) {
    return {card.x, card.y + kHeader, card.width, card.height - kHeader};
}

Jetstream::Rect Padded(const Jetstream::Rect& rect, F32 padding) {
    return {rect.x + padding, rect.y + padding, rect.width - 2.0f * padding, rect.height - 2.0f * padding};
}

Extent2D<F32> CloseOf(const Jetstream::Rect& card) {
    return {card.right() - 15.0f, card.y + 15.0f};
}

TileMeta LayoutLeaf(const std::string& pane, std::optional<F32> weight = std::nullopt) {
    return TileMeta{.weight = weight, .pane = pane};
}

TileMeta LayoutSplit(const std::string& split, std::vector<TileMeta> children, std::optional<F32> weight = std::nullopt) {
    return TileMeta{.split = split, .weight = weight, .children = std::move(children)};
}

std::vector<std::string> LayoutPanes(const TileMeta& tiles) {
    std::vector<std::string> panes;
    CollectTilePanes(tiles, panes);
    return panes;
}

struct PaneRig : FlowgraphFixture {
    MonoFontWindow window;
    Sakura::Context ctx = MakeContext();
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    PresenterContext context{state, callbacks};
    FlowgraphPanePresenter presenter{context};
    std::vector<Mail> mails;
    std::function<void(FlowgraphPane::Config&)> tweak;
    FlowgraphPane pane;
    std::vector<Sakura::Dock::Pane> panes;
    Sakura::Dock workspace;
    Sakura::Layer content;
    Sakura::FloatingLayer floating;
    LayerHost host;
    Jetstream::Rect screen = kScreen;

    PaneRig() {
        ctx.render = &window;
        state.flowgraph.items[kGraph] = std::shared_ptr<Flowgraph>(flowgraph.get(), [](Flowgraph*) {});
        callbacks.enqueueMail = [this](Mail&& mail) {
            mails.push_back(std::move(mail));
        };
        content.update({.content = &workspace});
        host.setLayers({&content, &floating});
    }

    FlowgraphPane::Config build() {
        auto config = presenter.build(kGraph, state.flowgraph.items.at(kGraph));
        if (tweak) {
            tweak(config);
        }
        return config;
    }

    void update() {
        pane.update(build());
        panes.clear();
        pane.appendPanes(panes);
        std::vector<Sakura::FloatingLayer::Window> windows;
        pane.appendWindows(windows);
        workspace.update({.tabs = true, .accepts = "main", .panes = panes});
        floating.update({.windows = std::move(windows)});
    }

    void place(std::optional<Extent2D<F32>> mouse = std::nullopt) {
        auto next = ctx;
        next.mouse = mouse;
        host.place(next, screen);
    }

    void frame() {
        update();
        place();
    }

    void deliver() {
        auto snapshots = std::exchange(state.flowgraph.blocks, {});
        LayoutActions actions(state.flowgraph);
        for (const auto& mail : mails) {
            if (const auto* layout = std::get_if<MailSetFlowgraphLayout>(&mail)) {
                CHECK(actions.handle(*layout) == Result::SUCCESS);
            }
            if (const auto* open = std::get_if<MailOpenDocumentation>(&mail)) {
                CHECK(actions.handle(*open) == Result::SUCCESS);
            }
            if (const auto* close = std::get_if<MailCloseDocumentation>(&mail)) {
                CHECK(actions.handle(*close) == Result::SUCCESS);
            }
        }
        mails.clear();
        state.flowgraph.blocks = std::move(snapshots);
    }

    void documentedBlocks() {
        REQUIRE(graph().blockCreate("a", TestFlowgraph::kSyntheticIsolatedType, {}, {}) == Result::SUCCESS);
        REQUIRE(graph().blockCreate("b", TestFlowgraph::kSyntheticSourceType, {}, {}) == Result::SUCCESS);
        auto& blocks = state.flowgraph.blocks[kGraph];
        blocks.clear();
        for (const auto& name : {"a", "b"}) {
            auto data = ViewBlock(graph(), name);
            data.state = Block::State::Creating;
            blocks.push_back({.name = name, .data = data});
        }
    }

    void openDocumentation(const std::string& block) {
        const auto config = build();
        REQUIRE(config.editor.onOpenDocumentation);
        config.editor.onOpenDocumentation(block);
        REQUIRE(count<MailOpenDocumentation>() == 1);
        CHECK(last<MailOpenDocumentation>()->flowgraph == kGraph);
        CHECK(last<MailOpenDocumentation>()->block == block);
        deliver();
        frame();
    }

    std::vector<Sakura::Markdown*> documentation() {
        return Find<Sakura::Markdown>(layoutDock());
    }

    std::string documentationText() {
        std::string text;
        for (auto* markdown : documentation()) {
            for (auto* label : Find<Sakura::Label>(*markdown)) {
                for (const auto& instance : label->config().instances) {
                    if (instance.visible) {
                        text += instance.str + "\n";
                    }
                }
            }
        }
        return text;
    }

    bool header(const std::string& title) {
        for (auto* text : Direct<Sakura::Text>(layoutDock())) {
            if (text->config().str == title) {
                return true;
            }
        }
        return false;
    }

    Flowgraph& graph() {
        return *flowgraph;
    }

    void detachConfigs(std::vector<std::string> ids, std::vector<std::string>* closed = nullptr) {
        tweak = [ids = std::move(ids), closed](FlowgraphPane::Config& config) {
            for (const auto& id : ids) {
                config.detachedConfigs.push_back({
                    .id = "config:" + id,
                    .title = id,
                    .onClose = [closed, id] {
                        if (closed) {
                            closed->push_back(id);
                        }
                    },
                });
            }
        };
    }

    Sakura::Component& layoutDock() {
        return *panes.front().content;
    }

    std::vector<Sakura::Overlay*> tiles() {
        return Direct<Sakura::Overlay>(layoutDock());
    }

    std::vector<Jetstream::Rect> configFrames() {
        std::vector<Jetstream::Rect> frames;
        for (auto* panel : Direct<Sakura::NodeFieldPanel>(layoutDock())) {
            frames.push_back(Peek::Frame(*panel));
        }
        std::sort(frames.begin(), frames.end(), [](const auto& a, const auto& b) {
            return a.x == b.x ? a.y < b.y : a.x < b.x;
        });
        return frames;
    }

    template<typename T>
    U64 count() const {
        U64 total = 0;
        for (const auto& mail : mails) {
            total += std::holds_alternative<T>(mail) ? 1 : 0;
        }
        return total;
    }

    template<typename T>
    const T* last() const {
        for (auto it = mails.rbegin(); it != mails.rend(); ++it) {
            if (const auto* mail = std::get_if<T>(&*it)) {
                return mail;
            }
        }
        return nullptr;
    }

    void drag(Extent2D<F32> from, Extent2D<F32> to) {
        host.send(Mouse(MouseEventType::Click, from.x, from.y));
        host.send(Mouse(MouseEventType::Move, from.x + 10.0f, from.y));
        host.send(Mouse(MouseEventType::Move, to.x, to.y));
        update();
        place(to);
    }
};

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

}  // namespace

TEST_CASE("Each flowgraph contributes one main tab whose content is its layout dock",
          "[core][sakura][flowgraph][pane]") {
    PaneRig rig;
    rig.state.interface.focusedFlowgraph = kGraph;

    const auto config = rig.build();
    CHECK(config.id == kGraph);
    CHECK(config.layoutGroup == "layout:graph");
    CHECK(config.editorPane == "editor:graph");
    CHECK_FALSE(config.layout.has_value());
    REQUIRE(config.describe);
    CHECK(config.describe("config:fft") == "fft Config");

    rig.frame();
    REQUIRE(rig.panes.size() == 1);
    const auto& main = rig.panes.front();
    CHECK(main.id == kGraph);
    CHECK(main.title == "graph");
    CHECK(main.group == "main");
    CHECK(main.icon.empty());
    CHECK(main.closable);
    REQUIRE(main.content != nullptr);
    CHECK(dynamic_cast<Sakura::Dock*>(main.content) != nullptr);

    REQUIRE(main.onClose);
    REQUIRE(main.onFocus);
    rig.mails.clear();
    main.onClose();
    main.onFocus();
    REQUIRE(rig.mails.size() == 2);
    REQUIRE(std::holds_alternative<MailCloseFlowgraph>(rig.mails[0]));
    CHECK(std::get<MailCloseFlowgraph>(rig.mails[0]).flowgraph == kGraph);
    REQUIRE(std::holds_alternative<MailFocusFlowgraph>(rig.mails[1]));
    CHECK(std::get<MailFocusFlowgraph>(rig.mails[1]).flowgraph == kGraph);

    FlowgraphPane other;
    other.update({.id = "other", .layoutGroup = "layout:other", .editorPane = "editor:other"});
    std::vector<Sakura::Dock::Pane> panes;
    rig.pane.appendPanes(panes);
    other.appendPanes(panes);
    REQUIRE(panes.size() == 2);
    CHECK(panes[1].id == "other");
    CHECK(panes[1].title == "other");
    CHECK(panes[1].group == "main");
    CHECK(panes[1].content != panes[0].content);
}

TEST_CASE("A flowgraph without detached views fills its tab with the editor and shows no tile header",
          "[core][sakura][flowgraph][pane]") {
    PaneRig rig;
    rig.frame();

    CHECK(Peek::Frame(rig.layoutDock()) == Jetstream::Rect{0.0f, kBar, 800.0f, 600.0f - kBar});
    CHECK(HeaderLabels(rig.layoutDock()).empty());
    const auto tiles = rig.tiles();
    REQUIRE(tiles.size() == 1);
    CHECK(Peek::Frame(*tiles[0]) == Padded({6.0f, kBar + 6.0f, 788.0f, 556.0f}, Sakura::Dock::ContentPadding));
    REQUIRE(rig.count<MailSetFlowgraphLayout>() == 1);
    const auto* layout = rig.last<MailSetFlowgraphLayout>();
    REQUIRE(layout->tiles.has_value());
    CHECK(layout->tiles->pane == std::optional<std::string>("flowgraph"));
    CHECK_FALSE(layout->tiles->children.has_value());
}

TEST_CASE("A detached view joins as a column beside the editor and the layout is reported once",
          "[core][sakura][flowgraph][pane]") {
    PaneRig rig;
    rig.screen = kWideScreen;
    rig.frame();
    rig.deliver();
    for (int frame = 0; frame < 3; ++frame) {
        rig.frame();
    }
    CHECK(rig.count<MailSetFlowgraphLayout>() == 0);

    rig.detachConfigs({"a"});
    rig.frame();
    REQUIRE(rig.count<MailSetFlowgraphLayout>() == 1);
    const auto* layout = rig.last<MailSetFlowgraphLayout>();
    CHECK(layout->flowgraph == kGraph);
    REQUIRE(layout->tiles.has_value());
    CHECK(layout->tiles->split == std::optional<std::string>("row"));
    CHECK(LayoutPanes(*layout->tiles) == std::vector<std::string>{"flowgraph", "config:a"});
    CHECK_FALSE(layout->tiles->children->at(0).weight.has_value());
    CHECK_FALSE(layout->tiles->children->at(1).weight.has_value());

    rig.deliver();
    for (int frame = 0; frame < 3; ++frame) {
        rig.frame();
    }
    CHECK(rig.mails.empty());

    const auto tiles = rig.tiles();
    REQUIRE(tiles.size() == 1);
    CHECK(Peek::Frame(*tiles[0]) == Padded(Body({6.0f, 38.0f, 762.0f, 556.0f}), Sakura::Dock::ContentPadding));
    CHECK(rig.configFrames() == std::vector<Jetstream::Rect>{
        Padded(Body({774.0f, 38.0f, 420.0f, 556.0f}), Sakura::Dock::ContentPadding)});
    CHECK(HeaderLabels(rig.layoutDock()).size() == 2);
}

TEST_CASE("Saved layouts are adopted through the presenter with their column widths and unknown panes dropped",
          "[core][sakura][flowgraph][pane]") {
    PaneRig rig;
    rig.screen = kWideScreen;
    rig.detachConfigs({"a"});
    rig.state.flowgraph.layouts[kGraph] = LayoutSplit("row", {
        LayoutLeaf("flowgraph", 700.0f),
        LayoutLeaf("config:a", 300.0f),
        LayoutLeaf("stack_0", 400.0f),
    });

    const auto config = rig.build();
    REQUIRE(config.layout.has_value());
    CHECK(config.layout->children == std::vector<DockTiles>{Leaf("editor:graph"), Leaf("config:a")});
    CHECK(config.layout->weights == std::vector<F32>{700.0f, 300.0f});

    rig.frame();
    rig.frame();
    CHECK(rig.mails.empty());

    const auto tiles = rig.tiles();
    REQUIRE(tiles.size() == 1);
    CHECK(Peek::Frame(*tiles[0]) == Padded(Body({6.0f, 38.0f, 882.0f, 556.0f}), Sakura::Dock::ContentPadding));
    CHECK(rig.configFrames() == std::vector<Jetstream::Rect>{
        Padded(Body({894.0f, 38.0f, 300.0f, 556.0f}), Sakura::Dock::ContentPadding)});
}

TEST_CASE("The editor tile only maximizes while detached view headers close their view",
          "[core][sakura][flowgraph][pane]") {
    PaneRig rig;
    rig.screen = kWideScreen;
    std::vector<std::string> closed;
    rig.detachConfigs({"a"}, &closed);
    rig.state.flowgraph.layouts[kGraph] = LayoutSplit("row", {LayoutLeaf("flowgraph"), LayoutLeaf("config:a")});
    rig.frame();
    REQUIRE(rig.mails.empty());

    const auto labels = HeaderLabels(rig.layoutDock());
    REQUIRE(labels.size() == 2);
    CHECK(labels[0].x == 18.0f);
    CHECK(labels[1].x == 786.0f);

    const auto editorButton = CloseOf({6.0f, 38.0f, 762.0f, 556.0f});
    rig.host.click(editorButton.x, editorButton.y);
    rig.frame();
    CHECK(rig.configFrames().empty());
    const auto restore = CloseOf({6.0f, 38.0f, 1188.0f, 556.0f});
    rig.host.click(restore.x, restore.y);
    rig.frame();
    CHECK(rig.configFrames().size() == 1);
    CHECK(rig.count<MailSetFlowgraphLayout>() == 0);
    CHECK(closed.empty());

    rig.mails.clear();
    const auto close = CloseOf({774.0f, 38.0f, 420.0f, 556.0f});
    rig.host.click(close.x, close.y);
    CHECK(closed == std::vector<std::string>{"a"});
    REQUIRE(rig.count<MailSetFlowgraphLayout>() == 1);
    const auto* layout = rig.last<MailSetFlowgraphLayout>();
    REQUIRE(layout->tiles.has_value());
    CHECK(LayoutPanes(*layout->tiles) == std::vector<std::string>{"flowgraph"});
}

TEST_CASE("Detached configs from the presenter carry their tile title and reattach on close",
          "[core][sakura][flowgraph][pane]") {
    PaneRig rig;
    rig.screen = kWideScreen;
    REQUIRE(rig.graph().blockCreate("gain", TestFlowgraph::kSyntheticIsolatedType, {}, {}) == Result::SUCCESS);
    REQUIRE(rig.graph().metadata().set("config", ConfigMeta{.detached = true}, "gain") == Result::SUCCESS);
    auto data = ViewBlock(rig.graph(), "gain");
    data.state = Block::State::Creating;
    rig.state.flowgraph.blocks[kGraph] = {{.name = "gain", .data = data}};
    rig.state.flowgraph.layouts[kGraph] = LayoutSplit("row", {LayoutLeaf("flowgraph"), LayoutLeaf("config:gain")});

    const auto config = rig.build();
    REQUIRE(config.detachedConfigs.size() == 1);
    const auto& detached = config.detachedConfigs.front();
    CHECK(detached.id == "config:gain");
    CHECK(detached.title == MakeConfigTileTitle("gain", data.title));
    CHECK(detached.configFields.empty());
    CHECK(config.detachedSurfaces.empty());

    rig.frame();
    rig.mails.clear();
    REQUIRE(rig.configFrames().size() == 1);
    const auto close = CloseOf({774.0f, 38.0f, 420.0f, 556.0f});
    rig.host.click(close.x, close.y);
    REQUIRE(rig.count<MailSetConfigDetached>() == 1);
    CHECK(rig.last<MailSetConfigDetached>()->flowgraph == kGraph);
    CHECK(rig.last<MailSetConfigDetached>()->block == "gain");
    CHECK_FALSE(rig.last<MailSetConfigDetached>()->detached);
}

TEST_CASE("Detached surfaces become columns with their image and close callback",
          "[core][sakura][flowgraph][pane]") {
    PaneRig rig;
    rig.screen = kWideScreen;
    std::vector<std::string> closed;
    const auto texture = std::make_shared<SurfaceTexture>();
    rig.tweak = [&](FlowgraphPane::Config& config) {
        config.detachedSurfaces = {{
            .id = "surface:a:default",
            .title = "A",
            .texture = texture,
            .onClose = [&] { closed.push_back("surface"); },
        }};
    };
    rig.state.flowgraph.layouts[kGraph] = LayoutSplit("row", {LayoutLeaf("flowgraph"), LayoutLeaf("surface:a:default")});
    rig.frame();
    REQUIRE(rig.mails.empty());

    const auto surfaces = Direct<Sakura::Stack>(rig.layoutDock());
    REQUIRE(surfaces.size() == 1);
    CHECK(Peek::Frame(*surfaces[0]) == Padded(Body({774.0f, 38.0f, 420.0f, 556.0f}), Sakura::Dock::ContentPadding));
    CHECK(Find<Sakura::SurfaceImage>(rig.layoutDock()).size() == 1);
    for (auto* text : Direct<Sakura::Text>(rig.layoutDock())) {
        CHECK(text->config().str != ICON_FA_CHART_AREA);
    }

    const auto close = CloseOf({774.0f, 38.0f, 420.0f, 556.0f});
    rig.host.click(close.x, close.y);
    CHECK(closed == std::vector<std::string>{"surface"});
}

TEST_CASE("Dragging a view onto the bottom of another stacks them in one column of the layout dock",
          "[core][sakura][flowgraph][pane]") {
    PaneRig rig;
    rig.screen = kWideScreen;
    rig.detachConfigs({"a", "b"});
    rig.state.flowgraph.layouts[kGraph] = LayoutSplit("row", {
        LayoutLeaf("flowgraph"),
        LayoutLeaf("config:a", 300.0f),
        LayoutLeaf("config:b", 300.0f),
    });
    rig.frame();
    REQUIRE(rig.mails.empty());
    CHECK(rig.configFrames() == std::vector<Jetstream::Rect>{
        Padded(Body({588.0f, 38.0f, 300.0f, 556.0f}), Sakura::Dock::ContentPadding),
        Padded(Body({894.0f, 38.0f, 300.0f, 556.0f}), Sakura::Dock::ContentPadding)});

    rig.drag({1000.0f, 40.0f}, {738.0f, 500.0f});
    REQUIRE(rig.host.drag.active);
    CHECK(rig.host.drag.pane == "config:b");
    CHECK(rig.host.drag.group == "layout:graph");
    REQUIRE(rig.host.drag.target.has_value());
    CHECK(rig.host.drag.target->owner == static_cast<const void*>(&rig.layoutDock()));
    CHECK(rig.host.drag.target->preview == Jetstream::Rect{588.0f, 316.0f, 300.0f, 278.0f});

    rig.host.send(Mouse(MouseEventType::Release, 738.0f, 500.0f));
    REQUIRE(rig.count<MailSetFlowgraphLayout>() == 1);
    const auto* layout = rig.last<MailSetFlowgraphLayout>();
    REQUIRE(layout->tiles.has_value());
    REQUIRE(layout->tiles->children->size() == 2);
    CHECK(layout->tiles->children->at(0).pane == std::optional<std::string>("flowgraph"));
    const auto& column = layout->tiles->children->at(1);
    CHECK(column.split == std::optional<std::string>("column"));
    CHECK(column.weight == std::optional<F32>(300.0f));
    CHECK(LayoutPanes(column) == std::vector<std::string>{"config:a", "config:b"});

    rig.deliver();
    rig.frame();
    CHECK(rig.mails.empty());
    const auto frames = rig.configFrames();
    REQUIRE(frames.size() == 2);
    CHECK(frames[0].x == frames[1].x);
    CHECK(frames[0].y < frames[1].y);
}

TEST_CASE("Identical per-frame flowgraph pane updates settle and stay clean",
          "[core][sakura][flowgraph][pane]") {
    PaneRig rig;
    const auto texture = std::make_shared<SurfaceTexture>();
    rig.tweak = [&](FlowgraphPane::Config& config) {
        config.detachedSurfaces = {{.id = "surface:a:default", .title = "A", .texture = texture}};
        config.detachedConfigs = {{.id = "config:b", .title = "B"}};
    };
    Frames frames(rig.host, kScreen, rig.ctx);
    const auto update = [&] {
        rig.update();
        rig.deliver();
    };
    REQUIRE(Settles(frames, update));
    CHECK(rig.state.flowgraph.layouts.contains(kGraph));
    CHECK(LayoutPanes(rig.state.flowgraph.layouts.at(kGraph)) ==
          std::vector<std::string>{"flowgraph", "surface:a:default", "config:b"});
}

TEST_CASE("Clicks and drags inside a detached surface column reach the surface and keep its focus",
          "[core][sakura][flowgraph][pane]") {
    PaneRig rig;
    rig.screen = kWideScreen;
    std::vector<MouseEventType> received;
    U64 focusLost = 0;
    const auto texture = std::make_shared<SurfaceTexture>();
    rig.tweak = [&](FlowgraphPane::Config& config) {
        config.detachedSurfaces = {{
            .id = "surface:a:default",
            .title = "A",
            .texture = texture,
            .onInput = [&](InputEvent event) {
                if (const auto* mouse = std::get_if<MouseEvent>(&event)) {
                    received.push_back(mouse->type);
                }
                if (const auto* focus = std::get_if<FocusEvent>(&event); focus && !focus->focused) {
                    ++focusLost;
                }
            },
        }};
    };
    rig.state.flowgraph.layouts[kGraph] = LayoutSplit("row", {LayoutLeaf("flowgraph"), LayoutLeaf("surface:a:default")});
    rig.frame();
    rig.frame();
    const auto images = Find<Sakura::SurfaceImage>(rig.layoutDock());
    REQUIRE(images.size() == 1);
    const auto image = Peek::Frame(*images[0]);
    const F32 x = image.x + image.width * 0.5f;
    const F32 y = image.y + image.height * 0.5f;

    rig.host.send(Wheel(x, y, 0.0f, 1.0f));
    rig.host.send(Mouse(MouseEventType::Click, x, y));
    ClickFocus(rig.host, Extent2D<F32>{x, y});
    CHECK(focusLost == 0);
    CHECK(Peek::CursorAt(*images[0], {x, y}) == SurfaceCursor::Default);
    rig.host.send(Mouse(MouseEventType::Move, x + 20.0f, y + 10.0f));
    rig.host.send(Mouse(MouseEventType::Release, x + 20.0f, y + 10.0f));
    CHECK(std::find(received.begin(), received.end(), MouseEventType::Scroll) != received.end());
    CHECK(std::find(received.begin(), received.end(), MouseEventType::Click) != received.end());
    CHECK(std::find(received.begin(), received.end(), MouseEventType::Move) != received.end());
    CHECK(std::find(received.begin(), received.end(), MouseEventType::Release) != received.end());
    CHECK(std::find(received.begin(), received.end(), MouseEventType::Leave) == received.end());

    ClickFocus(rig.host, Extent2D<F32>{20.0f, 20.0f});
    CHECK(focusLost == 1);
}

TEST_CASE("The editor's top right holds the search pill, the zoom pill to its left, and a dependency pill when review is needed",
          "[core][sakura][flowgraph][pane]") {
    PaneRig rig;
    rig.screen = kWideScreen;
    U64 reviews = 0;
    bool review = false;
    rig.tweak = [&](FlowgraphPane::Config& config) {
        config.dependencyReviewAvailable = review;
        config.dependencyReviewTone = Callout::Tone::Warning;
        config.dependencyReviewMessage = "Python dependencies need approval.";
        config.onReviewDependencies = [&] {
            ++reviews;
        };
    };
    rig.frame();
    auto pills = Find<Sakura::PillField>(rig.layoutDock());
    REQUIRE(pills.size() == 1);
    const auto search = Peek::Frame(*pills[0]);
    CHECK(search.width == FlowgraphBlockPicker::PillWidth);
    CHECK(search.right() == 1188.0f - FlowgraphBlockPicker::PillMargin);
    CHECK(search.y == 38.0f + Sakura::Dock::ContentPadding + FlowgraphBlockPicker::PillMargin);

    const auto steppers = Find<Sakura::PillStepper>(rig.layoutDock());
    REQUIRE(steppers.size() == 1);
    const auto zoom = Peek::Frame(*steppers[0]);
    CHECK(zoom.width == FlowgraphEditor::ZoomPillWidth);
    CHECK(zoom.right() == search.x - FlowgraphEditor::PillGap);
    CHECK(zoom.y == search.y);
    const auto label = [&] {
        for (auto* text : Find<Sakura::Text>(*steppers[0])) {
            if (text->config().str.ends_with("%")) {
                return text->config().str;
            }
        }
        return std::string();
    };
    CHECK(label() == "100%");
    rig.host.click(zoom.right() - 10.0f, zoom.y + 17.0f);
    rig.frame();
    CHECK(label() == "110%");
    rig.host.click(zoom.x + zoom.width * 0.5f, zoom.y + 17.0f);
    rig.frame();
    CHECK(label() == "100%");

    review = true;
    rig.frame();
    pills = Find<Sakura::PillField>(rig.layoutDock());
    REQUIRE(pills.size() == 2);
    const auto dependency = Peek::Frame(*pills[0]).x < Peek::Frame(*pills[1]).x ? Peek::Frame(*pills[0])
                                                                                : Peek::Frame(*pills[1]);
    CHECK(dependency.right() == zoom.x - FlowgraphEditor::PillGap);
    CHECK(dependency.y == search.y);
    rig.host.click(dependency.x + 20.0f, dependency.y + 10.0f);
    CHECK(reviews == 1);
}


TEST_CASE("Documentation from a node opens one tile beside the editor with that block's markdown",
          "[core][sakura][flowgraph][pane][documentation]") {
    PaneRig rig;
    rig.screen = kWideScreen;
    rig.documentedBlocks();
    rig.frame();
    rig.deliver();
    CHECK(rig.documentation().empty());

    rig.openDocumentation("a");
    CHECK(rig.count<MailSetFlowgraphLayout>() == 0);
    REQUIRE(rig.state.flowgraph.documentation.contains(kGraph));
    CHECK(rig.state.flowgraph.documentation.at(kGraph).block == "a");
    CHECK(LayoutPanes(rig.state.flowgraph.layouts.at(kGraph)) == std::vector<std::string>{"flowgraph", "documentation"});

    const auto docs = rig.documentation();
    REQUIRE(docs.size() == 1);
    CHECK(Peek::Frame(*docs[0]) == Padded(Body({774.0f, 38.0f, 420.0f, 556.0f}), Sakura::Dock::ContentPadding));
    CHECK(rig.documentationText().find("Flowgraph fixture block without") != std::string::npos);
    CHECK(rig.header("Synthetic Isolated Documentation (a)"));
    CHECK(rig.tiles().size() == 1);
    CHECK(Peek::Frame(*rig.tiles()[0]) == Padded(Body({6.0f, 38.0f, 762.0f, 556.0f}), Sakura::Dock::ContentPadding));
}

TEST_CASE("Documentation from another node swaps the content of the same single tile",
          "[core][sakura][flowgraph][pane][documentation]") {
    PaneRig rig;
    rig.screen = kWideScreen;
    rig.documentedBlocks();
    rig.frame();
    rig.deliver();
    rig.openDocumentation("a");
    const auto first = rig.documentation();
    REQUIRE(first.size() == 1);
    const auto frame = Peek::Frame(*first[0]);

    rig.openDocumentation("b");
    CHECK(rig.count<MailSetFlowgraphLayout>() == 0);
    CHECK(rig.state.flowgraph.documentation.at(kGraph).block == "b");
    CHECK(LayoutPanes(rig.state.flowgraph.layouts.at(kGraph)) == std::vector<std::string>{"flowgraph", "documentation"});
    const auto second = rig.documentation();
    REQUIRE(second.size() == 1);
    CHECK(second[0] == first[0]);
    CHECK(Peek::Frame(*second[0]) == frame);
    const auto text = rig.documentationText();
    CHECK(text.find("Flowgraph fixture source block.") != std::string::npos);
    CHECK(text.find("Flowgraph fixture block without") == std::string::npos);
    CHECK(rig.header("Synthetic Source Documentation (b)"));
    CHECK_FALSE(rig.header("Synthetic Isolated Documentation (a)"));

    rig.openDocumentation("b");
    CHECK(rig.documentation().size() == 1);
    CHECK(LayoutPanes(rig.state.flowgraph.layouts.at(kGraph)) == std::vector<std::string>{"flowgraph", "documentation"});
}

TEST_CASE("Closing the documentation tile removes it from the dock and the layout",
          "[core][sakura][flowgraph][pane][documentation]") {
    PaneRig rig;
    rig.screen = kWideScreen;
    rig.documentedBlocks();
    rig.frame();
    rig.deliver();
    rig.openDocumentation("a");
    REQUIRE(rig.documentation().size() == 1);

    const auto close = CloseOf({774.0f, 38.0f, 420.0f, 556.0f});
    rig.host.click(close.x, close.y);
    REQUIRE(rig.count<MailCloseDocumentation>() == 1);
    CHECK(rig.last<MailCloseDocumentation>()->flowgraph == kGraph);
    rig.deliver();
    rig.frame();
    rig.deliver();
    rig.frame();
    CHECK(rig.count<MailSetFlowgraphLayout>() == 0);
    CHECK_FALSE(rig.state.flowgraph.documentation.contains(kGraph));
    CHECK(LayoutPanes(rig.state.flowgraph.layouts.at(kGraph)) == std::vector<std::string>{"flowgraph"});
    CHECK(rig.documentation().empty());
    CHECK(HeaderLabels(rig.layoutDock()).empty());
    CHECK(Peek::Frame(*rig.tiles()[0]) == Padded({6.0f, kBar + 6.0f, 1188.0f, 556.0f}, Sakura::Dock::ContentPadding));
}

TEST_CASE("A documentation tile settles under identical updates and swapping its block only repaints",
          "[core][sakura][flowgraph][pane][documentation][retention]") {
    PaneRig rig;
    rig.documentedBlocks();
    Frames frames(rig.host, kScreen, rig.ctx);
    const auto update = [&] {
        rig.update();
        rig.deliver();
    };
    REQUIRE(Settles(frames, update));
    rig.build().editor.onOpenDocumentation("a");
    rig.deliver();
    REQUIRE(Settles(frames, update));

    const U64 restructures = frames.restructures;
    rig.build().editor.onOpenDocumentation("b");
    rig.deliver();
    CHECK(Kept(frames.run(update)));
    REQUIRE(Settles(frames, update));
    CHECK(frames.restructures == restructures);
}

TEST_CASE("Rename completion and late dock reports preserve detached pane order and widths through save and reload",
          "[core][sakura][flowgraph][pane][layout-rename]") {
    const int detached = GENERATE(0, 1, 2);
    const bool late = GENERATE(false, true);
    const bool configDetached = detached != 1;
    const bool surfaceDetached = detached != 0;
    PaneRig rig;
    rig.screen = kWideScreen;
    REQUIRE(rig.graph().blockCreate("old", TestFlowgraph::kSyntheticSourceType, {}, {}) == Result::SUCCESS);
    REQUIRE(rig.graph().metadata().set("config", ConfigMeta{.detached = configDetached}, "old") == Result::SUCCESS);
    REQUIRE(rig.graph().metadata().set("surface_default", SurfaceMeta{.detached = surfaceDetached}, "old") == Result::SUCCESS);
    TileMeta initial{.split = "row", .children = std::vector<TileMeta>{{.weight = 600, .pane = "flowgraph"}}};
    if (configDetached) initial.children->push_back({.weight = 210, .pane = "config:old"});
    if (surfaceDetached) initial.children->push_back({.weight = 310, .pane = "surface:old:default"});
    rig.state.flowgraph.layouts[kGraph] = initial;
    rig.state.flowgraph.restoredLayouts.insert(kGraph);
    REQUIRE(rig.graph().metadata().set("layout", initial) == Result::SUCCESS);

    const auto snapshot = [](PaneRig& target) {
        auto& blocks = target.state.flowgraph.blocks[kGraph];
        blocks.clear();
        std::vector<std::string> names;
        REQUIRE(target.graph().view().keys(names) == Result::SUCCESS);
        for (const auto& name : names) blocks.push_back({.name = name, .data = ViewBlock(target.graph(), name)});
    };
    const auto surfaces = [](PaneRig& target) {
        target.tweak = [&target](FlowgraphPane::Config& config) {
            for (const auto& [name, _] : target.state.flowgraph.blocks.at(kGraph)) {
                SurfaceMeta meta;
                if (target.graph().metadata().get("surface_default", meta, name) == Result::SUCCESS && meta.detached) {
                    config.detachedSurfaces.push_back({.id = MakeSurfaceTileKey(name, "default"), .title = name,
                                                       .texture = std::make_shared<SurfaceTexture>()});
                }
            }
        };
    };
    const auto drain = [](PaneRig& target) {
        LayoutActions layouts(target.state.flowgraph);
        auto pending = std::exchange(target.mails, {});
        for (const auto& mail : pending) {
            if (const auto* layout = std::get_if<MailSetFlowgraphLayout>(&mail)) REQUIRE(layouts.handle(*layout) == Result::SUCCESS);
            if (const auto* rename = std::get_if<MailRenameLayoutBlock>(&mail)) REQUIRE(layouts.handle(*rename) == Result::SUCCESS);
        }
    };
    surfaces(rig);
    snapshot(rig);
    for (U64 frame = 0; frame < 3; ++frame) { rig.frame(); drain(rig); }
    const TileMeta before = rig.state.flowgraph.layouts.at(kGraph);
    const auto configFrames = rig.configFrames();
    const auto images = Find<Sakura::SurfaceImage>(rig.layoutDock());
    const std::optional<Jetstream::Rect> surfaceFrame = images.empty() ? std::nullopt : std::optional(Peek::Frame(*images.front()));
    std::function<Result()> command;
    std::optional<Mail> completion;
    rig.callbacks.enqueueCommandWithMail = [&](std::function<Result()> fn, bool, Mail mail) {
        command = std::move(fn);
        completion = std::move(mail);
    };
    FlowgraphActions flowgraphs(rig.state, rig.callbacks);
    REQUIRE(flowgraphs.handle(MailRenameBlock{.flowgraph = kGraph, .oldId = "old", .newId = "new"}) == Result::SUCCESS);
    rig.mails.clear();
    REQUIRE(command);
    REQUIRE(command() == Result::SUCCESS);
    if (late) {
        snapshot(rig);
        rig.frame();
        REQUIRE(rig.count<MailSetFlowgraphLayout>() > 0);
        drain(rig);
    }
    REQUIRE(completion.has_value());
    rig.mails.push_back(std::move(*completion));
    drain(rig);
    snapshot(rig);
    for (U64 frame = 0; frame < 3; ++frame) { rig.frame(); drain(rig); }
    const auto expected = MapTiles(before, [](const std::string& pane) -> std::optional<std::string> {
        if (pane == "config:old") return "config:new";
        if (pane == "surface:old:default") return "surface:new:default";
        return pane;
    });
    REQUIRE(expected.has_value());
    CHECK(Parser::Hash(rig.state.flowgraph.layouts.at(kGraph)) == Parser::Hash(*expected));
    CHECK(rig.configFrames() == configFrames);
    if (surfaceFrame) {
        const auto renamedImages = Find<Sakura::SurfaceImage>(rig.layoutDock());
        REQUIRE(renamedImages.size() == 1);
        CHECK(Peek::Frame(*renamedImages.front()) == *surfaceFrame);
    }
    std::vector<char> saved;
    REQUIRE(rig.graph().exportToBlob(saved) == Result::SUCCESS);
    REQUIRE(rig.graph().blockDestroy("new") == Result::SUCCESS);
    rig.state.flowgraph.layouts.clear();
    rig.state.flowgraph.restoredLayouts.clear();
    rig.state.flowgraph.blocks.clear();
    REQUIRE(rig.graph().importFromBlob(saved) == Result::SUCCESS);
    LayoutActions layouts(rig.state.flowgraph);
    layouts.restoreFromMetadata();
    CHECK(Parser::Hash(rig.state.flowgraph.layouts.at(kGraph)) == Parser::Hash(*expected));
    snapshot(rig);
    rig.frame();
    CHECK(rig.configFrames() == configFrames);
    const auto panes = LayoutPanes(rig.state.flowgraph.layouts.at(kGraph));
    CHECK(std::count(panes.begin(), panes.end(), "config:new") == (configDetached ? 1 : 0));
    CHECK(std::count(panes.begin(), panes.end(), "surface:new:default") == (surfaceDetached ? 1 : 0));
    CHECK(std::none_of(panes.begin(), panes.end(), [](const auto& pane) { return pane.find(":old") != std::string::npos; }));
}
