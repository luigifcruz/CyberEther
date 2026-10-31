#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <jetstream/render/components/text.hh>
#include <jetstream/render/sakura/clipboard.hh>
#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/card.hh>
#include <jetstream/render/sakura/components/chrome_button.hh>
#include <jetstream/render/sakura/components/label.hh>
#include <jetstream/render/sakura/components/polyline.hh>
#include <jetstream/render/sakura/components/terminal.hh>
#include <jetstream/render/sakura/components/text.hh>
#include <jetstream/render/sakura/components/text_input.hh>

#include "../../../resources/fonts/compressed_inter.hh"
#include "component_harness.hh"
#include "render/sakura/detail/text_metrics.hh"
#include "scene.hh"

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>
#include <vector>

using namespace Jetstream;
using namespace SakuraTest;

namespace {

constexpr const char* kTags = "[core][sakura][terminal]";

using Terminal = Sakura::Terminal;
using Box = Sakura::Box;

constexpr ColorRGBA<F32> kRed = {1.0f, 0.0f, 0.0f, 1.0f};
constexpr ColorRGBA<F32> kWhite = {1.0f, 1.0f, 1.0f, 1.0f};
constexpr ColorRGBA<F32> kBlue = {0.0f, 0.0f, 1.0f, 1.0f};
constexpr Rect kBounds = {10.0f, 10.0f, 400.0f, 200.0f};
constexpr F32 kPadding = 8.0f;

struct Recorder {
    std::vector<Terminal::Grid> grids;
    std::vector<KeyEvent> keys;
    std::vector<KeyEvent> releases;
    std::vector<std::string> texts;
    std::vector<std::string> pastes;
    std::vector<I32> scrolls;
    std::vector<U64> rows;
    std::vector<Terminal::Selection> selections;
    std::vector<MouseEvent> pointers;
    std::vector<bool> focus;
    std::vector<I32> zooms;
    std::vector<std::string> links;
    std::vector<std::optional<std::string>> searches;
    std::vector<bool> steps;

    Terminal::Config wire(const Terminal::Screen* screen, U64 revision) {
        return {
            .id = "terminal",
            .screen = screen,
            .revision = revision,
            .padding = kPadding,
            .onResize = [this](Terminal::Grid grid) { grids.push_back(grid); },
            .onKey = [this](const KeyEvent& event) {
                (event.type == KeyEventType::Release ? releases : keys).push_back(event);
            },
            .onText = [this](std::string text, KeyModifiers) { texts.push_back(std::move(text)); },
            .onPaste = [this](std::string text) { pastes.push_back(std::move(text)); },
            .onScroll = [this](I32 delta, Extent2D<F32>, KeyModifiers) { scrolls.push_back(delta); },
            .onScrollTo = [this](U64 row) { rows.push_back(row); },
            .onSelect = [this](Terminal::Selection selection) { selections.push_back(selection); },
            .onPointer = [this](const MouseEvent& event) { pointers.push_back(event); },
            .onFocusChange = [this](bool focused) { focus.push_back(focused); },
            .onZoom = [this](I32 step) { zooms.push_back(step); },
            .onOpenLink = [this](std::string uri) { links.push_back(std::move(uri)); },
            .onSearch = [this](std::optional<std::string> query) { searches.push_back(std::move(query)); },
            .onSearchStep = [this](bool older) { steps.push_back(older); },
        };
    }
};

Terminal::Screen MakeScreen(U16 columns = 40, U16 rows = 8) {
    Terminal::Screen screen;
    screen.columns = columns;
    screen.rows = rows;
    screen.lines.resize(rows);
    screen.scrollTotal = rows;
    return screen;
}

Extent2D<F32> CellCenter(const Terminal::Grid& grid, U16 column, U16 row) {
    return {kBounds.x + kPadding + (static_cast<F32>(column) + 0.5f) * grid.cellWidth,
            kBounds.y + kPadding + (static_cast<F32>(row) + 0.5f) * grid.cellHeight};
}

const Sakura::Label& Glyphs(Sakura::Component& root, U64 index = 0) {
    const auto labels = Find<Sakura::Label>(root);
    REQUIRE(labels.size() > index);
    return *labels[index];
}

const Sakura::Box& Boxes(Sakura::Component& root, U64 index) {
    const auto boxes = Find<Sakura::Box>(root);
    REQUIRE(boxes.size() > index);
    return *boxes[index];
}

const Sakura::Box& Surface(Sakura::Component& root) {
    return Boxes(root, 0);
}

const Sakura::Box& Fills(Sakura::Component& root) {
    return Boxes(root, 1);
}

const Sakura::Box& CursorBox(Sakura::Component& root) {
    return Boxes(root, 2);
}

const Sakura::Box& Sprites(Sakura::Component& root) {
    return Boxes(root, 3);
}

const Sakura::Box& Decorations(Sakura::Component& root) {
    return Boxes(root, 4);
}

const Sakura::Polyline& SpriteStrokes(Sakura::Component& root) {
    const auto lines = Find<Sakura::Polyline>(root);
    REQUIRE_FALSE(lines.empty());
    return *lines.front();
}

const Sakura::Label& CursorGlyph(Sakura::Component& root) {
    return Glyphs(root, 4);
}

Rect GridCell(const Terminal::Grid& grid, U16 column, U16 row) {
    return {kBounds.x + kPadding + static_cast<F32>(column) * grid.cellWidth,
            kBounds.y + kPadding + static_cast<F32>(row) * grid.cellHeight,
            grid.cellWidth,
            grid.cellHeight};
}

std::vector<Box::Instance> InsideCell(const Sakura::Box& box, const Rect& cell) {
    std::vector<Box::Instance> found;
    for (const auto& instance : box.config().instances) {
        if (instance.rect.x >= cell.x && instance.rect.right() <= cell.right() && instance.rect.y >= cell.y &&
            instance.rect.bottom() <= cell.bottom()) {
            found.push_back(instance);
        }
    }
    return found;
}

struct Editor : Sakura::Component {
    void focus() { requestFocus(); }
    bool editsText() const override { return true; }
    Extent2D<F32> measure(const Sakura::Context&, Extent2D<F32>) override { return {100.0f, 20.0f}; }
};

struct Beside : Sakura::Component {
    Terminal& terminal;
    Editor editor;

    explicit Beside(Terminal& terminal) : terminal(terminal) {
        addChild(terminal);
        addChild(editor);
    }

    Extent2D<F32> measure(const Sakura::Context&, Extent2D<F32> available) override { return available; }

    void layout(const Sakura::Context& ctx) override {
        layoutChild(ctx, terminal, frame());
        layoutChild(ctx, editor, {frame().x, frame().bottom(), 100.0f, 20.0f});
    }
};

std::vector<std::string> LabelStrings(Sakura::Component& root) {
    std::vector<std::string> strings;
    for (const auto* label : Find<Sakura::Label>(root)) {
        for (const auto& instance : label->config().instances) {
            if (instance.visible) {
                strings.push_back(instance.str);
            }
        }
    }
    return strings;
}

}  // namespace

TEST_CASE("Terminals measure natural, fixed, and filling sizes before their first layout", kTags) {
    Scene scene;
    Terminal terminal;
    Host host(terminal);

    terminal.update({});
    CHECK(host.measureOf(scene.ctx, {300.0f, 200.0f}) == Extent2D<F32>{300.0f, 200.0f});
    const auto natural = host.measureOf(scene.ctx, {kInfinite, kInfinite});
    CHECK(std::isfinite(natural.x));
    CHECK(std::isfinite(natural.y));
    CHECK(natural.x > 300.0f);
    CHECK(natural.y > 200.0f);

    terminal.update({.size = {0.0f, 0.0f}});
    CHECK(host.measureOf(scene.ctx, {kInfinite, kInfinite}) == natural);
    CHECK(host.measureOf(scene.ctx, {300.0f, 200.0f}) == Extent2D<F32>{300.0f, 200.0f});

    terminal.update({.size = {250.0f, 120.0f}});
    CHECK(host.measureOf(scene.ctx, {kInfinite, kInfinite}) == Extent2D<F32>{250.0f, 120.0f});
    CHECK(host.measureOf(scene.ctx, {100.0f, 60.0f}) == Extent2D<F32>{100.0f, 60.0f});

    terminal.update({.size = {Sakura::Fill, 0.0f}});
    CHECK(host.measureOf(scene.ctx, {300.0f, kInfinite}) == Extent2D<F32>{300.0f, natural.y});
}

TEST_CASE("Terminals report their grid once for each size", kTags) {
    Scene scene;
    Recorder recorder;
    auto screen = MakeScreen();
    Terminal terminal;
    Place place(terminal, kBounds);
    Frames frames(place, kScreen, scene.context());
    const auto update = [&] {
        terminal.update(recorder.wire(&screen, 1));
    };
    REQUIRE(frames.settle(update));
    REQUIRE(recorder.grids.size() == 1);
    const auto grid = recorder.grids.front();
    REQUIRE(grid.cellWidth > 0.0f);
    REQUIRE(grid.cellHeight > 0.0f);
    CHECK(grid.columns == static_cast<U16>(std::floor((kBounds.width - 2.0f * kPadding) / grid.cellWidth)));
    CHECK(grid.rows == static_cast<U16>(std::floor((kBounds.height - 2.0f * kPadding) / grid.cellHeight)));
    CHECK(Retains(frames, update));
    CHECK(recorder.grids.size() == 1);

    place.move({10.0f, 10.0f, 260.0f, 200.0f});
    REQUIRE(frames.settle(update));
    REQUIRE(recorder.grids.size() == 2);
    CHECK(recorder.grids.back().columns < grid.columns);
    CHECK(recorder.grids.back().rows == grid.rows);

    auto replacement = MakeScreen();
    terminal.update(recorder.wire(&replacement, 1));
    REQUIRE(frames.settle([&] { terminal.update(recorder.wire(&replacement, 1)); }));
    CHECK(recorder.grids.size() == 3);
}

TEST_CASE("Terminals fill their frame by default and overlay the scrollbar", kTags) {
    Scene scene;
    Recorder recorder;
    auto screen = MakeScreen();
    screen.lines[0].spans = {{.column = 0, .width = 2, .text = "ls", .color = kWhite}};
    Terminal terminal;
    Place place(terminal, kBounds);
    Frames frames(place, kScreen, scene.context());
    auto update = [&] {
        auto config = recorder.wire(&screen, screen.scrollTotal);
        config.padding = Terminal::Config{}.padding;
        terminal.update(std::move(config));
    };
    REQUIRE(frames.settle(update));
    const auto grid = recorder.grids.back();
    CHECK(Terminal::Config{}.padding == 0.0f);
    CHECK(grid.columns == static_cast<U16>(std::floor(kBounds.width / grid.cellWidth)));
    CHECK(grid.rows == static_cast<U16>(std::floor(kBounds.height / grid.cellHeight)));
    CHECK(Glyphs(place, 0).config().instances[0].rect.x == Catch::Approx(kBounds.x));

    screen.scrollTotal = screen.rows + 50;
    REQUIRE(frames.settle(update));
    CHECK(recorder.grids.back().columns == grid.columns);
}

TEST_CASE("Terminal cells and padding scale with the pixel ratio", kTags) {
    Scene scene;
    Recorder single;
    Recorder doubled;
    auto screen = MakeScreen();

    Terminal first;
    Place firstPlace(first, kBounds);
    Frames firstFrames(firstPlace, kScreen, scene.context());
    REQUIRE(firstFrames.settle([&] { first.update(single.wire(&screen, 1)); }));

    auto ctx = scene.context();
    ctx.pixelRatio = 2.0f;
    Terminal second;
    Place secondPlace(second, {10.0f, 10.0f, 800.0f, 400.0f});
    Frames secondFrames(secondPlace, {0.0f, 0.0f, 1600.0f, 1200.0f}, ctx);
    REQUIRE(secondFrames.settle([&] { second.update(doubled.wire(&screen, 1)); }));

    REQUIRE(single.grids.size() == 1);
    REQUIRE(doubled.grids.size() == 1);
    CHECK(doubled.grids.front().cellHeight == Catch::Approx(2.0f * single.grids.front().cellHeight).margin(1.0f));
    CHECK(doubled.grids.front().cellWidth == Catch::Approx(2.0f * single.grids.front().cellWidth).margin(1.0f));
    CHECK(doubled.grids.front().rows == single.grids.front().rows);
    CHECK(doubled.grids.front().columns ==
          static_cast<U16>(std::floor((800.0f - 4.0f * kPadding) / doubled.grids.front().cellWidth)));

    const auto& background = Fills(secondPlace).config();
    screen.lines[0].backgrounds = {{.column = 0, .width = 1, .color = kBlue}};
    REQUIRE(secondFrames.settle([&] { second.update(doubled.wire(&screen, 2)); }));
    REQUIRE(background.instances.size() == 1);
    CHECK(background.instances[0].rect.x == Catch::Approx(10.0f + 2.0f * kPadding));
    CHECK(background.instances[0].rect.y == Catch::Approx(10.0f + 2.0f * kPadding));
}

TEST_CASE("Terminal glyphs share a label per color within sixteen cells and keep their cell columns", kTags) {
    Scene scene;
    Recorder recorder;
    auto screen = MakeScreen();
    screen.lines[0].spans = {
        {.column = 0, .width = 2, .text = "ls", .color = kRed},
        {.column = 3, .width = 3, .text = "foo", .color = kWhite},
        {.column = 5, .width = 1, .text = "x", .color = kRed},
        {.column = 12, .width = 4, .text = "bold", .color = kWhite, .bold = true},
        {.column = 17, .width = 3, .text = "bar", .color = kRed},
    };
    Terminal terminal;
    Place place(terminal, kBounds);
    Frames frames(place, kScreen, scene.context());
    REQUIRE(frames.settle([&] { terminal.update(recorder.wire(&screen, 1)); }));
    const auto grid = recorder.grids.front();

    const auto& regular = Glyphs(place, 0).config();
    REQUIRE(regular.instances.size() == 3);
    CHECK(regular.instances[0].str == "ls   x");
    CHECK(regular.instances[0].color == kRed);
    CHECK(regular.instances[0].rect.x == Catch::Approx(kBounds.x + kPadding));
    CHECK(regular.instances[1].str == "foo");
    CHECK(regular.instances[1].color == kWhite);
    CHECK(regular.instances[1].rect.x == Catch::Approx(kBounds.x + kPadding + 3.0f * grid.cellWidth));
    CHECK(regular.instances[2].str == "bar");
    CHECK(regular.instances[2].color == kRed);
    CHECK(regular.instances[2].rect.x == Catch::Approx(kBounds.x + kPadding + 17.0f * grid.cellWidth));
    CHECK(regular.fontName == "default_mono");

    const auto& bold = Glyphs(place, 1).config();
    REQUIRE(bold.instances.size() == 1);
    CHECK(bold.instances[0].str == "bold");
    CHECK(bold.instances[0].rect.x == Catch::Approx(kBounds.x + kPadding + 12.0f * grid.cellWidth));
    CHECK(bold.fontName == "default_mono_bold");
}

TEST_CASE("Terminal glyphs advance by whole-pixel cells at every pixel ratio", kTags) {
    for (const F32 ratio : {1.0f, 2.0f}) {
        Scene scene;
        auto ctx = scene.context();
        ctx.pixelRatio = ratio;
        Recorder recorder;
        auto screen = MakeScreen();
        screen.lines[0].spans = {{.column = 0, .width = 40, .text = std::string(40, 'm'), .color = kWhite}};
        Terminal terminal;
        Place place(terminal, kBounds);
        Frames frames(place, kScreen, ctx);
        REQUIRE(frames.settle([&] { terminal.update(recorder.wire(&screen, 1)); }));
        const auto grid = recorder.grids.back();

        const auto& regular = Glyphs(place, 0).config();
        REQUIRE(regular.instances.size() == 3);
        Sakura::TextMetrics metrics;
        metrics.setWindow(ctx.render);
        CHECK(grid.cellWidth == std::round(grid.cellWidth));
        for (U64 index = 0; index < regular.instances.size(); ++index) {
            const auto& chunk = regular.instances[index];
            const F32 run = metrics.measure(regular.fontName, chunk.str, chunk.fontSize);
            const U64 length = index < 2 ? 16 : 8;
            CHECK(chunk.str == std::string(length, 'm'));
            CHECK(run == Catch::Approx(static_cast<F32>(length) * grid.cellWidth).margin(0.01f));
            CHECK(chunk.rect.x - regular.instances[0].rect.x == Catch::Approx(16.0f * index * grid.cellWidth));
        }
    }
}

TEST_CASE("Terminal backgrounds, selections, and decorations draw cell boxes", kTags) {
    Scene scene;
    Recorder recorder;
    auto screen = MakeScreen();
    screen.lines[1].backgrounds = {{.column = 2, .width = 3, .color = kBlue}};
    screen.lines[2].selection = Terminal::Range{.start = 1, .end = 4};
    screen.lines[3].spans = {{.column = 0, .width = 3, .text = "url", .color = kWhite, .underline = true}};
    Terminal terminal;
    Place place(terminal, kBounds);
    const auto ctx = scene.context();
    Frames frames(place, kScreen, ctx);
    REQUIRE(frames.settle([&] { terminal.update(recorder.wire(&screen, 1)); }));
    const auto grid = recorder.grids.front();

    const auto& fills = Fills(place).config();
    REQUIRE(fills.instances.size() == 2);
    CHECK(fills.instances[0].backgroundColor == kBlue);
    CHECK(fills.instances[0].rect.x == Catch::Approx(kBounds.x + kPadding + 2.0f * grid.cellWidth));
    CHECK(fills.instances[0].rect.width == Catch::Approx(3.0f * grid.cellWidth));
    CHECK(fills.instances[0].rect.y == Catch::Approx(kBounds.y + kPadding + grid.cellHeight));
    CHECK(fills.instances[1].backgroundColor == ctx.color("editor_selection"));
    CHECK(fills.instances[1].rect.width == Catch::Approx(4.0f * grid.cellWidth));

    const auto& lines = Decorations(place).config();
    REQUIRE(lines.instances.size() == 1);
    CHECK(lines.instances[0].rect.width == Catch::Approx(3.0f * grid.cellWidth));
    CHECK(lines.instances[0].rect.bottom() == Catch::Approx(kBounds.y + kPadding + 4.0f * grid.cellHeight));
}

TEST_CASE("Terminals settle and repaint only when the screen changes", "[core][sakura][terminal][retention]") {
    Scene scene;
    Recorder recorder;
    auto screen = MakeScreen();
    U64 revision = 1;
    Terminal terminal;
    Place place(terminal, kBounds);
    Frames frames(place, kScreen, scene.context());
    const auto update = [&] {
        terminal.update(recorder.wire(&screen, revision));
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
    CHECK(QuietMoves(frames, update, {{60.0f, 60.0f}, {120.0f, 90.0f}}, [&](const MouseEvent& event) {
        return place.send(event);
    }));

    screen.lines[0].spans = {{.column = 0, .width = 5, .text = "hello", .color = kWhite}};
    ++revision;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));

    screen.lines[0].spans = {{.column = 0, .width = 5, .text = "world", .color = kRed}};
    screen.lines[2].backgrounds = {{.column = 0, .width = 2, .color = kBlue}};
    ++revision;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));

    ++revision;
    CHECK(Unpainted(frames.run(update)));
    CHECK(Retains(frames, update));
    CHECK(frames.restructures == 0);
}

TEST_CASE("Dragging across a terminal reports selection gestures by cell", kTags) {
    Scene scene;
    Recorder recorder;
    auto screen = MakeScreen();
    Terminal terminal;
    Place place(terminal, kBounds);
    Frames frames(place, kScreen, scene.context());
    const auto update = [&] {
        terminal.update(recorder.wire(&screen, 1));
    };
    REQUIRE(frames.settle(update));
    const auto grid = recorder.grids.front();

    const auto start = CellCenter(grid, 2, 1);
    const auto end = CellCenter(grid, 6, 3);
    CHECK(place.send(Mouse(MouseEventType::Click, start.x, start.y, MouseButton::Left, 2)));
    CHECK(place.send(Mouse(MouseEventType::Move, start.x, start.y)));
    CHECK(place.send(Mouse(MouseEventType::Move, end.x, end.y)));
    CHECK(place.send(Mouse(MouseEventType::Move, end.x, end.y)));
    CHECK(place.send(Mouse(MouseEventType::Release, end.x, end.y)));
    REQUIRE(recorder.selections.size() == 3);
    CHECK(recorder.selections[0] == Terminal::Selection{.gesture = Terminal::Gesture::Press, .column = 2, .row = 1,
                                                        .clicks = 2});
    CHECK(recorder.selections[1].gesture == Terminal::Gesture::Drag);
    CHECK(recorder.selections[1].column == 6);
    CHECK(recorder.selections[1].row == 3);
    CHECK(recorder.selections[2].gesture == Terminal::Gesture::Release);

    CHECK(place.send(Mouse(MouseEventType::Click, start.x, start.y)));
    place.send(Mouse(MouseEventType::Cancel, start.x, start.y));
    CHECK_FALSE(place.send(Mouse(MouseEventType::Move, end.x, end.y)));
    CHECK(recorder.selections.size() == 4);
}

TEST_CASE("Terminal wheels report whole rows and mouse tracking forwards pointer events", kTags) {
    Scene scene;
    Recorder recorder;
    auto screen = MakeScreen();
    Terminal terminal;
    Place place(terminal, kBounds);
    Frames frames(place, kScreen, scene.context());
    REQUIRE(frames.settle([&] { terminal.update(recorder.wire(&screen, 1)); }));
    const auto grid = recorder.grids.front();
    const auto cell = CellCenter(grid, 3, 2);

    CHECK(place.send(Wheel(cell.x, cell.y, 1.0f)));
    CHECK(place.send(Wheel(cell.x, cell.y, -0.2f)));
    CHECK(place.send(Wheel(cell.x, cell.y, -0.2f)));
    REQUIRE(recorder.scrolls.size() == 2);
    CHECK(recorder.scrolls[0] == -3);
    CHECK(recorder.scrolls[1] == 1);

    screen.mouseTracking = true;
    REQUIRE(frames.settle([&] { terminal.update(recorder.wire(&screen, 2)); }));
    CHECK(place.send(Mouse(MouseEventType::Click, cell.x, cell.y, MouseButton::Right)));
    CHECK(place.send(Mouse(MouseEventType::Release, cell.x, cell.y, MouseButton::Right)));
    REQUIRE(recorder.pointers.size() == 2);
    CHECK(recorder.pointers[0].type == MouseEventType::Click);
    CHECK(recorder.pointers[0].button == MouseButton::Right);
    CHECK(recorder.pointers[0].position.x == Catch::Approx(cell.x - kBounds.x - kPadding));
    CHECK(recorder.pointers[0].position.y == Catch::Approx(cell.y - kBounds.y - kPadding));
    CHECK(recorder.pointers[1].type == MouseEventType::Release);
    CHECK(recorder.selections.empty());

    CHECK(place.send(Mouse(MouseEventType::Click, cell.x, cell.y, MouseButton::Left)));
    CHECK_FALSE(place.send(Mouse(MouseEventType::Leave, cell.x, cell.y)));
    REQUIRE(recorder.pointers.size() == 4);
    CHECK(recorder.pointers[3].type == MouseEventType::Cancel);
    CHECK_FALSE(place.send(Mouse(MouseEventType::Release, cell.x, cell.y, MouseButton::Left)));
    CHECK(recorder.pointers.size() == 4);
}

TEST_CASE("Scrollbar drags move the terminal viewport", kTags) {
    Scene scene;
    Recorder recorder;
    auto screen = MakeScreen();
    screen.scrollTotal = 200;
    screen.scrollOffset = 192;
    Terminal terminal;
    Place place(terminal, kBounds);
    Frames frames(place, kScreen, scene.context());
    REQUIRE(frames.settle([&] { terminal.update(recorder.wire(&screen, 1)); }));

    const F32 x = kBounds.right() - Sakura::kScrollbarMargin - Sakura::kScrollbarThickness * 0.5f;
    CHECK(place.send(Mouse(MouseEventType::Click, x, kBounds.y + 10.0f)));
    CHECK(place.send(Mouse(MouseEventType::Release, x, kBounds.y + 10.0f)));
    REQUIRE_FALSE(recorder.rows.empty());
    CHECK(recorder.rows.back() < 192);
    CHECK(recorder.selections.empty());
}

TEST_CASE("The terminal scrollbar track stays clear of the rounded corners", kTags) {
    for (const F32 ratio : {1.0f, 2.0f}) {
        Scene scene;
        auto ctx = scene.context();
        ctx.pixelRatio = ratio;
        Recorder recorder;
        auto screen = MakeScreen();
        screen.scrollTotal = 200;
        screen.scrollOffset = 192;
        Terminal terminal;
        Place place(terminal, kBounds);
        Frames frames(place, kScreen, ctx);
        REQUIRE(frames.settle([&] {
            auto config = recorder.wire(&screen, 1);
            config.rounding = 10.0f;
            terminal.update(std::move(config));
        }));
        const auto track = ctx.color("editor_scrollbar_track");
        std::optional<Rect> found;
        for (const auto* box : Find<Sakura::Box>(place)) {
            for (const auto& instance : box->config().instances) {
                if (instance.visible && instance.backgroundColor == track) {
                    found = instance.rect;
                }
            }
        }
        REQUIRE(found.has_value());
        CHECK(found->y >= kBounds.y + 10.0f * ratio);
        CHECK(found->bottom() <= kBounds.bottom() - 10.0f * ratio);
    }
}

TEST_CASE("Focused terminals send keys to the shell and keep copy, paste, and app shortcuts", kTags) {
    Scene scene;
    Recorder recorder;
    auto screen = MakeScreen();
    Terminal terminal;
    Place place(terminal, kBounds);
    Frames frames(place, kScreen, scene.context());
    const auto update = [&] {
        terminal.update(recorder.wire(&screen, 1));
    };
    REQUIRE(frames.settle(update));
    CHECK(FocusOwner(place) == &terminal);
    REQUIRE(recorder.focus == std::vector<bool>{true});

    CHECK(Tap(place, KeyCode::Enter));
    CHECK(Tap(place, KeyCode::C, {.control = true}));
    REQUIRE(recorder.keys.size() == 2);
    CHECK(recorder.keys[0].key == KeyCode::Enter);
    CHECK(recorder.keys[1].modifiers.control);

    auto* window = scene.ctx.render;
    Sakura::SetClipboardText(window, "");
    CHECK(Tap(place, KeyCode::C, {.super = true}));
    CHECK(Sakura::GetClipboardText(window).empty());
    screen.selection = "selected text";
    CHECK(Tap(place, KeyCode::C, {.super = true}));
    CHECK(Sakura::GetClipboardText(window) == "selected text");
    screen.selection = "from shift control";
    CHECK(Tap(place, KeyCode::C, {.control = true, .shift = true}));
    CHECK(Sakura::GetClipboardText(window) == "from shift control");

    Sakura::SetClipboardText(window, "pasted");
    CHECK(Tap(place, KeyCode::V, {.super = true}));
    REQUIRE(recorder.pastes == std::vector<std::string>{"pasted"});
    for (const auto key : {KeyCode::C, KeyCode::V}) {
        CHECK(Sakura::Access::key(place, {.type = KeyEventType::Press,
                                          .key = key,
                                          .modifiers = {.control = true, .shift = true},
                                          .repeat = true}));
    }
    CHECK(recorder.keys.size() == 2);
    CHECK(recorder.pastes.size() == 1);

    CHECK_FALSE(Tap(place, KeyCode::N, {.super = true}));
    CHECK_FALSE(Tap(place, KeyCode::GraveAccent, {.control = true}));
    CHECK(recorder.keys.size() == 2);

    Type(place, "ls");
    Sakura::Access::text(place, {.codepoint = U'c', .modifiers = {.control = true}});
    CHECK(recorder.texts == std::vector<std::string>{"l", "s"});

    ClickFocus(place, std::nullopt);
    CHECK(FocusOwner(place) == nullptr);
    CHECK(recorder.focus == std::vector<bool>{true, false});
}

TEST_CASE("Command plus, minus, and zero zoom the terminal instead of reaching the shell", kTags) {
    Scene scene;
    Recorder recorder;
    auto screen = MakeScreen();
    Terminal terminal;
    Place place(terminal, kBounds);
    Frames frames(place, kScreen, scene.context());
    REQUIRE(frames.settle([&] { terminal.update(recorder.wire(&screen, 1)); }));
    REQUIRE(FocusOwner(place) == &terminal);

    CHECK(Tap(place, KeyCode::Equal, {.super = true}));
    CHECK(Tap(place, KeyCode::Equal, {.shift = true, .super = true}));
    CHECK(Tap(place, KeyCode::Minus, {.super = true}));
    CHECK(Tap(place, KeyCode::Digit0, {.super = true}));
    CHECK(Tap(place, KeyCode::Equal, {.control = true, .shift = true}));
    CHECK(recorder.zooms == std::vector<I32>{1, 1, -1, 0, 1});
    CHECK(recorder.keys.empty());

    CHECK(Tap(place, KeyCode::Minus));
    CHECK(recorder.zooms.size() == 5);
    CHECK(recorder.keys.size() == 1);
}

TEST_CASE("A larger terminal font size reports a smaller grid", kTags) {
    Scene scene;
    Recorder recorder;
    auto screen = MakeScreen();
    Terminal terminal;
    Place place(terminal, kBounds);
    Frames frames(place, kScreen, scene.context());
    F32 fontSize = Sakura::Typography::FontSize;
    const auto update = [&] {
        auto config = recorder.wire(&screen, 1);
        config.fontSize = fontSize;
        terminal.update(std::move(config));
    };
    REQUIRE(frames.settle(update));
    REQUIRE(recorder.grids.size() == 1);

    fontSize += 6.0f;
    REQUIRE(frames.settle(update));
    REQUIRE(recorder.grids.size() == 2);
    CHECK(recorder.grids[1].cellHeight > recorder.grids[0].cellHeight);
    CHECK(recorder.grids[1].columns < recorder.grids[0].columns);
}

TEST_CASE("Command clicking a terminal link opens it and command hover underlines it with a hand", kTags) {
    Scene scene;
    Recorder recorder;
    auto screen = MakeScreen();
    screen.lines[1].spans = {{.column = 4, .width = 11, .text = "example.org", .color = kRed}};
    screen.lines[1].links = {{.column = 4, .width = 11, .uri = "https://example.org"}};
    Terminal terminal;
    Place place(terminal, kBounds);
    Frames frames(place, kScreen, scene.context());
    REQUIRE(frames.settle([&] { terminal.update(recorder.wire(&screen, 1)); }));
    const auto grid = recorder.grids.front();
    const auto link = CellCenter(grid, 6, 1);
    const auto plain = CellCenter(grid, 20, 1);
    const auto command = [](MouseEvent event) {
        event.modifiers.super = true;
        return event;
    };
    const U64 decorations = Decorations(place).config().instances.size();

    place.send(command(Mouse(MouseEventType::Move, link.x, link.y)));
    REQUIRE(frames.settle([&] { terminal.update(recorder.wire(&screen, 1)); }));
    CHECK(Peek::CursorAt(terminal, link) == SurfaceCursor::Hand);
    REQUIRE(Decorations(place).config().instances.size() == decorations + 1);
    const auto underline = Decorations(place).config().instances.back();
    const Rect first = GridCell(grid, 4, 1);
    CHECK(underline.rect.x == first.x);
    CHECK(underline.rect.width == Catch::Approx(11.0f * grid.cellWidth));
    CHECK(underline.backgroundColor == kRed);

    place.send(Mouse(MouseEventType::Move, link.x, link.y));
    REQUIRE(frames.settle([&] { terminal.update(recorder.wire(&screen, 1)); }));
    CHECK(Peek::CursorAt(terminal, link) == SurfaceCursor::TextInput);
    CHECK(Decorations(place).config().instances.size() == decorations);

    CHECK(place.send(command(Mouse(MouseEventType::Click, link.x, link.y))));
    place.send(command(Mouse(MouseEventType::Release, link.x, link.y)));
    CHECK(recorder.links == std::vector<std::string>{"https://example.org"});
    CHECK(recorder.selections.empty());

    place.send(command(Mouse(MouseEventType::Click, plain.x, plain.y)));
    place.send(command(Mouse(MouseEventType::Release, plain.x, plain.y)));
    place.send(Mouse(MouseEventType::Click, link.x, link.y));
    place.send(Mouse(MouseEventType::Release, link.x, link.y));
    CHECK(recorder.links.size() == 1);
    CHECK_FALSE(recorder.selections.empty());
}

TEST_CASE("Command F opens a find bar that takes typing, steps through matches, and closes back to the shell",
          "[core][sakura][terminal][find]") {
    Scene scene;
    Recorder recorder;
    auto screen = MakeScreen();
    Terminal terminal;
    Place place(terminal, kBounds);
    Frames frames(place, kScreen, scene.context());
    std::optional<std::string> search;
    const auto update = [&] {
        auto config = recorder.wire(&screen, 1);
        config.search = search;
        terminal.update(std::move(config));
    };
    REQUIRE(frames.settle(update));
    REQUIRE(FocusOwner(place) == &terminal);
    CHECK(Find<Sakura::TextInput>(place).empty());

    CHECK(Tap(place, KeyCode::F, {.super = true}));
    REQUIRE(recorder.searches == std::vector<std::optional<std::string>>{std::string()});
    search = std::string();
    REQUIRE(frames.settle(update));
    const auto field = Find<Sakura::TextInput>(place);
    REQUIRE(field.size() == 1);
    CHECK(field[0]->focused());

    Type(place, "ab");
    CHECK(recorder.texts.empty());
    REQUIRE(recorder.searches.size() == 3);
    CHECK(recorder.searches.back() == std::optional<std::string>("ab"));
    search = std::string("ab");
    screen.matchCount = 4;
    screen.matchIndex = 1;
    REQUIRE(frames.settle(update));
    CHECK(ShownText(place, "2/4") != nullptr);

    CHECK(Tap(place, KeyCode::Enter));
    CHECK(Tap(place, KeyCode::Enter, {.shift = true}));
    CHECK(recorder.steps == std::vector<bool>{true, false});
    CHECK(recorder.keys.empty());

    const auto buttons = Find<Sakura::ChromeButton>(place);
    REQUIRE(buttons.size() == 3);
    const Rect older = Peek::Frame(*buttons[1]);
    const Extent2D<F32> center = {older.x + older.width * 0.5f, older.y + older.height * 0.5f};
    const auto focusChanges = recorder.focus;
    place.send(Mouse(MouseEventType::Click, center.x, center.y));
    ClickFocus(place, center);
    place.send(Mouse(MouseEventType::Release, center.x, center.y));
    REQUIRE(frames.settle(update));
    CHECK(recorder.steps == std::vector<bool>{true, false, true});
    CHECK(field[0]->focused());
    CHECK(recorder.focus == focusChanges);
    Type(place, "c");
    CHECK(recorder.texts.empty());
    CHECK(recorder.searches.back() == std::optional<std::string>("abc"));

    const auto shown = [&](const std::string& value) {
        const auto strings = LabelStrings(place);
        return std::find(strings.begin(), strings.end(), value) != strings.end();
    };
    REQUIRE(shown("2/4"));
    search = std::string();
    REQUIRE(frames.settle(update));
    CHECK_FALSE(shown("2/4"));
    search = std::string("ab");
    REQUIRE(frames.settle(update));

    CHECK(Tap(place, KeyCode::Escape));
    CHECK(recorder.searches.back() == std::nullopt);
    search.reset();
    REQUIRE(frames.settle(update));
    CHECK(Find<Sakura::TextInput>(place).empty());
    CHECK(FocusOwner(place) == &terminal);
    CHECK(Tap(place, KeyCode::Enter));
    CHECK(recorder.keys.size() == 1);
}

TEST_CASE("The terminal find bar settles, toggles as a repaint, and scales with the pixel ratio",
          "[core][sakura][terminal][find][retention]") {
    for (const F32 ratio : {1.0f, 2.0f}) {
        Scene scene;
        auto ctx = scene.context();
        ctx.pixelRatio = ratio;
        Recorder recorder;
        auto screen = MakeScreen();
        Terminal terminal;
        Place place(terminal, kScreen);
        Frames frames(place, kScreen, ctx);
        std::optional<std::string> search = std::string("needle");
        const auto update = [&] {
            auto config = recorder.wire(&screen, 1);
            config.search = search;
            terminal.update(std::move(config));
        };
        REQUIRE(frames.settle(update));
        CHECK(Retains(frames, update));
        const auto panel = [&]() -> std::optional<Rect> {
            const auto cards = Find<Sakura::Card>(place);
            if (cards.size() != 1) {
                return std::nullopt;
            }
            return Peek::Frame(*cards.front());
        };
        REQUIRE(panel().has_value());
        CHECK(panel()->width == Catch::Approx(320.0f * ratio));
        CHECK(panel()->y == Catch::Approx(kScreen.y + 14.0f * ratio));
        const F32 gutter = 8.0f + Sakura::kScrollbarThickness + 2.0f * Sakura::kScrollbarMargin;
        CHECK(panel()->right() == Catch::Approx(kScreen.right() - std::round(gutter * ratio)).margin(1.0f));

        const U64 restructures = frames.restructures;
        search.reset();
        CHECK(RepaintOnly(frames.run(update)));
        REQUIRE(frames.settle(update));
        search = std::string("needle");
        CHECK(RepaintOnly(frames.run(update)));
        REQUIRE(frames.settle(update));
        CHECK(frames.restructures == restructures);
    }
}

TEST_CASE("Terminal search matches tint their cells and the current match stands out", "[core][sakura][terminal][find]") {
    Scene scene;
    Recorder recorder;
    auto screen = MakeScreen();
    screen.lines[2].matches = {{.start = 3, .end = 5}};
    screen.lines[4].match = Terminal::Range{.start = 0, .end = 2};
    Terminal terminal;
    Place place(terminal, kBounds);
    Frames frames(place, kScreen, scene.context());
    REQUIRE(frames.settle([&] { terminal.update(recorder.wire(&screen, 1)); }));
    const auto grid = recorder.grids.front();
    const auto ctx = scene.context();
    auto current = ctx.color("accent_color");
    current.a *= 0.55f;
    const auto& fills = Fills(place).config().instances;
    const auto at = [&](U16 column, U16 row, U16 width) {
        const Rect cell = GridCell(grid, column, row);
        return std::find_if(fills.begin(), fills.end(), [&](const auto& fill) {
            return fill.rect.x == cell.x && fill.rect.y == cell.y &&
                   fill.rect.width == Catch::Approx(width * grid.cellWidth);
        });
    };
    const auto match = at(3, 2, 3);
    REQUIRE(match != fills.end());
    CHECK(match->backgroundColor == ctx.color("editor_selection_match"));
    const auto selected = at(0, 4, 3);
    REQUIRE(selected != fills.end());
    CHECK(selected->backgroundColor == current);
}

TEST_CASE("Terminals forward key releases only for keys whose press reached the shell", kTags) {
    Scene scene;
    Recorder recorder;
    auto screen = MakeScreen();
    Terminal terminal;
    Place place(terminal, kBounds);
    Frames frames(place, kScreen, scene.context());
    REQUIRE(frames.settle([&] { terminal.update(recorder.wire(&screen, 1)); }));
    REQUIRE(FocusOwner(place) == &terminal);

    CHECK(Tap(place, KeyCode::A));
    CHECK(Tap(place, KeyCode::C, {.super = true}));
    REQUIRE(recorder.keys.size() == 1);
    REQUIRE(recorder.releases.size() == 1);
    CHECK(recorder.releases[0].key == KeyCode::A);

    CHECK(Sakura::Access::key(place, {KeyEventType::Press, KeyCode::B, {}}));
    REQUIRE(recorder.keys.size() == 2);
    ClickFocus(place, std::nullopt);
    REQUIRE(FocusOwner(place) == nullptr);
    REQUIRE(recorder.releases.size() == 2);
    CHECK(recorder.releases[1].key == KeyCode::B);
}

TEST_CASE("Terminals take keyboard focus again when their focus request changes", kTags) {
    Scene scene;
    Recorder recorder;
    auto screen = MakeScreen();
    Terminal terminal;
    Place place(terminal, kBounds);
    Frames frames(place, kScreen, scene.context());
    U64 request = 0;
    const auto update = [&] {
        auto config = recorder.wire(&screen, 1);
        config.focusRequest = request;
        terminal.update(std::move(config));
    };
    REQUIRE(frames.settle(update));
    REQUIRE(FocusOwner(place) == &terminal);

    ClickFocus(place, std::nullopt);
    REQUIRE(FocusOwner(place) == nullptr);
    REQUIRE(frames.settle(update));
    CHECK(FocusOwner(place) == nullptr);

    request = 1;
    REQUIRE(frames.settle(update));
    CHECK(FocusOwner(place) == &terminal);
    CHECK(recorder.focus == std::vector<bool>{true, false, true});
}

TEST_CASE("A terminal focus request takes focus from another text editor", kTags) {
    Scene scene;
    Recorder recorder;
    auto screen = MakeScreen();
    Terminal terminal;
    Beside beside(terminal);
    Place place(beside, kBounds);
    Frames frames(place, kScreen, scene.context());
    U64 request = 0;
    const auto update = [&] {
        auto config = recorder.wire(&screen, 1);
        config.focusRequest = request;
        terminal.update(std::move(config));
    };
    REQUIRE(frames.settle(update));
    REQUIRE(FocusOwner(place) == &terminal);

    beside.editor.focus();
    REQUIRE(FocusOwner(place) == &beside.editor);
    REQUIRE(frames.settle(update));
    CHECK(FocusOwner(place) == &beside.editor);

    request = 1;
    REQUIRE(frames.settle(update));
    CHECK(FocusOwner(place) == &terminal);
}

TEST_CASE("Terminal cursors blink while focused and stay hollow when unfocused", "[core][sakura][terminal][cursor]") {
    Scene scene;
    Recorder recorder;
    auto screen = MakeScreen();
    screen.cursor = {.visible = true, .blinking = true, .column = 4, .row = 1, .text = "x", .textColor = kWhite};
    Terminal terminal;
    Place place(terminal, kBounds);
    Frames frames(place, kScreen, scene.context());
    const auto update = [&] {
        terminal.update(recorder.wire(&screen, 1));
    };
    REQUIRE(frames.settle(update));
    REQUIRE(FocusOwner(place) == &terminal);
    const auto& cursor = CursorBox(place).config();
    const auto& glyph = CursorGlyph(place).config();
    REQUIRE(cursor.instances.size() == 1);
    CHECK(cursor.instances[0].visible);
    CHECK(cursor.borderWidth == 0.0f);
    CHECK(glyph.instances[0].str == "x");
    CHECK(glyph.instances[0].color != kWhite);

    frames.timeStep = 0.3;
    bool hidden = false;
    for (int frame = 0; frame < 6 && !hidden; ++frame) {
        frames.run(update);
        hidden = !cursor.instances[0].visible;
    }
    CHECK(hidden);
    CHECK(glyph.instances[0].visible);
    CHECK(glyph.instances[0].str == "x");
    CHECK(glyph.instances[0].color == kWhite);

    frames.timeStep = 0.0;
    ClickFocus(place, std::nullopt);
    REQUIRE(frames.settle(update));
    CHECK(cursor.instances[0].visible);
    CHECK(cursor.borderWidth > 0.0f);
    CHECK(glyph.instances[0].color == kWhite);
    frames.timeStep = 0.3;
    CHECK(Retains(frames, update));
}

TEST_CASE("Terminal half blocks fill exactly half of their cell", "[core][sakura][terminal][sprites]") {
    Scene scene;
    Recorder recorder;
    auto screen = MakeScreen();
    screen.lines[1].spans = {
        {.column = 2, .text = "▀", .color = kRed},
        {.column = 4, .text = "▄", .color = kBlue},
        {.column = 6, .text = "█", .color = kWhite},
        {.column = 8, .text = "▌", .color = kWhite},
    };
    Terminal terminal;
    Place place(terminal, kBounds);
    Frames frames(place, kScreen, scene.context());
    REQUIRE(frames.settle([&] { terminal.update(recorder.wire(&screen, 1)); }));
    const auto grid = recorder.grids.front();
    const F32 half = std::round(grid.cellHeight * 0.5f);

    const Rect upperCell = GridCell(grid, 2, 1);
    const auto upper = InsideCell(Sprites(place), upperCell);
    REQUIRE(upper.size() == 1);
    CHECK(upper[0].rect == Rect{upperCell.x, upperCell.y, upperCell.width, half});
    CHECK(upper[0].backgroundColor == kRed);

    const Rect lowerCell = GridCell(grid, 4, 1);
    const auto lower = InsideCell(Sprites(place), lowerCell);
    REQUIRE(lower.size() == 1);
    CHECK(lower[0].rect == Rect{lowerCell.x, lowerCell.bottom() - half, lowerCell.width, half});
    CHECK(lower[0].backgroundColor == kBlue);
    CHECK(upper[0].rect.height + lower[0].rect.height ==
          Catch::Approx(grid.cellHeight).margin(std::fmod(grid.cellHeight, 2.0f)));

    const Rect fullCell = GridCell(grid, 6, 1);
    const auto full = InsideCell(Sprites(place), fullCell);
    REQUIRE(full.size() == 1);
    CHECK(full[0].rect == fullCell);

    const Rect leftCell = GridCell(grid, 8, 1);
    const auto left = InsideCell(Sprites(place), leftCell);
    REQUIRE(left.size() == 1);
    CHECK(left[0].rect == Rect{leftCell.x, leftCell.y, std::round(grid.cellWidth * 0.5f), leftCell.height});

    for (const auto& text : LabelStrings(place)) {
        CHECK(text.find("▀") == std::string::npos);
        CHECK(text.find("▄") == std::string::npos);
    }
}

TEST_CASE("Terminal box lines touch both cell edges and join their neighbors", "[core][sakura][terminal][sprites]") {
    Scene scene;
    Recorder recorder;
    auto screen = MakeScreen();
    screen.lines[0].spans = {
        {.column = 0, .text = "─", .color = kWhite},
        {.column = 1, .text = "─", .color = kWhite},
        {.column = 2, .text = "─", .color = kWhite},
    };
    for (U16 row = 1; row <= 3; ++row) {
        screen.lines[row].spans = {
            {.column = 5, .text = "│", .color = kWhite},
            {.column = 7, .text = "┃", .color = kWhite},
        };
    }
    screen.lines[5].spans = {
        {.column = 0, .text = "┌", .color = kWhite},
        {.column = 1, .text = "─", .color = kWhite},
        {.column = 2, .text = "┐", .color = kWhite},
    };
    Terminal terminal;
    Place place(terminal, kBounds);
    Frames frames(place, kScreen, scene.context());
    REQUIRE(frames.settle([&] { terminal.update(recorder.wire(&screen, 1)); }));
    const auto grid = recorder.grids.front();
    const auto& sprites = Sprites(place);

    const Rect first = GridCell(grid, 0, 0);
    const Rect last = GridCell(grid, 2, 0);
    const auto horizontal = InsideCell(sprites, {first.x, first.y, last.right() - first.x, first.height});
    REQUIRE(horizontal.size() == 1);
    CHECK(horizontal[0].rect.x == first.x);
    CHECK(horizontal[0].rect.right() == last.right());
    CHECK(horizontal[0].rect.height >= 1.0f);
    CHECK(horizontal[0].rect.y == first.y + std::floor((grid.cellHeight - horizontal[0].rect.height) * 0.5f));

    std::vector<Box::Instance> light;
    std::vector<Box::Instance> heavy;
    for (U16 row = 1; row <= 3; ++row) {
        const Rect cell = GridCell(grid, 5, row);
        const auto found = InsideCell(sprites, cell);
        REQUIRE(found.size() == 1);
        CHECK(found[0].rect.y == cell.y);
        CHECK(found[0].rect.bottom() == cell.bottom());
        light.push_back(found[0]);
        const auto thick = InsideCell(sprites, GridCell(grid, 7, row));
        REQUIRE(thick.size() == 1);
        heavy.push_back(thick[0]);
    }
    for (U64 index = 1; index < light.size(); ++index) {
        CHECK(light[index].rect.x == light[0].rect.x);
        CHECK(light[index].rect.width == light[0].rect.width);
        CHECK(light[index - 1].rect.bottom() == light[index].rect.y);
    }
    CHECK(heavy[0].rect.width == 2.0f * light[0].rect.width);
    CHECK(light[0].rect.width == horizontal[0].rect.height);

    const Rect corner = GridCell(grid, 0, 5);
    const Rect middle = GridCell(grid, 1, 5);
    const Rect end = GridCell(grid, 2, 5);
    const auto boxRow = InsideCell(sprites, {corner.x, corner.y, end.right() - corner.x, corner.height});
    std::optional<Rect> leftArm;
    std::optional<Rect> span;
    U64 downArms = 0;
    for (const auto& instance : boxRow) {
        if (instance.rect.right() == middle.x && instance.rect.height < corner.height) {
            leftArm = instance.rect;
        }
        if (instance.rect.x == middle.x && instance.rect.right() > middle.right()) {
            span = instance.rect;
        }
        if (instance.rect.bottom() == corner.bottom() && instance.rect.height < corner.height) {
            ++downArms;
        }
    }
    REQUIRE(leftArm.has_value());
    REQUIRE(span.has_value());
    CHECK(leftArm->y == span->y);
    CHECK(leftArm->height == span->height);
    CHECK(span->right() < end.right());
    CHECK(downArms == 2);
}

TEST_CASE("Terminal braille cells place each dot on its pattern position", "[core][sakura][terminal][sprites]") {
    Scene scene;
    Recorder recorder;
    auto screen = MakeScreen();
    screen.lines[0].spans = {
        {.column = 0, .text = "⠁", .color = kWhite},
        {.column = 2, .text = "⢀", .color = kWhite},
        {.column = 4, .text = "⣿", .color = kRed},
        {.column = 6, .text = "⠀", .color = kWhite},
    };
    Terminal terminal;
    Place place(terminal, kBounds);
    Frames frames(place, kScreen, scene.context());
    REQUIRE(frames.settle([&] { terminal.update(recorder.wire(&screen, 1)); }));
    const auto grid = recorder.grids.front();
    const auto& sprites = Sprites(place);

    const Rect topLeft = GridCell(grid, 0, 0);
    const auto first = InsideCell(sprites, topLeft);
    REQUIRE(first.size() == 1);
    CHECK(first[0].rect.right() <= topLeft.x + grid.cellWidth * 0.5f);
    CHECK(first[0].rect.bottom() <= topLeft.y + grid.cellHeight * 0.25f);
    CHECK(first[0].rect.width > 0.0f);
    CHECK(first[0].rect.width == first[0].rect.height);

    const Rect bottomRight = GridCell(grid, 2, 0);
    const auto eighth = InsideCell(sprites, bottomRight);
    REQUIRE(eighth.size() == 1);
    CHECK(eighth[0].rect.x >= bottomRight.x + grid.cellWidth * 0.5f);
    CHECK(eighth[0].rect.y >= bottomRight.y + grid.cellHeight * 0.75f);

    const auto all = InsideCell(sprites, GridCell(grid, 4, 0));
    REQUIRE(all.size() == 8);
    std::vector<F32> xs;
    std::vector<F32> ys;
    for (const auto& dot : all) {
        CHECK(dot.backgroundColor == kRed);
        CHECK(dot.rect.width == first[0].rect.width);
        CHECK(dot.rect.height == first[0].rect.height);
        if (std::find(xs.begin(), xs.end(), dot.rect.x) == xs.end()) {
            xs.push_back(dot.rect.x);
        }
        if (std::find(ys.begin(), ys.end(), dot.rect.y) == ys.end()) {
            ys.push_back(dot.rect.y);
        }
    }
    CHECK(xs.size() == 2);
    CHECK(ys.size() == 4);
    CHECK(InsideCell(sprites, GridCell(grid, 6, 0)).empty());
}

TEST_CASE("Terminal arcs, diagonals, and powerline separators draw cell geometry", "[core][sakura][terminal][sprites]") {
    Scene scene;
    Recorder recorder;
    auto screen = MakeScreen();
    screen.lines[0].spans = {
        {.column = 0, .text = "╭", .color = kWhite},
        {.column = 2, .text = "╱", .color = kWhite},
        {.column = 4, .text = "", .color = kBlue},
    };
    Terminal terminal;
    Place place(terminal, kBounds);
    Frames frames(place, kScreen, scene.context());
    REQUIRE(frames.settle([&] { terminal.update(recorder.wire(&screen, 1)); }));
    const auto grid = recorder.grids.front();

    const auto& strokes = SpriteStrokes(place).config().instances;
    REQUIRE(strokes.size() == 3);
    const Rect arcCell = GridCell(grid, 0, 0);
    CHECK(strokes[0].points.front().y == arcCell.bottom());
    CHECK(strokes[0].points.back().x == arcCell.right());
    CHECK_FALSE(strokes[0].filled);
    CHECK(strokes[1].points.size() == 2);
    const Rect powerCell = GridCell(grid, 4, 0);
    CHECK(strokes[2].filled);
    CHECK(strokes[2].color == kBlue);
    REQUIRE(strokes[2].points.size() == 3);
    CHECK(strokes[2].points[0] == Extent2D<F32>{powerCell.x, powerCell.y});
    CHECK(strokes[2].points[1] == Extent2D<F32>{powerCell.right(), powerCell.y + powerCell.height * 0.5f});
    CHECK(strokes[2].points[2] == Extent2D<F32>{powerCell.x, powerCell.bottom()});
    CHECK(SpriteStrokes(place).config().capacity > 0);
}

TEST_CASE("Terminal glyphs missing from every font render blank instead of tofu", kTags) {
    Scene scene;
    Recorder recorder;
    auto screen = MakeScreen();
    screen.lines[0].spans = {
        {.column = 0, .width = 1, .text = "a", .color = kWhite},
        {.column = 2, .width = 2, .text = "一", .color = kWhite},
        {.column = 5, .width = 1, .text = "", .color = kWhite},
        {.column = 7, .width = 1, .text = "é", .color = kWhite},
    };
    screen.cursor = {.visible = true, .column = 9, .row = 0, .text = "丁"};
    Terminal terminal;
    Place place(terminal, kBounds);
    Frames frames(place, kScreen, scene.context());
    const auto update = [&] {
        terminal.update(recorder.wire(&screen, 1));
    };
    REQUIRE(frames.settle(update));

    bool accented = false;
    for (const auto& text : LabelStrings(place)) {
        CHECK(text.find("一") == std::string::npos);
        CHECK(text.find("丁") == std::string::npos);
        CHECK(text.find("") == std::string::npos);
        CHECK(text.find("�") == std::string::npos);
        accented = accented || text.find("é") != std::string::npos;
    }
    CHECK(accented);
    CHECK(CursorGlyph(place).config().instances[0].visible == false);
    CHECK(Retains(frames, update));
}

TEST_CASE("Terminal prompt arrows render from the mono font or as cell geometry", kTags) {
    Scene scene;
    Recorder recorder;
    auto screen = MakeScreen();
    screen.lines[0].spans = {
        {.column = 0, .width = 1, .text = "\u276F", .color = kWhite},
        {.column = 2, .width = 1, .text = "\u279C", .color = kRed},
    };
    Terminal terminal;
    Place place(terminal, kBounds);
    Frames frames(place, kScreen, scene.context());
    REQUIRE(frames.settle([&] { terminal.update(recorder.wire(&screen, 1)); }));
    const auto strings = LabelStrings(place);
    const auto has = [&](const std::string& value) {
        return std::any_of(strings.begin(), strings.end(), [&](const std::string& text) {
            return text.find(value) != std::string::npos;
        });
    };
    CHECK(has("\u276F"));
    CHECK_FALSE(has("\u279C"));
    const auto& strokes = SpriteStrokes(place).config().instances;
    REQUIRE(strokes.size() == 2);
    const Rect cell = GridCell(recorder.grids.front(), 2, 0);
    for (const auto& stroke : strokes) {
        CHECK(stroke.color == kRed);
        for (const auto& point : stroke.points) {
            CHECK(cell.contains(point));
        }
    }
    CHECK(strokes[1].filled);
}

TEST_CASE("Terminal glyphs missing from the mono font fall back to another app font", kTags) {
    Scene scene;
    scene.window.load("default_body", inter_regular_compressed_data);
    const auto& mono = scene.window.font("default_mono");
    const auto& body = scene.window.font("default_body");
    U32 fallback = 0;
    for (U32 codepoint = 0x00A0; codepoint <= 0x2718 && fallback == 0; ++codepoint) {
        if (body->hasGlyph(static_cast<I32>(codepoint)) && !mono->hasGlyph(static_cast<I32>(codepoint))) {
            fallback = codepoint;
        }
    }
    if (fallback == 0) {
        SKIP("Every body glyph is also in the mono font.");
    }

    Recorder recorder;
    auto screen = MakeScreen();
    const std::string text = Render::Components::Text::Unicode::Encode(fallback);
    screen.lines[0].spans = {{.column = 3, .width = 1, .text = text, .color = kRed}};
    Terminal terminal;
    Place place(terminal, kBounds);
    Frames frames(place, kScreen, scene.context());
    REQUIRE(frames.settle([&] { terminal.update(recorder.wire(&screen, 1)); }));

    const auto& label = Glyphs(place, 2).config();
    CHECK(label.fontName == "default_body");
    REQUIRE(label.instances.size() == 1);
    CHECK(label.instances[0].str == text);
    CHECK(label.instances[0].color == kRed);
    CHECK(Glyphs(place, 0).config().instances.empty());
}

TEST_CASE("Terminal cells keep every codepoint of a multi-codepoint grapheme", kTags) {
    Scene scene;
    const auto& mono = scene.window.font("default_mono");
    std::vector<U32> covered;
    for (U32 codepoint = 0x00C0; codepoint <= 0x017F && covered.size() < 3; ++codepoint) {
        if (mono->hasGlyph(static_cast<I32>(codepoint))) {
            covered.push_back(codepoint);
        }
    }
    if (covered.size() < 3) {
        SKIP("The mono font lacks three accented letters.");
    }
    std::string text;
    for (const U32 codepoint : covered) {
        text += Render::Components::Text::Unicode::Encode(codepoint);
    }

    Recorder recorder;
    auto screen = MakeScreen();
    screen.lines[0].spans = {{.column = 1, .width = 1, .text = text, .color = kRed}};
    Terminal terminal;
    Place place(terminal, kBounds);
    Frames frames(place, kScreen, scene.context());
    REQUIRE(frames.settle([&] { terminal.update(recorder.wire(&screen, 1)); }));
    const auto strings = LabelStrings(place);
    CHECK(std::any_of(strings.begin(), strings.end(), [&](const std::string& value) {
        return value.find(text) != std::string::npos;
    }));
}

TEST_CASE("Terminal cells on the default background paint no fill", kTags) {
    Scene scene;
    Recorder recorder;
    auto screen = MakeScreen();
    screen.lines[0].spans = {{.column = 0, .width = 5, .text = "hello", .color = kWhite}};
    U64 revision = 1;
    Terminal terminal;
    Place place(terminal, kBounds);
    Frames frames(place, kScreen, scene.context());
    const auto update = [&] {
        terminal.update(recorder.wire(&screen, revision));
    };
    REQUIRE(frames.settle(update));
    const auto grid = recorder.grids.front();

    CHECK(Fills(place).config().instances.empty());
    for (const auto* box : Find<Sakura::Box>(place)) {
        for (const auto& instance : box->config().instances) {
            const bool covers = instance.rect.width >= 0.5f * kBounds.width &&
                                instance.rect.height >= 0.5f * kBounds.height;
            CHECK_FALSE((instance.visible && covers && instance.backgroundColor.a > 0.0f));
        }
    }

    screen.lines[0].backgrounds = {{.column = 1, .width = 2, .color = kBlue}};
    ++revision;
    CHECK(RepaintOnly(frames.run(update)));
    const auto& fills = Fills(place).config().instances;
    REQUIRE(fills.size() == 1);
    CHECK(fills[0].backgroundColor == kBlue);
    CHECK(fills[0].rect == Rect{GridCell(grid, 1, 0).x, GridCell(grid, 1, 0).y, 2.0f * grid.cellWidth, grid.cellHeight});
}

TEST_CASE("Terminal sprites settle and repaint only when they change", "[core][sakura][terminal][retention]") {
    Scene scene;
    Recorder recorder;
    auto screen = MakeScreen();
    U64 revision = 1;
    screen.lines[0].spans = {{.column = 0, .text = "─", .color = kWhite}};
    Terminal terminal;
    Place place(terminal, kBounds);
    Frames frames(place, kScreen, scene.context());
    const auto update = [&] {
        terminal.update(recorder.wire(&screen, revision));
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));

    screen.lines[0].spans = {{.column = 0, .text = "╭", .color = kRed}};
    ++revision;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    CHECK(frames.restructures == 0);
}

TEST_CASE("Terminal mouse tracking reports hover motion by cell and keeps alt text by option", kTags) {
    Scene scene;
    Recorder recorder;
    auto screen = MakeScreen();
    screen.mouseTracking = true;
    bool altSendsEscape = false;
    Terminal terminal;
    Place place(terminal, kBounds);
    Frames frames(place, kScreen, scene.context());
    const auto update = [&] {
        auto config = recorder.wire(&screen, 1);
        config.altSendsEscape = altSendsEscape;
        terminal.update(std::move(config));
    };
    REQUIRE(frames.settle(update));
    const auto grid = recorder.grids.front();

    const auto cell = CellCenter(grid, 3, 2);
    CHECK_FALSE(place.send(Mouse(MouseEventType::Move, cell.x, cell.y)));
    CHECK_FALSE(place.send(Mouse(MouseEventType::Move, cell.x + 1.0f, cell.y)));
    REQUIRE(recorder.pointers.size() == 1);
    CHECK(recorder.pointers[0].type == MouseEventType::Move);
    const auto next = CellCenter(grid, 4, 2);
    CHECK_FALSE(place.send(Mouse(MouseEventType::Move, next.x, next.y)));
    CHECK(recorder.pointers.size() == 2);
    CHECK(Retains(frames, update));

    Sakura::Access::text(place, {.codepoint = U'b', .modifiers = {.alt = true}});
    CHECK(recorder.texts == std::vector<std::string>{"b"});
    altSendsEscape = true;
    REQUIRE(frames.settle(update));
    Sakura::Access::text(place, {.codepoint = U'b', .modifiers = {.alt = true}});
    CHECK(recorder.texts.size() == 1);
}

TEST_CASE("Terminals report their grid again when a closed screen reopens at the same address", kTags) {
    Scene scene;
    Recorder recorder;
    auto screen = MakeScreen();
    const Terminal::Screen* shown = &screen;
    Terminal terminal;
    Place place(terminal, kBounds);
    Frames frames(place, kScreen, scene.context());
    const auto update = [&] {
        terminal.update(recorder.wire(shown, 1));
    };
    REQUIRE(frames.settle(update));
    REQUIRE(recorder.grids.size() == 1);

    shown = nullptr;
    terminal.update(recorder.wire(shown, 1));
    shown = &screen;
    REQUIRE(frames.settle(update));
    REQUIRE(recorder.grids.size() == 2);
    CHECK(recorder.grids[1] == recorder.grids[0]);
}

TEST_CASE("Color dense rows keep every background and decoration", kTags) {
    Scene scene;
    Recorder recorder;
    auto screen = MakeScreen(40, 8);
    for (U16 row = 0; row < 8; ++row) {
        for (U16 column = 0; column < 40; ++column) {
            const F32 shade = static_cast<F32>(column % 2);
            screen.lines[row].backgrounds.push_back({.column = column, .width = 1, .color = {shade, 0.0f, 1.0f, 1.0f}});
            screen.lines[row].spans.push_back({
                .column = column,
                .width = 1,
                .text = "_",
                .color = {0.0f, shade, 0.0f, 1.0f},
                .underline = true,
            });
        }
    }
    Terminal terminal;
    Place place(terminal, kBounds);
    Frames frames(place, kScreen, scene.context());
    REQUIRE(frames.settle([&] { terminal.update(recorder.wire(&screen, 1)); }));
    const auto grid = recorder.grids.front();
    const U64 rows = std::min<U64>(8, grid.rows);
    const U64 columns = std::min<U64>(40, grid.columns);
    CHECK(Fills(place).config().instances.size() == rows * columns);
    CHECK(Decorations(place).config().instances.size() == rows * columns);
}

TEST_CASE("Color dense rows keep every regular and bold glyph", kTags) {
    Scene scene;
    Recorder recorder;
    auto screen = MakeScreen(200, 80);
    for (U16 row = 0; row < 80; ++row) {
        for (U16 column = 0; column < 200; ++column) {
            screen.lines[row].spans.push_back({
                .column = column,
                .width = 1,
                .text = std::string(1, static_cast<char>('a' + column % 26)),
                .color = {static_cast<F32>(column) / 200.0f, static_cast<F32>(row) / 80.0f, 0.5f, 1.0f},
                .bold = row % 2 == 1,
            });
        }
    }
    Terminal terminal;
    Place place(terminal, kScreen);
    Frames frames(place, kScreen, scene.context());
    REQUIRE(frames.settle([&] { terminal.update(recorder.wire(&screen, 1)); }));
    const auto grid = recorder.grids.front();
    const U64 rows = std::min<U64>(80, grid.rows);
    const U64 columns = std::min<U64>(200, grid.columns);
    const auto shown = [](const Sakura::Label& label) {
        U64 count = 0;
        for (const auto& instance : label.config().instances) {
            count += static_cast<U64>(std::count_if(instance.str.begin(), instance.str.end(), [](char value) {
                return value != ' ';
            }));
        }
        return count;
    };
    CHECK(shown(Glyphs(place, 0)) == ((rows + 1) / 2) * columns);
    CHECK(shown(Glyphs(place, 1)) == (rows / 2) * columns);
}

TEST_CASE("Full braille, diagonal, and arrow screens keep every dot and stroke", "[core][sakura][terminal][sprites]") {
    Scene scene;
    Recorder recorder;
    auto screen = MakeScreen(200, 80);
    for (U16 row = 0; row < 80; ++row) {
        const char* text = row % 3 == 0 ? "\u28FF" : row % 3 == 1 ? "\u2571" : "\u279C";
        for (U16 column = 0; column < 200; ++column) {
            screen.lines[row].spans.push_back({.column = column, .width = 1, .text = text, .color = kWhite});
        }
    }
    Terminal terminal;
    Place place(terminal, kScreen);
    Frames frames(place, kScreen, scene.context());
    REQUIRE(frames.settle([&] { terminal.update(recorder.wire(&screen, 1)); }));
    const auto grid = recorder.grids.front();
    const U64 rows = std::min<U64>(80, grid.rows);
    const U64 columns = std::min<U64>(200, grid.columns);
    U64 braille = 0;
    U64 diagonal = 0;
    U64 arrow = 0;
    for (U64 row = 0; row < rows; ++row) {
        (row % 3 == 0 ? braille : row % 3 == 1 ? diagonal : arrow) += columns;
    }
    CHECK(Sprites(place).config().instances.size() == braille * 8);
    CHECK(SpriteStrokes(place).config().instances.size() == diagonal + arrow * 2);
}

TEST_CASE("Terminal glyph labels hold a whole grapheme in every instance", kTags) {
    Scene scene;
    Recorder recorder;
    auto screen = MakeScreen();
    Terminal terminal;
    Place place(terminal, kBounds);
    Frames frames(place, kScreen, scene.context());
    REQUIRE(frames.settle([&] { terminal.update(recorder.wire(&screen, 1)); }));
    for (const auto* label : Find<Sakura::Label>(place)) {
        CHECK(label->config().maxCharacters >= 16);
    }
}

TEST_CASE("Terminals paint their predominant background across the frame and keep default cells", kTags) {
    Scene scene;
    Recorder recorder;
    constexpr ColorRGBA<F32> kGrey = {0.1f, 0.1f, 0.1f, 1.0f};
    auto screen = MakeScreen(40, 8);
    screen.base = kGrey;
    screen.background = kGrey;
    screen.fill = kGrey;
    U64 revision = 1;
    screen.lines[0].backgrounds = {{.column = 0, .width = 2, .color = kBlue}, {.column = 4, .width = 36, .color = kBlue}};
    screen.lines[1].backgrounds = {{.column = 0, .width = 2, .color = kRed}};
    Terminal terminal;
    Place place(terminal, kBounds);
    Frames frames(place, kScreen, scene.context());
    const auto update = [&] {
        auto config = recorder.wire(&screen, revision);
        config.rounding = 12.0f;
        config.corners = Box::BottomLeft | Box::BottomRight;
        terminal.update(std::move(config));
    };
    REQUIRE(frames.settle(update));
    const auto grid = recorder.grids.front();
    CHECK(Fills(place).config().instances.size() == 3);
    CHECK_FALSE(Surface(place).config().instances[0].visible);
    CHECK(Surface(place).config().cornerRadius == 12.0f);
    CHECK(Fills(place).config().cornerRadius == 0.0f);

    screen.fill = kBlue;
    ++revision;
    CHECK(RepaintOnly(frames.run(update)));
    const auto& surface = Surface(place).config().instances;
    REQUIRE(surface.size() == 1);
    CHECK(surface[0].visible);
    CHECK(surface[0].rect == kBounds);
    CHECK(surface[0].backgroundColor == kBlue);
    CHECK(surface[0].corners == (Box::BottomLeft | Box::BottomRight));
    const auto& fills = Fills(place).config().instances;
    const auto shaded = [&](U16 column, U16 row) {
        std::optional<ColorRGBA<F32>> color;
        const Rect cell = GridCell(grid, column, row);
        for (U64 index = 0; index < fills.size(); ++index) {
            if (fills[index].rect.x <= cell.x && fills[index].rect.right() >= cell.right() &&
                fills[index].rect.y <= cell.y && fills[index].rect.bottom() >= cell.bottom()) {
                color = fills[index].backgroundColor;
            }
        }
        return color;
    };
    CHECK_FALSE(shaded(0, 0).has_value());
    CHECK(shaded(2, 0) == kGrey);
    CHECK(shaded(3, 0) == kGrey);
    CHECK_FALSE(shaded(5, 0).has_value());
    CHECK(shaded(0, 1) == kRed);
    CHECK(shaded(5, 1) == kGrey);
    CHECK(Retains(frames, update));
}

TEST_CASE("Terminals paint default cells when an app recolors the default background", kTags) {
    Scene scene;
    Recorder recorder;
    constexpr ColorRGBA<F32> kGrey = {0.1f, 0.1f, 0.1f, 1.0f};
    auto screen = MakeScreen(40, 8);
    screen.base = kGrey;
    screen.background = kBlue;
    screen.fill = kBlue;
    screen.lines[0].backgrounds = {{.column = 0, .width = 40, .color = kRed}};
    Terminal terminal;
    Place place(terminal, kBounds);
    Frames frames(place, kScreen, scene.context());
    REQUIRE(frames.settle([&] { terminal.update(recorder.wire(&screen, 1)); }));
    const auto& surface = Surface(place).config().instances;
    REQUIRE(surface.size() == 1);
    CHECK(surface[0].visible);
    CHECK(surface[0].backgroundColor == kBlue);
    for (const auto& fill : Fills(place).config().instances) {
        CHECK(fill.backgroundColor == kRed);
    }

    screen.fill = kRed;
    REQUIRE(frames.settle([&] { terminal.update(recorder.wire(&screen, 2)); }));
    CHECK(surface[0].backgroundColor == kRed);
    bool blue = false;
    for (const auto& fill : Fills(place).config().instances) {
        CHECK(fill.backgroundColor != kRed);
        blue = blue || fill.backgroundColor == kBlue;
    }
    CHECK(blue);
}
