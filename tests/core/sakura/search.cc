#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "fonts.hh"
#include "component_harness.hh"

#include "render/sakura/detail/access.hh"
#include "scene.hh"

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/button.hh>
#include <jetstream/render/sakura/components/overlay.hh>
#include <jetstream/render/sakura/components/pill_button.hh>
#include <jetstream/render/sakura/components/pill_field.hh>
#include <jetstream/render/sakura/components/pill_icon_button.hh>
#include <jetstream/render/sakura/components/pill_stepper.hh>
#include <jetstream/render/sakura/components/scroll_area.hh>
#include <jetstream/render/sakura/components/search_catalog.hh>
#include <jetstream/render/sakura/components/text_input.hh>
#include <jetstream/render/sakura/components/label.hh>
#include <jetstream/render/sakura/components/markdown.hh>
#include <jetstream/render/tools/icons_lucide.hh>
#include <jetstream/render/sakura/components/text.hh>
#include <jetstream/render/sakura/components/text_grid.hh>

#include <algorithm>
#include <cmath>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace Jetstream;
using namespace SakuraTest;

namespace {

using Item = Sakura::SearchCatalog::Item;
using Label = Sakura::Label;

std::vector<Item> Items() {
    return {
        {.id = "fft", .title = "FFT", .summary = "Fast Fourier transform", .group = "DSP", .badges = {"CPU", "Metal"},
         .documentation = "# FFT"},
        {.id = "filter", .title = "Filter", .summary = "FIR taps", .group = "DSP", .badges = {"CPU"}},
        {.id = "file", .title = "File Reader", .summary = "Reads samples", .group = "IO", .badges = {"CPU"}},
        {.id = "custom", .title = "Custom", .summary = "Fits spectral data", .group = "Extra", .badges = {"CPU"}},
    };
}

struct CatalogRig {
    Scene scene{Extent2D<F32>{1200.0f, 900.0f}};
    Sakura::SearchCatalog catalog;
    Host host{catalog};
    Sakura::SearchCatalog::Config config;
    std::vector<U64> selections;
    std::vector<std::pair<std::string, std::optional<U64>>> activations;
    Jetstream::Rect bounds;

    CatalogRig() {
        config.items = Items();
        config.groups = {"DSP", "IO"};
        config.onSelect = [this](U64 index) {
            selections.push_back(index);
        };
        config.onActivate = [this](const std::string& id, std::optional<U64> badge) {
            activations.emplace_back(id, badge);
        };
    }

    void place() {
        catalog.update(config);
        const auto size = host.measureOf(scene.ctx, {kInfinite, kInfinite});
        bounds = {0.0f, 0.0f, size.x, size.y};
        host.place(scene.ctx, bounds);
    }

    template<typename T>
    std::vector<T*> all() {
        std::vector<T*> found;
        std::function<void(Sakura::Component&)> visit = [&](Sakura::Component& node) {
            if (!node.visible()) {
                return;
            }
            if (auto* match = dynamic_cast<T*>(&node)) {
                found.push_back(match);
            }
            for (auto* child : Peek::Children(node)) {
                if (child) {
                    visit(*child);
                }
            }
        };
        visit(catalog);
        return found;
    }

    void click(F32 x, F32 y, U8 clicks = 1) {
        host.send(Mouse(MouseEventType::Click, x, y, MouseButton::Left, clicks));
        host.send(Mouse(MouseEventType::Release, x, y));
    }
};

}  // namespace

TEST_CASE("Search catalog ranking groups an empty query and tiers a typed one",
          "[core][sakura][search]") {
    const auto items = Items();
    CHECK(Sakura::SearchCatalog::Rank(items, {"IO", "DSP"}, "") == std::vector<U64>{2, 0, 1, 3});
    CHECK(Sakura::SearchCatalog::Rank(items, {}, "") == std::vector<U64>{0, 1, 2, 3});
    CHECK(Sakura::SearchCatalog::Rank(items, {"DSP"}, "fi") == std::vector<U64>{1, 2, 3});
    CHECK(Sakura::SearchCatalog::Rank(items, {"DSP"}, "FIL") == std::vector<U64>{1, 2});
    CHECK(Sakura::SearchCatalog::Rank(items, {"DSP"}, "reader") == std::vector<U64>{2});
    CHECK(Sakura::SearchCatalog::Rank(items, {"DSP"}, "extra") == std::vector<U64>{3});
    CHECK(Sakura::SearchCatalog::Rank(items, {"DSP"}, "zzz").empty());
}

TEST_CASE("Search catalog lists results in rank order and pairs the list with a docs card on either side",
          "[core][sakura][search]") {
    CatalogRig rig;
    rig.place();
    CHECK(rig.catalog.results() == std::vector<std::string>{"fft", "filter", "file", "custom"});
    CHECK(rig.bounds.width == 300.0f + 8.0f + 440.0f);
    CHECK(rig.bounds.height == 360.0f);
    const auto scrolls = rig.all<Sakura::ScrollArea>();
    REQUIRE(scrolls.size() == 1);
    CHECK(Peek::Frame(*scrolls[0]).x == 308.0f + 10.0f);

    rig.config.docsSide = Sakura::SearchCatalog::DocsSide::Left;
    rig.place();
    CHECK(Peek::Frame(*rig.all<Sakura::ScrollArea>()[0]).x == 10.0f);

    rig.config.query = "zzz";
    rig.place();
    CHECK(rig.catalog.results().empty());
    CHECK(rig.bounds.width == 300.0f);
    CHECK(rig.all<Sakura::ScrollArea>().empty());
}

TEST_CASE("Search catalog rows select on click, activate on double click, and docs buttons pick the backend",
          "[core][sakura][search]") {
    CatalogRig rig;
    rig.place();

    rig.click(100.0f, 126.0f);
    CHECK(rig.selections == std::vector<U64>{1});
    rig.click(100.0f, 86.0f, 2);
    REQUIRE(rig.activations.size() == 1);
    CHECK(rig.activations[0] == std::make_pair(std::string("fft"), std::optional<U64>()));

    rig.click(280.0f, 86.0f);
    CHECK(rig.selections == std::vector<U64>{1, 0});
    CHECK(rig.activations.size() == 1);
    CHECK(Sakura::Access::cursor(rig.catalog, {100.0f, 86.0f}) == SurfaceCursor::Hand);

    const auto buttons = rig.all<Sakura::Button>();
    REQUIRE(buttons.size() == 2);
    const auto frame = Peek::Frame(*buttons[1]);
    rig.click(frame.x + frame.width * 0.5f, frame.y + frame.height * 0.5f);
    REQUIRE(rig.activations.size() == 2);
    CHECK(rig.activations[1] == std::make_pair(std::string("fft"), std::optional<U64>(1)));
}

TEST_CASE("Search catalog scrolls with the wheel and reveals the selected result",
          "[core][sakura][search]") {
    CatalogRig rig;
    rig.config.items.clear();
    for (U64 index = 0; index < 20; ++index) {
        rig.config.items.push_back({.id = "item" + std::to_string(index), .title = "Item", .group = "DSP"});
    }
    rig.config.maxHeight = 300.0f;
    rig.place();

    rig.click(100.0f, 250.0f);
    CHECK(rig.selections.back() == 3);

    rig.host.send(Wheel(100.0f, 200.0f, 0.0f, -1.0f));
    rig.host.place(rig.scene.ctx, rig.bounds);
    rig.click(100.0f, 250.0f);
    CHECK(rig.selections.back() == 4);

    rig.config.selected = 19;
    rig.place();
    rig.click(100.0f, 270.0f);
    CHECK(rig.selections.back() == 19);
}

TEST_CASE("Search catalog follows trackpad scrolling point for point", "[core][sakura][search]") {
    CatalogRig rig;
    rig.config.items.clear();
    for (U64 index = 0; index < 20; ++index) {
        rig.config.items.push_back({.id = "item" + std::to_string(index), .title = "Item", .group = "DSP"});
    }
    rig.config.maxHeight = 300.0f;
    rig.place();
    const auto trackpad = [&](F32 dy) {
        auto event = Wheel(100.0f, 200.0f, 0.0f, dy);
        event.phase = ScrollPhase::Changed;
        rig.host.send(event);
        rig.host.place(rig.scene.ctx, rig.bounds);
    };

    rig.click(100.0f, 250.0f);
    CHECK(rig.selections.back() == 3);
    trackpad(-0.5f);
    rig.click(100.0f, 250.0f);
    CHECK(rig.selections.back() == 3);
    trackpad(-4.3f);
    rig.click(100.0f, 250.0f);
    CHECK(rig.selections.back() == 4);
}

TEST_CASE("Search catalog stacks an optional field above both cards", "[core][sakura][search]") {
    CatalogRig rig;
    Sakura::PillField field;
    field.update({.placeholder = "Search", .editing = true, .size = {0.0f, 34.0f}});
    rig.config.field = &field;
    rig.place();
    CHECK(Peek::Frame(field) == Jetstream::Rect{0.0f, 0.0f, 300.0f + 8.0f + 440.0f, 34.0f});
    CHECK(rig.bounds.height == 34.0f + 8.0f + 360.0f);
}

TEST_CASE("Pill fields open on click while idle and host a text input while editing",
          "[core][sakura][search]") {
    Scene scene{Extent2D<F32>{1200.0f, 900.0f}};
    U64 opens = 0;
    Sakura::PillField pill;
    Host host(pill);
    Sakura::PillField::Config config{
        .placeholder = "Search blocks",
        .size = {240.0f, 34.0f},
        .onOpen = [&] { ++opens; },
    };
    pill.update(config);
    host.place(scene.ctx, {0.0f, 0.0f, 240.0f, 34.0f});

    CHECK(host.measureOf(scene.ctx, {kInfinite, kInfinite}) == Extent2D<F32>{240.0f, 34.0f});
    CHECK(Peek::CursorAt(pill, {100.0f, 17.0f}) == SurfaceCursor::Hand);
    host.click(100.0f, 17.0f);
    CHECK(opens == 1);

    const auto inputs = [&] {
        U64 count = 0;
        for (auto* child : Peek::Children(pill)) {
            count += child && child->visible() && dynamic_cast<Sakura::TextInput*>(child) ? 1 : 0;
        }
        return count;
    };
    CHECK(inputs() == 0);

    config.editing = true;
    pill.update(config);
    host.place(scene.ctx, {0.0f, 0.0f, 240.0f, 34.0f});
    CHECK(inputs() == 1);
    CHECK_FALSE(Peek::CursorAt(pill, {100.0f, 17.0f}) == SurfaceCursor::Hand);
    host.click(100.0f, 17.0f);
    CHECK(opens == 1);

    config.size.x = 0.0f;
    config.editing = false;
    pill.update(config);
    const auto natural = host.measureOf(scene.ctx, {kInfinite, kInfinite});
    CHECK(natural.x > 2.0f * 14.0f);
    CHECK(natural.x < 240.0f);
}

TEST_CASE("Pill fields that start editing with a query append typed text after it", "[core][sakura][search]") {
    Scene scene{Extent2D<F32>{1200.0f, 900.0f}};
    std::vector<std::string> changes;
    Sakura::PillField pill;
    Host host(pill);
    Sakura::PillField::Config config{
        .placeholder = "Search blocks",
        .query = "fft",
        .size = {240.0f, 34.0f},
        .onChange = [&](const std::string& value) { changes.push_back(value); },
    };
    pill.update(config);
    host.place(scene.ctx, {0.0f, 0.0f, 240.0f, 34.0f});

    config.editing = true;
    pill.update(config);
    host.place(scene.ctx, {0.0f, 0.0f, 240.0f, 34.0f});
    Sakura::Access::settleFocus(host);
    REQUIRE(FocusOwner(host) != nullptr);
    Type(host, "x");
    CHECK(changes == std::vector<std::string>{"fftx"});
}

TEST_CASE("Search catalog settles under identical updates and hover only repaints",
          "[core][sakura][search]") {
    Scene scene{Extent2D<F32>{1200.0f, 900.0f}};
    Sakura::SearchCatalog catalog;
    Host host(catalog);
    Frames frames(host, {0.0f, 0.0f, 748.0f, 360.0f}, scene.ctx);
    Sakura::SearchCatalog::Config config{.items = Items(), .groups = {"DSP", "IO"}};
    const auto update = [&] {
        catalog.update(config);
    };
    REQUIRE(frames.settle(update));
    for (U64 frame = 0; frame < 3; ++frame) {
        CHECK(frames.run(update).clean());
    }
    host.send(Mouse(MouseEventType::Move, 100.0f, 130.0f));
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(frames.run(update).clean());
}

TEST_CASE("Search catalog centers its empty label in the list", "[core][sakura][search]") {
    CatalogRig rig;
    rig.config.query = "zzz";
    rig.config.emptyLabel = "No matching blocks.";
    rig.place();
    const Sakura::Label* empty = nullptr;
    for (auto* label : rig.all<Sakura::Label>()) {
        for (const auto& instance : label->config().instances) {
            if (instance.str == "No matching blocks.") {
                empty = label;
            }
        }
    }
    REQUIRE(empty);
    const auto& instance = empty->config().instances.front();
    CHECK(instance.visible);
    CHECK(instance.alignment == Extent2D<I32>{1, 1});
    CHECK(instance.rect == Jetstream::Rect{6.0f, 42.0f, 288.0f, 48.0f});
    CHECK(rig.bounds.height == 36.0f + 48.0f + 12.0f);
}

TEST_CASE("Search catalog list card matches the docs card height beside it",
          "[core][sakura][search]") {
    CatalogRig rig;
    rig.config.query = "fil";
    rig.place();
    REQUIRE(rig.catalog.results().size() == 2);
    CHECK(rig.bounds.height == 360.0f);
    std::vector<Jetstream::Rect> cards;
    for (auto* box : rig.all<Sakura::Box>()) {
        if (box->config().borderWidth > 0.0f && !box->config().topLit && box->config().softness == 0.0f &&
            box->config().cornerRadius == 12.0f) {
            for (const auto& instance : box->config().instances) {
                if (instance.rect.width >= 300.0f) {
                    cards.push_back(instance.rect);
                }
            }
        }
    }
    REQUIRE(cards.size() == 2);
    CHECK(cards[0] == Jetstream::Rect{0.0f, 0.0f, 300.0f, 360.0f});
    CHECK(cards[1].height == 360.0f);
}

TEST_CASE("Search catalog categories follow the configured order and narrow the ranking",
          "[core][sakura][search]") {
    const auto items = Items();
    CHECK(Sakura::SearchCatalog::Groups(items, {"IO", "Missing", "DSP"}) ==
          std::vector<std::string>{"IO", "DSP", "Extra"});
    CHECK(Sakura::SearchCatalog::Rank(items, {"DSP"}, "", "IO") == std::vector<U64>{2});
    CHECK(Sakura::SearchCatalog::Rank(items, {"DSP"}, "f", "DSP") == std::vector<U64>{0, 1});
    CHECK(Sakura::SearchCatalog::Rank(items, {"DSP"}, "reader", "DSP").empty());
}

TEST_CASE("Search catalog chips report their category and an active category hides the group headers",
          "[core][sakura][search]") {
    CatalogRig rig;
    std::vector<std::string> picked;
    rig.config.onGroup = [&](const std::string& group) {
        picked.push_back(group);
    };
    rig.place();

    const auto chips = [&] {
        std::vector<Label::Instance> found;
        for (auto* label : rig.all<Sakura::Label>()) {
            const auto& instances = label->config().instances;
            if (!instances.empty() && instances.front().str == "All") {
                found = instances;
            }
        }
        return found;
    };
    auto strip = chips();
    REQUIRE(strip.size() == 4);
    CHECK(strip[1].str == "DSP");
    CHECK(strip[2].str == "IO");
    CHECK(strip[3].str == "Extra");
    CHECK(strip[0].rect.y == 12.0f);
    CHECK(strip[0].color == rig.scene.ctx.color("text_primary"));

    const auto center = [](const Jetstream::Rect& rect) {
        return Extent2D<F32>{rect.x + rect.width * 0.5f, rect.y + rect.height * 0.5f};
    };
    const auto io = center(strip[2].rect);
    rig.click(io.x, io.y);
    const auto all = center(strip[0].rect);
    rig.click(all.x, all.y);
    CHECK(picked == std::vector<std::string>{"IO", ""});
    CHECK(Sakura::Access::cursor(rig.catalog, io) == SurfaceCursor::Hand);

    rig.config.activeGroup = "IO";
    rig.place();
    CHECK(rig.catalog.results() == std::vector<std::string>{"file"});
    strip = chips();
    REQUIRE(strip.size() == 4);
    CHECK(strip[2].color == rig.scene.ctx.color("text_primary"));
    CHECK(strip[0].color == rig.scene.ctx.color("text_secondary"));
    rig.click(100.0f, 62.0f);
    CHECK(rig.selections == std::vector<U64>{0});
}

TEST_CASE("Search catalog gives unnamed groups no chip and lists them last under Other",
          "[core][sakura][search]") {
    CatalogRig rig;
    rig.config.items.push_back({.id = "misc", .title = "Misc", .summary = "Anything"});
    rig.config.items.insert(rig.config.items.begin(), {.id = "first", .title = "First", .summary = "Unnamed"});
    CHECK(Sakura::SearchCatalog::Groups(rig.config.items, {"DSP", "IO"}) ==
          std::vector<std::string>{"DSP", "IO", "Extra", ""});
    rig.place();
    CHECK(rig.catalog.results() == std::vector<std::string>{"fft", "filter", "file", "custom", "first", "misc"});
    for (auto* label : rig.all<Sakura::Label>()) {
        const auto& instances = label->config().instances;
        if (!instances.empty() && instances.front().str == "All") {
            CHECK(instances.size() == 4);
            for (const auto& instance : instances) {
                CHECK(instance.str != "Other");
            }
        }
    }
}

TEST_CASE("Pill field placeholder keeps its position when the field starts editing",
          "[core][sakura][search]") {
    Scene scene{Extent2D<F32>{1200.0f, 900.0f}};
    Sakura::PillField pill;
    Host host(pill);
    Sakura::PillField::Config config{.icon = "Q", .placeholder = "Search blocks", .size = {240.0f, 34.0f}};
    const auto placeholderX = [&]() -> std::optional<F32> {
        std::function<std::optional<F32>(Sakura::Component&)> visit = [&](Sakura::Component& node) -> std::optional<F32> {
            if (!node.visible()) {
                return std::nullopt;
            }
            if (auto* text = dynamic_cast<Sakura::Text*>(&node); text && text->config().str == "Search blocks") {
                return Peek::Frame(*text).x;
            }
            if (auto* label = dynamic_cast<Sakura::Label*>(&node)) {
                for (const auto& instance : label->config().instances) {
                    if (instance.str == "Search blocks" && instance.visible) {
                        return instance.rect.x;
                    }
                }
            }
            for (auto* child : Peek::Children(node)) {
                if (child) {
                    if (const auto found = visit(*child)) {
                        return found;
                    }
                }
            }
            return std::nullopt;
        };
        return visit(pill);
    };

    pill.update(config);
    host.place(scene.ctx, {0.0f, 0.0f, 240.0f, 34.0f});
    const auto idle = placeholderX();
    REQUIRE(idle.has_value());

    config.editing = true;
    pill.update(config);
    host.place(scene.ctx, {0.0f, 0.0f, 240.0f, 34.0f});
    const auto editing = placeholderX();
    REQUIRE(editing.has_value());
    CHECK(*editing == *idle);
}

TEST_CASE("Pill steppers run their actions per part and gray out disabled sides",
          "[core][sakura][search]") {
    Scene scene{Extent2D<F32>{1200.0f, 900.0f}};
    std::vector<std::string> calls;
    Sakura::PillStepper stepper;
    Host host(stepper);
    Sakura::PillStepper::Config config{
        .label = "100%",
        .size = {120.0f, 34.0f},
        .onDecrement = [&] { calls.push_back("down"); },
        .onIncrement = [&] { calls.push_back("up"); },
        .onReset = [&] { calls.push_back("reset"); },
    };
    const auto place = [&] {
        stepper.update(config);
        host.place(scene.ctx, {0.0f, 0.0f, 120.0f, 34.0f});
    };
    const auto marks = [&] {
        for (auto* child : Peek::Children(stepper)) {
            if (auto* box = dynamic_cast<Sakura::Box*>(child); box && box->config().instances.size() == 3) {
                return box->config().instances;
            }
        }
        return std::vector<Sakura::Box::Instance>{};
    };
    place();
    CHECK(host.measureOf(scene.ctx, {kInfinite, kInfinite}) == Extent2D<F32>{120.0f, 34.0f});

    host.click(17.0f, 17.0f);
    host.click(60.0f, 17.0f);
    host.click(103.0f, 17.0f);
    CHECK(calls == std::vector<std::string>{"down", "reset", "up"});
    CHECK(Peek::CursorAt(stepper, {17.0f, 17.0f}) == SurfaceCursor::Hand);

    host.send(Mouse(MouseEventType::Move, 103.0f, 17.0f));
    place();
    auto bars = marks();
    REQUIRE(bars.size() == 3);
    CHECK(bars[0].rect.width == bars[1].rect.width);
    CHECK(bars[1].rect.width == bars[2].rect.height);
    CHECK(bars[0].rect.height == bars[1].rect.height);
    CHECK(bars[1].rect.y + bars[1].rect.height * 0.5f == bars[2].rect.y + bars[2].rect.height * 0.5f);
    CHECK(bars[1].rect.x + bars[1].rect.width * 0.5f == bars[2].rect.x + bars[2].rect.width * 0.5f);
    CHECK(bars[0].rect.y == bars[1].rect.y);
    CHECK(bars[1].backgroundColor == scene.ctx.color("text_secondary"));
    CHECK(bars[0].backgroundColor == scene.ctx.color("text_secondary"));
    const auto halo = [&]() -> std::optional<Jetstream::Rect> {
        for (auto* child : Peek::Children(stepper)) {
            if (auto* box = dynamic_cast<Sakura::Box*>(child); box && box->config().instances.size() == 1 &&
                                                                  box->config().cornerRadius == 12.0f) {
                return box->config().instances.front().rect;
            }
        }
        return std::nullopt;
    };
    REQUIRE(halo().has_value());
    CHECK(halo()->width == 24.0f);
    CHECK(halo()->height == 24.0f);
    CHECK(halo()->x + 12.0f == bars[2].rect.x + bars[2].rect.width * 0.5f);

    config.canIncrement = false;
    place();
    host.click(103.0f, 17.0f);
    CHECK(calls.size() == 3);
    bars = marks();
    REQUIRE(bars.size() == 3);
    CHECK(bars[2].backgroundColor == scene.ctx.color("text_disabled"));
    host.send(Mouse(MouseEventType::Move, 103.0f, 17.0f));
    place();
    CHECK_FALSE(halo().has_value());
    CHECK_FALSE(Peek::CursorAt(stepper, {103.0f, 17.0f}).has_value());
}

TEST_CASE("Pill fields and steppers fill with the pill field color", "[core][sakura][search]") {
    Scene scene{Extent2D<F32>{1200.0f, 900.0f}};
    const ColorRGBA<F32> fill = {0.1f, 0.2f, 0.3f, 0.97f};
    const Sakura::Palette palette = {{"pill_field_bg", fill}};
    auto ctx = scene.ctx;
    ctx.palette = palette;
    Sakura::PillField pill;
    Sakura::PillStepper stepper;
    for (Sakura::Component* component : std::initializer_list<Sakura::Component*>{&pill, &stepper}) {
        Host host(*component);
        if (component == &pill) {
            pill.update({.placeholder = "Search", .size = {240.0f, 34.0f}});
        } else {
            stepper.update({.label = "100%", .size = {100.0f, 34.0f}});
        }
        host.place(ctx, {0.0f, 0.0f, 240.0f, 34.0f});
        bool found = false;
        for (auto* box : Find<Sakura::Box>(*component)) {
            if (box->config().borderWidth == 1.5f) {
                REQUIRE(box->config().instances.size() == 1);
                CHECK(box->config().instances.front().backgroundColor == fill);
                found = true;
            }
        }
        CHECK(found);
    }
}

TEST_CASE("Search catalog pins the backend buttons in a footer below the scrolling docs",
          "[core][sakura][search]") {
    CatalogRig rig;
    rig.place();
    const auto scrolls = rig.all<Sakura::ScrollArea>();
    REQUIRE(scrolls.size() == 1);
    const auto docs = Peek::Frame(*scrolls[0]);
    const auto buttons = rig.all<Sakura::Button>();
    REQUIRE(buttons.size() == 2);
    const auto first = Peek::Frame(*buttons[0]);
    const auto second = Peek::Frame(*buttons[1]);
    CHECK(first.y > docs.bottom());
    CHECK(first.y == second.y);
    CHECK(second.x > first.right());
    CHECK(first.bottom() <= rig.bounds.bottom());

    const Sakura::Text* label = nullptr;
    for (auto* text : rig.all<Sakura::Text>()) {
        if (text->config().str == "Add on") {
            label = text;
        }
    }
    REQUIRE(label);
    const auto labelFrame = Peek::Frame(*label);
    CHECK(labelFrame.right() < first.x);
    CHECK(labelFrame.y + labelFrame.height * 0.5f == Catch::Approx(first.y + first.height * 0.5f).margin(1.0f));
}

TEST_CASE("Search catalog cards fill with the pill field color", "[core][sakura][search]") {
    CatalogRig rig;
    const ColorRGBA<F32> fill = {0.1f, 0.2f, 0.3f, 0.97f};
    const Sakura::Palette palette = {{"pill_field_bg", fill}};
    rig.scene.ctx.palette = palette;
    rig.place();
    U64 cards = 0;
    for (auto* box : rig.all<Sakura::Box>()) {
        if (box->config().cornerRadius == 12.0f && box->config().borderWidth == 1.0f && !box->config().topLit) {
            for (const auto& instance : box->config().instances) {
                if (instance.rect.width >= 300.0f) {
                    CHECK(instance.backgroundColor == fill);
                    ++cards;
                }
            }
        }
    }
    CHECK(cards == 2);
}

TEST_CASE("Pill button is a capsule with an icon and label that highlights and clicks",
          "[core][sakura][search]") {
    Scene scene{Extent2D<F32>{1200.0f, 900.0f}};
    Sakura::PillButton button;
    Host host(button);
    U64 clicks = 0;
    button.update({.icon = ICON_LC_CIRCLE_ARROW_UP, .label = "Update Available", .colorKey = "warning_yellow",
                   .onClick = [&]() {
                       ++clicks;
                   }});
    const auto size = host.measureOf(scene.ctx, {kInfinite, kInfinite});
    CHECK(size.y == 28.0f);
    host.place(scene.ctx, {100.0f, 10.0f, size.x, size.y});

    Sakura::Box* fill = nullptr;
    std::vector<Sakura::Text*> texts;
    std::vector<Sakura::Box*> boxes;
    for (auto* child : Peek::Children(button)) {
        if (auto* box = dynamic_cast<Sakura::Box*>(child)) {
            boxes.push_back(box);
        }
        if (auto* text = dynamic_cast<Sakura::Text*>(child)) {
            texts.push_back(text);
        }
    }
    REQUIRE(boxes.size() == 3);
    fill = boxes[1];
    CHECK_FALSE(boxes[0]->config().instances.front().visible);
    CHECK_FALSE(boxes[2]->config().instances.front().visible);
    REQUIRE(texts.size() == 2);
    CHECK(fill->config().cornerRadius == 14.0f);
    CHECK(fill->config().instances.front().backgroundColor == scene.ctx.color("pill_bg"));
    CHECK(texts[0]->config().str == ICON_LC_CIRCLE_ARROW_UP);
    CHECK(texts[1]->config().str == "Update Available");
    for (auto* text : texts) {
        CHECK(text->config().colorKey == "warning_yellow");
        const auto frame = Peek::Frame(*text);
        CHECK(frame.y + frame.height * 0.5f == Catch::Approx(24.0f).margin(1.0f));
    }
    CHECK(Peek::Frame(*texts[0]).right() < Peek::Frame(*texts[1]).x);
    CHECK(Peek::Frame(*texts[1]).right() <= 100.0f + size.x);

    host.send(Mouse(MouseEventType::Move, 120.0f, 26.0f));
    host.place(scene.ctx, {100.0f, 10.0f, size.x, size.y});
    CHECK(fill->config().instances.front().backgroundColor == scene.ctx.color("pill_hover_bg"));
    CHECK(Peek::CursorAt(button, {120.0f, 26.0f}) == SurfaceCursor::Hand);
    host.click(120.0f, 26.0f);
    CHECK(clicks == 1);
}

TEST_CASE("Raised pill button adds a soft drop shadow and a top lit edge", "[core][sakura][search]") {
    Scene scene{Extent2D<F32>{1200.0f, 900.0f}};
    Sakura::PillButton button;
    Host host(button);
    button.update({.label = "Update", .backgroundColorKey = "action_btn", .raised = true});
    host.place(scene.ctx, {100.0f, 10.0f, 120.0f, 28.0f});
    std::vector<Sakura::Box*> boxes;
    for (auto* child : Peek::Children(button)) {
        if (auto* box = dynamic_cast<Sakura::Box*>(child)) {
            boxes.push_back(box);
        }
    }
    REQUIRE(boxes.size() == 3);
    const auto& shadow = boxes[0]->config();
    CHECK(shadow.instances.front().visible);
    CHECK(shadow.softness == 4.0f);
    CHECK(shadow.instances.front().rect.y == 10.0f + 1.0f - 4.0f);
    CHECK(shadow.instances.front().backgroundColor == scene.ctx.color("tile_shadow"));
    const auto& edge = boxes[2]->config();
    CHECK(edge.instances.front().visible);
    CHECK(edge.topLit);
    CHECK(edge.borderWidth == 1.0f);
    CHECK(edge.borderColor == scene.ctx.color("action_btn_highlight"));
}

namespace {

const Sakura::Palette& Tones() {
    static const Sakura::Palette palette = [] {
        const std::vector<std::string> keys = {
            "pill_bg", "pill_hover_bg", "pill_active_bg", "pill_field_bg", "text_primary", "text_secondary",
            "text_disabled", "chrome_button_hover", "tile_border", "tile_shadow",
        };
        Sakura::Palette result;
        for (U64 index = 0; index < keys.size(); ++index) {
            const F32 value = static_cast<F32>(index + 1) / static_cast<F32>(keys.size() + 1);
            result[keys[index]] = {value, 1.0f - value, 0.5f, 1.0f};
        }
        return result;
    }();
    return palette;
}

std::vector<Jetstream::Rect> RowTiles(CatalogRig& rig, const char* colorKey) {
    std::vector<Jetstream::Rect> found;
    for (auto* box : rig.all<Sakura::Box>()) {
        if (box->config().cornerRadius != 8.0f || box->config().borderWidth != 0.0f) {
            continue;
        }
        for (const auto& instance : box->config().instances) {
            if (instance.rect.width >= 280.0f && instance.backgroundColor == rig.scene.ctx.color(colorKey)) {
                found.push_back(instance.rect);
            }
        }
    }
    return found;
}

std::vector<Label::Instance> ChipLabels(CatalogRig& rig) {
    for (auto* label : rig.all<Sakura::Label>()) {
        const auto& instances = label->config().instances;
        if (!instances.empty() && instances.front().str == "All") {
            return instances;
        }
    }
    return {};
}

const Sakura::Box* ChipFills(CatalogRig& rig) {
    const auto chips = ChipLabels(rig);
    if (chips.empty()) {
        return nullptr;
    }
    for (auto* box : rig.all<Sakura::Box>()) {
        const auto& instances = box->config().instances;
        if (box->config().borderWidth == 0.0f && instances.size() == chips.size() &&
            instances.front().rect == chips.front().rect) {
            return box;
        }
    }
    return nullptr;
}

std::vector<Label::Instance> HeaderLabels(CatalogRig& rig) {
    std::vector<Label::Instance> found;
    for (auto* label : rig.all<Sakura::Label>()) {
        for (const auto& instance : label->config().instances) {
            if (instance.fontSize == 12.0f && instance.alignment == Extent2D<I32>{0, 1} && instance.visible) {
                found.push_back(instance);
            }
        }
    }
    return found;
}

Extent2D<F32> Center(const Jetstream::Rect& rect) {
    return {rect.x + rect.width * 0.5f, rect.y + rect.height * 0.5f};
}

}  // namespace

TEST_CASE("Search catalog clamps a stale selection to the last result when the results shrink",
          "[core][sakura][search]") {
    CatalogRig rig;
    rig.scene.ctx.palette = Tones();
    rig.config.items[3].badges = {"CPU", "Metal", "CUDA"};
    rig.config.selected = 3;
    rig.place();
    REQUIRE(rig.catalog.results().size() == 4);
    CHECK(rig.all<Sakura::Button>().size() == 3);
    CHECK(RowTiles(rig, "pill_active_bg") == std::vector<Jetstream::Rect>{{10.0f, 259.0f, 280.0f, 46.0f}});

    Frames frames(rig.host, rig.bounds, rig.scene.ctx);
    const auto update = [&] {
        rig.catalog.update(rig.config);
    };
    REQUIRE(frames.settle(update));

    rig.config.query = "fil";
    REQUIRE(frames.settle(update));
    CHECK(DirtyFrames(frames, update) == 0);
    REQUIRE(rig.catalog.results() == std::vector<std::string>{"filter", "file"});
    CHECK(RowTiles(rig, "pill_active_bg") == std::vector<Jetstream::Rect>{{10.0f, 91.0f, 280.0f, 46.0f}});
    auto buttons = rig.all<Sakura::Button>();
    REQUIRE(buttons.size() == 1);
    const auto file = Center(Peek::Frame(*buttons[0]));
    rig.click(file.x, file.y);
    REQUIRE(rig.activations.size() == 1);
    CHECK(rig.activations[0] == std::make_pair(std::string("file"), std::optional<U64>(0)));

    rig.config.query.clear();
    rig.config.activeGroup = "DSP";
    REQUIRE(frames.settle(update));
    CHECK(DirtyFrames(frames, update) == 0);
    REQUIRE(rig.catalog.results() == std::vector<std::string>{"fft", "filter"});
    CHECK(RowTiles(rig, "pill_active_bg") == std::vector<Jetstream::Rect>{{10.0f, 91.0f, 280.0f, 46.0f}});
    buttons = rig.all<Sakura::Button>();
    REQUIRE(buttons.size() == 1);
    const auto filter = Center(Peek::Frame(*buttons[0]));
    rig.click(filter.x, filter.y);
    REQUIRE(rig.activations.size() == 2);
    CHECK(rig.activations[1] == std::make_pair(std::string("filter"), std::optional<U64>(0)));
    CHECK(rig.selections.empty());
}

TEST_CASE("Search catalog row text keeps its drawable capacity when the result count drops below 64 rows",
          "[core][sakura][search]") {
    CatalogRig rig;
    rig.config.items.clear();
    for (U64 index = 0; index < 70; ++index) {
        rig.config.items.push_back({.id = "item" + std::to_string(index), .title = "Item " + std::to_string(index),
                                    .summary = "Block", .group = "DSP"});
    }
    rig.config.groups = {"DSP"};
    rig.place();
    REQUIRE(rig.catalog.results().size() == 70);
    const auto capacities = [&] {
        std::vector<U64> found;
        for (auto* label : rig.all<Sakura::Label>()) {
            found.push_back(label->config().capacity);
        }
        return found;
    };
    const auto many = capacities();

    Frames frames(rig.host, rig.bounds, rig.scene.ctx);
    const auto update = [&] {
        rig.catalog.update(rig.config);
    };
    REQUIRE(frames.settle(update));
    const U64 rebuilds = frames.rebuilds;

    rig.config.query = "Item 1";
    REQUIRE(frames.settle(update));
    CHECK(rig.catalog.results().size() == 11);
    CHECK(capacities() == many);
    CHECK(frames.rebuilds == rebuilds);
    CHECK(DirtyFrames(frames, update) == 0);

    rig.config.query.clear();
    REQUIRE(frames.settle(update));
    CHECK(rig.catalog.results().size() == 70);
    CHECK(capacities() == many);
    CHECK(frames.rebuilds == rebuilds);
}

TEST_CASE("Search catalog chips wrap onto more rows and push the list down by the chip rows",
          "[core][sakura][search]") {
    CatalogRig rig;
    rig.config.items.clear();
    rig.config.groups.clear();
    const auto add = [&](const std::string& group) {
        rig.config.items.push_back({.id = group, .title = group, .group = group});
        rig.config.groups.push_back(group);
    };
    add("DSP");
    rig.place();
    const auto single = ChipLabels(rig);
    REQUIRE(single.size() == 2);
    const auto singleHeaders = HeaderLabels(rig);
    REQUIRE_FALSE(singleHeaders.empty());
    CHECK(singleHeaders.front().rect.y == 42.0f + 8.0f);

    for (U64 index = 1; index < 11; ++index) {
        add("Group " + std::to_string(index));
    }
    add("Software Defined Radio Tools");
    rig.place();
    const auto chips = ChipLabels(rig);
    REQUIRE(chips.size() == 13);
    std::vector<F32> rows;
    for (const auto& chip : chips) {
        CHECK(chip.visible);
        CHECK(chip.rect.x >= 12.0f);
        CHECK(chip.rect.right() <= 12.0f + 276.0f);
        rows.push_back(chip.rect.y);
    }
    CHECK(chips.back().str == "Software Defined Radio Tools");
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    REQUIRE(rows.size() > 1);
    for (U64 index = 0; index < rows.size(); ++index) {
        CHECK(rows[index] == 12.0f + static_cast<F32>(index) * 30.0f);
    }

    const auto* fills = ChipFills(rig);
    REQUIRE(fills);
    CHECK(fills->config().instances.size() == chips.size());
    CHECK(fills->config().instances.size() <= fills->config().capacity);
    for (auto* label : rig.all<Sakura::Label>()) {
        if (!label->config().instances.empty() && label->config().instances.front().str == "All") {
            CHECK(label->config().instances.size() <= label->config().capacity);
        }
    }

    const F32 shift = static_cast<F32>(rows.size() - 1) * 30.0f;
    const auto headers = HeaderLabels(rig);
    REQUIRE_FALSE(headers.empty());
    CHECK(headers.front().rect.y == singleHeaders.front().rect.y + shift);
    rig.click(100.0f, headers.front().rect.y + 24.0f + 10.0f);
    CHECK(rig.selections == std::vector<U64>{0});
}

TEST_CASE("Search catalog labels the unnamed group header OTHER in the empty query list",
          "[core][sakura][search]") {
    CatalogRig rig;
    rig.scene.ctx.palette = Tones();
    rig.config.items.push_back({.id = "misc", .title = "Misc", .summary = "Anything"});
    rig.place();
    std::vector<std::string> names;
    for (const auto& header : HeaderLabels(rig)) {
        names.push_back(header.str);
        CHECK(header.color == rig.scene.ctx.color("text_secondary"));
    }
    CHECK(names == std::vector<std::string>{"DSP", "IO", "EXTRA", "OTHER"});
    CHECK(HeaderLabels(rig).back().rect.y == 42.0f + 264.0f + 8.0f);
    rig.click(100.0f, 42.0f + 264.0f + 24.0f + 10.0f);
    CHECK(rig.selections == std::vector<U64>{4});

    rig.config.query = "misc";
    rig.place();
    CHECK(HeaderLabels(rig).empty());
}

TEST_CASE("Search catalog scrolls back up to reveal the first result after a wheel to the bottom",
          "[core][sakura][search]") {
    CatalogRig rig;
    rig.scene.ctx.palette = Tones();
    rig.config.items.clear();
    for (U64 index = 0; index < 20; ++index) {
        rig.config.items.push_back({.id = "item" + std::to_string(index), .title = "Item", .group = "DSP"});
    }
    rig.config.maxHeight = 300.0f;
    rig.config.selected = 5;
    rig.place();

    CHECK(rig.host.send(Wheel(100.0f, 200.0f, 0.0f, -40.0f)));
    rig.host.place(rig.scene.ctx, rig.bounds);
    rig.click(100.0f, 60.0f);
    CHECK(rig.selections.back() == 15);

    rig.config.selected = 0;
    rig.place();
    rig.click(100.0f, 60.0f);
    CHECK(rig.selections.back() == 0);
    CHECK(Sakura::Access::cursor(rig.catalog, {100.0f, 60.0f}) == SurfaceCursor::Hand);
    CHECK(RowTiles(rig, "pill_active_bg") == std::vector<Jetstream::Rect>{{10.0f, 43.0f, 280.0f, 46.0f}});
}

TEST_CASE("Search catalog leaves the wheel unconsumed when the list does not overflow",
          "[core][sakura][search]") {
    CatalogRig fits;
    fits.place();
    CHECK_FALSE(fits.host.send(Wheel(100.0f, 200.0f, 0.0f, -1.0f)));
    CHECK_FALSE(fits.host.send(Wheel(100.0f, 200.0f, 0.0f, 1.0f)));
    fits.host.place(fits.scene.ctx, fits.bounds);
    fits.click(100.0f, 86.0f);
    CHECK(fits.selections == std::vector<U64>{0});
}

TEST_CASE("Search catalog chip and row hovers repaint once and clear on leave",
          "[core][sakura][search]") {
    CatalogRig rig;
    rig.scene.ctx.palette = Tones();
    rig.place();
    Frames frames(rig.host, rig.bounds, rig.scene.ctx);
    const auto update = [&] {
        rig.catalog.update(rig.config);
    };
    REQUIRE(frames.settle(update));

    const auto chips = ChipLabels(rig);
    REQUIRE(chips.size() == 4);
    const auto fill = [&](U64 index) {
        const auto* box = ChipFills(rig);
        return box ? box->config().instances[index].backgroundColor : ColorRGBA<F32>{};
    };
    CHECK(fill(2) == rig.scene.ctx.color("pill_bg"));

    const auto io = Center(chips[2].rect);
    rig.host.send(Mouse(MouseEventType::Move, io.x, io.y));
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(fill(2) == rig.scene.ctx.color("pill_hover_bg"));
    CHECK(fill(1) == rig.scene.ctx.color("pill_bg"));
    CHECK(DirtyFrames(frames, update) == 0);
    rig.host.send(Mouse(MouseEventType::Move, io.x + 1.0f, io.y));
    CHECK(frames.run(update).clean());

    rig.host.send(Mouse(MouseEventType::Leave, -10.0f, -10.0f));
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(fill(2) == rig.scene.ctx.color("pill_bg"));
    CHECK(DirtyFrames(frames, update) == 0);

    CHECK(RowTiles(rig, "chrome_button_hover").empty());
    rig.host.send(Mouse(MouseEventType::Move, 100.0f, 130.0f));
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(RowTiles(rig, "chrome_button_hover") == std::vector<Jetstream::Rect>{{10.0f, 115.0f, 280.0f, 46.0f}});
    CHECK(DirtyFrames(frames, update) == 0);
    rig.host.send(Mouse(MouseEventType::Move, 120.0f, 140.0f));
    CHECK(frames.run(update).clean());

    rig.host.send(Mouse(MouseEventType::Leave, -10.0f, -10.0f));
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(RowTiles(rig, "chrome_button_hover").empty());
    CHECK(DirtyFrames(frames, update) == 0);

    rig.host.send(Mouse(MouseEventType::Move, 500.0f, 100.0f));
    CHECK(frames.run(update).clean());
    rig.host.send(Mouse(MouseEventType::Move, 200.0f, 330.0f));
    CHECK(frames.run(update).clean());
}

TEST_CASE("Pill stepper value hover draws a reset halo that repaints once and clears on leave",
          "[core][sakura][search]") {
    Scene scene{Extent2D<F32>{1200.0f, 900.0f}};
    U64 resets = 0;
    Sakura::PillStepper stepper;
    Host host(stepper);
    Sakura::PillStepper::Config config{.label = "100%", .size = {120.0f, 34.0f}, .onReset = [&] { ++resets; }};
    Frames frames(host, {0.0f, 0.0f, 120.0f, 34.0f}, scene.ctx);
    const auto update = [&] {
        stepper.update(config);
    };
    const auto halo = [&]() -> std::optional<Jetstream::Rect> {
        for (auto* child : Peek::Children(stepper)) {
            if (auto* box = dynamic_cast<Sakura::Box*>(child); box && box->config().instances.size() == 1 &&
                                                                  box->config().cornerRadius == 12.0f) {
                return box->config().instances.front().rect;
            }
        }
        return std::nullopt;
    };
    REQUIRE(frames.settle(update));
    CHECK_FALSE(halo().has_value());

    host.send(Mouse(MouseEventType::Move, 60.0f, 17.0f));
    CHECK(RepaintOnly(frames.run(update)));
    REQUIRE(halo().has_value());
    CHECK(*halo() == Jetstream::Rect{31.0f, 5.0f, 58.0f, 24.0f});
    CHECK(Peek::CursorAt(stepper, {60.0f, 17.0f}) == SurfaceCursor::Hand);
    CHECK(DirtyFrames(frames, update) == 0);
    host.send(Mouse(MouseEventType::Move, 70.0f, 20.0f));
    CHECK(frames.run(update).clean());

    host.send(Mouse(MouseEventType::Leave, -10.0f, -10.0f));
    CHECK(RepaintOnly(frames.run(update)));
    CHECK_FALSE(halo().has_value());
    CHECK(DirtyFrames(frames, update) == 0);
    CHECK(resets == 0);

    config.onReset = nullptr;
    REQUIRE(frames.settle(update));
    host.send(Mouse(MouseEventType::Move, 60.0f, 17.0f));
    CHECK(frames.run(update).clean());
    CHECK_FALSE(halo().has_value());
    CHECK_FALSE(Peek::CursorAt(stepper, {60.0f, 17.0f}).has_value());
}

TEST_CASE("Pill fields ignore presses released elsewhere and right clicks",
          "[core][sakura][search]") {
    Scene scene{Extent2D<F32>{1200.0f, 900.0f}};
    U64 opens = 0;
    Sakura::PillField pill;
    Host pillHost(pill);
    pill.update({.placeholder = "Search", .size = {240.0f, 34.0f}, .onOpen = [&] { ++opens; }});
    pillHost.place(scene.ctx, {0.0f, 0.0f, 240.0f, 34.0f});
    pillHost.send(Mouse(MouseEventType::Click, 100.0f, 17.0f));
    pillHost.send(Mouse(MouseEventType::Release, 300.0f, 17.0f));
    pillHost.click(100.0f, 17.0f, MouseButton::Right);
    CHECK(opens == 0);
    pillHost.click(100.0f, 17.0f);
    CHECK(opens == 1);
    CHECK_FALSE(pillHost.send(Wheel(100.0f, 17.0f, -1.0f)));
    CHECK_FALSE(pillHost.send(Mouse(MouseEventType::Move, 100.0f, 17.0f)));
}

TEST_CASE("Pill steppers ignore presses released on another part and right clicks",
          "[core][sakura][search]") {
    Scene scene{Extent2D<F32>{1200.0f, 900.0f}};
    std::vector<std::string> calls;
    Sakura::PillStepper stepper;
    Host stepperHost(stepper);
    stepper.update({
        .label = "100%",
        .size = {120.0f, 34.0f},
        .onDecrement = [&] { calls.push_back("down"); },
        .onIncrement = [&] { calls.push_back("up"); },
        .onReset = [&] { calls.push_back("reset"); },
    });
    stepperHost.place(scene.ctx, {0.0f, 0.0f, 120.0f, 34.0f});
    stepperHost.send(Mouse(MouseEventType::Click, 17.0f, 17.0f));
    stepperHost.send(Mouse(MouseEventType::Release, 103.0f, 17.0f));
    stepperHost.send(Mouse(MouseEventType::Click, 60.0f, 17.0f));
    stepperHost.send(Mouse(MouseEventType::Release, 300.0f, 17.0f));
    stepperHost.send(Mouse(MouseEventType::Click, 103.0f, 17.0f));
    stepperHost.send(Mouse(MouseEventType::Leave, -10.0f, -10.0f));
    stepperHost.send(Mouse(MouseEventType::Release, 103.0f, 17.0f));
    for (const F32 x : {17.0f, 60.0f, 103.0f}) {
        stepperHost.click(x, 17.0f, MouseButton::Right);
    }
    CHECK(calls.empty());
    stepperHost.click(103.0f, 17.0f);
    CHECK(calls == std::vector<std::string>{"up"});
}

TEST_CASE("Search catalog chips and rows ignore right clicks and releases from presses elsewhere",
          "[core][sakura][search]") {
    CatalogRig rig;
    std::vector<std::string> picked;
    rig.config.onGroup = [&](const std::string& group) {
        picked.push_back(group);
    };
    rig.place();
    const auto chips = ChipLabels(rig);
    REQUIRE(chips.size() == 4);
    const auto io = Center(chips[2].rect);
    rig.host.send(Mouse(MouseEventType::Click, io.x, io.y, MouseButton::Right));
    rig.host.send(Mouse(MouseEventType::Release, io.x, io.y, MouseButton::Right));
    rig.host.send(Mouse(MouseEventType::Click, 200.0f, 330.0f));
    rig.host.send(Mouse(MouseEventType::Release, io.x, io.y));
    CHECK(picked.empty());
    rig.host.send(Mouse(MouseEventType::Click, 100.0f, 126.0f, MouseButton::Right));
    rig.host.send(Mouse(MouseEventType::Click, 100.0f, 126.0f, MouseButton::Right, 2));
    CHECK(rig.activations.empty());
    CHECK(rig.selections.empty());
}

TEST_CASE("Icon only pill button is a square with the icon centered", "[core][sakura][search]") {
    Scene scene{Extent2D<F32>{1200.0f, 900.0f}};
    Sakura::PillButton button;
    Host host(button);
    U64 clicks = 0;
    button.update({.icon = ICON_LC_CIRCLE_ARROW_UP, .onClick = [&] { ++clicks; }});
    CHECK(host.measureOf(scene.ctx, {kInfinite, kInfinite}) == Extent2D<F32>{28.0f, 28.0f});
    host.place(scene.ctx, {100.0f, 10.0f, 28.0f, 28.0f});

    std::vector<Sakura::Text*> texts;
    for (auto* child : Peek::Children(button)) {
        if (auto* text = dynamic_cast<Sakura::Text*>(child)) {
            texts.push_back(text);
        }
    }
    REQUIRE(texts.size() == 2);
    CHECK(texts[0]->visible());
    CHECK_FALSE(texts[1]->visible());
    const auto icon = Peek::Frame(*texts[0]);
    CHECK(icon.x + icon.width * 0.5f == Catch::Approx(114.0f).margin(1.0f));
    CHECK(icon.y + icon.height * 0.5f == Catch::Approx(24.0f).margin(1.0f));

    button.update({.icon = ICON_LC_CIRCLE_ARROW_UP, .height = 34.0f});
    CHECK(host.measureOf(scene.ctx, {kInfinite, kInfinite}) == Extent2D<F32>{34.0f, 34.0f});
}

TEST_CASE("Search catalog docs scroll back to the top when the selected block changes", "[core][sakura][search]") {
    CatalogRig rig;
    std::string longDocs = "# Block";
    for (U64 line = 0; line < 80; ++line) {
        longDocs += jst::fmt::format("\n\nParagraph {} explains a detail of the block.", line);
    }
    rig.config.items[0].documentation = longDocs;
    rig.config.items[1].documentation = longDocs;
    rig.place();

    const auto scrolls = rig.all<Sakura::ScrollArea>();
    REQUIRE(scrolls.size() == 1);
    const auto markdowns = Find<Sakura::Markdown>(*scrolls.front());
    REQUIRE(markdowns.size() == 1);
    const F32 top = Peek::Frame(*markdowns.front()).y;
    const auto area = Peek::Frame(*scrolls.front());
    const auto scrollDown = [&] {
        for (U64 step = 0; step < 5; ++step) {
            rig.host.send(Wheel(area.x + area.width * 0.5f, area.y + area.height * 0.5f, -3.0f));
        }
        rig.place();
    };

    scrollDown();
    REQUIRE(Peek::Frame(*markdowns.front()).y < top);
    rig.place();
    CHECK(Peek::Frame(*markdowns.front()).y < top);

    rig.config.selected = 1;
    rig.place();
    CHECK(Peek::Frame(*markdowns.front()).y == top);

    scrollDown();
    REQUIRE(Peek::Frame(*markdowns.front()).y < top);
    rig.config.selected = 0;
    rig.place();
    CHECK(Peek::Frame(*markdowns.front()).y == top);
}

TEST_CASE("Search catalog bounds follow custom card widths and clamp the docs height to the max height",
          "[core][sakura][search]") {
    CatalogRig rig;
    rig.config.listWidth = 200.0f;
    rig.config.docsWidth = 500.0f;
    rig.place();
    CHECK(rig.bounds.width == 200.0f + 8.0f + 500.0f);
    CHECK(rig.bounds.height == 360.0f);
    const auto scrolls = rig.all<Sakura::ScrollArea>();
    REQUIRE(scrolls.size() == 1);
    CHECK(Peek::Frame(*scrolls[0]).x == 208.0f + 10.0f);
    CHECK(Peek::Frame(*scrolls[0]).width == 500.0f - 20.0f);

    rig.config.maxHeight = 200.0f;
    rig.place();
    CHECK(rig.bounds.height == 200.0f);
    U64 cards = 0;
    for (auto* box : rig.all<Sakura::Box>()) {
        if (box->config().cornerRadius == 12.0f && box->config().borderWidth == 1.0f && !box->config().topLit) {
            for (const auto& instance : box->config().instances) {
                if (instance.rect.width >= 200.0f) {
                    CHECK(instance.rect.height == 200.0f);
                    ++cards;
                }
            }
        }
    }
    CHECK(cards == 2);
}

TEST_CASE("Search catalog heights are natural, fixed, or filling and keep the docs card unless natural",
          "[core][sakura][search]") {
    CatalogRig rig;
    rig.config.query = "zzz";
    rig.catalog.update(rig.config);
    const auto natural = rig.host.measureOf(rig.scene.ctx, {kInfinite, kInfinite});
    CHECK(natural.x == 300.0f);
    CHECK(natural.y < 300.0f);

    rig.config.height = 400.0f;
    rig.catalog.update(rig.config);
    CHECK(rig.host.measureOf(rig.scene.ctx, {kInfinite, kInfinite}) == Extent2D<F32>{748.0f, 400.0f});
    CHECK(rig.host.measureOf(rig.scene.ctx, {kInfinite, 300.0f}).y == 300.0f);
    auto doubled = rig.scene.ctx;
    doubled.pixelRatio = 2.0f;
    CHECK(rig.host.measureOf(doubled, {kInfinite, kInfinite}) == Extent2D<F32>{1496.0f, 800.0f});
    rig.host.place(rig.scene.ctx, {0.0f, 0.0f, 748.0f, 400.0f});
    U64 cards = 0;
    for (auto* box : rig.all<Sakura::Box>()) {
        if (box->config().cornerRadius == 12.0f && box->config().borderWidth == 1.0f && !box->config().topLit) {
            for (const auto& instance : box->config().instances) {
                if (instance.rect.width >= 300.0f) {
                    CHECK(instance.rect.height == 400.0f);
                    ++cards;
                }
            }
        }
    }
    CHECK(cards == 2);

    rig.config.height = Sakura::Fill;
    rig.catalog.update(rig.config);
    CHECK(rig.host.measureOf(rig.scene.ctx, {1000.0f, 500.0f}) == Extent2D<F32>{748.0f, 500.0f});
    CHECK(rig.host.measureOf(rig.scene.ctx, {kInfinite, kInfinite}) == Extent2D<F32>{748.0f, natural.y});

    rig.config.query.clear();
    rig.config.height = 0.0f;
    rig.catalog.update(rig.config);
    CHECK(rig.host.measureOf(rig.scene.ctx, {kInfinite, kInfinite}).y == 360.0f);
    rig.config.height = 120.0f;
    rig.catalog.update(rig.config);
    CHECK(rig.host.measureOf(rig.scene.ctx, {kInfinite, kInfinite}).y == 120.0f);
    rig.config.height = Sakura::Fill;
    rig.catalog.update(rig.config);
    CHECK(rig.host.measureOf(rig.scene.ctx, {kInfinite, kInfinite}).y == 360.0f);
    CHECK(rig.host.measureOf(rig.scene.ctx, {kInfinite, 500.0f}).y == 500.0f);
}

TEST_CASE("Search catalog fixed heights size the whole component with the field above the cards",
          "[core][sakura][search]") {
    CatalogRig rig;
    Sakura::PillField field;
    field.update({.placeholder = "Search", .editing = true, .size = {0.0f, 34.0f}});
    rig.config.field = &field;
    rig.config.height = 400.0f;
    rig.place();
    CHECK(rig.bounds.height == 400.0f);
    CHECK(Peek::Frame(field) == Jetstream::Rect{0.0f, 0.0f, 748.0f, 34.0f});
    U64 cards = 0;
    for (auto* box : rig.all<Sakura::Box>()) {
        if (box->config().cornerRadius == 12.0f && box->config().borderWidth == 1.0f && !box->config().topLit) {
            for (const auto& instance : box->config().instances) {
                if (instance.rect.width >= 300.0f) {
                    CHECK(instance.rect.y == 42.0f);
                    CHECK(instance.rect.bottom() == 400.0f);
                    ++cards;
                }
            }
        }
    }
    CHECK(cards == 2);

    auto doubled = rig.scene.ctx;
    doubled.pixelRatio = 2.0f;
    CHECK(rig.host.measureOf(doubled, {kInfinite, kInfinite}).y == 800.0f);
    rig.config.height = Sakura::Fill;
    rig.catalog.update(rig.config);
    CHECK(rig.host.measureOf(rig.scene.ctx, {kInfinite, 500.0f}).y == 500.0f);
}

TEST_CASE("Search catalog treats a zero max height as no cap on its natural height", "[core][sakura][search]") {
    CatalogRig rig;
    rig.config.items.clear();
    for (U64 index = 0; index < 20; ++index) {
        rig.config.items.push_back({.id = "item" + std::to_string(index), .title = "Item", .group = "DSP"});
    }
    rig.config.maxHeight = 300.0f;
    rig.catalog.update(rig.config);
    CHECK(rig.host.measureOf(rig.scene.ctx, {kInfinite, kInfinite}).y == 300.0f);

    rig.config.maxHeight = 0.0f;
    rig.place();
    CHECK(rig.bounds.height == 12.0f + 36.0f + 24.0f + 20.0f * 48.0f);
    CHECK_FALSE(rig.host.send(Wheel(100.0f, 200.0f, 0.0f, -1.0f)));
}

TEST_CASE("Search catalog measures wrapped chips at the capped width it lays them out at",
          "[core][sakura][search]") {
    CatalogRig rig;
    rig.config.items.clear();
    rig.config.groups.clear();
    for (U64 index = 0; index < 12; ++index) {
        const std::string group = "Group " + std::to_string(index);
        rig.config.items.push_back({.id = group, .title = group, .group = group});
        rig.config.groups.push_back(group);
    }
    rig.config.maxHeight = 0.0f;
    const auto rows = [&] {
        std::vector<F32> found;
        for (const auto& chip : ChipLabels(rig)) {
            found.push_back(chip.rect.y);
        }
        std::sort(found.begin(), found.end());
        found.erase(std::unique(found.begin(), found.end()), found.end());
        return static_cast<F32>(found.size());
    };

    rig.place();
    const F32 wide = rig.bounds.height;
    const F32 wideRows = rows();

    const auto narrow = rig.host.measureOf(rig.scene.ctx, {200.0f, kInfinite});
    CHECK(narrow.x == 200.0f);
    rig.host.place(rig.scene.ctx, {0.0f, 0.0f, narrow.x, narrow.y});
    REQUIRE(rows() > wideRows);
    CHECK(narrow.y - wide == (rows() - wideRows) * 30.0f);
    CHECK_FALSE(rig.host.send(Wheel(100.0f, narrow.y - 20.0f, 0.0f, -1.0f)));
}

TEST_CASE("Search catalog drops the docs card when both cards do not fit and restores it when they do",
          "[core][sakura][search]") {
    CatalogRig rig;
    rig.config.docsSide = Sakura::SearchCatalog::DocsSide::Left;
    rig.catalog.update(rig.config);
    const auto narrow = rig.host.measureOf(rig.scene.ctx, {500.0f, kInfinite});
    CHECK(narrow.x == 300.0f);
    rig.host.place(rig.scene.ctx, {0.0f, 0.0f, 500.0f, narrow.y});
    CHECK(rig.all<Sakura::ScrollArea>().empty());

    U64 wide = 0;
    for (auto* box : rig.all<Sakura::Box>()) {
        if (box->config().cornerRadius == 12.0f && box->config().borderWidth == 1.0f && !box->config().topLit) {
            for (const auto& instance : box->config().instances) {
                CHECK(instance.rect.right() <= 500.0f);
                wide += instance.rect.width >= 300.0f ? 1 : 0;
            }
        }
    }
    CHECK(wide == 1);

    const F32 cramped = 120.0f;
    rig.host.place(rig.scene.ctx, {0.0f, 0.0f, cramped, narrow.y});
    CHECK(rig.all<Sakura::ScrollArea>().empty());
    for (auto* box : rig.all<Sakura::Box>()) {
        if (box->config().cornerRadius == 12.0f && box->config().borderWidth == 1.0f && !box->config().topLit) {
            for (const auto& instance : box->config().instances) {
                CHECK(instance.rect.right() <= cramped);
            }
        }
    }

    rig.place();
    CHECK(rig.bounds.width == 300.0f + 8.0f + 440.0f);
    CHECK(rig.all<Sakura::ScrollArea>().size() == 1);
}

TEST_CASE("Pill steppers let releases they did not press pass to the component that owns the press",
          "[core][sakura][search]") {
    const Scene scene;
    U64 calls = 0;
    Sakura::PillStepper stepper;
    stepper.update({
        .label = "100%",
        .size = {120.0f, 34.0f},
        .onDecrement = [&] { ++calls; },
        .onIncrement = [&] { ++calls; },
        .onReset = [&] { ++calls; },
    });
    Host host(stepper);
    host.place(scene.ctx, {0.0f, 0.0f, 120.0f, 34.0f});
    CHECK_FALSE(host.send(Mouse(MouseEventType::Release, 60.0f, 17.0f)));
    CHECK(calls == 0);
    CHECK(host.send(Mouse(MouseEventType::Click, 60.0f, 17.0f)));
    CHECK(host.send(Mouse(MouseEventType::Release, 60.0f, 17.0f)));
    CHECK(calls == 1);
}

TEST_CASE("Pills keep a left press across releases of other buttons", "[core][sakura][search]") {
    const Scene scene;
    U64 steps = 0;
    U64 opens = 0;
    Sakura::PillStepper stepper;
    stepper.update({.label = "100%", .size = {120.0f, 34.0f}, .onReset = [&] { ++steps; }});
    Sakura::PillField field;
    field.update({.placeholder = "Search", .size = {120.0f, 34.0f}, .onOpen = [&] { ++opens; }});
    for (Sakura::Component* component : std::initializer_list<Sakura::Component*>{&stepper, &field}) {
        Host host(*component);
        host.place(scene.ctx, {0.0f, 0.0f, 120.0f, 34.0f});
        CHECK(host.send(Mouse(MouseEventType::Click, 60.0f, 17.0f)));
        CHECK(host.send(Mouse(MouseEventType::Click, 60.0f, 17.0f, MouseButton::Right)));
        CHECK_FALSE(host.send(Mouse(MouseEventType::Release, 60.0f, 17.0f, MouseButton::Right)));
        CHECK_FALSE(host.send(Mouse(MouseEventType::Release, 60.0f, 17.0f, MouseButton::Middle)));
        CHECK(host.send(Mouse(MouseEventType::Release, 60.0f, 17.0f)));
    }
    CHECK(steps == 1);
    CHECK(opens == 1);
}

namespace {

constexpr const char* kIconButtonTags = "[core][sakura][pill-icon-button]";

const Sakura::Box& IconButtonHalo(const Sakura::PillIconButton& button) {
    return *dynamic_cast<const Sakura::Box*>(Peek::Children(button).at(1));
}

const Sakura::Text& IconButtonGlyph(const Sakura::PillIconButton& button) {
    return *dynamic_cast<const Sakura::Text*>(Peek::Children(button).at(2));
}

}  // namespace

TEST_CASE("Pill icon buttons measure square pills and honor fixed and filling sizes", kIconButtonTags) {
    const Scene scene;
    Sakura::PillIconButton button;
    Host host(button);

    button.update({.icon = ICON_LC_INFO, .size = {0.0f, 0.0f}});
    CHECK(host.measureOf(scene.ctx, {kInfinite, kInfinite}) == Extent2D<F32>{34.0f, 34.0f});
    CHECK(host.measureOf(scene.ctx, {20.0f, 10.0f}) == Extent2D<F32>{20.0f, 10.0f});

    button.update({.icon = ICON_LC_INFO});
    CHECK(host.measureOf(scene.ctx, {kInfinite, kInfinite}) == Extent2D<F32>{34.0f, 34.0f});

    button.update({.icon = ICON_LC_INFO, .size = {40.0f, 30.0f}});
    CHECK(host.measureOf(scene.ctx, {300.0f, 300.0f}) == Extent2D<F32>{40.0f, 30.0f});

    button.update({.icon = ICON_LC_INFO, .size = {Sakura::Fill, Sakura::Fill}});
    CHECK(host.measureOf(scene.ctx, {120.0f, 50.0f}) == Extent2D<F32>{120.0f, 50.0f});
    CHECK(host.measureOf(scene.ctx, {kInfinite, kInfinite}) == Extent2D<F32>{34.0f, 34.0f});

    auto dense = scene.ctx;
    dense.pixelRatio = 2.0f;
    button.update({.icon = ICON_LC_INFO});
    CHECK(host.measureOf(dense, {kInfinite, kInfinite}) == Extent2D<F32>{68.0f, 68.0f});
}

TEST_CASE("Pill icon buttons act on a left release over the button they were pressed on", kIconButtonTags) {
    const Scene scene;
    U64 clicks = 0;
    Sakura::PillIconButton button;
    Host host(button);
    button.update({.icon = ICON_LC_PAUSE, .onClick = [&] { ++clicks; }});
    host.place(scene.ctx, {0.0f, 0.0f, 34.0f, 34.0f});

    CHECK(Peek::CursorAt(button, {17.0f, 17.0f}) == SurfaceCursor::Hand);
    CHECK_FALSE(Peek::CursorAt(button, {60.0f, 17.0f}).has_value());

    CHECK(host.send(Mouse(MouseEventType::Click, 17.0f, 17.0f)));
    CHECK(clicks == 0);
    CHECK(host.send(Mouse(MouseEventType::Release, 17.0f, 17.0f)));
    CHECK(clicks == 1);

    CHECK_FALSE(host.send(Mouse(MouseEventType::Release, 17.0f, 17.0f)));
    CHECK(clicks == 1);

    host.send(Mouse(MouseEventType::Click, 17.0f, 17.0f));
    host.send(Mouse(MouseEventType::Release, 80.0f, 17.0f));
    CHECK(clicks == 1);

    host.click(17.0f, 17.0f, MouseButton::Right);
    CHECK(clicks == 1);

    host.send(Mouse(MouseEventType::Click, 17.0f, 17.0f));
    host.send(Mouse(MouseEventType::Cancel, 17.0f, 17.0f));
    host.send(Mouse(MouseEventType::Release, 17.0f, 17.0f));
    CHECK(clicks == 1);

    host.send(Mouse(MouseEventType::Click, 17.0f, 17.0f));
    host.send(Mouse(MouseEventType::Release, 17.0f, 17.0f, MouseButton::Right));
    host.send(Mouse(MouseEventType::Release, 17.0f, 17.0f));
    CHECK(clicks == 2);

    CHECK_FALSE(host.send(Wheel(17.0f, 17.0f, -1.0f)));
}

TEST_CASE("Pill icon buttons leave whole secondary gestures to the component beneath", kIconButtonTags) {
    const Scene scene;
    U64 clicks = 0;
    Probe behind({100.0f, 100.0f});
    behind.prepare();
    Sakura::PillIconButton button;
    button.update({.icon = ICON_LC_PAUSE, .onClick = [&] { ++clicks; }});
    Sakura::Overlay overlay;
    overlay.update({.items = {{.child = &behind}, {.child = &button, .anchor = Sakura::Anchor::TopLeft}}});
    Host host(overlay);
    host.place(scene.ctx, {0.0f, 0.0f, 100.0f, 100.0f});
    REQUIRE(Peek::Frame(button) == Rect{0.0f, 0.0f, 34.0f, 34.0f});
    const auto received = [&](MouseEventType type, MouseButton pressed) {
        return std::count_if(behind.events.begin(), behind.events.end(), [&](const MouseEvent& event) {
            return event.type == type && event.button == pressed;
        });
    };

    for (const auto secondary : {MouseButton::Right, MouseButton::Middle}) {
        CHECK(host.send(Mouse(MouseEventType::Click, 17.0f, 17.0f, secondary)));
        CHECK(host.send(Mouse(MouseEventType::Release, 17.0f, 17.0f, secondary)));
        CHECK(received(MouseEventType::Click, secondary) == 1);
        CHECK(received(MouseEventType::Release, secondary) == 1);
    }
    CHECK(clicks == 0);

    CHECK(host.send(Mouse(MouseEventType::Click, 17.0f, 17.0f)));
    CHECK(host.send(Mouse(MouseEventType::Click, 17.0f, 17.0f, MouseButton::Right)));
    CHECK(host.send(Mouse(MouseEventType::Release, 17.0f, 17.0f, MouseButton::Right)));
    CHECK(host.send(Mouse(MouseEventType::Release, 17.0f, 17.0f)));
    CHECK(clicks == 1);
    CHECK(received(MouseEventType::Click, MouseButton::Right) == 2);
    CHECK(received(MouseEventType::Release, MouseButton::Right) == 2);
    CHECK(received(MouseEventType::Click, MouseButton::Left) == 0);
    CHECK(received(MouseEventType::Release, MouseButton::Left) == 0);
}

TEST_CASE("Pill icon buttons settle and repaint only for hover and icon changes", kIconButtonTags) {
    Scene scene;
    Sakura::PillIconButton button;
    Place place(button, {10.0f, 10.0f, 34.0f, 34.0f});
    Frames frames(place, kScreen, scene.context());
    Sakura::PillIconButton::Config config{.icon = ICON_LC_PAUSE, .iconColorKey = "run_state_running", .tinted = true};
    const auto update = [&] {
        button.update(config);
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
    CHECK_FALSE(IconButtonHalo(button).config().instances.front().visible);
    CHECK(IconButtonGlyph(button).config().colorKey == "run_state_running");

    CheckHoverRepaintsOnce(frames, place, update, {27.0f, 27.0f}, {200.0f, 200.0f});
    CheckLeaveRepaintsOnce(frames, place, update, {27.0f, 27.0f});

    place.send(Mouse(MouseEventType::Move, 27.0f, 27.0f));
    CHECK(RepaintOnly(frames.run(update)));
    const auto& halo = IconButtonHalo(button).config();
    REQUIRE(halo.instances.size() == 1);
    CHECK(halo.instances.front().visible);
    CHECK(halo.instances.front().rect == Rect{10.0f, 10.0f, 34.0f, 34.0f});
    CHECK(halo.cornerRadius == 17.0f);
    CHECK(halo.instances.front().backgroundColor == DistinctPalette().at("chrome_button_hover"));
    place.send(Mouse(MouseEventType::Leave, 27.0f, 27.0f));
    REQUIRE(frames.settle(update));

    config.icon = ICON_LC_PLAY;
    config.iconColorKey = "run_state_paused";
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    CHECK(IconButtonGlyph(button).config().colorKey == "run_state_paused");
    CHECK(frames.rebuilds == 1);
    CHECK(frames.restructures == 0);
}

TEST_CASE("Pill icon buttons scale their halo with the pixel ratio", kIconButtonTags) {
    Scene scene;
    Sakura::PillIconButton button;
    Host host(button);
    button.update({.icon = ICON_LC_INFO});
    auto ctx = scene.context();
    ctx.pixelRatio = 2.0f;
    host.place(ctx, {0.0f, 0.0f, 68.0f, 68.0f});
    host.send(Mouse(MouseEventType::Move, 34.0f, 34.0f));
    host.place(ctx, {0.0f, 0.0f, 68.0f, 68.0f});
    const auto& halo = IconButtonHalo(button).config();
    REQUIRE(halo.instances.size() == 1);
    CHECK(halo.instances.front().visible);
    CHECK(halo.instances.front().rect == Rect{3.0f, 3.0f, 62.0f, 62.0f});
    CHECK(halo.cornerRadius == 31.0f);
    const Rect glyph = Peek::Frame(IconButtonGlyph(button));
    CHECK(glyph.x == std::round(glyph.x));
    CHECK(glyph.y == std::round(glyph.y));
}

TEST_CASE("Pill icon buttons draw a colored ring only when tinted", kIconButtonTags) {
    Scene scene;
    Sakura::PillIconButton button;
    Host host(button);
    const auto ctx = scene.context();
    const auto hovered = [&](bool tinted) {
        button.update({.icon = ICON_LC_PAUSE, .iconColorKey = "run_state_running", .tinted = tinted});
        host.place(ctx, {0.0f, 0.0f, 34.0f, 34.0f});
        host.send(Mouse(MouseEventType::Move, 17.0f, 17.0f));
        host.place(ctx, {0.0f, 0.0f, 34.0f, 34.0f});
        return IconButtonHalo(button).config();
    };

    const auto plain = hovered(false);
    REQUIRE(plain.instances.size() == 1);
    CHECK(plain.instances.front().rect == Rect{1.5f, 1.5f, 31.0f, 31.0f});
    CHECK(plain.borderWidth == 0.0f);

    const auto tinted = hovered(true);
    REQUIRE(tinted.instances.size() == 1);
    CHECK(tinted.instances.front().rect == Rect{0.0f, 0.0f, 34.0f, 34.0f});
    CHECK(tinted.borderWidth == 1.5f);
    CHECK(tinted.borderColor.r == ctx.color("run_state_running").r);
    CHECK(tinted.borderColor.a == Catch::Approx(ctx.color("run_state_running").a * 0.7f));
}

TEST_CASE("Pill icon buttons show their tooltip below after the hover delay and drop it on leave", kIconButtonTags) {
    Scene scene;
    Sakura::PillIconButton button;
    LayerSlot slot;
    LayerHost host({&slot});
    slot.hold(button, {700.0f, 12.0f, 34.0f, 34.0f});
    Frames frames(host, kScreen, scene.context());
    frames.timeStep = 0.1;
    frames.ctx.hoverDelay = 0.25f;
    const auto update = [&] {
        button.update({.icon = ICON_LC_SAVE, .tooltip = "Save flowgraph"});
    };
    const auto tooltip = [&]() -> std::optional<Jetstream::Rect> {
        for (auto* text : Find<Sakura::Text>(host.popups)) {
            if (text->config().str == "Save flowgraph") {
                return Peek::Frame(*text);
            }
        }
        return std::nullopt;
    };
    REQUIRE(frames.settle(update));
    CHECK_FALSE(tooltip().has_value());

    Hover(host, frames, 717.0f, 29.0f);
    frames.run(update);
    CHECK_FALSE(tooltip().has_value());
    for (U64 frame = 0; frame < 6; ++frame) {
        frames.run(update);
    }
    REQUIRE(tooltip().has_value());
    CHECK(tooltip()->y > 46.0f);
    CHECK(tooltip()->right() <= 734.0f);
    CHECK(Retains(frames, update));

    Hover(host, frames, 300.0f, 300.0f);
    frames.run(update);
    CHECK_FALSE(tooltip().has_value());
    CHECK(Retains(frames, update));
}
