#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <jetstream/render/sakura/components/vstack.hh>

#include "harness.hh"

#include <cmath>
#include <vector>

using namespace Jetstream;
using VStack = Sakura::VStack;

namespace {

// Renders a fixed-height item (in ImGui pixels) for a given child slot.
template<int Pixels>
Sakura::VStack::Child FixedItem() {
    return [](const Sakura::Context&) {
        ImGui::Dummy(ImVec2(0.0f, Pixels));
    };
}

void renderFrame(SakuraTest::HeadlessUi& ui,
                const Sakura::Context& ctx,
                const VStack& stack,
                const std::vector<VStack::Child>& children) {
    ui.frame([&] {
        stack.render(ctx, children);
    });
}

}  // namespace

TEST_CASE("VStack stays unmanaged without items", "[core][sakura][vstack]") {
    SakuraTest::HeadlessUi ui;
    const auto ctx = ui.sakura();

    VStack stack;
    VStack::Config config;
    config.id = "unmanaged";

    REQUIRE(stack.update(config));
    REQUIRE(stack.layout() == nullptr);

    renderFrame(ui, ctx, stack, {FixedItem<100>()});

    REQUIRE(stack.layout() == nullptr);
}

TEST_CASE("VStack reports provisional allocation before its first measurement",
          "[core][sakura][vstack][warmup]") {
    SakuraTest::HeadlessUi ui;
    const auto ctx = ui.sakura();

    VStack stack;
    VStack::Config config;
    config.id = "provisional";
    config.height = 500.0f;
    config.items = {
        {.id = "fixed", .flex = std::nullopt},
        {.id = "flex", .flex = VStack::Flex{.minimum = 150.0f, .grow = 1.0f}},
    };

    REQUIRE(stack.update(config));
    const auto* layout = stack.layout();
    REQUIRE(layout != nullptr);
    REQUIRE_FALSE(layout->measured);

    // No measurement yet: fixedHeight is unknown (0), so the single flex
    // item is provisionally allocated the full stack height.
    REQUIRE(layout->itemHeight(0) == std::nullopt);
    REQUIRE(layout->itemHeight(1) == Catch::Approx(500.0f));
    REQUIRE(layout->minimumHeight == std::nullopt);

    // Out-of-range queries report no allocation.
    REQUIRE(layout->itemHeight(999) == std::nullopt);
}

TEST_CASE("VStack resolves measured layout on the update after render",
          "[core][sakura][vstack][measurement]") {
    SakuraTest::HeadlessUi ui;
    const auto ctx = ui.sakura();

    VStack stack;
    VStack::Config config;
    config.id = "measured";
    config.height = 500.0f;
    config.items = {
        {.id = "fixed", .flex = std::nullopt},
        {.id = "flex", .flex = VStack::Flex{.minimum = 150.0f, .grow = 1.0f}},
    };

    std::vector<VStack::Child> children = {
        FixedItem<100>(),  // fixed chrome: 100 logical units
        FixedItem<123>(),  // flex item currently drawn at 123 units
    };

    REQUIRE(stack.update(config));
    renderFrame(ui, ctx, stack, children);
    REQUIRE(stack.update(config));

    const auto* layout = stack.layout();
    REQUIRE(layout != nullptr);
    REQUIRE(layout->measured);

    // Measured: flexible 123 resolved against the stack; fixed chrome = 100.
    REQUIRE(layout->fixedHeight == Catch::Approx(100.0f));

    // Available 400, single grow item fills it exactly.
    REQUIRE(layout->itemHeight(0) == std::nullopt);
    REQUIRE(layout->itemHeight(1) == Catch::Approx(400.0f));

    // Minimum node height: fixed 100 + capped flexible minimum 150.
    REQUIRE(layout->minimumHeight.has_value());
    REQUIRE(*layout->minimumHeight == Catch::Approx(250.0f));
}

TEST_CASE("VStack distributes grow proportionally across flexible items",
          "[core][sakura][vstack][flex]") {
    SakuraTest::HeadlessUi ui;
    const auto ctx = ui.sakura();

    VStack stack;
    VStack::Config config;
    config.id = "grow";
    config.height = 700.0f;
    config.items = {
        {.id = "fixed", .flex = std::nullopt},
        {.id = "flex-1", .flex = VStack::Flex{.minimum = 150.0f, .grow = 1.0f}},
        {.id = "flex-2", .flex = VStack::Flex{.minimum = 150.0f, .grow = 3.0f}},
    };

    std::vector<VStack::Child> children = {
        FixedItem<100>(),
        FixedItem<50>(),
        FixedItem<50>(),
    };

    REQUIRE(stack.update(config));
    renderFrame(ui, ctx, stack, children);
    REQUIRE(stack.update(config));

    const auto* layout = stack.layout();
    REQUIRE(layout != nullptr);
    REQUIRE(layout->fixedHeight == Catch::Approx(100.0f));

    // Available 600, sum minimums 300, leftover 300 over grow 4 -> 75/unit.
    REQUIRE(layout->itemHeight(1) == Catch::Approx(225.0f));  // 150 + 75
    REQUIRE(layout->itemHeight(2) == Catch::Approx(375.0f));  // 150 + 225
}

TEST_CASE("VStack preserves flexible bases and shrinks them toward minimums",
          "[core][sakura][vstack][flex]") {
    SakuraTest::HeadlessUi ui;
    const auto ctx = ui.sakura();

    VStack stack;
    VStack::Config config;
    config.id = "basis";
    config.height = 650.0f;
    config.items = {
        {.id = "fixed", .flex = std::nullopt},
        {.id = "flex-1", .flex = VStack::Flex{.minimum = 100.0f,
                                               .grow = 1.0f,
                                               .basis = 200.0f}},
        {.id = "flex-2", .flex = VStack::Flex{.minimum = 100.0f,
                                               .grow = 1.0f,
                                               .basis = 300.0f}},
    };

    std::vector<VStack::Child> children = {
        FixedItem<100>(),
        FixedItem<200>(),
        FixedItem<300>(),
    };

    REQUIRE(stack.update(config));
    renderFrame(ui, ctx, stack, children);
    REQUIRE(stack.update(config));

    // The 50 units above the combined bases are shared by grow weight.
    REQUIRE(stack.layout()->itemHeight(1) == Catch::Approx(225.0f));
    REQUIRE(stack.layout()->itemHeight(2) == Catch::Approx(325.0f));

    config.height = 450.0f;
    REQUIRE(stack.update(config));

    // Available height is now halfway from the minimum sum to the basis sum.
    REQUIRE(stack.layout()->itemHeight(1) == Catch::Approx(150.0f));
    REQUIRE(stack.layout()->itemHeight(2) == Catch::Approx(200.0f));
}

TEST_CASE("VStack shrinks flexible items proportionally and fills exactly",
          "[core][sakura][vstack][flex]") {
    SakuraTest::HeadlessUi ui;
    const auto ctx = ui.sakura();

    VStack stack;
    VStack::Config config;
    config.id = "shrink";
    config.height = 200.0f;
    config.items = {
        {.id = "fixed", .flex = std::nullopt},
        {.id = "flex-1", .flex = VStack::Flex{.minimum = 150.0f, .grow = 1.0f}},
        {.id = "flex-2", .flex = VStack::Flex{.minimum = 150.0f, .grow = 1.0f}},
    };

    std::vector<VStack::Child> children = {
        FixedItem<100>(),
        FixedItem<50>(),
        FixedItem<50>(),
    };

    REQUIRE(stack.update(config));
    renderFrame(ui, ctx, stack, children);
    REQUIRE(stack.update(config));

    const auto* layout = stack.layout();
    REQUIRE(layout != nullptr);

    // Items compress proportionally; the last one absorbs the remainder.
    const auto first = layout->itemHeight(1);
    const auto second = layout->itemHeight(2);
    REQUIRE(first.has_value());
    REQUIRE(second.has_value());
    REQUIRE(*first == Catch::Approx(50.0f).margin(0.01f));
    REQUIRE(*second == Catch::Approx(50.0f).margin(0.01f));
    REQUIRE(*first + *second == Catch::Approx(100.0f).margin(0.01f));
}

TEST_CASE("VStack does not stretch zero-grow items when space remains",
          "[core][sakura][vstack][flex]") {
    SakuraTest::HeadlessUi ui;
    const auto ctx = ui.sakura();

    VStack stack;
    VStack::Config config;
    config.id = "no-grow";
    config.height = 400.0f;
    config.items = {
        {.id = "fixed", .flex = std::nullopt},
        {.id = "flex", .flex = VStack::Flex{.minimum = 150.0f, .grow = 0.0f}},
    };

    std::vector<VStack::Child> children = {
        FixedItem<100>(),
        FixedItem<123>(),
    };

    REQUIRE(stack.update(config));
    renderFrame(ui, ctx, stack, children);
    REQUIRE(stack.update(config));

    const auto* layout = stack.layout();
    REQUIRE(layout != nullptr);

    // Zero grow keeps the minimum; the remainder stays unused.
    REQUIRE(layout->itemHeight(1) == Catch::Approx(150.0f));

    // Minimum height still reflects the flexible minimum, not the space.
    REQUIRE(layout->minimumHeight.has_value());
    REQUIRE(*layout->minimumHeight == Catch::Approx(250.0f));
}

TEST_CASE("VStack includes configured spacing in the measured fixed height",
          "[core][sakura][vstack][measurement]") {
    SakuraTest::HeadlessUi ui;
    const auto ctx = ui.sakura();

    VStack stack;
    VStack::Config config;
    config.id = "spacing";
    config.spacing = 10.0f;
    config.height = 500.0f;
    config.items = {
        {.id = "fixed-1", .flex = std::nullopt},
        {.id = "fixed-2", .flex = std::nullopt},
        {.id = "flex", .flex = VStack::Flex{.minimum = 150.0f, .grow = 1.0f}},
    };

    std::vector<VStack::Child> children = {
        FixedItem<100>(),
        FixedItem<100>(),
        FixedItem<123>(),
    };

    REQUIRE(stack.update(config));
    renderFrame(ui, ctx, stack, children);
    REQUIRE(stack.update(config));

    const auto* layout = stack.layout();
    REQUIRE(layout != nullptr);

    // Two 10-unit spacers land in the fixed height: 343 - 123 = 220.
    REQUIRE(layout->fixedHeight == Catch::Approx(220.0f));
    REQUIRE(layout->itemHeight(2) == Catch::Approx(280.0f));
}

TEST_CASE("VStack preserves measurements when only the height changes",
          "[core][sakura][vstack][contract]") {
    SakuraTest::HeadlessUi ui;
    const auto ctx = ui.sakura();

    VStack stack;
    VStack::Config config;
    config.id = "resize";
    config.height = 500.0f;
    config.items = {
        {.id = "fixed", .flex = std::nullopt},
        {.id = "flex", .flex = VStack::Flex{.minimum = 150.0f, .grow = 1.0f}},
    };

    std::vector<VStack::Child> children = {
        FixedItem<100>(),
        FixedItem<123>(),
    };

    REQUIRE(stack.update(config));
    renderFrame(ui, ctx, stack, children);
    REQUIRE(stack.update(config));
    REQUIRE(stack.layout()->measured);
    REQUIRE(stack.layout()->itemHeight(1) == Catch::Approx(400.0f));

    // Height alone is not a contract change: no re-render needed.
    config.height = 300.0f;
    REQUIRE(stack.update(config));
    REQUIRE(stack.layout()->measured);
    REQUIRE(stack.layout()->itemHeight(1) == Catch::Approx(200.0f));
}

TEST_CASE("VStack reports the true flexible minimum regardless of the current height",
          "[core][sakura][vstack][minimum]") {
    SakuraTest::HeadlessUi ui;
    const auto ctx = ui.sakura();

    VStack stack;
    VStack::Config config;
    config.id = "true-minimum";
    config.height = 500.0f;
    config.items = {
        {.id = "fixed", .flex = std::nullopt},
        {.id = "flex", .flex = VStack::Flex{.minimum = 150.0f, .grow = 1.0f}},
    };

    std::vector<VStack::Child> children = {
        FixedItem<100>(),
        FixedItem<123>(),
    };

    REQUIRE(stack.update(config));
    renderFrame(ui, ctx, stack, children);
    REQUIRE(stack.update(config));
    REQUIRE(stack.layout()->measured);
    REQUIRE(*stack.layout()->minimumHeight == Catch::Approx(250.0f));

    // Shrinking the stack does not lower the minimum: it is the true
    // flexible minimum, not a cap at the current height.
    config.height = 120.0f;
    REQUIRE(stack.update(config));
    REQUIRE(*stack.layout()->minimumHeight == Catch::Approx(250.0f));

    // A grow-weight change does not lower it either: the minimum tracks
    // the SUM of flexible minimums, not the available height.
    config.items[1].flex = VStack::Flex{.minimum = 150.0f, .grow = 3.0f};
    REQUIRE(stack.update(config));
    renderFrame(ui, ctx, stack, children);
    REQUIRE(stack.update(config));
    REQUIRE(stack.layout()->measured);
    REQUIRE(*stack.layout()->minimumHeight == Catch::Approx(250.0f));
}

TEST_CASE("VStack invalidates measurements when the item contract changes",
          "[core][sakura][vstack][contract]") {
    SakuraTest::HeadlessUi ui;
    const auto ctx = ui.sakura();

    VStack stack;
    VStack::Config config;
    config.id = "contract";
    config.height = 500.0f;
    config.items = {
        {.id = "fixed", .flex = std::nullopt},
        {.id = "flex", .flex = VStack::Flex{.minimum = 150.0f, .grow = 1.0f}},
    };

    std::vector<VStack::Child> children = {
        FixedItem<100>(),
        FixedItem<123>(),
    };

    REQUIRE(stack.update(config));
    renderFrame(ui, ctx, stack, children);
    REQUIRE(stack.update(config));
    REQUIRE(stack.layout()->measured);

    // Adding an item invalidates the measurement; provisional meanwhile.
    config.items.push_back({.id = "flex-2",
                            .flex = VStack::Flex{.minimum = 150.0f, .grow = 1.0f}});
    REQUIRE(stack.update(config));
    REQUIRE_FALSE(stack.layout()->measured);

    std::vector<VStack::Child> nextChildren = {
        FixedItem<100>(),
        FixedItem<123>(),
        FixedItem<50>(),
    };
    renderFrame(ui, ctx, stack, nextChildren);
    REQUIRE(stack.update(config));
    REQUIRE(stack.layout()->measured);
    REQUIRE(stack.layout()->fixedHeight == Catch::Approx(100.0f));

    // Available 400, sum minimums 300, leftover 100 over grow 2 -> 50/unit.
    REQUIRE(stack.layout()->itemHeight(1) == Catch::Approx(200.0f));
    REQUIRE(stack.layout()->itemHeight(2) == Catch::Approx(200.0f));
}

TEST_CASE("VStack normalizes invalid configuration values",
          "[core][sakura][vstack][robustness]") {
    SakuraTest::HeadlessUi ui;
    const auto ctx = ui.sakura();

    VStack stack;
    VStack::Config config;
    config.id = "normalize";
    config.spacing = -2.0f;
    config.height = -5.0f;
    config.items = {
        {.id = "flex", .flex = VStack::Flex{.minimum = std::nanf(""), .grow = -1.0f}},
    };

    REQUIRE(stack.update(config));

    const auto* layout = stack.layout();
    REQUIRE(layout != nullptr);

    // Negative height clamps to zero: the flex item is allocated nothing.
    const auto height = layout->itemHeight(0);
    REQUIRE(height.has_value());
    REQUIRE(*height == Catch::Approx(0.0f));

    REQUIRE(stack.update(config));
    std::vector<VStack::Child> children = {FixedItem<50>()};
    renderFrame(ui, ctx, stack, children);
    REQUIRE(stack.update(config));

    layout = stack.layout();
    REQUIRE(std::isfinite(layout->fixedHeight));
    if (layout->minimumHeight.has_value()) {
        REQUIRE(std::isfinite(*layout->minimumHeight));
        REQUIRE(*layout->minimumHeight >= 0.0f);
    }
}

TEST_CASE("VStack measures in logical units independent of the scaling factor",
          "[core][sakura][vstack][scaling]") {
    SakuraTest::HeadlessUi ui(2.0f);
    const auto ctx = ui.sakura();

    VStack stack;
    VStack::Config config;
    config.id = "scaled";
    config.height = 250.0f;
    config.items = {
        {.id = "fixed", .flex = std::nullopt},
        {.id = "flex", .flex = VStack::Flex{.minimum = 150.0f, .grow = 1.0f}},
    };

    // Children draw in device pixels: 100px and 123px at scale 2.
    std::vector<VStack::Child> children = {
        FixedItem<100>(),
        FixedItem<123>(),
    };

    REQUIRE(stack.update(config));
    renderFrame(ui, ctx, stack, children);
    REQUIRE(stack.update(config));

    const auto* layout = stack.layout();
    REQUIRE(layout != nullptr);
    REQUIRE(layout->measured);
    REQUIRE(layout->fixedHeight == Catch::Approx(50.0f));  // 100px / 2

    // Available 200 logical units go to the single flexible item.
    REQUIRE(layout->itemHeight(1) == Catch::Approx(200.0f));
}

TEST_CASE("VStack resets managed state when children do not match items",
          "[core][sakura][vstack][robustness]") {
    SakuraTest::HeadlessUi ui;
    const auto ctx = ui.sakura();

    VStack stack;
    VStack::Config config;
    config.id = "mismatch";
    config.height = 500.0f;
    config.items = {
        {.id = "fixed", .flex = std::nullopt},
        {.id = "flex", .flex = VStack::Flex{.minimum = 150.0f, .grow = 1.0f}},
    };

    std::vector<VStack::Child> children = {
        FixedItem<100>(),
        FixedItem<123>(),
    };

    REQUIRE(stack.update(config));
    renderFrame(ui, ctx, stack, children);
    REQUIRE(stack.update(config));
    REQUIRE(stack.layout()->measured);

    // One child for two items: unmanaged fallback drops the measurement.
    renderFrame(ui, ctx, stack, {children[0]});
    REQUIRE(stack.update(config));
    REQUIRE_FALSE(stack.layout()->measured);
}
