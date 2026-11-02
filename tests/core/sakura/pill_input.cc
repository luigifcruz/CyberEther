#include <catch2/catch_test_macros.hpp>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/pill_input.hh>
#include <jetstream/render/sakura/components/text.hh>
#include <jetstream/render/sakura/components/text_input.hh>

#include "component_harness.hh"
#include "harness.hh"
#include "scene.hh"

#include <string>
#include <vector>

using namespace Jetstream;
using namespace SakuraTest;

namespace {

constexpr Rect kField = {10.0f, 10.0f, 260.0f, 34.0f};

bool RingShown(Sakura::PillInput& field, const Sakura::Context& ctx) {
    for (auto* box : Find<Sakura::Box>(field)) {
        const auto& config = box->config();
        if (config.borderWidth > 0.0f && config.borderColor == ctx.color("action_btn_outline") &&
            !config.instances.empty() && config.instances.front().visible) {
            return true;
        }
    }
    return false;
}

}  // namespace

TEST_CASE("Pill inputs fill the available width at pill height and scale with the pixel ratio",
          "[core][sakura][pill-input]") {
    const Scene scene;
    Sakura::PillInput field;
    field.update({.icon = "@", .value = "alpha"});
    Host host(field);
    CHECK(host.measureOf(scene.ctx, {300.0f, 500.0f}) == Extent2D<F32>{300.0f, 34.0f});
    CHECK(host.measureOf(scene.ctx, {kInfinite, kInfinite}) == Extent2D<F32>{200.0f, 34.0f});
    auto doubled = scene.ctx;
    doubled.pixelRatio = 2.0f;
    CHECK(host.measureOf(doubled, {kInfinite, kInfinite}) == Extent2D<F32>{400.0f, 68.0f});
    field.update({.icon = "@", .value = "alpha", .width = 150.0f});
    CHECK(host.measureOf(scene.ctx, {300.0f, 500.0f}) == Extent2D<F32>{150.0f, 34.0f});
}

TEST_CASE("Pill inputs place the icon before the text and commit edits on Enter", "[core][sakura][pill-input]") {
    const Scene base;
    auto ctx = base.context();
    ctx.palette = DistinctPalette();
    std::vector<std::string> changes;
    Sakura::PillInput field;
    field.update({.icon = "@", .value = "alpha",
                  .onChange = [&](const std::string& value) { changes.push_back(value); }});
    Host host(field);
    host.place(ctx, kField);

    auto inputs = Find<Sakura::TextInput>(field);
    REQUIRE(inputs.size() == 1);
    auto* icon = ShownText(field, "@");
    REQUIRE(icon != nullptr);
    CHECK(Peek::Frame(*icon).right() < Peek::Frame(*inputs[0]).x);
    CHECK(Peek::Frame(*inputs[0]).right() < kField.right());
    CHECK_FALSE(RingShown(field, ctx));

    const auto center = Peek::Frame(*inputs[0]);
    CHECK(host.click(center.x + 20.0f, center.y + center.height * 0.5f));
    REQUIRE(inputs[0]->focused());
    host.place(ctx, kField);
    CHECK(RingShown(field, ctx));

    Type(host, "!");
    CHECK(changes.empty());
    Tap(host, KeyCode::Enter);
    REQUIRE(changes.size() == 1);
    CHECK(changes[0].size() == 6);
    CHECK(changes[0].find('!') != std::string::npos);
}

TEST_CASE("Pill inputs focus their text from a click on the icon and ignore input while disabled",
          "[core][sakura][pill-input]") {
    const Scene scene;
    std::vector<std::string> changes;
    Sakura::PillInput field;
    const auto configure = [&](bool disabled) {
        field.update({.icon = "@", .value = "alpha", .disabled = disabled,
                      .onChange = [&](const std::string& value) { changes.push_back(value); }});
    };
    configure(false);
    Host host(field);
    host.place(scene.ctx, kField);
    auto* input = Find<Sakura::TextInput>(field).at(0);
    const Extent2D<F32> iconPoint = {kField.x + 16.0f, kField.y + 17.0f};
    CHECK(Peek::CursorAt(field, iconPoint) == SurfaceCursor::TextInput);
    CHECK(host.click(iconPoint.x, iconPoint.y));
    CHECK(input->focused());

    configure(true);
    host.place(scene.ctx, kField);
    CHECK_FALSE(input->focused());
    CHECK_FALSE(Peek::CursorAt(field, iconPoint).has_value());
    CHECK_FALSE(host.click(iconPoint.x, iconPoint.y));
    host.place(scene.ctx, kField);
    CHECK_FALSE(input->focused());
    for (auto* text : Find<Sakura::Text>(field)) {
        CHECK(text->config().opacity < 1.0f);
    }
    CHECK(changes.empty());
}

TEST_CASE("Pill inputs settle under identical updates and repaint without rebuilding when disabled",
          "[core][sakura][pill-input][retention]") {
    Scene scene;
    Sakura::PillInput field;
    Place place(field, kField);
    Frames frames(place, kScreen, scene.context());
    bool disabled = false;
    const auto update = [&] {
        field.update({.icon = "@", .value = "alpha", .hint = "Address", .disabled = disabled,
                      .onChange = [](const std::string&) {}});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
    disabled = true;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    CHECK(frames.restructures == 0);
}
