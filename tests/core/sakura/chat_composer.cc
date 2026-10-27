#include <catch2/catch_test_macros.hpp>

#include "harness.hh"
#include "fonts.hh"
#include "component_harness.hh"
#include "scene.hh"

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/chat_composer.hh>
#include <jetstream/render/sakura/components/label.hh>
#include <jetstream/render/sakura/components/model_picker.hh>
#include <jetstream/render/sakura/components/stack.hh>
#include <jetstream/render/sakura/components/text_grid.hh>
#include <jetstream/render/sakura/components/text_input.hh>
#include <jetstream/render/sakura/typography.hh>
#include <jetstream/render/tools/icons_lucide.hh>

#include "render/sakura/detail/text_metrics.hh"
#include "render/sakura/metrics.hh"

#include <cmath>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

using namespace Jetstream;
using namespace SakuraTest;

namespace {

constexpr const char* kPickerTags = "[core][sakura][model-picker]";
constexpr const char* kPickerRetentionTags = "[core][sakura][model-picker][retention]";
constexpr const char* kComposerTags = "[core][sakura][chat-composer]";
constexpr const char* kComposerRetentionTags = "[core][sakura][chat-composer][retention]";

constexpr Rect kField = {100.0f, 300.0f, 160.0f, 23.0f};
constexpr Extent2D<F32> kFieldCenter = {180.0f, 311.0f};
constexpr Extent2D<F32> kModelRow = {150.0f, 372.0f};
constexpr Extent2D<F32> kSecondRow = {150.0f, 372.0f};
constexpr Extent2D<F32> kFirstRow = {150.0f, 347.0f};
constexpr Extent2D<F32> kPanelTitle = {150.0f, 345.0f};
constexpr Extent2D<F32> kEffortBottom = {45.0f, 425.0f};
constexpr F32 kTrackY = 404.0f;
constexpr F32 kTitleY = 339.0f;
constexpr F32 kListTop = 335.0f;
constexpr F32 kRowHeight = 25.0f;
constexpr F32 kKnobSize = 26.0f;
constexpr F32 kMediumKnob = 137.0f;
constexpr F32 kHighKnob = 227.0f;

constexpr F32 kPad = 7.0f;
constexpr F32 kTextPad = 6.0f;
constexpr F32 kControl = 29.0f;
constexpr F32 kComposerWidth = 400.0f;

std::vector<Sakura::Label::Instance> Texts(Sakura::Component& root, const std::string& str) {
    std::vector<Sakura::Label::Instance> found;
    for (auto* label : Find<Sakura::Label>(root)) {
        for (const auto& instance : label->config().instances) {
            if (instance.visible && instance.str == str) {
                found.push_back(instance);
            }
        }
    }
    return found;
}

bool Shows(Sakura::Component& root, const std::string& str) {
    return !Texts(root, str).empty();
}

bool ShowsAt(Sakura::Component& root, const std::string& str, F32 y) {
    for (const auto& instance : Texts(root, str)) {
        if (instance.rect.y == y) {
            return true;
        }
    }
    return false;
}

std::vector<Rect> Squares(Sakura::Component& root, F32 size) {
    std::vector<Rect> found;
    for (auto* box : Find<Sakura::Box>(root)) {
        for (const auto& instance : box->config().instances) {
            if (instance.visible && instance.rect.width == size && instance.rect.height == size) {
                found.push_back(instance.rect);
            }
        }
    }
    return found;
}

std::vector<const Sakura::Box::Config*> BoxesWithCapacity(Sakura::Component& root, U64 capacity) {
    std::vector<const Sakura::Box::Config*> found;
    for (auto* box : Find<Sakura::Box>(root)) {
        if (box->config().capacity == capacity) {
            found.push_back(&box->config());
        }
    }
    return found;
}

U64 VisibleInstances(const Sakura::Box::Config& config) {
    U64 count = 0;
    for (const auto& instance : config.instances) {
        count += instance.visible ? 1 : 0;
    }
    return count;
}

U64 VoiceBars(Sakura::Component& root) {
    const auto boxes = BoxesWithCapacity(root, 5);
    REQUIRE(boxes.size() == 1);
    return VisibleInstances(*boxes.front());
}

std::vector<U64> Capacities(Sakura::Component& root) {
    std::vector<U64> capacities;
    for (auto* label : Find<Sakura::Label>(root)) {
        capacities.push_back(label->config().capacity);
    }
    for (auto* box : Find<Sakura::Box>(root)) {
        capacities.push_back(box->config().capacity);
    }
    return capacities;
}

F32 TextWidth(const Sakura::Context& ctx, const std::string& str, F32 size = Sakura::Typography::FontSize) {
    Sakura::TextMetrics metrics;
    metrics.setWindow(ctx.render);
    return metrics.measure(Sakura::Typography::BodyFont, str, size);
}

Sakura::Context Doubled(const Sakura::Context& ctx) {
    auto next = ctx;
    next.pixelRatio = 2.0f;
    return next;
}

std::vector<Sakura::ModelPicker::Option> Efforts() {
    return {{.id = "low", .label = "Low"}, {.id = "medium", .label = "Medium"}, {.id = "high", .label = "High"}};
}

struct PickerRig {
    Scene scene;
    Sakura::ModelPicker picker;
    LayerSlot slot;
    LayerHost host{std::vector<Sakura::Layer*>{&slot}};
    Frames frames{host, kScreen, scene.context()};
    Sakura::ModelPicker::Config config;
    std::vector<std::string> selected;
    std::vector<std::string> efforts;

    explicit PickerRig(Rect field = kField) {
        config.models = {{.id = "provider/model-a", .label = "Same label"},
                         {.id = "provider/model-b", .label = "Same label"}};
        config.value = "provider/model-a";
        config.efforts = Efforts();
        config.effort = "medium";
        config.onSelect = [this](const std::string& id) { selected.push_back(id); };
        config.onEffort = [this](const std::string& id) { efforts.push_back(id); };
        slot.hold(picker, field);
    }

    PickerRig(const PickerRig&) = delete;
    PickerRig& operator=(const PickerRig&) = delete;

    std::function<void()> updater() {
        return [this] { picker.update(config); };
    }

    FrameStats step() {
        return frames.run(updater());
    }

    bool settle(U64 limit = 8) {
        return frames.settle(updater(), limit);
    }

    void click(Extent2D<F32> point) {
        Press(host, frames, point.x, point.y);
        Release(host, frames, point.x, point.y);
    }

    bool open() const {
        return host.popups.interactiveAt(kModelRow);
    }

    F32 knob() {
        const auto knobs = Squares(host, kKnobSize * frames.ctx.pixelRatio);
        REQUIRE(knobs.size() == 1);
        return knobs.front().x;
    }
};

struct ComposerPoints {
    Rect grid;
    F32 controlsY = 0.0f;
    Extent2D<F32> send;
    Extent2D<F32> attach;
};

ComposerPoints PointsOf(Sakura::Component& root, const Rect& bounds) {
    const auto grids = Find<Sakura::TextGrid>(root);
    REQUIRE(grids.size() == 1);
    const Rect grid = Peek::Frame(*grids.front());
    const F32 top = grid.bottom() + kTextPad;
    return {
        .grid = grid,
        .controlsY = top,
        .send = {bounds.right() - kPad - 0.5f * kControl, top + 0.5f * kControl},
        .attach = {bounds.x + kPad + 0.5f * kControl, top + 0.5f * kControl},
    };
}

struct ComposerRig {
    Scene scene;
    Sakura::ChatComposer composer;
    Host host{composer};
    Sakura::ChatComposer::Config config;
    std::vector<std::string> submits;
    U64 cancels = 0;
    U64 voices = 0;
    U64 attaches = 0;
    U64 clears = 0;
    Rect bounds;

    ComposerRig() {
        config.id = "composer";
        config.style.placeholder = "Ask";
        config.onSubmit = [this](const std::string& text) { submits.push_back(text); };
        config.onCancel = [this] { ++cancels; };
        config.onAttach = [this] { ++attaches; };
    }

    ComposerRig(const ComposerRig&) = delete;
    ComposerRig& operator=(const ComposerRig&) = delete;

    void place() {
        composer.update(config);
        const auto ctx = scene.context();
        bounds = {0.0f, 0.0f, kComposerWidth, host.measureOf(ctx, {kComposerWidth, kInfinite}).y};
        host.place(ctx, bounds);
    }

    ComposerPoints points() {
        return PointsOf(host, bounds);
    }

    const Sakura::TextGrid* grid() {
        const auto grids = Find<Sakura::TextGrid>(host);
        REQUIRE(grids.size() == 1);
        return grids.front();
    }
};

}  // namespace

TEST_CASE("Model pickers measure their model, effort, and caret before the first layout", kPickerTags) {
    const Scene scene;
    Sakura::ModelPicker picker;
    Sakura::ModelPicker::Config config{
        .models = {{.id = "a", .label = "Alpha"}},
        .value = "a",
        .efforts = Efforts(),
        .effort = "medium",
    };
    picker.update(config);
    Host host(picker);

    const auto first = host.measureOf(scene.ctx, {kInfinite, kInfinite});
    const F32 chrome = 2.0f * 10.0f + 3.0f + 12.0f;
    CHECK(first == Extent2D<F32>{std::ceil(TextWidth(scene.ctx, "Alpha") + 6.0f + TextWidth(scene.ctx, "Medium") + chrome),
                                 23.0f});
    host.place(scene.ctx, {0.0f, 0.0f, first.x, first.y});
    CHECK(host.measureOf(scene.ctx, {kInfinite, kInfinite}) == first);
    CHECK(host.measureOf(scene.ctx, {60.0f, 10.0f}) == Extent2D<F32>{60.0f, 10.0f});

    config.efforts.clear();
    config.effort.clear();
    picker.update(config);
    CHECK(host.measureOf(scene.ctx, {kInfinite, kInfinite}).x == std::ceil(TextWidth(scene.ctx, "Alpha") + chrome));

    config.models.clear();
    picker.update(config);
    CHECK(host.measureOf(scene.ctx, {kInfinite, kInfinite}).x == std::ceil(TextWidth(scene.ctx, "No models") + chrome));

    config.fontSize = 30.0f;
    picker.update(config);
    CHECK(host.measureOf(scene.ctx, {kInfinite, kInfinite}).y == 46.0f);
}

TEST_CASE("Model pickers scale their field and panel with the pixel ratio", kPickerTags) {
    const Scene scene;
    Sakura::ModelPicker picker;
    picker.update({.models = {{.id = "a", .label = "Alpha"}}, .value = "a", .efforts = Efforts(), .effort = "low"});
    Host host(picker);
    const auto single = host.measureOf(scene.ctx, {kInfinite, kInfinite});
    const auto doubled = host.measureOf(Doubled(scene.ctx), {kInfinite, kInfinite});
    CHECK(doubled.y == 2.0f * single.y);
    CHECK(std::abs(doubled.x - 2.0f * single.x) <= 2.0f);

    PickerRig rig({200.0f, 200.0f, 320.0f, 46.0f});
    rig.frames.verify = false;
    rig.frames.ctx.pixelRatio = 2.0f;
    rig.config.rounding = 14.5f;
    REQUIRE(rig.settle());
    rig.click({360.0f, 223.0f});
    REQUIRE(rig.settle());
    REQUIRE(rig.host.popups.interactiveAt({100.0f, 300.0f}));

    CHECK(Squares(rig.host, 52.0f).size() == 1);
    bool panel = false;
    bool field = false;
    for (auto* box : Find<Sakura::Box>(rig.host)) {
        const auto& config = box->config();
        if (config.instances.size() != 1 || !config.instances.front().visible) {
            continue;
        }
        const Rect rect = config.instances.front().rect;
        if (rect.width == 440.0f && rect.height == 197.0f) {
            panel = panel || (config.cornerRadius == 30.0f && config.borderWidth == 2.0f);
        }
        if (rect == Rect{200.0f, 200.0f, 320.0f, 46.0f}) {
            field = field || config.cornerRadius == 29.0f;
        }
    }
    CHECK(panel);
    CHECK(field);
}

TEST_CASE("Model pickers open an effort panel and select models by identity", kPickerTags) {
    PickerRig rig;
    rig.frames.verify = false;
    REQUIRE(rig.settle());
    CHECK_FALSE(rig.open());

    rig.click(kFieldCenter);
    REQUIRE(rig.settle());
    REQUIRE(rig.open());
    CHECK(rig.host.popups.interactiveAt(kEffortBottom));
    CHECK(ShowsAt(rig.host, "Medium", kTitleY));
    CHECK(ShowsAt(rig.host, "Medium", kField.y));
    CHECK(Shows(rig.host, ICON_LC_CHEVRON_RIGHT));

    rig.click(kModelRow);
    REQUIRE(rig.settle());
    CHECK_FALSE(rig.host.popups.interactiveAt(kEffortBottom));
    CHECK(ShowsAt(rig.host, "Same label", kListTop));
    CHECK(ShowsAt(rig.host, "Same label", kListTop + kRowHeight));
    REQUIRE(Texts(rig.host, ICON_LC_CHECK).size() == 1);
    CHECK(Texts(rig.host, ICON_LC_CHECK).front().rect.y == kListTop);

    rig.click(kFirstRow);
    REQUIRE(rig.settle());
    CHECK(rig.selected.empty());
    CHECK(rig.host.popups.interactiveAt(kEffortBottom));

    rig.click(kModelRow);
    REQUIRE(rig.settle());
    rig.click(kSecondRow);
    REQUIRE(rig.settle());
    CHECK(rig.selected == std::vector<std::string>{"provider/model-b"});
    CHECK(rig.open());
    CHECK_FALSE(rig.host.popups.interactiveAt(kEffortBottom));

    rig.config.value = "provider/model-b";
    REQUIRE(rig.settle());
    CHECK(rig.open());
    CHECK(rig.host.popups.interactiveAt(kEffortBottom));

    rig.click(kFieldCenter);
    REQUIRE(rig.settle());
    CHECK_FALSE(rig.open());
}

TEST_CASE("Model pickers without efforts list models and close once the selection arrives", kPickerTags) {
    PickerRig rig;
    rig.frames.verify = false;
    rig.config.efforts.clear();
    rig.config.effort.clear();
    REQUIRE(rig.settle());

    rig.click(kFieldCenter);
    REQUIRE(rig.settle());
    REQUIRE(rig.open());
    CHECK(ShowsAt(rig.host, "Same label", kListTop + kRowHeight));
    CHECK(Squares(rig.host, kKnobSize).empty());

    rig.click(kSecondRow);
    REQUIRE(rig.settle());
    CHECK(rig.selected == std::vector<std::string>{"provider/model-b"});
    CHECK(rig.open());

    rig.config.value = "provider/model-b";
    REQUIRE(rig.settle());
    CHECK_FALSE(rig.open());

    rig.click(kFieldCenter);
    REQUIRE(rig.settle());
    rig.click(kSecondRow);
    REQUIRE(rig.settle());
    CHECK_FALSE(rig.open());
    CHECK(rig.selected.size() == 1);
}

TEST_CASE("Model pickers close and fade when disabled and stay shut after enabling", kPickerTags) {
    PickerRig rig;
    rig.frames.verify = false;
    REQUIRE(rig.settle());
    rig.click(kFieldCenter);
    REQUIRE(rig.settle());
    REQUIRE(rig.open());

    rig.config.disabled = true;
    REQUIRE(rig.settle());
    CHECK_FALSE(rig.open());
    const auto labels = Texts(rig.host, "Same label");
    REQUIRE(labels.size() == 1);
    CHECK(labels.front().color.a == Sakura::kDisabledAlpha);
    CHECK_FALSE(Peek::CursorAt(rig.picker, kFieldCenter).has_value());

    rig.click(kFieldCenter);
    REQUIRE(rig.settle());
    CHECK_FALSE(rig.open());

    rig.config.disabled = false;
    REQUIRE(rig.settle());
    CHECK_FALSE(rig.open());
    CHECK(Peek::CursorAt(rig.picker, kFieldCenter) == SurfaceCursor::Hand);
    rig.click(kSecondRow);
    REQUIRE(rig.settle());
    CHECK(rig.selected.empty());
}

TEST_CASE("Model pickers close when hidden and stay shut when shown again", kPickerTags) {
    PickerRig rig;
    rig.frames.verify = false;
    REQUIRE(rig.settle());
    rig.click(kFieldCenter);
    REQUIRE(rig.settle());
    REQUIRE(rig.open());

    rig.picker.setVisible(false);
    REQUIRE(rig.settle());
    CHECK_FALSE(rig.open());

    rig.picker.setVisible(true);
    REQUIRE(rig.settle());
    CHECK_FALSE(rig.open());
}

TEST_CASE("Model picker sliders pick the nearest stop and keep the committed effort locally", kPickerTags) {
    PickerRig rig;
    rig.frames.verify = false;
    REQUIRE(rig.settle());
    rig.click(kFieldCenter);
    REQUIRE(rig.settle());
    REQUIRE(rig.open());
    CHECK(Sakura::Access::cursor(rig.host, {150.0f, kTrackY}) == SurfaceCursor::Hand);
    CHECK(Sakura::Access::cursor(rig.host, kModelRow) == SurfaceCursor::Hand);
    CHECK(Sakura::Access::cursor(rig.host, kPanelTitle) == SurfaceCursor::Default);

    Press(rig.host, rig.frames, 238.0f, kTrackY);
    REQUIRE(rig.settle());
    CHECK(ShowsAt(rig.host, "High", kField.y));
    Hover(rig.host, rig.frames, 70.0f, kTrackY);
    REQUIRE(rig.settle());
    CHECK(ShowsAt(rig.host, "Low", kField.y));
    CHECK(rig.efforts.empty());
    Release(rig.host, rig.frames, 70.0f, kTrackY);
    REQUIRE(rig.settle());
    CHECK(rig.efforts == std::vector<std::string>{"low"});
    CHECK(ShowsAt(rig.host, "Low", kField.y));
    CHECK(ShowsAt(rig.host, "Low", kTitleY));

    REQUIRE(rig.settle());
    CHECK(ShowsAt(rig.host, "Low", kField.y));

    rig.config.effort = "high";
    REQUIRE(rig.settle());
    CHECK(ShowsAt(rig.host, "High", kField.y));
    rig.click({238.0f, kTrackY});
    REQUIRE(rig.settle());
    CHECK(rig.efforts.size() == 1);
}

TEST_CASE("Model picker slider drags cancel without committing an effort", kPickerTags) {
    PickerRig rig;
    rig.frames.verify = false;
    REQUIRE(rig.settle());
    rig.click(kFieldCenter);
    REQUIRE(rig.settle());

    Press(rig.host, rig.frames, 238.0f, kTrackY);
    REQUIRE(rig.settle());
    REQUIRE(ShowsAt(rig.host, "High", kField.y));
    rig.host.send(Mouse(MouseEventType::Cancel, 238.0f, kTrackY));
    REQUIRE(rig.settle());
    CHECK(ShowsAt(rig.host, "Medium", kField.y));
    Release(rig.host, rig.frames, 238.0f, kTrackY);
    REQUIRE(rig.settle());
    CHECK(rig.efforts.empty());
    CHECK(rig.open());
}

TEST_CASE("Model picker lists fit the space beside the field and scroll by rows", kPickerTags) {
    PickerRig rig;
    rig.frames.verify = false;
    rig.config.models.clear();
    for (U64 i = 0; i < 30; ++i) {
        const std::string index = (i < 10 ? "0" : "") + std::to_string(i);
        rig.config.models.push_back({.id = "model-" + index, .label = "Model " + index, .detail = "Detail"});
    }
    rig.config.value = "model-00";
    rig.config.efforts.clear();
    rig.config.effort.clear();
    REQUIRE(rig.settle());
    rig.click(kFieldCenter);
    REQUIRE(rig.settle());

    const F32 top = 13.0f;
    CHECK(rig.host.popups.interactiveAt({150.0f, 100.0f}));
    CHECK(ShowsAt(rig.host, "Model 00", top));
    CHECK(ShowsAt(rig.host, "Model 10", top + 10.0f * kRowHeight));
    CHECK_FALSE(Shows(rig.host, "Model 11"));
    CHECK(ShowsAt(rig.host, "Detail", top));
    const auto capacities = Capacities(rig.host);

    rig.host.send(Wheel(150.0f, 100.0f, -1.0f));
    REQUIRE(rig.settle());
    CHECK_FALSE(ShowsAt(rig.host, "Model 00", top));
    CHECK(ShowsAt(rig.host, "Model 00", kField.y));
    CHECK(ShowsAt(rig.host, "Model 01", top));
    CHECK(ShowsAt(rig.host, "Model 11", top + 10.0f * kRowHeight));
    CHECK(Texts(rig.host, ICON_LC_CHECK).empty());

    for (U64 i = 0; i < 40; ++i) {
        rig.host.send(Wheel(150.0f, 100.0f, -1.0f));
    }
    REQUIRE(rig.settle());
    CHECK(ShowsAt(rig.host, "Model 29", top + 10.0f * kRowHeight));
    CHECK(Capacities(rig.host) == capacities);

    rig.host.send(Wheel(150.0f, 100.0f, 1.0f));
    REQUIRE(rig.settle());
    CHECK(ShowsAt(rig.host, "Model 28", top + 10.0f * kRowHeight));

    auto trackpad = Wheel(150.0f, 100.0f, kRowHeight / 10.0f * 0.5f);
    trackpad.phase = ScrollPhase::Changed;
    rig.host.send(trackpad);
    REQUIRE(rig.settle());
    CHECK(ShowsAt(rig.host, "Model 28", top + 10.0f * kRowHeight));
    rig.host.send(trackpad);
    REQUIRE(rig.settle());
    CHECK(ShowsAt(rig.host, "Model 27", top + 10.0f * kRowHeight));

    auto down = trackpad;
    down.scroll.y = -trackpad.scroll.y;
    rig.host.send(down);
    REQUIRE(rig.settle());
    CHECK(ShowsAt(rig.host, "Model 27", top + 10.0f * kRowHeight));
    rig.click(kFieldCenter);
    REQUIRE(rig.settle());
    REQUIRE_FALSE(rig.host.popups.interactiveAt({150.0f, 100.0f}));
    rig.click(kFieldCenter);
    REQUIRE(rig.settle());
    REQUIRE(rig.host.popups.interactiveAt({150.0f, 100.0f}));
    rig.host.send(down);
    REQUIRE(rig.settle());
    CHECK(ShowsAt(rig.host, "Model 00", top));
}

TEST_CASE("Model picker lists leave their endpoints on small inward trackpad deltas", kPickerTags) {
    PickerRig rig;
    rig.frames.verify = false;
    rig.config.models.clear();
    for (U64 i = 0; i < 30; ++i) {
        const std::string index = (i < 10 ? "0" : "") + std::to_string(i);
        rig.config.models.push_back({.id = "model-" + index, .label = "Model " + index, .detail = "Detail"});
    }
    rig.config.value = "model-00";
    rig.config.efforts.clear();
    rig.config.effort.clear();
    REQUIRE(rig.settle());
    rig.click(kFieldCenter);
    REQUIRE(rig.settle());

    const F32 top = 13.0f;
    REQUIRE(ShowsAt(rig.host, "Model 00", top));

    auto inward = Wheel(150.0f, 100.0f, -kRowHeight / 10.0f * 0.5f);
    inward.phase = ScrollPhase::Changed;
    rig.host.send(inward);
    REQUIRE(rig.settle());
    CHECK(ShowsAt(rig.host, "Model 00", top));
    rig.host.send(inward);
    REQUIRE(rig.settle());
    CHECK(ShowsAt(rig.host, "Model 01", top));

    for (U64 i = 0; i < 40; ++i) {
        rig.host.send(Wheel(150.0f, 100.0f, -1.0f));
    }
    REQUIRE(rig.settle());
    REQUIRE(ShowsAt(rig.host, "Model 29", top + 10.0f * kRowHeight));

    auto up = inward;
    up.scroll.y = -inward.scroll.y;
    rig.host.send(up);
    REQUIRE(rig.settle());
    CHECK(ShowsAt(rig.host, "Model 29", top + 10.0f * kRowHeight));
    rig.host.send(up);
    REQUIRE(rig.settle());
    CHECK(ShowsAt(rig.host, "Model 28", top + 10.0f * kRowHeight));
}

TEST_CASE("Closed model pickers settle and repaint once per hover transition and visual change",
          kPickerRetentionTags) {
    Scene scene;
    Sakura::ModelPicker picker;
    Place place(picker, kField);
    Frames frames(place, kScreen, scene.context());
    std::string value = "a";
    std::string effort = "medium";
    std::string textColorKey = "text_primary";
    bool disabled = false;
    const auto update = [&] {
        picker.update({
            .models = {{.id = "a", .label = "Alpha", .detail = "Fast"}, {.id = "b", .label = "Beta"}},
            .value = value,
            .efforts = Efforts(),
            .effort = effort,
            .disabled = disabled,
            .textColorKey = textColorKey,
        });
    };
    const auto send = [&](const MouseEvent& event) {
        return place.send(event);
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
    CHECK(QuietMoves(frames, update, {{600.0f, 100.0f}, {610.0f, 100.0f}}, send));
    CheckHoverRepaintsOnce(frames, place, update, kFieldCenter, {600.0f, 100.0f});
    CheckLeaveRepaintsOnce(frames, place, update, kFieldCenter);
    place.send(Mouse(MouseEventType::Leave, kFieldCenter.x, kFieldCenter.y));
    REQUIRE(frames.settle(update));

    value = "b";
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    effort = "high";
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    textColorKey = "text_secondary";
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    disabled = true;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    CHECK(QuietMoves(frames, update, {kFieldCenter, {kFieldCenter.x + 1.0f, kFieldCenter.y}}, send));
    CHECK(frames.restructures == 0);
}

TEST_CASE("Model picker panels restructure once when first opened and reopen without rebuilding",
          kPickerRetentionTags) {
    PickerRig rig;
    const auto update = rig.updater();
    REQUIRE(rig.settle());
    CHECK(Retains(rig.frames, update));
    const U64 restructures = rig.frames.restructures;
    const U64 rebuilds = rig.frames.rebuilds;

    rig.click(kFieldCenter);
    REQUIRE(rig.settle());
    REQUIRE(rig.open());
    CHECK(rig.frames.restructures == restructures + 1);
    CHECK(Retains(rig.frames, update));
    CheckHoverRepaintsOnce(rig.frames, rig.host, update, kModelRow, kPanelTitle);

    rig.click(kModelRow);
    CHECK(RepaintOnly(rig.step()));
    REQUIRE(rig.settle());
    CHECK(Retains(rig.frames, update));
    CheckHoverRepaintsOnce(rig.frames, rig.host, update, kSecondRow, {150.0f, 388.0f});

    rig.click(kFieldCenter);
    CHECK(RepaintOnly(rig.step()));
    REQUIRE(rig.settle());
    CHECK_FALSE(rig.open());
    CHECK(Retains(rig.frames, update));

    rig.click(kFieldCenter);
    CHECK(Kept(rig.step()));
    REQUIRE(rig.settle());
    REQUIRE(rig.open());
    CHECK(Retains(rig.frames, update));
    CHECK(rig.frames.restructures == restructures + 1);
    CHECK(rig.frames.rebuilds == rebuilds);

    rig.config.effort = "high";
    CHECK(RepaintOnly(rig.step()));
    REQUIRE(rig.settle(64));
    rig.config.popupColorKey = "popup_bg";
    CHECK(RepaintOnly(rig.step()));
    CHECK(Retains(rig.frames, update));
    CHECK(rig.frames.restructures == restructures + 1);
}

TEST_CASE("Model picker list scrolling repaints without changing capacity", kPickerRetentionTags) {
    PickerRig rig;
    for (U64 i = 0; i < 30; ++i) {
        rig.config.models.push_back({.id = "extra-" + std::to_string(i), .label = "Extra " + std::to_string(i)});
    }
    const auto update = rig.updater();
    REQUIRE(rig.settle());
    rig.click(kFieldCenter);
    REQUIRE(rig.settle());
    const auto capacities = Capacities(rig.host);
    rig.click(kModelRow);
    REQUIRE(rig.settle());
    CHECK(Capacities(rig.host) == capacities);
    const U64 restructures = rig.frames.restructures;

    rig.host.send(Wheel(150.0f, 100.0f, -1.0f));
    CHECK(RepaintOnly(rig.step()));
    CHECK(Retains(rig.frames, update));
    CHECK(Capacities(rig.host) == capacities);
    CHECK(rig.frames.restructures == restructures);
}

TEST_CASE("Model picker thumbs ease to the selected stop on the frame clock and hold while it is frozen",
          kPickerRetentionTags) {
    PickerRig rig;
    const auto update = rig.updater();
    REQUIRE(rig.settle());
    rig.click(kFieldCenter);
    REQUIRE(rig.settle());
    CHECK(rig.knob() == kMediumKnob);

    rig.config.effort = "high";
    CHECK(RepaintOnly(rig.step()));
    CHECK(rig.knob() == kMediumKnob);
    CHECK(Retains(rig.frames, update));

    rig.frames.timeStep = 1.0 / 60.0;
    rig.frames.ctx.deltaTime = 1.0f / 60.0f;
    CHECK(RepaintOnly(rig.step()));
    const F32 moving = rig.knob();
    CHECK(moving > kMediumKnob);
    CHECK(moving < kHighKnob);
    CHECK(RepaintOnly(rig.step()));
    CHECK(rig.knob() > moving);
    REQUIRE(rig.settle(64));
    CHECK(rig.knob() == kHighKnob);
    CHECK(Retains(rig.frames, update));
}

TEST_CASE("Chat composers measure their input, controls, and margins before the first layout", kComposerTags) {
    ComposerRig rig;
    const auto ctx = rig.scene.context();
    rig.composer.update(rig.config);
    const auto first = rig.host.measureOf(ctx, {kComposerWidth, kInfinite});
    CHECK(first.x == kComposerWidth);
    rig.place();
    CHECK(rig.host.measureOf(ctx, {kComposerWidth, kInfinite}) == first);
    const auto points = rig.points();
    CHECK(points.grid.x == kPad);
    CHECK(points.grid.y == kPad + kTextPad);
    CHECK(points.grid.width == kComposerWidth - 2.0f * kPad);
    CHECK(points.controlsY + kControl <= first.y - kPad);
    CHECK(rig.host.measureOf(ctx, {kInfinite, kInfinite}).x == Sakura::kNaturalControlWidth);

    rig.config.style.compact = true;
    rig.composer.update(rig.config);
    const auto compact = rig.host.measureOf(ctx, {kComposerWidth, kInfinite});
    CHECK(first.y - compact.y == kControl);
    rig.place();
    CHECK(rig.points().grid.width == kComposerWidth - 2.0f * kPad - kControl - kTextPad);

    rig.config.style.compact = false;
    rig.config.style.margin = {10.0f, 4.0f, 10.0f, 6.0f};
    rig.composer.update(rig.config);
    CHECK(rig.host.measureOf(ctx, {kComposerWidth, kInfinite}).y == first.y + 10.0f);
    rig.place();
    CHECK(rig.points().grid.x == 10.0f + kPad);
    CHECK(rig.points().grid.width == kComposerWidth - 20.0f - 2.0f * kPad);

    rig.config.style.margin = {};
    rig.config.style.minLines = 3.0f;
    rig.composer.update(rig.config);
    CHECK(rig.host.measureOf(ctx, {kComposerWidth, kInfinite}).y > first.y + 30.0f);

    rig.config.style.minLines = 1.0f;
    rig.config.style.icon = ICON_LC_CHECK;
    rig.place();
    CHECK(rig.points().grid.x == kPad + 1.8f * Sakura::Typography::FontSize);
    CHECK(Shows(rig.host, ICON_LC_CHECK));
}

TEST_CASE("Chat composers scale padding, margins, and rounding with the pixel ratio", kComposerTags) {
    Scene scene;
    Sakura::ChatComposer composer;
    composer.update({.id = "composer", .style = {.margin = {10.0f, 10.0f, 10.0f, 10.0f}}});
    Host host(composer);
    const auto ctx = scene.context();
    const auto doubled = Doubled(ctx);
    const auto single = host.measureOf(ctx, {kComposerWidth, kInfinite});
    const auto twice = host.measureOf(doubled, {2.0f * kComposerWidth, kInfinite});
    CHECK(std::abs(twice.y - 2.0f * single.y) <= 2.0f);

    host.place(doubled, {0.0f, 0.0f, 2.0f * kComposerWidth, twice.y});
    const auto boxes = Find<Sakura::Box>(host);
    REQUIRE_FALSE(boxes.empty());
    const auto& background = boxes.front()->config();
    REQUIRE(background.instances.size() == 1);
    CHECK(background.instances.front().rect.x == 20.0f);
    CHECK(background.instances.front().rect.y == 20.0f);
    CHECK(background.cornerRadius == 2.0f * (kPad + 0.5f * kControl));
    CHECK(background.borderWidth == 2.0f);
    const auto grids = Find<Sakura::TextGrid>(host);
    REQUIRE(grids.size() == 1);
    CHECK(Peek::Frame(*grids.front()).x == 20.0f + 2.0f * kPad);
    CHECK(Peek::Frame(*grids.front()).y == 20.0f + 2.0f * (kPad + kTextPad));

    composer.update({.id = "composer", .style = {.rounding = 4.0f}});
    host.place(doubled, {0.0f, 0.0f, 2.0f * kComposerWidth, twice.y});
    CHECK(Find<Sakura::Box>(host).front()->config().cornerRadius == 8.0f);
}

TEST_CASE("Chat composers cancel while busy and route the send button by state", kComposerTags) {
    ComposerRig rig;
    rig.config.busy = true;
    rig.config.onVoice = [&rig] { ++rig.voices; };
    rig.config.focusRequest = 1;
    rig.place();
    REQUIRE(FocusOwner(rig.host) == rig.grid());
    CHECK(Shows(rig.host, ICON_LC_SQUARE));
    CHECK(VoiceBars(rig.host) == 0);
    CHECK(Shows(rig.host, "Ask"));
    rig.host.click(rig.points().send.x, rig.points().send.y);
    CHECK(rig.cancels == 1);

    rig.config.busy = false;
    rig.place();
    CHECK_FALSE(Shows(rig.host, ICON_LC_SQUARE));
    CHECK_FALSE(Shows(rig.host, ICON_LC_ARROW_UP));
    CHECK(VoiceBars(rig.host) == 5);
    rig.host.click(rig.points().send.x, rig.points().send.y);
    CHECK(rig.voices == 1);
    CHECK(rig.cancels == 1);

    Type(rig.host, "hello");
    rig.place();
    CHECK(Shows(rig.host, ICON_LC_ARROW_UP));
    CHECK(VoiceBars(rig.host) == 0);
    CHECK_FALSE(Shows(rig.host, "Ask"));
    rig.host.click(rig.points().send.x, rig.points().send.y);
    CHECK(rig.submits == std::vector<std::string>{"hello"});
    CHECK(rig.voices == 1);

    rig.config.clearRequest = 1;
    rig.place();
    CHECK(Shows(rig.host, "Ask"));
    CHECK(VoiceBars(rig.host) == 5);

    rig.config.onVoice = {};
    rig.place();
    const auto arrows = Texts(rig.host, ICON_LC_ARROW_UP);
    REQUIRE(arrows.size() == 1);
    CHECK(arrows.front().color.a == Sakura::kDisabledAlpha);
    CHECK(Sakura::Access::cursor(rig.host, rig.points().send) != SurfaceCursor::Hand);
    rig.host.click(rig.points().send.x, rig.points().send.y);
    CHECK(rig.submits.size() == 1);
    CHECK(rig.cancels == 1);
}

TEST_CASE("Chat composers submit on Enter and keep their draft until cleared", kComposerTags) {
    ComposerRig rig;
    rig.config.focusRequest = 1;
    rig.place();
    Type(rig.host, "draft");
    rig.place();
    Tap(rig.host, KeyCode::Enter);
    rig.place();
    CHECK(rig.submits == std::vector<std::string>{"draft"});
    CHECK_FALSE(Shows(rig.host, "Ask"));

    rig.config.busy = true;
    rig.place();
    Tap(rig.host, KeyCode::Enter);
    rig.place();
    CHECK(rig.submits.size() == 1);

    rig.config.busy = false;
    rig.config.clearRequest = 1;
    rig.place();
    CHECK(Shows(rig.host, "Ask"));
    Type(rig.host, "   ");
    rig.place();
    Tap(rig.host, KeyCode::Enter);
    rig.place();
    CHECK(rig.submits.size() == 1);

    rig.place();
    rig.config.clearRequest = 1;
    rig.place();
    CHECK_FALSE(Shows(rig.host, "Ask"));
    rig.config.clearRequest = 2;
    rig.place();
    CHECK(Shows(rig.host, "Ask"));
}

TEST_CASE("Chat composer focus requests focus the input each time the counter changes", kComposerTags) {
    ComposerRig rig;
    rig.place();
    CHECK(FocusOwner(rig.host) == nullptr);

    rig.config.focusRequest = 1;
    rig.place();
    CHECK(FocusOwner(rig.host) == rig.grid());

    ClickFocus(rig.host, std::nullopt);
    rig.place();
    CHECK(FocusOwner(rig.host) == nullptr);
    rig.place();
    CHECK(FocusOwner(rig.host) == nullptr);

    rig.config.focusRequest = 2;
    rig.place();
    CHECK(FocusOwner(rig.host) == rig.grid());
}

TEST_CASE("Chat composer focus requests take focus from another text editor", kComposerTags) {
    const Scene scene;
    Sakura::TextInput other;
    other.update({.value = "note"});
    Sakura::ChatComposer composer;
    Sakura::Stack stack;
    stack.update({.items = {{.child = &other}, {.child = &composer}}});
    Host host(stack);
    Sakura::ChatComposer::Config config{.id = "composer"};
    const auto place = [&] {
        composer.update(config);
        host.place(scene.ctx, {0.0f, 0.0f, kComposerWidth, 300.0f});
    };
    place();

    const auto others = Find<Sakura::TextGrid>(other);
    const auto inputs = Find<Sakura::TextGrid>(composer);
    REQUIRE(others.size() == 1);
    REQUIRE(inputs.size() == 1);
    REQUIRE(Sakura::Access::focus(*others.front()));
    REQUIRE(FocusOwner(host) == others.front());

    config.focusRequest = 1;
    place();
    CHECK(FocusOwner(host) == inputs.front());
}

TEST_CASE("Chat composer controls place attach, clear, the model picker, and the usage ring", kComposerTags) {
    ComposerRig rig;
    rig.config.onClear = [&rig] { ++rig.clears; };
    rig.config.selector = Sakura::ChatComposer::Selector{.options = {{.id = "m", .label = "Model"}}, .value = "m"};
    rig.config.usage = Sakura::ChatComposer::Usage{.value = 0.5f, .title = "Context", .details = {"1k of 2k"}};
    rig.place();
    const auto points = rig.points();
    const auto ctx = rig.scene.context();

    const auto pickers = Find<Sakura::ModelPicker>(rig.host);
    REQUIRE(pickers.size() == 1);
    const Rect picker = Peek::Frame(*pickers.front());
    const F32 sendX = kComposerWidth - kPad - kControl;
    CHECK(picker.right() == sendX - kTextPad);
    CHECK(picker.y == points.controlsY);
    CHECK(picker.height == kControl);
    CHECK(picker.width == std::ceil(TextWidth(ctx, "Model") + 2.0f * 10.0f + 3.0f + 12.0f));

    const auto rings = BoxesWithCapacity(rig.host, 48);
    REQUIRE(rings.size() == 1);
    const auto& ring = *rings.front();
    REQUIRE(ring.instances.size() == 48);
    U64 filled = 0;
    for (const auto& dot : ring.instances) {
        filled += dot.visible && dot.backgroundColor == ctx.color("text_primary") ? 1 : 0;
    }
    CHECK(filled == 24);
    REQUIRE(ring.clip.has_value());
    CHECK(*ring.clip == Rect{picker.x - kControl, points.controlsY, kControl, kControl});

    rig.host.click(points.attach.x, points.attach.y);
    CHECK(rig.attaches == 1);
    const F32 clearWidth = std::ceil(TextWidth(ctx, "Clear") + kControl);
    const auto clear = Texts(rig.host, "Clear");
    REQUIRE(clear.size() == 1);
    CHECK(clear.front().rect == Rect{kPad + kControl, points.controlsY, clearWidth, kControl});
    rig.host.click(kPad + kControl + 0.5f * clearWidth, points.send.y);
    CHECK(rig.clears == 1);
    CHECK(rig.cancels == 0);
    CHECK(rig.submits.empty());

    rig.config.style.clearButton = false;
    rig.place();
    CHECK_FALSE(Shows(rig.host, "Clear"));
    rig.config.style.clearButton = true;
    rig.config.onClear = {};
    rig.place();
    CHECK_FALSE(Shows(rig.host, "Clear"));

    rig.config.style.compact = true;
    rig.place();
    CHECK(Find<Sakura::ModelPicker>(rig.host).empty());
    CHECK_FALSE(Shows(rig.host, ICON_LC_PLUS));
    CHECK(VisibleInstances(*BoxesWithCapacity(rig.host, 48).front()) == 0);
}

TEST_CASE("Chat composer usage rings show their tooltip only while hovered", kComposerRetentionTags) {
    Scene scene;
    Sakura::ChatComposer composer;
    LayerSlot slot;
    LayerHost host({&slot});
    const Rect bounds = {0.0f, 300.0f, kComposerWidth, 120.0f};
    slot.hold(composer, bounds);
    Frames frames(host, kScreen, scene.context());
    const auto update = [&] {
        composer.update({
            .id = "composer",
            .usage = Sakura::ChatComposer::Usage{.value = 0.25f, .title = "Context",
                                                 .details = {"1k of 4k", "25%", "hidden"}},
        });
    };
    REQUIRE(frames.settle(update));
    const auto points = PointsOf(host, bounds);
    const F32 ringX = kComposerWidth - kPad - kControl - kTextPad - 0.5f * kControl;
    const Extent2D<F32> ring = {ringX, points.send.y};
    const U64 restructures = frames.restructures;
    CHECK_FALSE(Shows(host, "Context"));

    Hover(host, frames, ring.x, ring.y);
    REQUIRE(frames.settle(update));
    CHECK(frames.restructures == restructures + 1);
    REQUIRE(Shows(host, "Context"));
    CHECK(Shows(host, "1k of 4k"));
    CHECK(Shows(host, "25%"));
    CHECK_FALSE(Shows(host, "hidden"));
    CHECK(Texts(host, "Context").front().rect.y < points.controlsY - kTextPad);
    CHECK_FALSE(host.popups.interactiveAt({ring.x, Texts(host, "Context").front().rect.y + 1.0f}));
    Hover(host, frames, ring.x + 1.0f, ring.y);
    CHECK(frames.run(update).clean());

    Hover(host, frames, 100.0f, points.send.y);
    CHECK(RepaintOnly(frames.run(update)));
    CHECK_FALSE(Shows(host, "Context"));
    CHECK(Retains(frames, update));

    Hover(host, frames, ring.x, ring.y);
    CHECK(Kept(frames.run(update)));
    REQUIRE(frames.settle(update));
    CHECK(Shows(host, "Context"));
    host.send(Mouse(MouseEventType::Leave, ring.x, ring.y));
    CHECK(RepaintOnly(frames.run(update)));
    CHECK_FALSE(Shows(host, "Context"));
    CHECK(frames.restructures == restructures + 1);
}

TEST_CASE("Chat composers settle and repaint only for visual changes", kComposerRetentionTags) {
    Scene scene;
    Sakura::ChatComposer composer;
    Place place(composer, {10.0f, 10.0f, kComposerWidth, 100.0f});
    Frames frames(place, kScreen, scene.context());
    std::string placeholder = "Ask";
    std::string icon;
    std::string background = "card";
    bool busy = false;
    std::string model = "m";
    F32 usage = 0.25f;
    const auto update = [&] {
        composer.update({
            .id = "composer",
            .style = {.placeholder = placeholder, .icon = icon, .backgroundColorKey = background},
            .busy = busy,
            .selector = Sakura::ChatComposer::Selector{
                .options = {{.id = "m", .label = "Model"}, {.id = "n", .label = "Next"}}, .value = model},
            .usage = Sakura::ChatComposer::Usage{.value = usage, .title = "Context"},
            .onSubmit = [](const std::string&) {},
            .onCancel = [] {},
            .onAttach = [] {},
        });
    };
    const auto send = [&](const MouseEvent& event) {
        return place.send(event);
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
    CHECK(QuietMoves(frames, update, {{600.0f, 400.0f}, {610.0f, 400.0f}, {200.0f, 30.0f}, {210.0f, 30.0f}}, send));

    placeholder = "Ask more";
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    busy = true;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    model = "n";
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    usage = 0.75f;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    background = "agent_composer";
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    icon = ICON_LC_CHECK;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    CHECK(frames.restructures == 0);
}

TEST_CASE("Chat composer buttons repaint once per hover transition", kComposerRetentionTags) {
    Scene scene;
    Sakura::ChatComposer composer;
    const Rect bounds = {10.0f, 10.0f, kComposerWidth, 100.0f};
    Place place(composer, bounds);
    Frames frames(place, kScreen, scene.context());
    const auto update = [&] {
        composer.update({.id = "composer", .busy = true, .onCancel = [] {}, .onAttach = [] {}});
    };
    REQUIRE(frames.settle(update));
    const auto points = PointsOf(place, bounds);
    CheckHoverRepaintsOnce(frames, place, update, points.send, {600.0f, 400.0f});
    CheckLeaveRepaintsOnce(frames, place, update, points.send);
    place.send(Mouse(MouseEventType::Leave, points.send.x, points.send.y));
    CheckHoverRepaintsOnce(frames, place, update, points.attach, {600.0f, 400.0f});
    CheckLeaveRepaintsOnce(frames, place, update, points.attach);
    CHECK(Sakura::Access::cursor(place, points.send) == SurfaceCursor::Hand);
    CHECK(Sakura::Access::cursor(place, points.attach) == SurfaceCursor::Hand);
}

TEST_CASE("Chat composer buttons drop a cancelled press and ignore releases outside", kComposerTags) {
    ComposerRig rig;
    rig.config.busy = true;
    rig.place();
    const auto send = rig.points().send;

    rig.host.send(Mouse(MouseEventType::Click, send.x, send.y));
    rig.host.send(Mouse(MouseEventType::Cancel, send.x, send.y));
    rig.host.send(Mouse(MouseEventType::Release, send.x, send.y));
    CHECK(rig.cancels == 0);

    rig.host.send(Mouse(MouseEventType::Click, send.x, send.y));
    rig.host.send(Mouse(MouseEventType::Release, 600.0f, 400.0f));
    CHECK(rig.cancels == 0);

    rig.host.click(send.x, send.y);
    CHECK(rig.cancels == 1);
}

TEST_CASE("Typing into a chat composer switches the send button from voice to submit without another update",
          kComposerRetentionTags) {
    Scene scene;
    Sakura::ChatComposer composer;
    Place place(composer, {10.0f, 10.0f, kComposerWidth, 100.0f});
    Frames frames(place, kScreen, scene.context());
    frames.verify = false;
    frames.ctx.windowFocused = false;
    const auto update = [&] {
        composer.update({.id = "composer", .style = {.placeholder = "Ask"}, .onVoice = [] {}, .focusRequest = 1});
    };
    REQUIRE(frames.settle(update));
    REQUIRE(FocusOwner(place) != nullptr);
    CHECK(VoiceBars(place) == 5);
    CHECK(Shows(place, "Ask"));

    Type(place, "a");
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(VoiceBars(place) == 0);
    CHECK(Shows(place, ICON_LC_ARROW_UP));
    CHECK_FALSE(Shows(place, "Ask"));
    CHECK(Retains(frames, update));
}

TEST_CASE("Chat composer model pickers mount once and park when the selector goes away", kComposerRetentionTags) {
    Scene scene;
    Sakura::ChatComposer composer;
    Place place(composer, {10.0f, 10.0f, kComposerWidth, 100.0f});
    Frames frames(place, kScreen, scene.context());
    std::optional<Sakura::ChatComposer::Selector> selector;
    const auto update = [&] {
        composer.update({.id = "composer", .selector = selector});
    };
    REQUIRE(frames.settle(update));
    const U64 restructures = frames.restructures;
    CHECK(Find<Sakura::ModelPicker>(place).empty());

    selector = Sakura::ChatComposer::Selector{.options = {{.id = "m", .label = "Model"}}, .value = "m"};
    REQUIRE(frames.settle(update));
    CHECK(frames.restructures == restructures + 1);
    CHECK(Find<Sakura::ModelPicker>(place).size() == 1);
    CHECK(Retains(frames, update));

    selector.reset();
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Find<Sakura::ModelPicker>(place).empty());
    CHECK(Retains(frames, update));

    selector = Sakura::ChatComposer::Selector{.options = {{.id = "m", .label = "Model"}}, .value = "m"};
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    CHECK(frames.restructures == restructures + 1);
}
