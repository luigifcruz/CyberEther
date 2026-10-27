#include <catch2/catch_test_macros.hpp>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/button.hh>
#include <jetstream/render/sakura/components/code_editor.hh>
#include <jetstream/render/sakura/components/combo.hh>
#include <jetstream/render/sakura/components/console.hh>
#include <jetstream/render/sakura/components/curves.hh>
#include <jetstream/render/sakura/components/div.hh>
#include <jetstream/render/sakura/components/divider.hh>
#include <jetstream/render/sakura/components/grid.hh>
#include <jetstream/render/sakura/components/image.hh>
#include <jetstream/render/sakura/components/label.hh>
#include <jetstream/render/sakura/components/log_view.hh>
#include <jetstream/render/sakura/components/markdown.hh>
#include <jetstream/render/sakura/components/markdown_editor.hh>
#include <jetstream/render/sakura/components/panel.hh>
#include <jetstream/render/sakura/components/pill_button.hh>
#include <jetstream/render/sakura/components/pill_field.hh>
#include <jetstream/render/sakura/components/pill_stepper.hh>
#include <jetstream/render/sakura/components/polyline.hh>
#include <jetstream/render/sakura/components/progress_bar.hh>
#include <jetstream/render/sakura/components/node/progress_bar.hh>
#include <jetstream/render/sakura/components/qr_code.hh>
#include <jetstream/render/sakura/components/reveal.hh>
#include <jetstream/render/sakura/components/scroll_area.hh>
#include <jetstream/render/sakura/components/scroller.hh>
#include <jetstream/render/sakura/components/search_catalog.hh>
#include <jetstream/render/sakura/components/setting_field.hh>
#include <jetstream/render/sakura/components/spacer.hh>
#include <jetstream/render/sakura/components/split_view.hh>
#include <jetstream/render/sakura/components/stack.hh>
#include <jetstream/render/sakura/components/status_bar.hh>
#include <jetstream/render/sakura/components/table.hh>
#include <jetstream/render/sakura/components/text.hh>
#include <jetstream/render/sakura/components/text_editor.hh>
#include <jetstream/render/sakura/components/text_grid.hh>
#include <jetstream/render/sakura/components/text_input.hh>
#include <jetstream/render/sakura/components/text_markdown.hh>
#include <jetstream/render/sakura/components/text_view.hh>
#include <jetstream/render/sakura/components/toggle.hh>

#include "harness.hh"
#include "fonts.hh"
#include "component_harness.hh"
#include "scene.hh"

#include <cmath>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

using namespace Jetstream;
using namespace SakuraTest;

namespace {

struct Clipper : Sakura::Component {
    Probe& child;

    explicit Clipper(Probe& child) : child(child) {
        addChild(child);
    }

    void clipping(bool clipsChildren) {
        setClipsChildren(clipsChildren);
    }

 protected:
    void layout(const Sakura::Context& ctx) override {
        layoutChild(ctx, child, {frame().x, frame().y, 300.0f, 300.0f});
    }
};

std::optional<Sakura::Label::Instance> HintOf(Sakura::Component& root, const std::string& hint) {
    for (auto* label : Find<Sakura::Label>(root)) {
        for (const auto& instance : label->config().instances) {
            if (instance.str == hint) {
                return instance;
            }
        }
    }
    return std::nullopt;
}

}  // namespace

TEST_CASE("Identical container updates leave a settled tree retained", "[core][sakura][retention]") {
    Sakura::Box first;
    Sakura::Box second;
    Sakura::Stack stack;
    Sakura::Panel panel;
    Host host(panel);
    Frames frames(host);
    const auto update = [&] {
        first.update({.instances = {{.rect = {0.0f, 0.0f, 10.0f, 10.0f}}}});
        second.update({.instances = {{.rect = {0.0f, 20.0f, 10.0f, 10.0f}}}});
        stack.update({.items = {{.child = &first}, {.child = &second}}});
        panel.update({.id = "panel", .content = &stack});
    };
    REQUIRE(frames.settle(update));
    for (int frame = 0; frame < 5; ++frame) {
        CHECK(frames.run(update).clean());
    }

    second.update({.instances = {{.rect = {0.0f, 24.0f, 10.0f, 10.0f}}}});
    const auto changed = frames.run();
    CHECK(changed.repainted);
    CHECK_FALSE(changed.rebuilt);
    CHECK(frames.run(update).repainted);
    CHECK(frames.run(update).clean());
}

TEST_CASE("Settling builds the tree once and paints it once", "[core][sakura][retention]") {
    Scene scene;
    Sakura::Box box;
    Host host(box);
    Frames frames(host, kScreen, scene.context());
    const auto update = [&] {
        box.update({.instances = {{.rect = {10.0f, 10.0f, 40.0f, 20.0f}}}});
    };
    const auto first = frames.run(update);
    CHECK(first.rebuilt);
    CHECK(first.repainted);
    CHECK(frames.run(update).clean());
    CHECK(Retains(frames, update));
    CHECK(frames.rebuilds == 1);
    CHECK(frames.restructures == 0);
    CHECK(frames.repaints == 1);
}

TEST_CASE("Boxes repaint only when their instances or clip change", "[core][sakura][retention]") {
    Scene scene;
    Sakura::Box box;
    Host host(box);
    Frames frames(host, kScreen, scene.context());
    ColorRGBA<F32> color = {1.0f, 0.0f, 0.0f, 1.0f};
    std::optional<Rect> clip;
    std::vector<Sakura::Box::Instance> extra;
    const auto update = [&] {
        std::vector<Sakura::Box::Instance> instances = {{.rect = {10.0f, 10.0f, 40.0f, 20.0f},
                                                           .backgroundColor = color}};
        instances.insert(instances.end(), extra.begin(), extra.end());
        box.update({.instances = std::move(instances), .clip = clip, .cornerRadius = 4.0f});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
    CHECK(QuietMoves(frames, update, {{20.0f, 20.0f}, {300.0f, 300.0f}}, [&](const MouseEvent& e) {
        return host.send(e);
    }));

    color = {0.0f, 1.0f, 0.0f, 1.0f};
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));

    clip = Rect{0.0f, 0.0f, 30.0f, 30.0f};
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));

    extra = {{.rect = {60.0f, 10.0f, 10.0f, 10.0f}}, {.rect = {80.0f, 10.0f, 10.0f, 10.0f}}};
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    CHECK(frames.rebuilds == 1);
}

TEST_CASE("Labels repaint only when their text instances change", "[core][sakura][retention]") {
    Scene scene;
    Sakura::Label label;
    Host host(label);
    Frames frames(host, kScreen, scene.context());
    std::string str = "Frequency";
    ColorRGBA<F32> color = {1.0f, 1.0f, 1.0f, 1.0f};
    const auto update = [&] {
        label.update({.instances = {{.rect = {10.0f, 10.0f, 100.0f, 20.0f}, .str = str, .color = color}}});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));

    str = "Bandwidth";
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));

    color = {0.5f, 0.5f, 0.5f, 1.0f};
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
}

TEST_CASE("Labels repaint only when their rotation changes", "[core][sakura][retention]") {
    Scene scene;
    Sakura::Label label;
    Host host(label);
    Frames frames(host, kScreen, scene.context());
    F32 rotation = 0.0f;
    const auto update = [&] {
        label.update({.instances = {{.rect = {10.0f, 10.0f, 32.0f, 200.0f}, .str = "Spine", .rotation = rotation}}});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));

    rotation = -90.0f;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    rotation = 0.0f;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    CHECK(frames.rebuilds == 1);
}

TEST_CASE("Polylines repaint only when their geometry or clip change", "[core][sakura][retention]") {
    Scene scene;
    Sakura::Polyline line;
    Host host(line);
    Frames frames(host, kScreen, scene.context());
    ColorRGBA<F32> color = {1.0f, 1.0f, 1.0f, 1.0f};
    F32 thickness = 2.0f;
    std::optional<Rect> clip;
    const auto update = [&] {
        line.update({
            .instances = {{.points = {{10.0f, 10.0f}, {50.0f, 40.0f}, {90.0f, 10.0f}},
                           .color = color,
                           .thickness = thickness}},
            .clip = clip,
        });
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));

    color = {1.0f, 0.0f, 0.0f, 1.0f};
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));

    thickness = 3.0f;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));

    clip = Rect{0.0f, 0.0f, 50.0f, 50.0f};
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
}

TEST_CASE("Curves repaint only when their instances change", "[core][sakura][retention]") {
    Scene scene;
    Sakura::Curves curves;
    Host host(curves);
    Frames frames(host, kScreen, scene.context());
    Extent2D<F32> end = {200.0f, 100.0f};
    const auto update = [&] {
        curves.update({.instances = {{.p0 = {10.0f, 10.0f},
                                      .p1 = {80.0f, 10.0f},
                                      .p2 = {120.0f, 100.0f},
                                      .p3 = end,
                                      .thickness = 2.0f,
                                      .segments = 24}}});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));

    end = {220.0f, 120.0f};
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
}

TEST_CASE("Images without a texture settle and repaint on visual changes", "[core][sakura][retention]") {
    Scene scene;
    Sakura::Image image;
    Host host(image);
    Frames frames(host, kScreen, scene.context());
    F32 opacity = 1.0f;
    Rect rect = {10.0f, 10.0f, 64.0f, 64.0f};
    const auto update = [&] {
        image.update({.rect = rect, .cornerRadius = 4.0f, .opacity = opacity});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));

    opacity = 0.5f;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));

    rect = {20.0f, 20.0f, 64.0f, 64.0f};
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
}

TEST_CASE("Plain text settles and ignores mouse moves", "[core][sakura][retention]") {
    Scene scene;
    Sakura::Text text;
    Place place(text, {10.0f, 10.0f, 200.0f, 20.0f});
    Frames frames(place, kScreen, scene.context());
    const auto update = [&] {
        text.update({.str = "Sample rate"});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
    CHECK(QuietMoves(frames, update, {{20.0f, 15.0f}, {40.0f, 15.0f}, {500.0f, 500.0f}}, [&](const MouseEvent& e) {
        return place.send(e);
    }));
}

TEST_CASE("Text changes repaint without rebuilding", "[core][sakura][retention]") {
    Scene scene;
    Sakura::Text text;
    Place place(text, {10.0f, 10.0f, 200.0f, 20.0f});
    Frames frames(place, kScreen, scene.context());
    Sakura::Text::Config config = {.str = "Sample rate"};
    const auto update = [&] {
        text.update(config);
    };
    REQUIRE(frames.settle(update));

    config.str = "Center frequency";
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));

    config.tone = Sakura::Text::Tone::Secondary;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));

    config.colorKey = "cyber_blue";
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));

    config.opacity = 0.5f;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));

    config.align = Sakura::Text::Align::Right;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    CHECK(frames.rebuilds == 1);
}

TEST_CASE("Text tones each settle under identical updates", "[core][sakura][retention]") {
    Scene scene;
    for (const auto tone : {Sakura::Text::Tone::Primary, Sakura::Text::Tone::Secondary,
                            Sakura::Text::Tone::Disabled, Sakura::Text::Tone::Accent,
                            Sakura::Text::Tone::Success, Sakura::Text::Tone::Warning}) {
        CAPTURE(static_cast<int>(tone));
        Sakura::Text text;
        Place place(text, {10.0f, 10.0f, 200.0f, 20.0f});
        Frames frames(place, kScreen, scene.context());
        const auto update = [&] {
            text.update({.str = "Gain", .tone = tone});
        };
        REQUIRE(frames.settle(update));
        CHECK(Retains(frames, update));
    }
}

TEST_CASE("Wrapped text settles and rewraps with a repaint on resize", "[core][sakura][retention]") {
    Scene scene;
    Sakura::Text text;
    Place place(text, {10.0f, 10.0f, 120.0f, 200.0f});
    Frames frames(place, kScreen, scene.context());
    const auto update = [&] {
        text.update({.str = "The quick brown fox jumps over the lazy dog near the river bank", .wrapped = true});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));

    place.move({10.0f, 10.0f, 240.0f, 200.0f});
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
}

TEST_CASE("Text with icon glyphs settles", "[core][sakura][retention]") {
    Scene scene;
    Sakura::Text text;
    Place place(text, {10.0f, 10.0f, 200.0f, 20.0f});
    Frames frames(place, kScreen, scene.context());
    const auto update = [&] {
        text.update({.str = "\xef\x80\x93 Save", .font = Sakura::Text::Font::Bold});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
}

TEST_CASE("Markdown settles and repaints when its value changes", "[core][sakura][retention]") {
    Scene scene;
    Sakura::Markdown markdown;
    Place place(markdown, {10.0f, 10.0f, 300.0f, 200.0f});
    Frames frames(place, kScreen, scene.context());
    std::string value = "# Title\n\nSome **bold** text and `code`.\n\n- one\n- two";
    const auto update = [&] {
        markdown.update({.value = value});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
    CHECK(QuietMoves(frames, update, {{50.0f, 50.0f}, {60.0f, 60.0f}}, [&](const MouseEvent& e) {
        return place.send(e);
    }));

    value += "\n- three";
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    CHECK(frames.restructures == 0);
}

TEST_CASE("Text markdown settles under identical updates", "[core][sakura][retention]") {
    Scene scene;
    Sakura::TextMarkdown markdown;
    Place place(markdown, {10.0f, 10.0f, 400.0f, 300.0f});
    Frames frames(place, kScreen, scene.context());
    std::string value =
        "## Notes\n\n> [!NOTE]\n> Callout body\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n```python\nx = 1\n```";
    const auto update = [&] {
        markdown.update({.id = "notes", .value = value, .scrollbar = true});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));

    value = "## Notes\n\nShort.";
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    CHECK(frames.restructures == 0);
}

TEST_CASE("Text grids settle under identical updates and repaint on value changes",
          "[core][sakura][retention]") {
    Scene scene;
    Sakura::TextGrid grid;
    Place place(grid, {10.0f, 10.0f, 300.0f, 200.0f});
    Frames frames(place, kScreen, scene.context());
    std::string value = "first line\nsecond line\nthird line";
    const auto update = [&] {
        grid.update({.id = "grid", .value = value, .lineNumbers = true});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
    CHECK(QuietMoves(frames, update, {{50.0f, 20.0f}, {60.0f, 40.0f}, {500.0f, 500.0f}}, [&](const MouseEvent& e) {
        return place.send(e);
    }));

    value += "\nfourth line";
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    CHECK(frames.restructures == 0);
}

TEST_CASE("Unfocused editable text grids ignore the clock", "[core][sakura][retention]") {
    Scene scene;
    Sakura::TextGrid grid;
    Place place(grid, {10.0f, 10.0f, 300.0f, 200.0f});
    Frames frames(place, kScreen, scene.context());
    const auto update = [&] {
        grid.update({.id = "idle-editor", .value = "x = 1", .editable = true});
    };
    REQUIRE(frames.settle(update));
    for (int frame = 0; frame < 6; ++frame) {
        frames.ctx.time += 0.3;
        CHECK(frames.run(update).clean());
    }
    CHECK(frames.restructures == 0);
}

TEST_CASE("Focused editable text grids repaint only when the caret blinks", "[core][sakura][retention]") {
    Scene scene;
    Sakura::TextGrid grid;
    Place place(grid, {10.0f, 10.0f, 300.0f, 200.0f});
    Frames frames(place, kScreen, scene.context());
    frames.ctx.windowFocused = true;
    const auto update = [&] {
        grid.update({.id = "blink-editor", .value = "x = 1", .editable = true});
    };
    REQUIRE(frames.settle(update));
    grid.focus();
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));

    frames.ctx.time = 0.25;
    CHECK(frames.run(update).clean());
    frames.ctx.time = 0.6;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(frames.run(update).clean());

    for (int frame = 0; frame < 6; ++frame) {
        frames.ctx.time += 0.5;
        CHECK(RepaintOnly(frames.run(update)));
    }

    frames.ctx.windowFocused = false;
    frames.run(update);
    for (int frame = 0; frame < 4; ++frame) {
        frames.ctx.time += 0.5;
        CHECK(frames.run(update).clean());
    }
    frames.ctx.windowFocused = true;
    CHECK(RepaintOnly(frames.run(update)));

    grid.blur();
    CHECK(RepaintOnly(frames.run(update)));
    for (int frame = 0; frame < 4; ++frame) {
        frames.ctx.time += 0.5;
        CHECK(frames.run(update).clean());
    }
    CHECK(frames.restructures == 0);
}

TEST_CASE("Text views settle and wheel scrolling repaints without rebuilding", "[core][sakura][retention]") {
    Scene scene;
    Sakura::TextView view;
    Place place(view, {10.0f, 10.0f, 300.0f, 100.0f});
    Frames frames(place, kScreen, scene.context());
    std::string value;
    for (int line = 0; line < 40; ++line) {
        value += "line " + std::to_string(line) + "\n";
    }
    const auto update = [&] {
        view.update({.id = "view", .value = value, .lineNumbers = true});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));

    CHECK(place.send(Wheel(100.0f, 50.0f, -1.0f)));
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    CHECK(frames.restructures == 0);
}

TEST_CASE("Text editors settle under identical updates", "[core][sakura][retention]") {
    Scene scene;
    Sakura::TextEditor editor;
    Place place(editor, {10.0f, 10.0f, 300.0f, 200.0f});
    Frames frames(place, kScreen, scene.context());
    std::string value = "def main():\n    return 42\n";
    const auto update = [&] {
        editor.update({.id = "editor", .value = value});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));

    value = "def main():\n    return 7\n";
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    CHECK(frames.restructures == 0);
}

TEST_CASE("Log views settle and repaint when output is appended", "[core][sakura][retention]") {
    Scene scene;
    Sakura::LogView log;
    Place place(log, {10.0f, 10.0f, 300.0f, 120.0f});
    Frames frames(place, kScreen, scene.context());
    std::string value = "boot\nready";
    const auto update = [&] {
        log.update({.value = value, .height = 120.0f, .stickToBottom = true});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));

    value += "\nstreaming";
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    CHECK(frames.restructures == 0);
}

TEST_CASE("Status bars settle and repaint on status changes", "[core][sakura][retention]") {
    Scene scene;
    Sakura::StatusBar bar;
    Place place(bar, {0.0f, 100.0f, 400.0f, 24.0f});
    Frames frames(place, kScreen, scene.context());
    std::string status = "Ready";
    Sakura::StatusTone tone = Sakura::StatusTone::Info;
    bool expanded = false;
    const auto update = [&] {
        bar.update({.id = "status",
                    .status = status,
                    .tone = tone,
                    .toggleText = "Console",
                    .toggleExpanded = expanded,
                    .onToggle = [] {}});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
    CHECK(QuietMoves(frames, update, {{20.0f, 110.0f}, {380.0f, 110.0f}}, [&](const MouseEvent& e) {
        return place.send(e);
    }));

    status = "Running";
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));

    tone = Sakura::StatusTone::Error;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));

    expanded = true;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
}

TEST_CASE("Panels settle in natural and filling sizes", "[core][sakura][retention]") {
    Scene scene;
    for (const F32 width : {Sakura::Fill, 0.0f}) {
        for (const F32 height : {0.0f, Sakura::Fill}) {
            CAPTURE(width, height);
            Sakura::Text text;
            Sakura::Panel panel;
            Place place(panel, {10.0f, 10.0f, 300.0f, 200.0f});
            Frames frames(place, kScreen, scene.context());
            const auto update = [&] {
                text.update({.str = "Panel body"});
                panel.update({.id = "panel", .content = &text, .size = {width, height}});
            };
            REQUIRE(frames.settle(update));
            CHECK(Retains(frames, update));
        }
    }
}

TEST_CASE("Panel background changes repaint without rebuilding", "[core][sakura][retention]") {
    Scene scene;
    Sakura::Text text;
    Sakura::Panel panel;
    Place place(panel, {10.0f, 10.0f, 300.0f, 200.0f});
    Frames frames(place, kScreen, scene.context());
    std::string key = "background";
    bool transparent = false;
    const auto update = [&] {
        text.update({.str = "Panel body"});
        panel.update({.id = "panel", .content = &text, .transparent = transparent, .backgroundColorKey = key});
    };
    REQUIRE(frames.settle(update));

    key = "card";
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));

    transparent = true;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
}

TEST_CASE("Stacks settle in both axes and repaint when an item resizes", "[core][sakura][retention]") {
    Scene scene;
    for (const auto axis : {Sakura::Stack::Axis::Vertical, Sakura::Stack::Axis::Horizontal}) {
        CAPTURE(static_cast<int>(axis));
        Sakura::Text first;
        Sakura::Text second;
        Sakura::Spacer spacer;
        Sakura::Stack stack;
        Place place(stack, {10.0f, 10.0f, 400.0f, 200.0f});
        Frames frames(place, kScreen, scene.context());
        F32 size = 30.0f;
        const auto update = [&] {
            first.update({.str = "Alpha"});
            second.update({.str = "Beta"});
            spacer.update({.lines = 2});
            stack.update({.axis = axis,
                          .items = {{.child = &first, .size = size},
                                    {.child = &spacer},
                                    {.child = &second, .grow = 1.0f}}});
        };
        REQUIRE(frames.settle(update));
        CHECK(Retains(frames, update));

        size = 60.0f;
        CHECK(RepaintOnly(frames.run(update)));
        CHECK(Retains(frames, update));
    }
}

TEST_CASE("Hiding and showing a stack item repaints without rebuilding", "[core][sakura][retention]") {
    Scene scene;
    Sakura::Text first;
    Sakura::Text second;
    Sakura::Stack stack;
    Place place(stack, {10.0f, 10.0f, 400.0f, 200.0f});
    Frames frames(place, kScreen, scene.context());
    const auto update = [&] {
        first.update({.str = "Alpha"});
        second.update({.str = "Beta"});
        stack.update({.items = {{.child = &first}, {.child = &second}}});
    };
    REQUIRE(frames.settle(update));

    first.setVisible(false);
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));

    first.setVisible(true);
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    CHECK(frames.rebuilds == 1);
    CHECK(frames.restructures == 0);
}

TEST_CASE("Children hidden before the first build restructure once when shown", "[core][sakura][retention]") {
    Scene scene;
    Sakura::Text first;
    Sakura::Text second;
    Sakura::Stack stack;
    Place place(stack, {10.0f, 10.0f, 400.0f, 200.0f});
    Frames frames(place, kScreen, scene.context());
    const auto update = [&] {
        first.update({.str = "Alpha"});
        second.update({.str = "Beta"});
        stack.update({.items = {{.child = &first}, {.child = &second}}});
    };
    update();
    second.setVisible(false);
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));

    second.setVisible(true);
    const auto shown = frames.run(update);
    CHECK(shown.restructured);
    CHECK_FALSE(shown.rebuilt);
    CHECK(shown.repainted);
    CHECK(Retains(frames, update));
}

TEST_CASE("Replacing stack items restructures once then settles", "[core][sakura][retention]") {
    Scene scene;
    Sakura::Text first;
    Sakura::Text second;
    Sakura::Text third;
    Sakura::Stack stack;
    Place place(stack, {10.0f, 10.0f, 400.0f, 200.0f});
    Frames frames(place, kScreen, scene.context());
    bool swapped = false;
    const auto update = [&] {
        first.update({.str = "Alpha"});
        second.update({.str = "Beta"});
        third.update({.str = "Gamma"});
        stack.update({.items = {{.child = &first}, {.child = swapped ? &third : &second}}});
    };
    REQUIRE(frames.settle(update));

    swapped = true;
    const auto changed = frames.run(update);
    CHECK(changed.restructured);
    CHECK_FALSE(changed.rebuilt);
    CHECK(Retains(frames, update));

    swapped = false;
    const auto restored = frames.run(update);
    CHECK_FALSE(restored.rebuilt);
    CHECK(restored.repainted);
    CHECK(Retains(frames, update));
}

TEST_CASE("Dividers settle under identical updates", "[core][sakura][retention]") {
    Scene scene;
    Sakura::Divider divider;
    Place place(divider, {10.0f, 10.0f, 300.0f, 20.0f});
    Frames frames(place, kScreen, scene.context());
    bool separator = true;
    const auto update = [&] {
        divider.update({.separator = separator});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));

    separator = false;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
}

TEST_CASE("Interactive divs settle and repaint once per hover transition", "[core][sakura][retention]") {
    Scene scene;
    Sakura::Text text;
    Sakura::Div div;
    Place place(div, {10.0f, 10.0f, 200.0f, 60.0f});
    Frames frames(place, kScreen, scene.context());
    bool selected = false;
    const auto update = [&] {
        text.update({.str = "Card"});
        div.update({.content = &text, .padding = 8.0f, .rounding = 6.0f, .selected = selected, .onClick = [] {}});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
    CheckHoverRepaintsOnce(frames, place, update, {50.0f, 30.0f}, {400.0f, 300.0f});
    CheckLeaveRepaintsOnce(frames, place, update, {50.0f, 30.0f});

    selected = true;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
}

TEST_CASE("Reveals lay out only while animating and stay clean while holding or with a frozen clock",
          "[core][sakura][retention]") {
    Probe child({100.0f, 40.0f});
    child.prepare();
    Sakura::Reveal reveal;
    Place place(reveal, {10.0f, 10.0f, 200.0f, 50.0f});
    Frames frames(place, kScreen, MakeContext());
    bool shown = false;
    const auto update = [&] {
        reveal.update({.child = &child, .shown = shown, .gap = 10.0f, .minimumDuration = 1.0, .linger = 0.5});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
    CHECK_FALSE(child.visible());

    shown = true;
    CHECK(frames.run(update).laidOut);
    CHECK(child.visible());
    CHECK(Retains(frames, update));

    frames.timeStep = 1.0 / 60.0;
    REQUIRE(frames.settle(update, 60));
    CHECK(Retains(frames, update));

    shown = false;
    CHECK(frames.run(update).laidOut);
    const U64 held = frames.layouts;
    for (U64 frame = 0; frame < 20; ++frame) {
        frames.run(update);
    }
    CHECK(frames.layouts == held);
    CHECK(child.visible());

    for (U64 frame = 0; frame < 60; ++frame) {
        frames.run(update);
    }
    REQUIRE(frames.settle(update, 120));
    CHECK(Retains(frames, update));
    CHECK_FALSE(child.visible());
}

TEST_CASE("Collapsed reveals stop laying out an animated child and settle", "[core][sakura][retention]") {
    struct Ticker : Probe {
        using Probe::Probe;

     protected:
        void layout(const Sakura::Context& ctx) override {
            Probe::layout(ctx);
            requestFrame();
        }
    };

    Ticker child({100.0f, 40.0f});
    child.prepare();
    Sakura::Reveal reveal;
    Place place(reveal, {10.0f, 10.0f, 200.0f, 50.0f});
    Frames frames(place, kScreen, MakeContext());
    frames.timeStep = 1.0 / 60.0;
    bool shown = true;
    const auto update = [&] {
        reveal.update({.child = &child, .shown = shown, .gap = 10.0f});
    };
    for (U64 frame = 0; frame < 30; ++frame) {
        frames.run(update);
    }
    CHECK(child.visible());
    CHECK(child.layouts > 20);

    shown = false;
    REQUIRE(frames.settle(update, 120));
    CHECK_FALSE(child.visible());
    const U64 laid = child.layouts;
    CHECK(Retains(frames, update));
    CHECK(child.layouts == laid);
}

TEST_CASE("Passive divs ignore hover changes", "[core][sakura][retention]") {
    Scene scene;
    Sakura::Text text;
    Sakura::Div div;
    Place place(div, {10.0f, 10.0f, 200.0f, 60.0f});
    Frames frames(place, kScreen, scene.context());
    const auto update = [&] {
        text.update({.str = "Card"});
        div.update({.content = &text, .padding = 8.0f});
    };
    REQUIRE(frames.settle(update));
    CHECK(QuietMoves(frames, update, {{50.0f, 30.0f}, {60.0f, 30.0f}, {400.0f, 300.0f}, {50.0f, 30.0f}},
                     [&](const MouseEvent& e) { return place.send(e); }));
    place.send(Mouse(MouseEventType::Leave, 50.0f, 30.0f));
    CHECK(frames.run(update).clean());
    place.send(Mouse(MouseEventType::Cancel, 50.0f, 30.0f));
    CHECK(frames.run(update).clean());
    CHECK(Retains(frames, update));
}

TEST_CASE("Scroll areas settle and wheel scrolling repaints without rebuilding",
          "[core][sakura][retention]") {
    Scene scene;
    Sakura::Text text;
    Sakura::ScrollArea area;
    Place place(area, {10.0f, 10.0f, 300.0f, 120.0f});
    Frames frames(place, kScreen, scene.context());
    std::string body;
    for (int line = 0; line < 30; ++line) {
        body += "Line " + std::to_string(line) + "\n";
    }
    const auto update = [&] {
        text.update({.str = body});
        area.update({.content = &text, .height = 120.0f});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
    CHECK(QuietMoves(frames, update, {{50.0f, 50.0f}, {60.0f, 60.0f}}, [&](const MouseEvent& e) {
        return place.send(e);
    }));

    CHECK(place.send(Wheel(100.0f, 50.0f, -1.0f)));
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
}

TEST_CASE("Scrollers move their thumb on the frame of a wheel scroll or a thumb drag",
          "[core][sakura][retention]") {
    Scene scene;
    Sakura::Scroller scroll;
    Place place(scroll, {10.0f, 10.0f, 200.0f, 100.0f});
    Frames frames(place, kScreen, scene.context());
    frames.verify = false;
    F32 scrollY = 0.0f;
    const auto update = [&] {
        scroll.update({.id = "scroll",
                       .contentHeight = 600.0f,
                       .scrollY = scrollY,
                       .onScrollY = [&](F32 next) { scrollY = next; }});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
    CHECK(QuietMoves(frames, update, {{50.0f, 50.0f}, {205.0f, 50.0f}}, [&](const MouseEvent& e) {
        return place.send(e);
    }));
    const auto resting = Drawn(place);

    CHECK(place.send(Wheel(100.0f, 50.0f, -1.0f)));
    CHECK(scrollY > 0.0f);
    CHECK(RepaintOnly(frames.run()));
    CHECK(Drawn(place) != resting);
    CHECK(Unpainted(frames.run(update)));
    CHECK(Retains(frames, update));

    const auto wheeled = Drawn(place);
    CHECK(place.send(Mouse(MouseEventType::Click, 205.0f, 30.0f)));
    CHECK(place.send(Mouse(MouseEventType::Move, 205.0f, 90.0f)));
    CHECK(RepaintOnly(frames.run()));
    CHECK(Drawn(place) != wheeled);
    CHECK(place.send(Mouse(MouseEventType::Release, 205.0f, 90.0f)));
    CHECK(Unpainted(frames.run(update)));
    CHECK(Retains(frames, update));
}

TEST_CASE("Scrollers keep their thumb still while content grows by less than a pixel of thumb",
          "[core][sakura][retention]") {
    Scene scene;
    Sakura::Scroller scroll;
    Place place(scroll, {10.0f, 10.0f, 200.0f, 100.0f});
    Frames frames(place, kScreen, scene.context());
    frames.verify = false;
    F32 contentHeight = 20000.0f;
    const auto update = [&] {
        scroll.update({.id = "scroll", .contentHeight = contentHeight, .scrollY = 4000.0f});
    };
    REQUIRE(frames.settle(update));

    for (U64 frame = 0; frame < 5; ++frame) {
        contentHeight += 16.0f;
        CHECK(Unpainted(frames.run(update)));
    }

    contentHeight = 2000.0f;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
}

TEST_CASE("Progress bars stay clean while their value moves within one pixel",
          "[core][sakura][retention]") {
    Scene scene;
    Sakura::ProgressBar bar;
    Place place(bar, {10.0f, 10.0f, 300.0f, 20.0f});
    Frames frames(place, kScreen, scene.context());
    F32 value = 0.3f;
    const auto update = [&] {
        bar.update({.value = value, .overlay = "30%"});
    };
    REQUIRE(frames.settle(update));

    for (U64 frame = 0; frame < 5; ++frame) {
        value += 0.0001f;
        CHECK(Unpainted(frames.run(update)));
    }

    value = 0.6f;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(frames.run(update).clean());
}

TEST_CASE("Progress bars keep exact endpoints at fractional bounds", "[core][sakura][retention]") {
    constexpr Rect bounds = {10.75f, 10.0f, 200.5f, 24.0f};
    const auto fillAt = [&](auto& bar, const auto& update, const F32 value) {
        Scene scene;
        Place place(bar, bounds);
        Frames frames(place, kScreen, scene.context());
        update(value);
        frames.run();
        const auto boxes = Find<Sakura::Box>(place);
        REQUIRE(boxes.size() == 2);
        return boxes[1]->config();
    };
    const auto check = [&](auto& bar, const auto& update) {
        const auto empty = fillAt(bar, update, 0.0f);
        CHECK_FALSE(empty.instances.front().visible);

        const auto half = fillAt(bar, update, 0.5f);
        REQUIRE(half.clip.has_value());
        CHECK(half.instances.front().visible);
        CHECK(half.clip->right() == std::round(half.clip->right()));

        const auto full = fillAt(bar, update, 1.0f);
        REQUIRE(full.clip.has_value());
        CHECK(full.instances.front().visible);
        CHECK(full.clip->x == bounds.x);
        CHECK(full.clip->right() == bounds.right());
    };

    Sakura::ProgressBar bar;
    check(bar, [&](const F32 value) { bar.update({.value = value, .overlay = "bar"}); });
    Sakura::NodeProgressBar node;
    check(node, [&](const F32 value) { node.update({.id = "node", .value = value, .overlay = "node"}); });
}

TEST_CASE("Scrollers follow an owner that resets a wheel scroll to the offset it configured before",
          "[core][sakura][retention]") {
    Scene scene;
    Sakura::Scroller scroll;
    Place place(scroll, {10.0f, 10.0f, 200.0f, 100.0f});
    Frames frames(place, kScreen, scene.context());
    frames.verify = false;
    Extent2D<F32> offset = {0.0f, 0.0f};
    std::vector<Extent2D<F32>> reports;
    const auto update = [&] {
        scroll.update({.id = "scroll",
                       .contentWidth = 900.0f,
                       .contentHeight = 600.0f,
                       .scrollX = offset.x,
                       .scrollY = offset.y,
                       .onScrollY = [&](F32 next) { reports.push_back({offset.x, next}); },
                       .onScrollX = [&](F32 next) { reports.push_back({next, offset.y}); }});
    };
    REQUIRE(frames.settle(update));
    const auto resting = Drawn(place);

    CHECK(place.send(Wheel(100.0f, 50.0f, -1.0f, -1.0f)));
    REQUIRE(reports.size() == 2);
    const Extent2D<F32> first = {reports[1].x, reports[0].y};
    REQUIRE(first.x > 0.0f);
    REQUIRE(first.y > 0.0f);
    frames.run(update);
    REQUIRE(frames.settle(update));
    CHECK(Drawn(place) == resting);

    reports.clear();
    CHECK(place.send(Wheel(100.0f, 50.0f, -1.0f, -1.0f)));
    REQUIRE(reports.size() == 2);
    CHECK(reports[1].x == first.x);
    CHECK(reports[0].y == first.y);
}

TEST_CASE("Scrollers clamp a configured offset against the viewport they are laid out in",
          "[core][sakura][retention]") {
    Scene scene;
    Sakura::Scroller scroll;
    Place place(scroll, {10.0f, 10.0f, 200.0f, 100.0f});
    Frames frames(place, kScreen, scene.context());
    frames.verify = false;
    Extent2D<F32> offset = {0.0f, 0.0f};
    const auto configure = [&](Sakura::Scroller& target) {
        target.update({.id = "scroll",
                       .contentWidth = 900.0f,
                       .contentHeight = 600.0f,
                       .scrollX = offset.x,
                       .scrollY = offset.y,
                       .onScrollY = [&](F32 next) { offset.y = next; },
                       .onScrollX = [&](F32 next) { offset.x = next; }});
    };
    const auto update = [&] {
        configure(scroll);
    };
    REQUIRE(frames.settle(update));

    offset = {800.0f, 550.0f};
    place.move({10.0f, 10.0f, 100.0f, 50.0f});
    frames.run(update);
    const auto shrunk = Drawn(place);
    CHECK(frames.run(update).clean());

    Sakura::Scroller fresh;
    Place freshPlace(fresh, {10.0f, 10.0f, 100.0f, 50.0f});
    Frames freshFrames(freshPlace, kScreen, scene.context());
    freshFrames.verify = false;
    REQUIRE(freshFrames.settle([&] { configure(fresh); }));
    CHECK(shrunk == Drawn(freshPlace));
    CHECK(offset.x == 800.0f);
    CHECK(offset.y == 550.0f);
}

TEST_CASE("Split views settle and repaint when the divider moves", "[core][sakura][retention]") {
    Scene scene;
    Sakura::Text left;
    Sakura::Text right;
    Sakura::SplitView split;
    Place place(split, {10.0f, 10.0f, 500.0f, 200.0f});
    Frames frames(place, kScreen, scene.context());
    F32 leftWidth = 150.0f;
    const auto update = [&] {
        left.update({.str = "Left"});
        right.update({.str = "Right"});
        split.update({.left = &left, .right = &right, .leftWidth = leftWidth, .height = 200.0f});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));

    leftWidth = 200.0f;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
}

TEST_CASE("Grids settle and repaint when an item changes", "[core][sakura][retention]") {
    Scene scene;
    std::vector<std::unique_ptr<Sakura::Text>> items;
    for (int index = 0; index < 6; ++index) {
        items.push_back(std::make_unique<Sakura::Text>());
    }
    Sakura::Grid grid;
    Place place(grid, {10.0f, 10.0f, 400.0f, 200.0f});
    Frames frames(place, kScreen, scene.context());
    std::string label = "Cell";
    const auto update = [&] {
        std::vector<Sakura::Component*> cells;
        for (U64 index = 0; index < items.size(); ++index) {
            items[index]->update({.str = label + " " + std::to_string(index)});
            cells.push_back(items[index].get());
        }
        grid.update({.columns = 3, .items = cells});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));

    label = "Item";
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
}

TEST_CASE("Tables settle under identical updates and repaint on cell changes", "[core][sakura][retention]") {
    Scene scene;
    Sakura::Table table;
    Place place(table, {10.0f, 10.0f, 400.0f, 200.0f});
    Frames frames(place, kScreen, scene.context());
    std::string value = "1.0";
    const auto update = [&] {
        table.update({.columns = {"Name", "Value"}, .rows = {{"Gain", value}, {"Rate", "2.0"}}});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
    CHECK(QuietMoves(frames, update, {{50.0f, 40.0f}, {60.0f, 60.0f}, {500.0f, 500.0f}}, [&](const MouseEvent& e) {
        return place.send(e);
    }));

    value = "3.0";
    const auto changed = frames.run(update);
    CHECK_FALSE(changed.rebuilt);
    CHECK(changed.repainted);
    CHECK(Retains(frames, update));
}

TEST_CASE("Headerless tables with rows settle under identical updates", "[core][sakura][retention]") {
    Scene scene;
    Sakura::Table table;
    Place place(table, {10.0f, 10.0f, 400.0f, 200.0f});
    Frames frames(place, kScreen, scene.context());
    const auto update = [&] {
        table.update({.columns = {"Key", "Value"},
                      .rows = {{"Driver", "soapy"}, {"Serial", "1234"}},
                      .showHeaders = false,
                      .wrapped = true});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
}

TEST_CASE("Empty headerless tables settle under identical updates", "[core][sakura][retention]") {
    Scene scene;
    Sakura::Table table;
    Place place(table, {10.0f, 10.0f, 400.0f, 200.0f});
    Frames frames(place, kScreen, scene.context());
    std::vector<std::vector<std::string>> rows;
    const auto update = [&] {
        table.update({.columns = {"Key", "Value"}, .rows = rows, .showHeaders = false});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
    CHECK(QuietMoves(frames, update, {{50.0f, 40.0f}, {500.0f, 500.0f}}, [&](const MouseEvent& e) {
        return place.send(e);
    }));

    rows = {{"Driver", "soapy"}};
    const auto filled = frames.run(update);
    CHECK_FALSE(filled.rebuilt);
    CHECK(filled.repainted);
    CHECK(Retains(frames, update));

    rows.clear();
    const auto emptied = frames.run(update);
    CHECK_FALSE(emptied.rebuilt);
    CHECK(emptied.repainted);
    CHECK(Retains(frames, update));
}

TEST_CASE("Tree tables repaint once per row hover transition", "[core][sakura][retention]") {
    Scene scene;
    Sakura::Table table;
    Place place(table, {10.0f, 10.0f, 400.0f, 200.0f});
    Frames frames(place, kScreen, scene.context());
    const auto update = [&] {
        table.update({
            .columns = {"Name", "Value"},
            .nodes = {{.id = "a", .label = "Alpha", .cells = {"1"}, .children = {{.label = "Child", .cells = {"2"}}},
                       .open = true},
                      {.id = "b", .label = "Beta", .cells = {"3"}}},
        });
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
    CheckHoverRepaintsOnce(frames, place, update, {200.0f, 40.0f}, {500.0f, 500.0f});
    CheckLeaveRepaintsOnce(frames, place, update, {200.0f, 40.0f});
}

TEST_CASE("Buttons settle and repaint once per hover transition", "[core][sakura][retention]") {
    Scene scene;
    Sakura::Button button;
    Place place(button, {10.0f, 10.0f, 120.0f, 30.0f});
    Frames frames(place, kScreen, scene.context());
    const auto update = [&] {
        button.update({.str = "Run", .onClick = [] {}});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
    CheckHoverRepaintsOnce(frames, place, update, {50.0f, 20.0f}, {400.0f, 300.0f});
    CheckLeaveRepaintsOnce(frames, place, update, {50.0f, 20.0f});
}

TEST_CASE("Button presses and releases repaint then settle", "[core][sakura][retention]") {
    Scene scene;
    U64 clicks = 0;
    Sakura::Button button;
    Place place(button, {10.0f, 10.0f, 120.0f, 30.0f});
    Frames frames(place, kScreen, scene.context());
    const auto update = [&] {
        button.update({.str = "Run", .onClick = [&] { ++clicks; }});
    };
    REQUIRE(frames.settle(update));
    place.send(Mouse(MouseEventType::Move, 50.0f, 20.0f));
    REQUIRE(frames.settle(update));

    CHECK(place.send(Mouse(MouseEventType::Click, 50.0f, 20.0f)));
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));

    CHECK(place.send(Mouse(MouseEventType::Release, 50.0f, 20.0f)));
    CHECK(clicks == 1);
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
}

TEST_CASE("Button label and variant changes repaint without rebuilding", "[core][sakura][retention]") {
    Scene scene;
    Sakura::Button button;
    Place place(button, {10.0f, 10.0f, 120.0f, 30.0f});
    Frames frames(place, kScreen, scene.context());
    Sakura::Button::Config config = {.str = "Run"};
    const auto update = [&] {
        button.update(config);
    };
    REQUIRE(frames.settle(update));

    config.str = "Stop";
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));

    config.variant = Sakura::Button::Variant::Action;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));

    config.disabled = true;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
}

TEST_CASE("Text buttons repaint their underline once per hover transition",
          "[core][sakura][retention]") {
    Scene scene;
    Sakura::Button button;
    Place place(button, {10.0f, 10.0f, 120.0f, 30.0f});
    Frames frames(place, kScreen, scene.context());
    const auto update = [&] {
        button.update({.str = "Learn more", .variant = Sakura::Button::Variant::Text, .onClick = [] {}});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
    CheckHoverRepaintsOnce(frames, place, update, {30.0f, 15.0f}, {400.0f, 300.0f});
    CheckLeaveRepaintsOnce(frames, place, update, {30.0f, 15.0f});
}

TEST_CASE("Navigation items settle and repaint on selection and hover", "[core][sakura][retention]") {
    Scene scene;
    Sakura::NavigationItem item;
    Place place(item, {10.0f, 10.0f, 200.0f, 40.0f});
    Frames frames(place, kScreen, scene.context());
    bool selected = false;
    const auto update = [&] {
        item.update({.label = "Overview", .selected = selected, .onSelect = [] {}});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
    CheckHoverRepaintsOnce(frames, place, update, {50.0f, 25.0f}, {400.0f, 300.0f});
    CheckLeaveRepaintsOnce(frames, place, update, {50.0f, 25.0f});

    selected = true;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
}

TEST_CASE("Progress bars settle and repaint when the value changes", "[core][sakura][retention]") {
    Scene scene;
    Sakura::ProgressBar bar;
    Place place(bar, {10.0f, 10.0f, 300.0f, 20.0f});
    Frames frames(place, kScreen, scene.context());
    F32 value = 0.25f;
    std::string overlay = "25%";
    const auto update = [&] {
        bar.update({.value = value, .overlay = overlay});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
    CHECK(QuietMoves(frames, update, {{50.0f, 20.0f}, {400.0f, 300.0f}}, [&](const MouseEvent& e) {
        return place.send(e);
    }));

    value = 0.5f;
    overlay = "50%";
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));

    value = 0.0f;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
}

TEST_CASE("QR codes settle and repaint when the data changes", "[core][sakura][retention]") {
    Scene scene;
    Sakura::QrCode code;
    Place place(code, {10.0f, 10.0f, 200.0f, 200.0f});
    Frames frames(place, kScreen, scene.context());
    std::vector<U8> data = {1, 0, 1, 0, 1, 0, 1, 0, 1};
    const auto update = [&] {
        code.update({.data = data, .width = 3, .onClick = [] {}});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
    CHECK(QuietMoves(frames, update, {{50.0f, 50.0f}, {400.0f, 300.0f}}, [&](const MouseEvent& e) {
        return place.send(e);
    }));

    data = {0, 1, 0, 1, 0, 1, 0, 1, 0};
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
}

TEST_CASE("Closed combos settle and repaint once per hover transition", "[core][sakura][retention]") {
    Scene scene;
    Sakura::Combo combo;
    Place place(combo, {10.0f, 10.0f, 200.0f, 30.0f});
    Frames frames(place, kScreen, scene.context());
    std::string value = "Two";
    const auto update = [&] {
        combo.update({.id = "combo", .options = {"One", "Two", "Three"}, .value = value});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
    CheckHoverRepaintsOnce(frames, place, update, {50.0f, 20.0f}, {400.0f, 300.0f});
    CheckLeaveRepaintsOnce(frames, place, update, {50.0f, 20.0f});

    value = "Three";
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
}

TEST_CASE("Text inputs settle under identical updates and repaint on value changes",
          "[core][sakura][retention]") {
    Scene scene;
    Sakura::TextInput input;
    Place place(input, {10.0f, 10.0f, 200.0f, 30.0f});
    Frames frames(place, kScreen, scene.context());
    std::string value;
    const auto update = [&] {
        input.update({.value = value, .hint = "Search"});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
    CHECK(QuietMoves(frames, update, {{50.0f, 20.0f}, {60.0f, 20.0f}, {400.0f, 300.0f}}, [&](const MouseEvent& e) {
        return place.send(e);
    }));

    value = "query";
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    CHECK(frames.restructures == 0);
}

TEST_CASE("Text inputs repaint only when their password or disabled state changes",
          "[core][sakura][retention]") {
    Scene scene;
    Sakura::TextInput input;
    Place place(input, {10.0f, 10.0f, 200.0f, 30.0f});
    Frames frames(place, kScreen, scene.context());
    bool password = false;
    bool disabled = false;
    const auto update = [&] {
        input.update({.value = "secret", .password = password, .disabled = disabled});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));

    password = true;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));

    disabled = true;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    CHECK(QuietMoves(frames, update, {{50.0f, 20.0f}, {60.0f, 20.0f}}, [&](const MouseEvent& e) {
        return place.send(e);
    }));

    disabled = false;
    password = false;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    CHECK(frames.restructures == 0);
}

TEST_CASE("Code editors report their content overflow only when it changes", "[core][sakura][retention]") {
    Scene scene;
    Sakura::CodeEditor view;
    Place place(view, {10.0f, 10.0f, 300.0f, 120.0f});
    Frames frames(place, kScreen, scene.context());
    frames.verify = false;
    std::string value = "x = 1\n";
    std::vector<Extent2D<F32>> reports;
    const auto update = [&] {
        view.update({.id = "code", .value = value, .height = 120.0f,
                     .onContentOverflow = [&](Extent2D<F32> overflow) { reports.push_back(overflow); }});
    };
    REQUIRE(frames.settle(update));
    REQUIRE_FALSE(reports.empty());
    const U64 settled = reports.size();
    const F32 first = reports.back().y;
    CHECK(first < 0.0f);
    CHECK(Retains(frames, update));
    CHECK(reports.size() == settled);

    for (int line = 0; line < 30; ++line) {
        value += "y = 2\n";
    }
    CHECK(RepaintOnly(frames.run(update)));
    REQUIRE(reports.size() == settled + 1);
    CHECK(reports.back().y > 0.0f);
    REQUIRE(frames.settle(update));
    CHECK(reports.size() == settled + 1);
}

TEST_CASE("Markdown editors report their vertical content overflow only when it changes",
          "[core][sakura][retention]") {
    Scene scene;
    Sakura::MarkdownEditor view;
    Place place(view, {10.0f, 10.0f, 300.0f, 160.0f});
    Frames frames(place, kScreen, scene.context());
    frames.verify = false;
    std::string value = "# Note";
    std::vector<Extent2D<F32>> reports;
    const auto update = [&] {
        view.update({.id = "note", .value = value,
                     .onContentOverflow = [&](Extent2D<F32> overflow) { reports.push_back(overflow); }});
    };
    REQUIRE(frames.settle(update));
    REQUIRE_FALSE(reports.empty());
    const U64 settled = reports.size();
    CHECK(reports.back().x == 0.0f);
    CHECK(reports.back().y < 0.0f);
    CHECK(Retains(frames, update));
    CHECK(reports.size() == settled);

    for (int line = 0; line < 30; ++line) {
        value += "\n\nParagraph.";
    }
    frames.run(update);
    REQUIRE(frames.settle(update));
    CHECK(reports.size() > settled);
    CHECK(reports.back().y > 0.0f);
    const U64 grown = reports.size();
    CHECK(Retains(frames, update));
    CHECK(reports.size() == grown);
}

TEST_CASE("Typing the first character into an empty text input hides its hint without another layout trigger",
          "[core][sakura][retention]") {
    Scene scene;
    Sakura::TextInput input;
    Place place(input, {10.0f, 10.0f, 200.0f, 30.0f});
    Frames frames(place, kScreen, scene.context());
    frames.verify = false;
    frames.ctx.windowFocused = false;
    const auto update = [&] {
        input.update({.value = "", .hint = "Search", .submit = Sakura::TextInput::Submit::OnCommit, .focus = true});
    };
    REQUIRE(frames.settle(update));
    REQUIRE(input.focused());
    REQUIRE(HintOf(place, "Search").has_value());
    CHECK(HintOf(place, "Search")->visible);

    Type(place, "a");
    CHECK(RepaintOnly(frames.run(update)));
    REQUIRE(HintOf(place, "Search").has_value());
    CHECK_FALSE(HintOf(place, "Search")->visible);
}

TEST_CASE("Enabling child clipping after placement recomputes the clips of placed children",
          "[core][sakura][retention]") {
    Probe child({300.0f, 300.0f});
    Clipper clipper(child);
    Place place(clipper, {10.0f, 10.0f, 100.0f, 100.0f});
    Frames frames(place);
    frames.verify = false;
    REQUIRE(frames.settle());
    CHECK(child.currentClip() == kScreen);

    clipper.clipping(true);
    frames.run();
    CHECK(child.currentClip() == Rect{10.0f, 10.0f, 100.0f, 100.0f});
    CHECK(child.laidClip == Rect{10.0f, 10.0f, 100.0f, 100.0f});

    clipper.clipping(false);
    frames.run();
    CHECK(child.currentClip() == kScreen);
}

TEST_CASE("Focused text inputs repaint only on caret blinks", "[core][sakura][retention]") {
    Scene scene;
    Sakura::TextInput input;
    Place place(input, {10.0f, 10.0f, 200.0f, 30.0f});
    Frames frames(place, kScreen, scene.context());
    frames.ctx.windowFocused = true;
    const auto update = [&] {
        input.update({.value = "query", .focus = true});
    };
    REQUIRE(frames.settle(update));
    REQUIRE(input.focused());
    CHECK(Retains(frames, update));

    frames.ctx.time = 0.6;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    CHECK(frames.restructures == 0);
}

TEST_CASE("Setting fields settle and repaint when the description changes", "[core][sakura][retention]") {
    Scene scene;
    Sakura::Toggle control;
    Sakura::SettingField field;
    Place place(field, {10.0f, 10.0f, 300.0f, 200.0f});
    Frames frames(place, kScreen, scene.context());
    std::string description = "Applies on restart.";
    const auto update = [&] {
        control.update({.value = true});
        field.update({.label = "Telemetry", .description = description, .control = &control});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));

    description = "Applies immediately.";
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    CHECK(frames.restructures == 0);
}

TEST_CASE("Code editor views settle under identical updates", "[core][sakura][retention]") {
    Scene scene;
    Sakura::CodeEditor view;
    Place place(view, {10.0f, 10.0f, 400.0f, 300.0f});
    Frames frames(place, kScreen, scene.context());
    std::string status = "Ready";
    std::vector<std::string> output = {"hello"};
    const auto update = [&] {
        view.update({.id = "code",
                     .value = "print('hello')\n",
                     .consoleOutput = output,
                     .status = status,
                     .height = 300.0f,
                     .consoleVisible = true,
                     .onChange = [](std::string) {}});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
    CHECK(QuietMoves(frames, update, {{50.0f, 50.0f}, {60.0f, 60.0f}}, [&](const MouseEvent& e) {
        return place.send(e);
    }));

    status = "Error";
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));

    output.push_back("world");
    CHECK(Unpainted(frames.run(update)));
    CHECK(Retains(frames, update));
    CHECK(frames.restructures == 0);
}

TEST_CASE("Expanding a code editor console repaints without restructuring", "[core][sakura][retention]") {
    Scene scene;
    Sakura::CodeEditor view;
    Place place(view, {10.0f, 10.0f, 400.0f, 300.0f});
    Frames frames(place, kScreen, scene.context());
    const auto update = [&] {
        view.update({.id = "code",
                     .value = "print('hello')\n",
                     .consoleOutput = {"hello"},
                     .status = "Ready",
                     .height = 300.0f,
                     .consoleVisible = true});
    };
    REQUIRE(frames.settle(update));
    CHECK(place.send(Mouse(MouseEventType::Click, 395.0f, 298.0f)));
    const auto expanded = frames.run(update);
    CHECK_FALSE(expanded.rebuilt);
    CHECK(expanded.repainted);
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
    CHECK(frames.restructures == 0);
}

TEST_CASE("Console views settle and repaint on new output", "[core][sakura][retention]") {
    Scene scene;
    Sakura::Console view;
    Place place(view, {10.0f, 10.0f, 400.0f, 160.0f});
    Frames frames(place, kScreen, scene.context());
    std::vector<std::string> output;
    const auto update = [&] {
        view.update({.id = "console", .output = output, .status = "Idle"});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));

    output = {"first", "second"};
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    CHECK(frames.restructures == 0);
}

TEST_CASE("Markdown editor views settle in preview and editing modes", "[core][sakura][retention]") {
    Scene scene;
    Sakura::MarkdownEditor view;
    Place place(view, {10.0f, 10.0f, 400.0f, 300.0f});
    Frames frames(place, kScreen, scene.context());
    bool editing = false;
    const auto update = [&] {
        view.update(
            {.id = "note", .value = "# Note\n\nBody text.", .editing = editing, .onChange = [](std::string) {}});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
    CheckHoverRepaintsOnce(frames, place, update, {200.0f, 290.0f}, {600.0f, 500.0f});
    CheckLeaveRepaintsOnce(frames, place, update, {200.0f, 290.0f});

    editing = true;
    const auto opened = frames.run(update);
    CHECK(opened.restructured);
    CHECK_FALSE(opened.rebuilt);
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));

    editing = false;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
}

TEST_CASE("A composed control panel stays retained under identical updates", "[core][sakura][retention]") {
    Scene scene;
    Sakura::Text title;
    Sakura::Button button;
    Sakura::Toggle checkbox;
    Sakura::ProgressBar progress;
    Sakura::TextInput input;
    Sakura::Stack stack;
    Sakura::Panel panel;
    Place place(panel, {10.0f, 10.0f, 360.0f, 400.0f});
    Frames frames(place, kScreen, scene.context());
    F32 value = 0.4f;
    bool enabled = true;
    const auto update = [&] {
        title.update({.str = "Receiver", .font = Sakura::Text::Font::H2});
        button.update({.str = "Start", .variant = Sakura::Button::Variant::Action, .onClick = [] {}});
        checkbox.update({.value = enabled, .onChange = [&](bool next) { enabled = next; }});
        progress.update({.value = value, .overlay = "Buffer"});
        input.update({.value = "100.0 MHz", .hint = "Frequency", .onChange = [](const std::string&) {}});
        stack.update({.items = {{.child = &title},
                                {.child = &button},
                                {.child = &checkbox},
                                {.child = &progress},
                                {.child = &input}}});
        panel.update({.id = "receiver", .content = &stack});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update, 10));
    CHECK(frames.rebuilds == 1);
    CHECK(frames.restructures == 0);
    for (int frame = 0; frame < 5; ++frame) {
        frames.ctx.time += 0.4;
        CHECK(frames.run(update).clean());
    }
    CHECK(QuietMoves(frames, update, {{300.0f, 30.0f}, {310.0f, 30.0f}, {600.0f, 500.0f}}, [&](const MouseEvent& e) {
        return place.send(e);
    }));

    value = 0.8f;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));

    enabled = false;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));

    checkbox.setVisible(false);
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    checkbox.setVisible(true);
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    CHECK(frames.rebuilds == 1);
    CHECK(frames.restructures == 0);
}

TEST_CASE("Grids restructure once when an item is added and park removed items with a repaint",
          "[core][sakura][retention]") {
    Scene scene;
    std::vector<std::unique_ptr<Sakura::Text>> items;
    for (int index = 0; index < 4; ++index) {
        items.push_back(std::make_unique<Sakura::Text>());
    }
    Sakura::Grid grid;
    Place place(grid, {10.0f, 10.0f, 400.0f, 200.0f});
    Frames frames(place, kScreen, scene.context());
    U64 count = 3;
    const auto update = [&] {
        std::vector<Sakura::Component*> cells;
        for (U64 index = 0; index < count; ++index) {
            items[index]->update({.str = "Cell " + std::to_string(index)});
            cells.push_back(items[index].get());
        }
        grid.update({.columns = 2, .items = cells});
    };
    REQUIRE(frames.settle(update));
    REQUIRE(frames.restructures == 0);

    count = 4;
    const auto added = frames.run(update);
    CHECK(added.restructured);
    CHECK_FALSE(added.rebuilt);
    CHECK(Retains(frames, update));
    CHECK(frames.restructures == 1);

    count = 2;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));

    count = 4;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    CHECK(frames.restructures == 1);
    CHECK(frames.rebuilds == 1);
}

TEST_CASE("Text components hidden in a stack repaint without rebuilding when shown again",
          "[core][sakura][retention]") {
    Scene scene;
    Sakura::Text text;
    Sakura::TextView view;
    Sakura::TextGrid grid;
    Sakura::Markdown markdown;
    Sakura::Stack stack;
    Place place(stack, {10.0f, 10.0f, 400.0f, 500.0f});
    Frames frames(place, kScreen, scene.context());
    const auto update = [&] {
        text.update({.str = "Sample rate"});
        view.update({.id = "view", .value = "first\nsecond", .lineNumbers = true});
        grid.update({.id = "grid", .value = "alpha\nbeta"});
        markdown.update({.value = "# Title\n\nSome **bold** text."});
        stack.update({.items = {{.child = &text, .size = 20.0f},
                                {.child = &view, .size = 80.0f},
                                {.child = &grid, .size = 80.0f},
                                {.child = &markdown, .size = 120.0f}}});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));

    for (Sakura::Component* component : std::vector<Sakura::Component*>{&text, &view, &grid, &markdown}) {
        component->setVisible(false);
        CHECK(RepaintOnly(frames.run(update)));
        CHECK(Retains(frames, update));

        component->setVisible(true);
        CHECK(RepaintOnly(frames.run(update)));
        CHECK(Retains(frames, update));
    }
    CHECK(frames.rebuilds == 1);
    CHECK(frames.restructures == 0);
}

TEST_CASE("Pill fields settle, ignore hover, restructure once on the first edit, and repaint on typing",
          "[core][sakura][retention]") {
    Scene scene;
    auto ctx = scene.context();
    ctx.palette = DistinctPalette();
    ctx.windowFocused = false;
    Sakura::PillField pill;
    Place place(pill, {10.0f, 10.0f, 240.0f, 34.0f});
    Frames frames(place, kScreen, ctx);
    bool editing = false;
    std::string query = "fil";
    const auto update = [&] {
        pill.update({.icon = "Q",
                     .placeholder = "Search blocks",
                     .query = query,
                     .editing = editing,
                     .size = {240.0f, 34.0f},
                     .onOpen = [] {},
                     .onChange = [&](const std::string& next) { query = next; }});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
    CHECK(QuietMoves(frames, update, {{50.0f, 27.0f}, {60.0f, 27.0f}, {400.0f, 300.0f}, {50.0f, 27.0f}},
                     [&](const MouseEvent& e) { return place.send(e); }));
    place.send(Mouse(MouseEventType::Leave, 50.0f, 27.0f));
    CHECK(frames.run(update).clean());

    query = "filt";
    CHECK(Unpainted(frames.run(update)));

    query.clear();
    editing = true;
    const auto opened = frames.run(update);
    CHECK(opened.restructured);
    CHECK_FALSE(opened.rebuilt);
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));

    Type(place, "fft");
    CHECK(query == "fft");
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));

    query = "external";
    CHECK(Unpainted(frames.run(update)));
    query = "fft";

    editing = false;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));

    editing = true;
    CHECK(RepaintOnly(frames.run(update)));
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
    CHECK(frames.restructures == 1);
    CHECK(frames.rebuilds == 1);
}

TEST_CASE("Pill buttons settle, repaint once per hover transition, and repaint on label changes",
          "[core][sakura][retention]") {
    Scene scene;
    auto ctx = scene.context();
    ctx.palette = DistinctPalette();
    Sakura::PillButton button;
    Place place(button, {10.0f, 10.0f, 120.0f, 28.0f});
    Frames frames(place, kScreen, ctx);
    std::string label = "Add";
    const auto update = [&] {
        button.update({.icon = "+", .label = label, .onClick = [] {}});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
    CheckHoverRepaintsOnce(frames, place, update, {50.0f, 24.0f}, {400.0f, 300.0f});
    CheckLeaveRepaintsOnce(frames, place, update, {50.0f, 24.0f});

    label = "Remove";
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    CHECK(frames.restructures == 0);
    CHECK(frames.rebuilds == 1);
}

TEST_CASE("Pill steppers repaint without restructuring when their label changes", "[core][sakura][retention]") {
    Scene scene;
    auto ctx = scene.context();
    ctx.palette = DistinctPalette();
    Sakura::PillStepper stepper;
    Place place(stepper, {10.0f, 10.0f, 120.0f, 34.0f});
    Frames frames(place, kScreen, ctx);
    std::string label = "100%";
    const auto update = [&] {
        stepper.update({.label = label, .size = {120.0f, 34.0f}, .onDecrement = [] {}, .onIncrement = [] {}});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));

    label = "125%";
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    CHECK(frames.restructures == 0);
    CHECK(frames.rebuilds == 1);
}

TEST_CASE("Search catalog query changes repaint and only a new docs button restructures",
          "[core][sakura][retention]") {
    Scene scene;
    auto ctx = scene.context();
    ctx.palette = DistinctPalette();
    Sakura::SearchCatalog catalog;
    Place place(catalog, {0.0f, 0.0f, 748.0f, 360.0f});
    Frames frames(place, kScreen, ctx);
    std::string query = "filter";
    const std::vector<Sakura::SearchCatalog::Item> items = {
        {.id = "fft", .title = "FFT", .summary = "Fast Fourier transform", .group = "DSP", .badges = {"CPU", "Metal"},
         .documentation = "# FFT"},
        {.id = "filter", .title = "Filter", .summary = "FIR taps", .group = "DSP", .badges = {"CPU"},
         .documentation = "# Filter"},
        {.id = "file", .title = "File Reader", .summary = "Reads samples", .group = "IO", .badges = {"CPU"}},
    };
    const auto update = [&] {
        catalog.update({.items = items, .groups = {"DSP", "IO"}, .query = query});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
    REQUIRE(catalog.results() == std::vector<std::string>{"filter"});

    query = "fi";
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));

    query = "zzz";
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    CHECK(catalog.results().empty());

    query = "filter";
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    CHECK(frames.restructures == 0);

    query = "";
    const auto grown = frames.run(update);
    CHECK(grown.restructured);
    CHECK_FALSE(grown.rebuilt);
    CHECK(Retains(frames, update));
    CHECK(frames.restructures == 1);

    query = "filter";
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    CHECK(frames.restructures == 1);
    CHECK(frames.rebuilds == 1);
}

TEST_CASE("Scroller springs keep animating after a same-frame relayout and settle without input",
          "[core][sakura][retention]") {
    Scene scene;
    Sakura::Scroller scroll;
    Place place(scroll, {10.0f, 10.0f, 200.0f, 100.0f});
    Frames frames(place, kScreen, scene.context());
    frames.verify = false;
    frames.timeStep = 1.0 / 60.0;
    F32 scrollY = 0.0f;
    const auto update = [&] {
        scroll.update({.id = "spring", .contentHeight = 600.0f, .scrollY = scrollY,
                       .onScrollY = [&](F32 next) { scrollY = next; }});
    };
    REQUIRE(frames.settle(update));
    const auto thumb = [&] {
        return Direct<Sakura::Box>(scroll)[1]->config().instances.at(0).rect.height;
    };
    const F32 resting = thumb();
    const auto send = [&](F32 dy, ScrollPhase phase) {
        auto event = Wheel(100.0f, 50.0f, dy);
        event.phase = phase;
        place.send(event);
    };

    send(0.0f, ScrollPhase::Began);
    send(4.0f, ScrollPhase::Changed);
    frames.run(update);
    REQUIRE(thumb() < resting);
    send(0.0f, ScrollPhase::Ended);
    frames.run(update);
    frames.run(update);
    frames.run(update);
    REQUIRE(thumb() < resting);
    Sakura::Access::invalidateLayout(place);
    Sakura::Access::layout(place, frames.ctx, frames.bounds);

    F32 previous = thumb();
    U64 moving = 0;
    for (U64 frame = 0; frame < 90 && thumb() != resting; ++frame) {
        frames.run(update);
        moving += thumb() != previous ? 1 : 0;
        previous = thumb();
    }
    CHECK(moving > 0);
    CHECK(thumb() == resting);
}
