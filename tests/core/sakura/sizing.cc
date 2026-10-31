#include <catch2/catch_test_macros.hpp>

#include <jetstream/render/sakura/base.hh>
#include <jetstream/render/tools/icons_lucide.hh>

#include "component_harness.hh"
#include "fonts.hh"
#include "scene.hh"

#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace Jetstream;
using namespace SakuraTest;

namespace {

struct Subject {
    std::string name;
    std::unique_ptr<Sakura::Component> component;
};

template<typename T>
Subject Make(const std::string& name, typename T::Config config) {
    auto component = std::make_unique<T>();
    component->update(std::move(config));
    return {name, std::move(component)};
}

template<typename T>
T* Own(std::vector<std::unique_ptr<Sakura::Component>>& owned, typename T::Config config) {
    auto component = std::make_unique<T>();
    component->update(std::move(config));
    T* raw = component.get();
    owned.push_back(std::move(component));
    return raw;
}

bool Finite(Extent2D<F32> size) {
    return std::isfinite(size.x) && std::isfinite(size.y) && size.x >= 0.0f && size.y >= 0.0f;
}

Sakura::Context Doubled(const Sakura::Context& ctx) {
    auto next = ctx;
    next.pixelRatio = 2.0f;
    return next;
}

F32 TextWidth(const Sakura::Context& ctx, Sakura::Text::Config config) {
    Sakura::Text text;
    text.update(std::move(config));
    Host host(text);
    return host.measureOf(ctx, {kInfinite, kInfinite}).x;
}

}  // namespace

TEST_CASE("Components measure to finite sizes within the available space", "[core][sakura][sizing]") {
    Scene scene;
    std::vector<std::unique_ptr<Probe>> probes;
    const auto probe = [&](Extent2D<F32> desired = {120.0f, 45.0f}) {
        probes.push_back(std::make_unique<Probe>(desired));
        return probes.back().get();
    };
    std::vector<std::unique_ptr<Sakura::Component>> owned;
    const auto nodeField = [&](const std::string& label) {
        return Own<Sakura::NodeField>(owned, {.label = label, .content = Own<Sakura::NodeTextInput>(owned, {.value = "1.0"})});
    };

    std::vector<Subject> subjects;
    subjects.push_back(Make<Sakura::Panel>("panel", {.content = probe()}));
    subjects.push_back(Make<Sakura::Panel>("natural panel", {.content = probe(), .size = {0.0f, 0.0f}}));
    subjects.push_back(Make<Sakura::Stack>("vertical stack", {.items = {{.child = probe()}, {.child = probe()}}}));
    subjects.push_back(Make<Sakura::Stack>("horizontal stack", {.axis = Sakura::Stack::Axis::Horizontal,
                                                                .items = {{.child = probe()}, {.child = probe()}}}));
    subjects.push_back(Make<Sakura::Div>("div", {.content = probe(), .padding = 4.0f}));
    subjects.push_back(Make<Sakura::ScrollArea>("scroll area", {.content = probe(), .height = 0.0f}));
    subjects.push_back(Make<Sakura::Grid>("grid", {.items = {probe(), probe(), probe()}}));
    subjects.push_back(Make<Sakura::Reveal>("reveal", {.child = probe(), .shown = true, .gap = 8.0f}));
    subjects.push_back(Make<Sakura::Spacer>("spacer", {.lines = 3}));
    subjects.push_back(Make<Sakura::Divider>("divider", {}));
    subjects.push_back(Make<Sakura::Button>("button", {.str = "Run"}));
    subjects.push_back(Make<Sakura::Button>("filling button", {.str = "Run", .size = {Sakura::Fill, Sakura::Fill}}));
    subjects.push_back(Make<Sakura::TextInput>("text input", {.value = "alpha"}));
    subjects.push_back(Make<Sakura::ProgressBar>("progress bar", {.value = 0.5f}));
    subjects.push_back(Make<Sakura::Combo>("combo", {.options = {"Alpha", "Beta"}, .value = "Alpha"}));
    subjects.push_back(Make<Sakura::Text>("text", {.str = "alpha beta gamma"}));
    subjects.push_back(Make<Sakura::Text>("wrapped text", {.str = "alpha beta gamma", .wrapped = true}));
    subjects.push_back(Make<Sakura::Markdown>("markdown", {.value = "# Title\n\nBody text."}));
    subjects.push_back(Make<Sakura::LogView>("log view", {.value = "first\nsecond", .height = 0.0f}));
    subjects.push_back(Make<Sakura::HighlightText>("highlight text", {.str = "alpha beta gamma",
                                                                      .marks = {{.begin = 6, .end = 10}}}));
    subjects.push_back(Make<Sakura::Console>("console", {.output = {"first", "second"}, .height = 0.0f}));
    subjects.push_back(Make<Sakura::CodeEditor>("code editor", {.value = "x = 1\ny = 2\n", .height = 0.0f}));
    subjects.push_back(Make<Sakura::Terminal>("terminal", {}));
    subjects.push_back(Make<Sakura::Terminal>("natural terminal", {.size = {0.0f, 0.0f}}));
    subjects.push_back(Make<Sakura::MarkdownEditor>("markdown editor", {.value = "Notes"}));
    subjects.push_back(Make<Sakura::PillField>("pill field", {.placeholder = "Search blocks"}));
    subjects.push_back(Make<Sakura::PillStepper>("pill stepper", {.label = "100%"}));
    subjects.push_back(Make<Sakura::PillIconButton>("pill icon button", {.icon = ICON_LC_INFO}));
    subjects.push_back(Make<Sakura::PillButton>("pill button", {.label = "Open"}));
    subjects.push_back(Make<Sakura::QrCode>("qr code", {.data = std::vector<U8>(21 * 21, 1), .width = 21}));
    subjects.push_back(Make<Sakura::TipBanner>("tip banner", {.str = "Drag blocks to move them.", .icon = ICON_LC_LIGHTBULB}));
    subjects.push_back(Make<Sakura::Table>("table", {.columns = {"Key", "Value"}, .rows = {{"Gain", "1.0"}}}));
    subjects.push_back(Make<Sakura::ChromeButton>("chrome button", {.label = "File"}));
    subjects.push_back(Make<Sakura::ChromeButton>("glyph chrome button", {.glyph = Sakura::ChromeButton::Glyph::Plus}));
    subjects.push_back(Make<Sakura::SearchCatalog>("search catalog", {
        .items = {{.id = "fft", .title = "FFT", .summary = "Fast transform", .group = "DSP",
                   .badges = {"CPU"}, .documentation = "# FFT\n\nTransforms samples."},
                  {.id = "filter", .title = "Filter", .summary = "Taps", .group = "DSP"}},
        .field = Own<Sakura::PillField>(owned, {.placeholder = "Search blocks"}),
    }));
    subjects.push_back(Make<Sakura::SearchCatalog>("empty search catalog", {.query = "nothing"}));
    subjects.push_back(Make<Sakura::NavigationItem>("navigation item", {.label = "General"}));
    subjects.push_back(Make<Sakura::SidebarItem>("sidebar item", {.icon = ICON_LC_SETTINGS, .label = "General"}));
    subjects.push_back(Make<Sakura::Toggle>("toggle", {.value = true}));
    subjects.push_back(Make<Sakura::Badge>("badge", {.label = "Valid"}));
    subjects.push_back(Make<Sakura::FlowgraphCard>("flowgraph card", {
        .diagram = "tone[Tone] --> add[Add] --> spectrum[Spectrum]\nnoise[Noise] --> add",
        .category = "Basics",
        .title = "First Steps",
        .description = "Build a spectrum analyzer.",
    }));
    subjects.push_back(Make<Sakura::PillTabs>("pill tabs", {
        .items = {{.icon = ICON_LC_LAYOUT_GRID, .label = "All"}, {.label = "Plugins"}},
    }));
    subjects.push_back(Make<Sakura::PillTabs>("compact pill tabs", {
        .items = {{.label = "All"}, {.label = "Basics"}, {.label = "Receivers"}},
        .selected = 1,
        .compact = true,
    }));
    subjects.push_back(Make<Sakura::WaveLine>("wave line", {.span = 0.6f}));
    subjects.push_back(Make<Sakura::FeatureCard>("feature card", {
        .title = "Update available",
        .description = "Restart to apply it.",
        .content = probe(),
    }));
    subjects.push_back(Make<Sakura::SettingCard>("setting card", {
        .rows = {
            {.label = "Automatic gain", .description = "Adjusts the gain to keep the signal level steady.",
             .control = Own<Sakura::Toggle>(owned, {.value = true})},
            {.label = "Sample rate", .control = Own<Sakura::Combo>(owned, {.options = {"1 MHz"}, .value = "1 MHz"})},
        },
    }));
    subjects.push_back(Make<Sakura::SettingField>("setting field", {
        .label = "Automatic gain",
        .description = "Adjusts the gain to keep the signal level steady.",
        .control = Own<Sakura::Toggle>(owned, {.value = true}),
    }));
    subjects.push_back(Make<Sakura::NodeLabel>("node label", {.str = "Gain"}));
    subjects.push_back(Make<Sakura::NodeLabel>("boxed node label", {.str = "Gain", .boxed = true}));
    subjects.push_back(Make<Sakura::NodeProgressBar>("node progress bar", {.value = 0.5f, .overlay = "50%"}));
    subjects.push_back(Make<Sakura::NodeTable>("node table", {.columns = {"Key", "Value"}, .rows = {{"Gain", "1.0"}}}));
    subjects.push_back(Make<Sakura::NodeTextInput>("node text input", {.value = "alpha"}));
    subjects.push_back(Make<Sakura::NodeFloatInput>("node float input", {.value = 1.0f}));
    subjects.push_back(Make<Sakura::NodeFloatInput>("stepped node float input", {.value = 1.0f, .step = 0.1f}));
    subjects.push_back(Make<Sakura::NodeBoolInput>("node bool input", {.value = true}));
    subjects.push_back(Make<Sakura::NodeCombo>("node combo", {.options = {"Alpha", "Beta"}, .value = "Alpha"}));
    subjects.push_back(Make<Sakura::NodeInputRow>("node input row", {
        .shape = "[8]",
        .axes = "0",
        .combos = {{.options = {"F32", "CF32"}, .value = "F32"}},
    }));
    {
        auto field = std::make_unique<Sakura::NodeField>();
        field->update({.label = "Gain", .content = Own<Sakura::NodeTextInput>(owned, {.value = "1.0"})});
        subjects.push_back({"node field", std::move(field)});
    }
    subjects.push_back(Make<Sakura::NodeFieldGrid>("node field grid", {
        .items = {{.child = nodeField("Rate")},
                  {.child = nodeField("Gain")},
                  {.child = nodeField("Mode"), .fullWidth = true}},
    }));
    subjects.push_back(Make<Sakura::NodeContentStack>("node content stack", {
        .items = {{.child = probe()}, {.child = probe(), .flex = Sakura::NodeContentStack::Flex{.minimum = 30.0f}}},
    }));
    subjects.push_back(Make<Sakura::NodeFieldPanel>("node field panel", {
        .content = Own<Sakura::NodeContentStack>(owned, {.items = {{.child = probe()}, {.child = nodeField("Rate")}}}),
    }));

    subjects.push_back(Make<Sakura::ModelPicker>("model picker", {
        .models = {{.id = "a", .label = "Alpha", .detail = "local"}, {.id = "b", .label = "Beta"}},
        .value = "a",
        .efforts = {{.id = "low", .label = "Low"}, {.id = "high", .label = "High"}},
        .effort = "high",
    }));
    subjects.push_back(Make<Sakura::ChatComposer>("chat composer", {
        .style = {.placeholder = "Ask anything..."},
        .selector = Sakura::ChatComposer::Selector{.options = {{.id = "a", .label = "Alpha"}}, .value = "a"},
        .usage = Sakura::ChatComposer::Usage{.value = 0.4f, .title = "Context window:"},
    }));
    subjects.push_back(Make<Sakura::ModelPicker>("empty model picker", {}));
    subjects.push_back(Make<Sakura::ChatComposer>("compact chat composer", {.style = {.compact = true}}));
    subjects.push_back(Make<Sakura::ChatComposer>("tall chat composer", {
        .style = {.minLines = 3.0f, .margin = {10.0f, 4.0f, 10.0f, 6.0f}},
        .onClear = [] {},
    }));
    {
        Sakura::Chat::Snapshot snapshot;
        snapshot.messages.push_back({.role = Sakura::Chat::Role::User, .text = "Hello"});
        snapshot.messages.push_back({.role = Sakura::Chat::Role::Assistant,
                                     .parts = {{.kind = Sakura::Chat::PartKind::Markdown, .key = 0, .revision = 1,
                                                .text = "Hi there."}}});
        subjects.push_back(Make<Sakura::Chat>("narrow chat", {.id = "narrow", .snapshot = snapshot, .maxContentWidth = 320.0f}));
        subjects.push_back(Make<Sakura::Chat>("large chat", {.id = "large", .snapshot = snapshot, .fontSize = 22.0f}));
        subjects.push_back(Make<Sakura::Chat>("chat", {.id = "chat", .snapshot = std::move(snapshot)}));
    }
    subjects.push_back(Make<Sakura::Voice>("voice", {}));
    subjects.push_back(Make<Sakura::Spinner>("spinner", {}));
    subjects.push_back(Make<Sakura::Spinner>("filling spinner", {.diameter = Sakura::Fill}));

    const std::vector<Extent2D<F32>> spaces = {
        {kInfinite, kInfinite}, {300.0f, kInfinite}, {kInfinite, 200.0f}, {300.0f, 200.0f}, {40.0f, 10.0f},
    };
    for (auto& subject : subjects) {
        Host host(*subject.component);
        for (const auto& available : spaces) {
            CAPTURE(subject.name, available.x, available.y);
            const auto measured = host.measureOf(scene.ctx, available);
            CHECK(Finite(measured));
            CHECK(measured.x <= available.x);
            CHECK(measured.y <= available.y);
        }
    }
}

TEST_CASE("Expanded reveals measure within the available space and rewrap text at capped widths",
          "[core][sakura][sizing]") {
    Scene scene;
    Probe probe({120.0f, 45.0f});
    probe.prepare();
    Sakura::Text text;
    text.update({.str = "alpha beta gamma delta epsilon zeta eta theta iota kappa", .wrapped = true});
    Sakura::Reveal probed;
    probed.update({.child = &probe, .shown = true, .gap = 8.0f});
    Sakura::Reveal wrapped;
    wrapped.update({.child = &text, .shown = true, .gap = 8.0f});
    Host probedHost(probed);
    Host wrappedHost(wrapped);
    auto ctx = scene.ctx;
    for (auto* host : {&probedHost, &wrappedHost}) {
        ctx.time = 1.0;
        host->place(ctx, {0.0f, 0.0f, 300.0f, 200.0f});
        ctx.time = 2.0;
        host->place(ctx, {0.0f, 0.0f, 300.0f, 200.0f});
    }
    CHECK(probedHost.measureOf(ctx, {300.0f, kInfinite}).y == 53.0f);

    const std::vector<Extent2D<F32>> spaces = {
        {kInfinite, kInfinite}, {300.0f, kInfinite}, {kInfinite, 200.0f}, {300.0f, 200.0f}, {40.0f, 10.0f},
    };
    for (auto* host : {&probedHost, &wrappedHost}) {
        for (const auto& available : spaces) {
            CAPTURE(host == &probedHost, available.x, available.y);
            const auto measured = host->measureOf(ctx, available);
            CHECK(Finite(measured));
            CHECK(measured.x <= available.x);
            CHECK(measured.y <= available.y);
        }
    }

    const F32 wide = wrappedHost.measureOf(ctx, {600.0f, kInfinite}).y;
    const F32 narrow = wrappedHost.measureOf(ctx, {80.0f, kInfinite}).y;
    CHECK(wide > 8.0f);
    CHECK(narrow > wide);
}

TEST_CASE("Sizes resolve to natural, fixed, or filling lengths", "[core][sakura][sizing]") {
    Scene scene;
    Probe content({120.0f, 45.0f});
    Sakura::Div div;
    Host host(div);

    div.update({.content = &content, .size = {0.0f, 0.0f}});
    CHECK(host.measureOf(scene.ctx, {300.0f, 200.0f}) == Extent2D<F32>{120.0f, 45.0f});
    CHECK(host.measureOf(scene.ctx, {100.0f, 200.0f}) == Extent2D<F32>{100.0f, 45.0f});

    div.update({.content = &content, .size = {80.0f, 30.0f}});
    CHECK(host.measureOf(scene.ctx, {300.0f, 200.0f}) == Extent2D<F32>{80.0f, 30.0f});
    CHECK(host.measureOf(scene.ctx, {60.0f, 20.0f}) == Extent2D<F32>{60.0f, 20.0f});

    div.update({.content = &content, .size = {Sakura::Fill, Sakura::Fill}});
    CHECK(host.measureOf(scene.ctx, {300.0f, 200.0f}) == Extent2D<F32>{300.0f, 200.0f});
    CHECK(host.measureOf(scene.ctx, {kInfinite, kInfinite}) == Extent2D<F32>{120.0f, 45.0f});
}

TEST_CASE("Filling containers claim no space when the axis is unbounded", "[core][sakura][sizing]") {
    Scene scene;
    Probe content({120.0f, 45.0f});
    Sakura::Panel panel;
    Host host(panel);

    panel.update({.content = &content, .size = {Sakura::Fill, Sakura::Fill}});
    CHECK(host.measureOf(scene.ctx, {300.0f, 200.0f}) == Extent2D<F32>{300.0f, 200.0f});
    CHECK(host.measureOf(scene.ctx, {kInfinite, kInfinite}) == Extent2D<F32>{0.0f, 0.0f});

    Probe filler({0.0f, 0.0f});
    Sakura::Stack stack;
    stack.update({.items = {{.child = &panel, .grow = 1.0f}, {.child = &filler}}});
    Host stackHost(stack);
    stackHost.place(scene.ctx, {0.0f, 0.0f, 300.0f, 400.0f});
    CHECK(content.laidFrame.height == 400.0f - Sakura::Stack::Spacing);
}

TEST_CASE("Filling buttons fall back to their natural size when unbounded", "[core][sakura][sizing]") {
    Scene scene;
    Sakura::Button button;
    Host host(button);

    button.update({.str = "Run"});
    const auto natural = host.measureOf(scene.ctx, {kInfinite, kInfinite});
    REQUIRE(natural.x > 0.0f);

    button.update({.str = "Run", .size = {Sakura::Fill, 0.0f}});
    CHECK(host.measureOf(scene.ctx, {kInfinite, kInfinite}) == natural);
    CHECK(host.measureOf(scene.ctx, {300.0f, kInfinite}).x == 300.0f);
}

TEST_CASE("Pill steppers measure their label before the first layout", "[core][sakura][sizing]") {
    Scene scene;
    Sakura::PillStepper narrow;
    Sakura::PillStepper wide;
    Host narrowHost(narrow);
    Host wideHost(wide);

    narrow.update({.label = "1%"});
    wide.update({.label = "1000%"});
    CHECK(wideHost.measureOf(scene.ctx, {kInfinite, kInfinite}).x >
          narrowHost.measureOf(scene.ctx, {kInfinite, kInfinite}).x);
}

TEST_CASE("Auto height code editors measure their content before the first layout", "[core][sakura][sizing]") {
    Scene scene;
    Sakura::CodeEditor editor;
    Host host(editor);

    editor.update({.value = "a\nb\nc\nd\ne\nf\ng\nh\n", .height = 0.0f, .maxAutoHeightWindowRatio = 0.0f});
    const auto first = host.measureOf(scene.ctx, {400.0f, kInfinite});
    host.place(scene.ctx, {0.0f, 0.0f, 400.0f, first.y});
    CHECK(host.measureOf(scene.ctx, {400.0f, kInfinite}) == first);

    editor.update({.value = "a\n", .height = 0.0f, .maxAutoHeightWindowRatio = 0.0f});
    CHECK(host.measureOf(scene.ctx, {400.0f, kInfinite}).y < first.y);
}

TEST_CASE("Natural width divs measure wrapped content at their capped width", "[core][sakura][sizing]") {
    Scene scene;
    Flow content(300.0f, 15.0f);
    Sakura::Div div;
    div.update({.content = &content, .size = {0.0f, 0.0f}});
    Host host(div);

    CHECK(host.measureOf(scene.ctx, {kInfinite, kInfinite}) == Extent2D<F32>{300.0f, 15.0f});
    CHECK(host.measureOf(scene.ctx, {100.0f, kInfinite}) == Extent2D<F32>{100.0f, 45.0f});
}

TEST_CASE("Tables measure rows at their capped width", "[core][sakura][sizing]") {
    Scene scene;
    Flow wide(300.0f, 15.0f);
    Flow wider(300.0f, 15.0f);
    Sakura::Table table;
    Host host(table);

    for (const auto size : {Extent2D<F32>{0.0f, 0.0f}, Extent2D<F32>{400.0f, 0.0f}}) {
        CAPTURE(size.x);
        table.update({.columns = {"Value"}, .cellRows = {{&wide}, {&wider}}, .size = size, .wrapped = true});
        const auto loose = host.measureOf(scene.ctx, {kInfinite, kInfinite});
        const auto tight = host.measureOf(scene.ctx, {150.0f, kInfinite});
        CHECK(tight.x == 150.0f);
        CHECK(tight.y > loose.y);
    }
}

TEST_CASE("Scroll areas scroll by a wheel step that follows the pixel ratio", "[core][sakura][sizing]") {
    Scene scene;
    const auto scrolled = [&](F32 pixelRatio) {
        Probe content({100.0f, 4000.0f});
        content.consume = false;
        Sakura::ScrollArea area;
        area.update({.content = &content, .height = 100.0f});
        Host host(area);
        auto ctx = scene.ctx;
        ctx.pixelRatio = pixelRatio;
        host.place(ctx, {0.0f, 0.0f, 300.0f, 200.0f});
        const F32 before = content.laidFrame.y;
        host.send(Wheel(50.0f, 50.0f, -1.0f));
        host.place(ctx, {0.0f, 0.0f, 300.0f, 200.0f});
        return before - content.laidFrame.y;
    };
    const F32 single = scrolled(1.0f);
    REQUIRE(single > 0.0f);
    CHECK(scrolled(2.0f) == 2.0f * single);
}

TEST_CASE("Resized table columns keep their logical width across pixel ratios", "[core][sakura][sizing]") {
    Scene scene;
    Probe first({20.0f, 15.0f});
    Probe second({20.0f, 15.0f});
    Sakura::Table table;
    table.update({.columns = {"Name", "Value"}, .cellRows = {{&first, &second}}, .showHeaders = false, .resizable = true});
    Host host(table);
    const Rect bounds = {0.0f, 0.0f, 400.0f, 100.0f};
    host.place(scene.ctx, bounds);

    const F32 border = second.laidFrame.x - 6.0f;
    host.send(Mouse(MouseEventType::Click, border, 10.0f));
    host.send(Mouse(MouseEventType::Move, border + 40.0f, 10.0f));
    host.send(Mouse(MouseEventType::Release, border + 40.0f, 10.0f));
    host.place(scene.ctx, bounds);
    const F32 single = first.laidFrame.width;

    auto doubled = scene.ctx;
    doubled.pixelRatio = 2.0f;
    host.place(doubled, bounds);
    CHECK(first.laidFrame.width == 2.0f * single);
}

TEST_CASE("Filling scroll areas take the available height and claim none when unbounded", "[core][sakura][sizing]") {
    Scene scene;
    Probe content({120.0f, 45.0f});
    Sakura::ScrollArea area;
    area.update({.content = &content, .height = Sakura::Fill});
    Host host(area);

    CHECK(host.measureOf(scene.ctx, {300.0f, 200.0f}) == Extent2D<F32>{300.0f, 200.0f});
    CHECK(host.measureOf(scene.ctx, {300.0f, kInfinite}) == Extent2D<F32>{300.0f, 0.0f});
}

TEST_CASE("Natural width grids fit their widest item per column and fill the height on request",
          "[core][sakura][sizing]") {
    Scene scene;
    Probe first({120.0f, 45.0f});
    Probe second({100.0f, 45.0f});
    Probe third({80.0f, 45.0f});
    Sakura::Grid grid;
    Host host(grid);

    grid.update({.size = {0.0f, 0.0f}, .items = {&first, &second, &third}});
    CHECK(host.measureOf(scene.ctx, {kInfinite, kInfinite}) == Extent2D<F32>{256.0f, 98.0f});
    CHECK(host.measureOf(scene.ctx, {200.0f, kInfinite}) == Extent2D<F32>{200.0f, 98.0f});

    grid.update({.size = {0.0f, Sakura::Fill}, .items = {&first, &second, &third}});
    CHECK(host.measureOf(scene.ctx, {400.0f, 150.0f}) == Extent2D<F32>{256.0f, 150.0f});
    CHECK(host.measureOf(scene.ctx, {kInfinite, kInfinite}) == Extent2D<F32>{256.0f, 0.0f});
}

TEST_CASE("Fixed width panels keep their width and measure content at it", "[core][sakura][sizing]") {
    Scene scene;
    Flow content(300.0f, 15.0f);
    Sakura::Panel panel;
    panel.update({.content = &content, .size = {150.0f, 0.0f}});
    Host host(panel);

    CHECK(host.measureOf(scene.ctx, {kInfinite, kInfinite}) == Extent2D<F32>{150.0f, 30.0f});
    CHECK(host.measureOf(scene.ctx, {400.0f, kInfinite}) == Extent2D<F32>{150.0f, 30.0f});
    CHECK(host.measureOf(scene.ctx, {100.0f, kInfinite}) == Extent2D<F32>{100.0f, 45.0f});
}

TEST_CASE("Progress bars and combos fall back to their natural width", "[core][sakura][sizing]") {
    Scene scene;
    Sakura::ProgressBar bar;
    Host barHost(bar);

    bar.update({.value = 0.5f, .size = {0.0f, 20.0f}});
    CHECK(barHost.measureOf(scene.ctx, {kInfinite, kInfinite}) == Extent2D<F32>{200.0f, 20.0f});
    CHECK(barHost.measureOf(scene.ctx, {300.0f, kInfinite}) == Extent2D<F32>{200.0f, 20.0f});
    CHECK(barHost.measureOf(scene.ctx, {150.0f, 10.0f}) == Extent2D<F32>{150.0f, 10.0f});

    bar.update({.value = 0.5f});
    CHECK(barHost.measureOf(scene.ctx, {kInfinite, kInfinite}) == Extent2D<F32>{200.0f, 20.0f});
    CHECK(barHost.measureOf(scene.ctx, {300.0f, kInfinite}) == Extent2D<F32>{300.0f, 20.0f});

    Sakura::Combo combo;
    Host comboHost(combo);
    combo.update({.options = {"Alpha", "Beta"}, .value = "Alpha", .width = 0.0f});
    CHECK(comboHost.measureOf(scene.ctx, {kInfinite, kInfinite}) == Extent2D<F32>{200.0f, 25.0f});
    CHECK(comboHost.measureOf(scene.ctx, {300.0f, kInfinite}) == Extent2D<F32>{200.0f, 25.0f});
    CHECK(comboHost.measureOf(scene.ctx, {150.0f, 20.0f}) == Extent2D<F32>{150.0f, 20.0f});

    combo.update({.options = {"Alpha", "Beta"}, .value = "Alpha", .width = 120.0f});
    CHECK(comboHost.measureOf(scene.ctx, {300.0f, kInfinite}) == Extent2D<F32>{120.0f, 25.0f});

    combo.update({.options = {"Alpha", "Beta"}, .value = "Alpha"});
    CHECK(comboHost.measureOf(scene.ctx, {300.0f, kInfinite}) == Extent2D<F32>{300.0f, 25.0f});
    CHECK(comboHost.measureOf(scene.ctx, {kInfinite, kInfinite}) == Extent2D<F32>{200.0f, 25.0f});
}

TEST_CASE("Filling pill fields take the available space", "[core][sakura][sizing]") {
    Scene scene;
    Sakura::PillField field;
    Host host(field);

    field.update({.placeholder = "Search blocks"});
    const auto natural = host.measureOf(scene.ctx, {kInfinite, kInfinite});
    REQUIRE(natural.x > 28.0f);
    CHECK(natural.y == 34.0f);

    field.update({.placeholder = "Search blocks", .size = {Sakura::Fill, Sakura::Fill}});
    CHECK(host.measureOf(scene.ctx, {300.0f, 200.0f}) == Extent2D<F32>{300.0f, 200.0f});
    CHECK(host.measureOf(scene.ctx, {300.0f, kInfinite}) == Extent2D<F32>{300.0f, 34.0f});
}

TEST_CASE("Natural width pill steppers fit their label and cap to the available width", "[core][sakura][sizing]") {
    Scene scene;
    Sakura::PillStepper stepper;
    Host host(stepper);

    stepper.update({.label = "100%"});
    const auto natural = host.measureOf(scene.ctx, {kInfinite, kInfinite});
    CHECK(natural.x > 70.0f);
    CHECK(natural.x == std::ceil(natural.x));
    CHECK(natural.y == 34.0f);
    CHECK(host.measureOf(scene.ctx, {300.0f, kInfinite}) == natural);
    CHECK(host.measureOf(scene.ctx, {50.0f, 20.0f}) == Extent2D<F32>{50.0f, 20.0f});

    stepper.update({.label = "100%", .size = {140.0f, 30.0f}});
    CHECK(host.measureOf(scene.ctx, {300.0f, kInfinite}) == Extent2D<F32>{140.0f, 30.0f});
}

TEST_CASE("Pill buttons take their fixed height", "[core][sakura][sizing]") {
    Scene scene;
    Sakura::PillButton button;
    Host host(button);

    button.update({.label = "Open", .height = 40.0f});
    const auto labeled = host.measureOf(scene.ctx, {kInfinite, kInfinite});
    CHECK(labeled.y == 40.0f);
    CHECK(labeled.x > 20.0f);
    CHECK(host.measureOf(scene.ctx, {300.0f, 30.0f}).y == 30.0f);

    button.update({.icon = "+", .height = 40.0f});
    CHECK(host.measureOf(scene.ctx, {kInfinite, kInfinite}) == Extent2D<F32>{40.0f, 40.0f});
}

TEST_CASE("Log views and consoles size their height naturally, fixed, or filling", "[core][sakura][sizing]") {
    Scene scene;
    Sakura::LogView log;
    Host logHost(log);

    log.update({.value = "first\nsecond", .height = 100.0f});
    CHECK(logHost.measureOf(scene.ctx, {300.0f, kInfinite}) == Extent2D<F32>{300.0f, 100.0f});
    CHECK(logHost.measureOf(scene.ctx, {300.0f, 50.0f}) == Extent2D<F32>{300.0f, 50.0f});

    log.update({.value = "first\nsecond", .height = Sakura::Fill});
    CHECK(logHost.measureOf(scene.ctx, {300.0f, 200.0f}) == Extent2D<F32>{300.0f, 200.0f});
    CHECK(logHost.measureOf(scene.ctx, {300.0f, kInfinite}) == Extent2D<F32>{300.0f, 0.0f});

    Sakura::Console console;
    Host consoleHost(console);

    console.update({.output = {"first", "second"}, .height = 0.0f});
    const auto two = consoleHost.measureOf(scene.ctx, {300.0f, kInfinite});
    CHECK(two.x == 300.0f);
    CHECK(two.y > 0.0f);
    CHECK(consoleHost.measureOf(scene.ctx, {300.0f, 10.0f}) == Extent2D<F32>{300.0f, 10.0f});
    console.update({.output = {"first", "second", "third", "fourth"}, .height = 0.0f});
    CHECK(consoleHost.measureOf(scene.ctx, {300.0f, kInfinite}).y > two.y);

    console.update({.output = {"first", "second"}, .height = 100.0f});
    CHECK(consoleHost.measureOf(scene.ctx, {300.0f, kInfinite}) == Extent2D<F32>{300.0f, 100.0f});
    CHECK(consoleHost.measureOf(scene.ctx, {300.0f, 50.0f}) == Extent2D<F32>{300.0f, 50.0f});

    console.update({.output = {"first", "second"}, .height = Sakura::Fill});
    CHECK(consoleHost.measureOf(scene.ctx, {300.0f, 200.0f}) == Extent2D<F32>{300.0f, 200.0f});
    CHECK(consoleHost.measureOf(scene.ctx, {300.0f, kInfinite}) == Extent2D<F32>{300.0f, 0.0f});
}

TEST_CASE("Filling code editors and text inputs take the available height", "[core][sakura][sizing]") {
    Scene scene;
    Sakura::CodeEditor editor;
    Host editorHost(editor);

    editor.update({.value = "x = 1\n", .height = Sakura::Fill});
    CHECK(editorHost.measureOf(scene.ctx, {300.0f, 200.0f}) == Extent2D<F32>{300.0f, 200.0f});
    CHECK(editorHost.measureOf(scene.ctx, {300.0f, kInfinite}) == Extent2D<F32>{300.0f, 0.0f});

    Sakura::TextInput input;
    Host inputHost(input);

    input.update({.value = "alpha", .height = Sakura::Fill});
    CHECK(inputHost.measureOf(scene.ctx, {300.0f, 200.0f}) == Extent2D<F32>{300.0f, 200.0f});
    CHECK(inputHost.measureOf(scene.ctx, {300.0f, kInfinite}) == Extent2D<F32>{300.0f, 25.0f});
    CHECK(inputHost.measureOf(scene.ctx, {kInfinite, kInfinite}) == Extent2D<F32>{200.0f, 25.0f});
}

TEST_CASE("Text and text views scale with the pixel ratio", "[core][sakura][sizing]") {
    Scene scene;
    const auto doubled = Doubled(scene.ctx);

    Sakura::Text text;
    Host textHost(text);
    text.update({.str = "alpha beta gamma"});
    const auto line = textHost.measureOf(scene.ctx, {kInfinite, kInfinite});
    REQUIRE(line.x > 0.0f);
    CHECK(textHost.measureOf(doubled, {kInfinite, kInfinite}) == Extent2D<F32>{2.0f * line.x, 2.0f * line.y});

    text.update({.str = "alpha beta gamma delta epsilon", .wrapped = true});
    const auto wrapped = textHost.measureOf(scene.ctx, {100.0f, kInfinite});
    REQUIRE(wrapped.y > line.y);
    CHECK(textHost.measureOf(doubled, {200.0f, kInfinite}) == Extent2D<F32>{200.0f, 2.0f * wrapped.y});

    Sakura::Markdown markdown;
    Host markdownHost(markdown);
    markdown.update({.value = "# Title\n\nBody text that wraps across a few lines of the view."});
    const auto document = markdownHost.measureOf(scene.ctx, {200.0f, kInfinite});
    REQUIRE(document.y > 0.0f);
    CHECK(markdownHost.measureOf(doubled, {400.0f, kInfinite}) == Extent2D<F32>{400.0f, 2.0f * document.y});

    Sakura::LogView log;
    Host logHost(log);
    log.update({.value = "first\nsecond line that wraps across the view", .height = 0.0f});
    const auto logged = logHost.measureOf(scene.ctx, {200.0f, kInfinite});
    REQUIRE(logged.y > 0.0f);
    CHECK(logHost.measureOf(doubled, {400.0f, kInfinite}) == Extent2D<F32>{400.0f, 2.0f * logged.y});

    Sakura::Console console;
    Host consoleHost(console);
    console.update({.output = {"first", "second"}, .status = "Done", .height = 0.0f});
    const auto output = consoleHost.measureOf(scene.ctx, {200.0f, kInfinite});
    REQUIRE(output.y > 0.0f);
    CHECK(consoleHost.measureOf(doubled, {400.0f, kInfinite}) == Extent2D<F32>{400.0f, 2.0f * output.y});

    Sakura::CodeEditor editor;
    Host editorHost(editor);
    editor.update({.value = "x = 1\ny = 2\n", .height = 0.0f, .maxAutoHeightWindowRatio = 0.0f});
    const auto code = editorHost.measureOf(scene.ctx, {200.0f, kInfinite});
    REQUIRE(code.y > 0.0f);
    CHECK(editorHost.measureOf(doubled, {400.0f, kInfinite}) == Extent2D<F32>{400.0f, 2.0f * code.y});

    Sakura::MarkdownEditor notes;
    Host notesHost(notes);
    notes.update({.value = "Notes\n\nSecond paragraph.", .maxAutoHeightWindowRatio = 0.0f});
    const auto note = notesHost.measureOf(scene.ctx, {200.0f, kInfinite});
    REQUIRE(note.y > 0.0f);
    const auto doubledNote = notesHost.measureOf(doubled, {400.0f, kInfinite});
    CHECK(doubledNote.x == 400.0f);
    CHECK(doubledNote.y == 2.0f * note.y - 1.0f);
}

TEST_CASE("Container paddings and widths scale with the pixel ratio", "[core][sakura][sizing]") {
    Scene scene;
    const auto doubled = Doubled(scene.ctx);
    Probe content({120.0f, 45.0f});

    Sakura::Div div;
    Host divHost(div);
    div.update({.content = &content, .size = {0.0f, 0.0f}, .padding = 4.0f});
    CHECK(divHost.measureOf(scene.ctx, {kInfinite, kInfinite}) == Extent2D<F32>{128.0f, 53.0f});
    CHECK(divHost.measureOf(doubled, {kInfinite, kInfinite}) == Extent2D<F32>{136.0f, 61.0f});

    Probe first({120.0f, 45.0f});
    Probe second({120.0f, 45.0f});
    Probe third({120.0f, 45.0f});
    Sakura::Grid grid;
    Host gridHost(grid);
    grid.update({.size = {0.0f, 0.0f}, .items = {&first, &second, &third}});
    CHECK(gridHost.measureOf(scene.ctx, {kInfinite, kInfinite}) == Extent2D<F32>{256.0f, 98.0f});
    CHECK(gridHost.measureOf(doubled, {kInfinite, kInfinite}) == Extent2D<F32>{272.0f, 106.0f});

    Probe panelContent({120.0f, 45.0f});
    Sakura::Panel panel;
    Host panelHost(panel);
    panel.update({.content = &panelContent, .maxWidth = 100.0f});
    CHECK(panelHost.measureOf(scene.ctx, {300.0f, kInfinite}) == Extent2D<F32>{100.0f, 45.0f});
    CHECK(panelHost.measureOf(doubled, {300.0f, kInfinite}) == Extent2D<F32>{200.0f, 45.0f});
}

TEST_CASE("Controls scale their frames and paddings with the pixel ratio", "[core][sakura][sizing]") {
    Scene scene;
    const auto doubled = Doubled(scene.ctx);
    const Extent2D<F32> unbounded = {kInfinite, kInfinite};

    Sakura::TextInput input;
    Host inputHost(input);
    input.update({.value = "alpha"});
    CHECK(inputHost.measureOf(scene.ctx, unbounded) == Extent2D<F32>{200.0f, 25.0f});
    CHECK(inputHost.measureOf(doubled, unbounded) == Extent2D<F32>{400.0f, 50.0f});
    input.update({.value = "alpha", .padding = {12.0f, 8.0f}});
    CHECK(inputHost.measureOf(scene.ctx, unbounded).y == 31.0f);
    CHECK(inputHost.measureOf(doubled, unbounded).y == 62.0f);

    Sakura::ProgressBar bar;
    Host barHost(bar);
    bar.update({.value = 0.5f});
    CHECK(barHost.measureOf(scene.ctx, unbounded) == Extent2D<F32>{200.0f, 20.0f});
    CHECK(barHost.measureOf(doubled, unbounded) == Extent2D<F32>{400.0f, 40.0f});

    Sakura::Combo combo;
    Host comboHost(combo);
    combo.update({.options = {"Alpha", "Beta"}, .value = "Alpha"});
    CHECK(comboHost.measureOf(scene.ctx, unbounded) == Extent2D<F32>{200.0f, 25.0f});
    CHECK(comboHost.measureOf(doubled, unbounded) == Extent2D<F32>{400.0f, 50.0f});

    Sakura::NavigationItem item;
    Host itemHost(item);
    item.update({.label = "General"});
    const auto navigation = itemHost.measureOf(scene.ctx, unbounded);
    REQUIRE(navigation.x > 24.0f);
    CHECK(itemHost.measureOf(doubled, unbounded) == Extent2D<F32>{2.0f * navigation.x, 2.0f * navigation.y});
    CHECK(itemHost.measureOf(doubled, {300.0f, kInfinite}) == Extent2D<F32>{300.0f, 2.0f * navigation.y});
}

TEST_CASE("Pill controls scale their paddings and heights with the pixel ratio", "[core][sakura][sizing]") {
    Scene scene;
    const auto doubled = Doubled(scene.ctx);
    const Extent2D<F32> unbounded = {kInfinite, kInfinite};

    Sakura::PillField field;
    Host fieldHost(field);
    field.update({.placeholder = "Search blocks"});
    const Sakura::Text::Config placeholder = {.str = "Search blocks", .tone = Sakura::Text::Tone::Secondary};
    CHECK(fieldHost.measureOf(scene.ctx, unbounded) ==
          Extent2D<F32>{std::ceil(28.0f + TextWidth(scene.ctx, placeholder)), 34.0f});
    CHECK(fieldHost.measureOf(doubled, unbounded) ==
          Extent2D<F32>{std::ceil(56.0f + TextWidth(doubled, placeholder)), 68.0f});

    Sakura::PillStepper stepper;
    Host stepperHost(stepper);
    stepper.update({.label = "100%"});
    const Sakura::Text::Config value = {.str = "100%",
                                        .font = Sakura::Text::Font::Display,
                                        .colorKey = "text_primary",
                                        .scale = 14.0f / Sakura::Typography::FontSize};
    CHECK(stepperHost.measureOf(scene.ctx, unbounded) ==
          Extent2D<F32>{std::ceil(70.0f + TextWidth(scene.ctx, value)), 34.0f});
    CHECK(stepperHost.measureOf(doubled, unbounded) ==
          Extent2D<F32>{std::ceil(140.0f + TextWidth(doubled, value)), 68.0f});

    Sakura::PillButton button;
    Host buttonHost(button);
    button.update({.label = "Open"});
    const Sakura::Text::Config label = {
        .str = "Open", .font = Sakura::Text::Font::Display, .colorKey = "text_primary", .scale = 14.0f / 15.0f};
    CHECK(buttonHost.measureOf(scene.ctx, unbounded) ==
          Extent2D<F32>{std::ceil(20.0f + TextWidth(scene.ctx, label)), 28.0f});
    CHECK(buttonHost.measureOf(doubled, unbounded) ==
          Extent2D<F32>{std::ceil(40.0f + TextWidth(doubled, label)), 56.0f});
}

TEST_CASE("Filling pill fields and steppers keep their natural width when unbounded", "[core][sakura][sizing]") {
    Scene scene;
    Sakura::PillField field;
    Host fieldHost(field);
    field.update({.placeholder = "Search blocks"});
    const auto natural = fieldHost.measureOf(scene.ctx, {kInfinite, kInfinite});
    REQUIRE(natural.x > 0.0f);
    field.update({.placeholder = "Search blocks", .size = {Sakura::Fill, Sakura::Fill}});
    CHECK(fieldHost.measureOf(scene.ctx, {kInfinite, kInfinite}) == natural);
    CHECK(fieldHost.measureOf(scene.ctx, {300.0f, 50.0f}) == Extent2D<F32>{300.0f, 50.0f});

    Sakura::PillStepper stepper;
    Host stepperHost(stepper);
    stepper.update({.label = "100%"});
    const auto stepped = stepperHost.measureOf(scene.ctx, {kInfinite, kInfinite});
    REQUIRE(stepped.x > 0.0f);
    stepper.update({.label = "100%", .size = {Sakura::Fill, 0.0f}});
    CHECK(stepperHost.measureOf(scene.ctx, {kInfinite, kInfinite}) == stepped);
    CHECK(stepperHost.measureOf(scene.ctx, {300.0f, kInfinite}).x == 300.0f);
}

TEST_CASE("Pill buttons resolve their height with the size convention", "[core][sakura][sizing]") {
    Scene scene;
    Sakura::PillButton button;
    Host host(button);
    button.update({.label = "Open", .height = Sakura::Fill});
    CHECK(host.measureOf(scene.ctx, {300.0f, 50.0f}).y == 50.0f);
    CHECK(host.measureOf(scene.ctx, {kInfinite, kInfinite}).y == 28.0f);
    button.update({.label = "Open", .height = 0.0f});
    CHECK(host.measureOf(scene.ctx, {kInfinite, kInfinite}).y == 28.0f);
}

TEST_CASE("Text grids cap their measured height to the available height", "[core][sakura][sizing]") {
    const auto ctx = MakeContext();
    Sakura::TextGrid grid;
    Host host(grid);
    grid.update({.id = "capped", .value = "a\nb\nc\nd\ne\nf\ng\nh\ni\nj"});
    const auto tall = host.measureOf(ctx, {300.0f, kInfinite});
    REQUIRE(tall.y > 40.0f);
    CHECK(host.measureOf(ctx, {300.0f, 40.0f}).y == 40.0f);
    CHECK(host.measureOf(ctx, {300.0f, kInfinite}).y == tall.y);
}
