#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <jetstream/render/sakura/components/surface_view.hh>
#include <jetstream/render/sakura/components/window.hh>
#include <jetstream/render/sakura/clipboard.hh>
#include <jetstream/render/sakura/components/retained/button.hh>
#include <jetstream/render/sakura/components/retained/canvas.hh>
#include <jetstream/render/sakura/components/retained/code_editor.hh>
#include <jetstream/render/sakura/components/retained/scroll_view.hh>
#include <jetstream/render/sakura/components/retained/text_grid.hh>
#include <imgui_internal.h>

#include "harness.hh"
#include "render/sakura/input.hh"
#include "render/sakura/detail/input_router.hh"
#include "superluminal/surface_interaction.hh"
#include "domains/visualization/frame/module_impl.hh"
#include "domains/visualization/signal_view/waterfall_3d.hh"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <vector>

using namespace Jetstream;

namespace {

struct InputSurface {
    std::vector<InputEvent> events;
    Sakura::SurfaceView view;
    detail::SurfaceInputState input;
    const char* id;
    ImVec2 origin;
    bool superluminal;

    InputSurface(const char* id, ImVec2 origin, bool superluminal)
        : id(id), origin(origin), superluminal(superluminal) {
        view.update({
            .id = id,
            .texture = 1,
            .size = {150.0f, 100.0f},
            .onInput = [this](InputEvent event) { events.push_back(event); },
        });
    }

    void render(const Sakura::Context& ctx) {
        ImGui::SetCursorScreenPos(origin);
        if (superluminal) {
            ImGui::InvisibleButton(id, {150.0f, 100.0f},
                                   ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
            detail::ForwardSuperluminalSurfaceInputEvents(origin, {150.0f, 100.0f}, input,
                [this](InputEvent event) { events.push_back(event); });
        } else {
            view.render(ctx);
        }
    }

    std::vector<KeyEvent> keys() const {
        std::vector<KeyEvent> result;
        for (const auto& event : events) {
            if (const auto* key = std::get_if<KeyEvent>(&event)) result.push_back(*key);
        }
        return result;
    }

    bool focus(bool focused) const {
        return std::any_of(events.begin(), events.end(), [focused](const auto& event) {
            const auto* focus = std::get_if<FocusEvent>(&event);
            return focus && focus->focused == focused;
        });
    }
};

template<typename Type>
std::vector<Type> Only(const std::vector<Sakura::FrameEvent>& events) {
    std::vector<Type> result;
    for (const auto& event : events) {
        if (const auto* typed = std::get_if<Type>(&event)) {
            result.push_back(*typed);
        }
    }
    return result;
}

struct FrameInputProbe : Modules::FrameImpl {
    Result validate() override { return Result::SUCCESS; }
    Result define() override { return Result::SUCCESS; }
    Result create() override { return Result::SUCCESS; }
    Result destroy() override { return Result::SUCCESS; }
    Result reconfigure() override { return Result::SUCCESS; }
    using Modules::FrameImpl::processInputEvents;
    using Modules::FrameImpl::view;
};

template<typename Tag, typename Tag::Type member>
struct InputMemberAccess {
    friend typename Tag::Type InputMember(Tag) { return member; }
};

struct WaterfallInputMember {
    using Type = void (Modules::SignalViewWaterfall3D::*)(std::vector<InputEvent>&&);
    friend Type InputMember(WaterfallInputMember);
};

struct WaterfallCameraMember {
    using Type = Modules::detail::Waterfall3DCamera Modules::SignalViewWaterfall3D::*;
    friend Type InputMember(WaterfallCameraMember);
};

template struct InputMemberAccess<WaterfallInputMember, &Modules::SignalViewWaterfall3D::processInputEvents>;
template struct InputMemberAccess<WaterfallCameraMember, &Modules::SignalViewWaterfall3D::cameraGoal>;

template<typename Tag, auto member>
struct RetainedMemberAccess {
    friend auto RetainedMember(Tag) { return member; }
};

struct CanvasStorage { friend auto RetainedMember(CanvasStorage); };
struct CanvasInput { friend auto RetainedMember(CanvasInput); };
struct CodeEditorCanvas { friend auto RetainedMember(CodeEditorCanvas); };
template struct RetainedMemberAccess<CanvasStorage, &Sakura::Retained::Canvas::impl>;
template struct RetainedMemberAccess<CanvasInput, &Sakura::Retained::Canvas::ForwardInput>;
template struct RetainedMemberAccess<CodeEditorCanvas, &Sakura::Retained::CodeEditor::canvas>;

bool SendRetained(Sakura::Retained::Canvas& canvas, const InputEvent& event) {
    return RetainedMember(CanvasInput{})((canvas.*RetainedMember(CanvasStorage{})).get(), event);
}

struct CancellationSink : Sakura::Component {
    U64 cancellations = 0;
    bool event(const MouseEvent& event) override {
        if (event.type == MouseEventType::Cancel) {
            ++cancellations;
            return true;
        }
        return false;
    }
};

struct RetainedInputRoot : Sakura::Component {
    Sakura::Component& widget;
    CancellationSink sink;

    explicit RetainedInputRoot(Sakura::Component& widget) : widget(widget) {
        setClipsChildren(true);
        add(widget);
        add(sink);
    }

    void layout(const Sakura::Context& ctx) override {
        layoutChild(ctx, widget, {20, 20, 200, 200});
        layoutChild(ctx, sink, {0, 0, 1, 1});
    }
};

struct RetainedSurface {
    struct Texture final : Render::Texture {
        Texture() : Render::Texture(Config{.size = {256, 256}}) {}
        Result create() override { return Result::SUCCESS; }
        Result destroy() override { return Result::SUCCESS; }
        uint64_t raw() const override { return 1; }
    };

    SakuraTest::HeadlessUi ui{1.0f, {500, 500}};
    Sakura::Retained::Canvas& canvas;
    std::function<void(const Sakura::Context&)> layout;
    std::vector<InputEvent> events;
    std::vector<bool> consumed;
    Sakura::SurfaceView::Config config;
    Sakura::SurfaceView view;

    RetainedSurface(Sakura::Retained::Canvas& canvas,
                    std::function<void(const Sakura::Context&)> layout)
        : canvas(canvas), layout(std::move(layout)) {
        config.id = "retained-input";
        config.texture = 1;
        config.textureSource = std::make_shared<Texture>();
        config.size = {256, 256};
        config.onInput = [this](const InputEvent& event) {
            events.push_back(event);
            consumed.push_back(SendRetained(this->canvas, event));
        };
        view.update(config);
    }

    void draw() {
        layout(ui.sakura());
        ImGui::SetCursorScreenPos({0, 0});
        view.render(ui.sakura());
    }

    void frame(ImVec2 point, bool down) {
        ui.setMouse(point, down);
        ui.frame([&] { draw(); });
    }

    void cancel(int boundary) {
        events.clear();
        consumed.clear();
        if (boundary == 0) {
            ImGui::GetIO().AddMouseButtonEvent(1, true);
            frame({350, 350}, true);
        } else if (boundary == 1) {
            ui.frame([] {});
        } else if (boundary == 2 || boundary == 3) {
            if (boundary == 2) config.id += "-replacement";
            else config.textureSource = std::make_shared<Texture>();
            view.update(config);
        } else {
            CHECK_FALSE(SendRetained(canvas, MouseEvent{.type = MouseEventType::Cancel, .position = {-1, -1}}));
        }
    }
};

}  // namespace

TEST_CASE("Surface keyboard focus survives pointer exit and forwards press repeat release",
          "[core][sakura][surface][keyboard]") {
    const bool superluminal = GENERATE(false, true);
    const auto key = GENERATE(ImGuiKey_B, ImGuiKey_Space);
    const auto code = key == ImGuiKey_Space ? KeyCode::Space : KeyCode::B;
    SakuraTest::HeadlessUi ui;
    InputSurface surface("keys", {20, 20}, superluminal);
    const auto frame = [&](ImVec2 position, bool down) {
        surface.events.clear();
        ui.setMouse(position, down);
        ui.frame([&] { surface.render(ui.sakura()); });
    };
    frame({70, 70}, false);
    ImGui::GetIO().AddKeyEvent(ImGuiKey_A, true);
    frame({70, 70}, false);
    REQUIRE(surface.keys().empty());
    ImGui::GetIO().AddKeyEvent(ImGuiKey_A, false);
    frame({70, 70}, true);
    REQUIRE(surface.focus(true));
    frame({70, 70}, false);
    REQUIRE(surface.keys().empty());

    ImGui::GetIO().AddKeyEvent(ImGuiMod_Ctrl, true);
    ImGui::GetIO().AddKeyEvent(key, true);
    frame({350, 250}, false);
    REQUIRE(surface.keys().size() == 1);
    REQUIRE(surface.keys()[0].key == code);
    REQUIRE(surface.keys()[0].type == KeyEventType::Press);
    REQUIRE(surface.keys()[0].modifiers.control);
    REQUIRE_FALSE(surface.keys()[0].repeat);
    REQUIRE(std::none_of(surface.events.begin(), surface.events.end(), [](const auto& event) {
        return std::holds_alternative<MouseEvent>(event);
    }));

    bool repeated = false;
    for (int i = 0; i < 40; ++i) {
        frame({350, 250}, false);
        for (const auto& event : surface.keys()) {
            REQUIRE(event.key == code);
            REQUIRE(event.type == KeyEventType::Press);
            REQUIRE(event.repeat);
            repeated = true;
        }
    }
    REQUIRE(repeated);
    ImGui::GetIO().AddKeyEvent(key, false);
    frame({350, 250}, false);
    REQUIRE(surface.keys().size() == 1);
    REQUIRE(surface.keys()[0].type == KeyEventType::Release);
    REQUIRE_FALSE(surface.keys()[0].repeat);
    frame({350, 250}, false);
    REQUIRE(surface.keys().empty());
}

TEST_CASE("Surface focus switches exclusively and yields to text fields",
          "[core][sakura][surface][keyboard][focus]") {
    const bool superluminal = GENERATE(false, true);
    const bool reverse = GENERATE(false, true);
    SakuraTest::HeadlessUi ui;
    InputSurface first("first", {10, 10}, superluminal);
    InputSurface second("second", {210, 10}, superluminal);
    char text[64] = {};
    const auto frame = [&](ImVec2 position, bool down) {
        first.events.clear();
        second.events.clear();
        ui.setMouse(position, down);
        ui.frame([&] {
            (reverse ? second : first).render(ui.sakura());
            (reverse ? first : second).render(ui.sakura());
            ImGui::SetCursorScreenPos({10, 200});
            ImGui::InputText("text", text, sizeof(text));
        });
    };
    frame({60, 60}, false);
    frame({60, 60}, true);
    frame({60, 60}, false);
    ImGui::GetIO().AddKeyEvent(ImGuiKey_Space, true);
    frame({60, 60}, false);
    REQUIRE(first.keys().size() == 1);
    REQUIRE(second.keys().empty());

    frame({260, 60}, true);
    REQUIRE(first.focus(false));
    REQUIRE(first.keys().size() == 1);
    REQUIRE(first.keys()[0].key == KeyCode::Space);
    REQUIRE(first.keys()[0].type == KeyEventType::Release);
    REQUIRE(second.focus(true));
    frame({260, 60}, false);
    ImGui::GetIO().AddKeyEvent(ImGuiKey_Space, false);
    frame({260, 60}, false);
    REQUIRE(first.keys().empty());
    REQUIRE(second.keys().empty());

    ImGui::GetIO().AddKeyEvent(ImGuiKey_C, true);
    frame({260, 60}, false);
    REQUIRE(second.keys().size() == 1);
    frame({50, 210}, true);
    REQUIRE(second.focus(false));
    REQUIRE(second.keys().size() == 1);
    REQUIRE(second.keys()[0].type == KeyEventType::Release);
    frame({50, 210}, false);
    ImGui::GetIO().AddKeyEvent(ImGuiKey_C, false);
    ImGui::GetIO().AddKeyEvent(ImGuiKey_D, true);
    ImGui::GetIO().AddInputCharacter('d');
    frame({50, 210}, false);
    REQUIRE(first.keys().empty());
    REQUIRE(second.keys().empty());
    REQUIRE(std::string(text) == "d");
}

TEST_CASE("Surface forwards discrete key and mouse ordering with modifier snapshots",
          "[core][sakura][surface][keyboard][ordering]") {
    const bool superluminal = GENERATE(false, true);
    SakuraTest::HeadlessUi ui;
    InputSurface surface("ordered", {20, 20}, superluminal);
    const auto frame = [&](bool down) {
        surface.events.clear();
        ui.setMouse({95, 70}, down);
        ui.frame([&] { surface.render(ui.sakura()); });
    };
    frame(false);
    frame(true);
    frame(false);

    auto& io = ImGui::GetIO();
    io.ConfigInputTrickleEventQueue = false;
    io.AddKeyEvent(ImGuiMod_Shift, true);
    io.AddKeyEvent(ImGuiKey_LeftShift, true);
    io.AddMouseButtonEvent(ImGuiMouseButton_Left, true);
    io.AddKeyEvent(ImGuiMod_Shift, false);
    io.AddKeyEvent(ImGuiKey_LeftShift, false);
    frame(false);
    REQUIRE(surface.events.size() == 4);
    const auto& press = std::get<KeyEvent>(surface.events[0]);
    REQUIRE(press.type == KeyEventType::Press);
    REQUIRE(press.modifiers.shift);
    const auto& click = std::get<MouseEvent>(surface.events[1]);
    REQUIRE(click.type == MouseEventType::Click);
    REQUIRE(click.modifiers.shift);
    REQUIRE(click.position.x == Catch::Approx(0.5f));
    REQUIRE(click.position.y == Catch::Approx(0.5f));
    const auto& release = std::get<KeyEvent>(surface.events[2]);
    REQUIRE(release.type == KeyEventType::Release);
    REQUIRE_FALSE(release.modifiers.shift);
    const auto& move = std::get<MouseEvent>(surface.events[3]);
    REQUIRE(move.type == MouseEventType::Move);
    REQUIRE_FALSE(move.modifiers.shift);
    REQUIRE(move.scroll.x == 0);
    REQUIRE(move.scroll.y == 0);
}

TEST_CASE("Window shortcuts deliver Space to only one surface",
          "[core][sakura][surface][keyboard][space-routing]") {
    const bool superluminal = GENERATE(false, true);
    const bool reverse = GENERATE(false, true);
    SakuraTest::HeadlessUi ui;
    InputSurface first("first", {10, 10}, superluminal);
    InputSurface second("second", {210, 10}, superluminal);
    const auto frame = [&] {
        first.events.clear();
        second.events.clear();
        ui.setMouse({260, 60}, false);
        ui.frame([&] {
            (reverse ? second : first).render(ui.sakura());
            (reverse ? first : second).render(ui.sakura());
        });
    };
    frame();
    ImGui::GetIO().AddKeyEvent(ImGuiKey_Space, true);
    frame();
    REQUIRE(first.keys().size() + second.keys().size() == 1);
    auto& recipient = first.keys().empty() ? second : first;
    REQUIRE(recipient.keys()[0].type == KeyEventType::Press);
    REQUIRE_FALSE(recipient.keys()[0].repeat);
    ImGui::GetIO().AddKeyEvent(ImGuiKey_Space, false);
    frame();
    REQUIRE(first.keys().size() + second.keys().size() == 1);
    REQUIRE(recipient.keys()[0].type == KeyEventType::Release);
}

TEST_CASE("Space targets the selected plot window regardless of hover and yields to text fields",
          "[core][sakura][surface][keyboard][space-routing]") {
    const bool superluminal = GENERATE(false, true);
    const bool reverse = GENERATE(false, true);
    SakuraTest::HeadlessUi ui(1.0f, {640, 360});
    InputSurface first("first", {20, 60}, superluminal);
    InputSurface second("second", {320, 60}, superluminal);
    char text[64] = {};
    bool textActive = false;
    const auto frame = [&](ImVec2 position, bool down = false) {
        first.events.clear();
        second.events.clear();
        ui.setMouse(position, down);
        ui.frame([&] {
            const auto draw = [&](InputSurface& surface, ImVec2 origin) {
                ImGui::SetNextWindowPos(origin);
                ImGui::SetNextWindowSize({280, 280});
                if (ImGui::Begin(surface.id, nullptr, ImGuiWindowFlags_NoMove)) {
                    surface.render(ui.sakura());
                    if (&surface == &second) {
                        ImGui::SetCursorScreenPos({320, 210});
                        ImGui::InputText("text", text, sizeof(text));
                        textActive = ImGui::IsItemActive();
                    }
                }
                ImGui::End();
            };
            if (reverse) {
                draw(second, {300, 0});
                draw(first, {0, 0});
            } else {
                draw(first, {0, 0});
                draw(second, {300, 0});
            }
        });
    };
    frame({100, 8});
    frame({100, 8}, true);
    frame({100, 8});
    ImGui::GetIO().AddKeyEvent(ImGuiKey_Space, true);
    frame({620, 320});
    REQUIRE(first.keys().size() == 1);
    REQUIRE(first.keys()[0].type == KeyEventType::Press);
    REQUIRE(second.keys().empty());
    ImGui::GetIO().AddKeyEvent(ImGuiKey_Space, false);
    frame({620, 320});
    REQUIRE(first.keys().size() == 1);
    REQUIRE(first.keys()[0].type == KeyEventType::Release);

    ImGui::GetIO().AddKeyEvent(ImGuiKey_Space, true);
    frame({370, 110});
    REQUIRE(first.keys().size() == 1);
    REQUIRE(first.keys()[0].type == KeyEventType::Press);
    REQUIRE(second.keys().empty());
    ImGui::GetIO().AddKeyEvent(ImGuiKey_Space, false);
    frame({370, 110});

    frame({400, 8}, true);
    frame({400, 8});
    ImGui::GetIO().AddKeyEvent(ImGuiKey_Space, true);
    frame({70, 110});
    REQUIRE(first.keys().empty());
    REQUIRE(second.keys().size() == 1);
    REQUIRE(second.keys()[0].type == KeyEventType::Press);
    bool repeated = false;
    for (int i = 0; i < 40; ++i) {
        frame({70, 110});
        REQUIRE(first.keys().empty());
        for (const auto& key : second.keys()) {
            REQUIRE(key.repeat);
            repeated = true;
        }
    }
    REQUIRE(repeated);
    ImGui::GetIO().AddKeyEvent(ImGuiKey_Space, false);
    frame({70, 110});
    REQUIRE(second.keys().size() == 1);
    REQUIRE(second.keys()[0].type == KeyEventType::Release);

    frame({350, 220}, true);
    frame({350, 220});
    REQUIRE(textActive);
    ImGui::GetIO().AddKeyEvent(ImGuiKey_Space, true);
    ImGui::GetIO().AddInputCharacter(' ');
    frame({70, 110});
    REQUIRE(textActive);
    REQUIRE(std::string(text) == " ");
    REQUIRE(first.keys().empty());
    REQUIRE(second.keys().empty());
}

TEST_CASE("Window Space shortcuts retain their place and modifiers in mixed input batches",
          "[core][sakura][surface][keyboard][space-routing][ordering]") {
    const bool superluminal = GENERATE(false, true);
    SakuraTest::HeadlessUi ui;
    InputSurface surface("space-order", {20, 20}, superluminal);
    ui.setMouse({70, 70}, false);
    ui.frame([&] { surface.render(ui.sakura()); });
    surface.events.clear();
    auto& io = ImGui::GetIO();
    io.ConfigInputTrickleEventQueue = false;
    io.AddMouseWheelEvent(0, 1);
    io.AddKeyEvent(ImGuiMod_Shift, true);
    io.AddKeyEvent(ImGuiKey_Space, true);
    io.AddKeyEvent(ImGuiMod_Shift, false);
    io.AddMouseWheelEvent(0, -1);
    io.AddKeyEvent(ImGuiKey_Space, false);
    ui.frame([&] { surface.render(ui.sakura()); });
    REQUIRE(surface.events.size() == 5);
    REQUIRE(std::get<MouseEvent>(surface.events[0]).scroll.y == 1);
    const auto& press = std::get<KeyEvent>(surface.events[1]);
    REQUIRE(press.key == KeyCode::Space);
    REQUIRE(press.type == KeyEventType::Press);
    REQUIRE(press.modifiers.shift);
    REQUIRE(std::get<MouseEvent>(surface.events[2]).scroll.y == -1);
    const auto& release = std::get<KeyEvent>(surface.events[3]);
    REQUIRE(release.key == KeyCode::Space);
    REQUIRE(release.type == KeyEventType::Release);
    REQUIRE_FALSE(release.modifiers.shift);
    REQUIRE(std::get<MouseEvent>(surface.events[4]).type == MouseEventType::Move);
}

TEST_CASE("Window Space shortcuts release when hidden or blocked by a popup",
          "[core][sakura][surface][keyboard][space-routing][hidden]") {
    const bool superluminal = GENERATE(false, true);
    SakuraTest::HeadlessUi ui;
    InputSurface surface("window-hidden", {20, 20}, superluminal);
    ui.setMouse({70, 70}, false);
    ui.frame([&] { surface.render(ui.sakura()); });
    ImGui::GetIO().AddKeyEvent(ImGuiKey_Space, true);
    ui.frame([&] { surface.render(ui.sakura()); });
    REQUIRE(surface.keys().size() == 1);
    REQUIRE_FALSE(surface.focus(true));
    surface.events.clear();
    SECTION("hidden surface") {
        ui.frame([] {});
        REQUIRE(surface.events.size() == 1);
        REQUIRE(surface.keys()[0].type == KeyEventType::Release);
        surface.events.clear();
        ui.frame([&] { surface.render(ui.sakura()); });
        REQUIRE(surface.keys().empty());
    }
    SECTION("open popup") {
        ui.frame([&] {
            ImGui::OpenPopup("popup");
            if (ImGui::BeginPopup("popup")) {
                ImGui::TextUnformatted("Popup");
                ImGui::EndPopup();
            }
            surface.render(ui.sakura());
        });
        REQUIRE(surface.keys().size() == 1);
        REQUIRE(surface.keys()[0].type == KeyEventType::Release);
    }
}

TEST_CASE("Surface preserves batched wheel deltas and interleaved modifier changes",
          "[core][sakura][surface][keyboard][ordering][wheel]") {
    const bool superluminal = GENERATE(false, true);
    const bool shift = GENERATE(false, true);
    const F32 secondDelta = GENERATE(1.0f, -1.0f);
    SakuraTest::HeadlessUi ui;
    InputSurface surface("wheel-order", {20, 20}, superluminal);
    const auto frame = [&](bool down) {
        surface.events.clear();
        ui.setMouse({95, 70}, down);
        ui.frame([&] { surface.render(ui.sakura()); });
    };
    frame(false);
    frame(true);
    frame(false);

    auto& io = ImGui::GetIO();
    REQUIRE(io.ConfigInputTrickleEventQueue);
    if (shift) {
        io.AddKeyEvent(ImGuiMod_Shift, true);
        io.AddKeyEvent(ImGuiKey_LeftShift, true);
        frame(false);
    }
    io.AddMouseWheelEvent(0.5f, 1.0f);
    io.AddKeyEvent(ImGuiMod_Shift, !shift);
    io.AddKeyEvent(ImGuiKey_LeftShift, !shift);
    io.AddMouseWheelEvent(0.5f * secondDelta, secondDelta);
    frame(false);

    REQUIRE(surface.events.size() == 4);
    const auto& first = std::get<MouseEvent>(surface.events[0]);
    REQUIRE(first.type == MouseEventType::Scroll);
    REQUIRE(first.scroll.x == 0.5f);
    REQUIRE(first.scroll.y == 1.0f);
    REQUIRE(first.modifiers.shift == shift);
    REQUIRE(first.position.x == Catch::Approx(0.5f));
    REQUIRE(first.position.y == Catch::Approx(0.5f));

    const auto& key = std::get<KeyEvent>(surface.events[1]);
    REQUIRE(key.key == KeyCode::LeftShift);
    REQUIRE(key.type == (shift ? KeyEventType::Release : KeyEventType::Press));
    REQUIRE(key.modifiers.shift == !shift);

    const auto& second = std::get<MouseEvent>(surface.events[2]);
    REQUIRE(second.type == MouseEventType::Scroll);
    REQUIRE(second.scroll.x == 0.5f * secondDelta);
    REQUIRE(second.scroll.y == secondDelta);
    REQUIRE(second.modifiers.shift == !shift);
    REQUIRE(second.position.x == first.position.x);
    REQUIRE(second.position.y == first.position.y);

    const auto& move = std::get<MouseEvent>(surface.events[3]);
    REQUIRE(move.type == MouseEventType::Move);
    REQUIRE(move.modifiers.shift == !shift);
    REQUIRE(move.scroll.x == 0.0f);
    REQUIRE(move.scroll.y == 0.0f);

    frame(false);
    REQUIRE(surface.events.size() == 1);
    REQUIRE(std::get<MouseEvent>(surface.events[0]).type == MouseEventType::Move);
}

TEST_CASE("Application focus loss balances held keys and cancels a surface drag",
          "[core][sakura][surface][keyboard][focus]") {
    const bool superluminal = GENERATE(false, true);
    SakuraTest::HeadlessUi ui;
    InputSurface surface("focus-loss", {20, 20}, superluminal);
    SurfaceInteractionState interaction;
    interaction.zoom = 2;
    const auto frame = [&](bool down) {
        surface.events.clear();
        ui.setMouse({95, 70}, down);
        ui.frame([&] { surface.render(ui.sakura()); });
        auto events = surface.events;
        interaction = ProcessSurfaceInteraction(interaction, {}, std::move(events));
    };
    frame(false);
    frame(true);
    ImGui::GetIO().AddKeyEvent(ImGuiKey_Escape, true);
    frame(true);
    REQUIRE(surface.keys().size() == 1);
    REQUIRE(interaction.dragging);

    ImGui::GetIO().AddFocusEvent(false);
    frame(true);
    REQUIRE(surface.focus(false));
    REQUIRE(surface.keys().size() == 1);
    REQUIRE(surface.keys()[0].type == KeyEventType::Release);
    REQUIRE_FALSE(interaction.dragging);
    ImGui::GetIO().AddFocusEvent(true);
    frame(false);
    REQUIRE(surface.keys().empty());
    REQUIRE_FALSE(surface.focus(true));
}

TEST_CASE("Surface events retain native Control and Command names on every platform",
          "[core][sakura][surface][keyboard][modifiers]") {
    const bool superluminal = GENERATE(false, true);
    const bool macOS = GENERATE(false, true);
    SakuraTest::HeadlessUi ui;
    auto& io = ImGui::GetIO();
    io.ConfigMacOSXBehaviors = macOS;
    InputSurface surface("modifiers", {20, 20}, superluminal);
    const auto frame = [&](bool down) {
        surface.events.clear();
        ui.setMouse({95, 70}, down);
        ui.frame([&] { surface.render(ui.sakura()); });
    };
    frame(false);
    frame(true);
    frame(false);
    io.AddKeyEvent(ImGuiMod_Ctrl, true);
    io.AddKeyEvent(ImGuiKey_LeftCtrl, true);
    io.AddKeyEvent(ImGuiMod_Super, true);
    io.AddKeyEvent(ImGuiKey_RightSuper, true);
    io.AddKeyEvent(ImGuiMod_Alt, true);
    io.AddKeyEvent(ImGuiMod_Shift, true);
    frame(false);
    REQUIRE(surface.keys().size() == 2);
    REQUIRE(surface.keys()[0].key == KeyCode::LeftCtrl);
    REQUIRE(surface.keys()[0].modifiers.control);
    REQUIRE_FALSE(surface.keys()[0].modifiers.super);
    REQUIRE(surface.keys()[1].key == KeyCode::RightSuper);
    REQUIRE(surface.keys()[1].modifiers.control);
    REQUIRE(surface.keys()[1].modifiers.super);

    io.MouseWheel = 1.0f;
    io.MouseWheelH = -2.0f;
    frame(false);
    REQUIRE(surface.events.size() == 2);
    const auto& scroll = std::get<MouseEvent>(surface.events[0]);
    REQUIRE(scroll.type == MouseEventType::Scroll);
    REQUIRE(scroll.scroll.x == -2.0f);
    REQUIRE(scroll.scroll.y == 1.0f);
    REQUIRE(scroll.modifiers.control);
    REQUIRE(scroll.modifiers.shift);
    REQUIRE(scroll.modifiers.alt);
    REQUIRE(scroll.modifiers.super);
    REQUIRE(scroll.position.x == Catch::Approx(0.5f));
    const auto& move = std::get<MouseEvent>(surface.events[1]);
    REQUIRE(move.type == MouseEventType::Move);
    REQUIRE(move.scroll.x == 0);
    REQUIRE(move.scroll.y == 0);
}

TEST_CASE("A hidden surface releases keys before reappearing and yields focus to other widgets",
          "[core][sakura][surface][keyboard][focus][hidden]") {
    const bool superluminal = GENERATE(false, true);
    SakuraTest::HeadlessUi ui;
    InputSurface surface("hidden", {20, 20}, superluminal);
    const auto frame = [&](bool down) {
        surface.events.clear();
        ui.setMouse({95, 70}, down);
        ui.frame([&] { surface.render(ui.sakura()); });
    };
    frame(false);
    frame(true);
    frame(false);
    ImGui::GetIO().AddKeyEvent(ImGuiKey_LeftArrow, true);
    frame(false);
    REQUIRE(surface.keys().size() == 1);
    surface.events.clear();
    ui.frame([] {});
    REQUIRE(surface.events.size() == 2);
    REQUIRE(std::get<KeyEvent>(surface.events[0]).type == KeyEventType::Release);
    REQUIRE(std::get<KeyEvent>(surface.events[0]).key == KeyCode::LeftArrow);
    REQUIRE_FALSE(std::get<FocusEvent>(surface.events[1]).focused);
    REQUIRE(surface.focus(false));

    surface.events.clear();
    char text[64] = {};
    bool otherFocused = false;
    const auto otherWidget = [&] {
        ImGui::SetCursorScreenPos({20, 200});
        ImGui::InputText("other-widget", text, sizeof(text));
        otherFocused = ImGui::IsItemActive();
    };
    ui.setMouse({50, 210}, true);
    ui.frame(otherWidget);
    REQUIRE(otherFocused);
    REQUIRE(surface.events.empty());
    ui.setMouse({50, 210}, false);
    ui.frame(otherWidget);
    ImGui::GetIO().AddKeyEvent(ImGuiKey_LeftArrow, false);
    ui.frame(otherWidget);
    ImGui::GetIO().AddInputCharacter('b');
    ui.frame(otherWidget);
    REQUIRE(std::string(text) == "b");
    REQUIRE(surface.events.empty());

    frame(false);
    REQUIRE(surface.keys().empty());
    REQUIRE_FALSE(surface.focus(true));
    REQUIRE_FALSE(surface.focus(false));
}

TEST_CASE("Collapsed windows release surface input while their content is skipped",
          "[core][sakura][surface][keyboard][focus][hidden]") {
    const bool superluminal = GENERATE(false, true);
    SakuraTest::HeadlessUi ui;
    InputSurface surface("collapsed", {20, 40}, superluminal);
    Sakura::Window window;
    window.update({.id = "surface-window", .title = "Surface", .size = {300, 200}});
    SurfaceInteractionState interaction;
    interaction.zoom = 2;
    bool rendered = false;
    const auto frame = [&](bool down, bool collapsed) {
        surface.events.clear();
        rendered = false;
        ui.setMouse({95, 90}, down);
        ui.frame([&] {
            ImGui::SetNextWindowPos({0, 0});
            ImGui::SetNextWindowCollapsed(collapsed);
            window.render(ui.sakura(), [&](const Sakura::Context& ctx) {
                rendered = true;
                surface.render(ctx);
            });
        });
        auto events = surface.events;
        interaction = ProcessSurfaceInteraction(interaction, {}, std::move(events));
    };
    frame(false, false);
    frame(false, false);
    frame(true, false);
    REQUIRE(surface.focus(true));
    ImGui::GetIO().AddKeyEvent(ImGuiKey_A, true);
    frame(true, false);
    REQUIRE(interaction.dragging);
    REQUIRE(surface.keys().size() == 1);

    frame(true, true);
    REQUIRE_FALSE(rendered);
    REQUIRE(surface.keys().size() == 1);
    REQUIRE(surface.keys()[0].type == KeyEventType::Release);
    REQUIRE(surface.focus(false));
    REQUIRE_FALSE(interaction.dragging);
    frame(false, true);
    REQUIRE_FALSE(rendered);
    REQUIRE(surface.events.empty());
    frame(false, false);
    REQUIRE(rendered);
    REQUIRE(surface.keys().empty());
    REQUIRE_FALSE(surface.focus(true));
    REQUIRE_FALSE(surface.focus(false));
}

TEST_CASE("Surface visibility tracking handles either context or surface destruction first",
          "[core][sakura][surface][keyboard][focus][hidden][lifetime]") {
    const bool superluminal = GENERATE(false, true);
    const auto key = GENERATE(ImGuiKey_A, ImGuiKey_Space);
    std::vector<InputEvent> events;
    auto ui = std::make_unique<SakuraTest::HeadlessUi>();
    auto view = std::make_unique<Sakura::SurfaceView>();
    auto input = std::make_unique<detail::SurfaceInputState>();
    const auto emit = [&](const InputEvent& event) { events.push_back(event); };
    view->update({.id = "lifetime", .texture = 1, .size = {150, 100}, .onInput = emit});
    const auto frame = [&](bool down) {
        events.clear();
        ui->setMouse({95, 70}, down);
        ui->frame([&] {
            ImGui::SetCursorScreenPos({20, 20});
            if (superluminal) {
                ImGui::InvisibleButton("lifetime", {150, 100},
                                       ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
                detail::ForwardSuperluminalSurfaceInputEvents({20, 20}, {150, 100}, *input, emit);
            } else {
                view->render(ui->sakura());
            }
        });
    };
    frame(false);
    frame(true);
    frame(false);
    ImGui::GetIO().AddKeyEvent(key, true);
    frame(false);
    REQUIRE(std::get<KeyEvent>(events[0]).type == KeyEventType::Press);
    events.clear();

    SECTION("context destroyed first") {
        ui.reset();
        REQUIRE(events.size() == 2);
        REQUIRE(std::get<KeyEvent>(events[0]).type == KeyEventType::Release);
        REQUIRE_FALSE(std::get<FocusEvent>(events[1]).focused);
        events.clear();
        view.reset();
        input.reset();
        REQUIRE(events.empty());
    }
    SECTION("surface destroyed first") {
        view.reset();
        input.reset();
        REQUIRE(events.size() == 2);
        REQUIRE(std::get<KeyEvent>(events[0]).type == KeyEventType::Release);
        REQUIRE_FALSE(std::get<FocusEvent>(events[1]).focused);
        events.clear();
        ui->frame([] {});
        ui.reset();
        REQUIRE(events.empty());
    }
}

TEST_CASE("The ImGui bridge samples ordered input with native modifiers once per frame",
          "[core][sakura][surface][bridge]") {
    SakuraTest::HeadlessUi ui;
    auto& io = ImGui::GetIO();
    io.ConfigInputTrickleEventQueue = false;
    io.ConfigMacOSXBehaviors = true;
    Sakura::InputFrame frame;

    io.AddKeyEvent(ImGuiMod_Super, true);
    io.AddKeyEvent(ImGuiKey_LeftSuper, true);
    io.AddKeyEvent(ImGuiKey_S, true);
    io.AddInputCharacter('x');
    io.AddMouseWheelEvent(0.0f, 1.0f);
    ui.setMouse({30.0f, 40.0f}, true);
    ui.frame([&] {
        Sakura::Private::SampleInput(frame);
        Sakura::Private::ApplyCursor(SurfaceCursor::Hand);
        CHECK(ImGui::GetMouseCursor() == ImGuiMouseCursor_Hand);
    });
    CHECK(frame.serial == 1);
    CHECK(frame.modifiers.super);
    CHECK_FALSE(frame.modifiers.control);
    CHECK(frame.pointer == Extent2D<F32>{30.0f, 40.0f});
    CHECK(frame.buttons[0]);

    const auto mouse = Only<MouseEvent>(frame.events);
    REQUIRE(mouse.size() == 2);
    const auto click = std::find_if(mouse.begin(), mouse.end(), [](const auto& event) {
        return event.type == MouseEventType::Click;
    });
    const auto scroll = std::find_if(mouse.begin(), mouse.end(), [](const auto& event) {
        return event.type == MouseEventType::Scroll;
    });
    REQUIRE(click != mouse.end());
    REQUIRE(scroll != mouse.end());
    CHECK(click->button == MouseButton::Left);
    CHECK(click->clicks == 1);
    CHECK(click->modifiers.super);
    CHECK(scroll->scroll == Extent2D<F32>{0.0f, 1.0f});

    const auto keys = Only<KeyEvent>(frame.events);
    REQUIRE(keys.size() == 2);
    CHECK(keys[0].key == KeyCode::LeftSuper);
    CHECK(keys[1].key == KeyCode::S);
    CHECK(keys[1].type == KeyEventType::Press);
    CHECK(keys[1].modifiers.super);
    CHECK_FALSE(keys[1].repeat);

    const auto text = Only<TextEvent>(frame.events);
    REQUIRE(text.size() == 1);
    CHECK(text[0].codepoint == U'x');

    io.AddKeyEvent(ImGuiKey_S, false);
    io.AddKeyEvent(ImGuiKey_LeftSuper, false);
    io.AddKeyEvent(ImGuiMod_Super, false);
    ui.setMouse({30.0f, 40.0f}, false);
    ui.frame([&] {
        Sakura::Private::SampleInput(frame);
        Sakura::Private::ApplyCursor(SurfaceCursor::Default);
        CHECK(ImGui::GetMouseCursor() == ImGuiMouseCursor_Arrow);
    });
    CHECK(frame.serial == 2);
    CHECK_FALSE(frame.modifiers.super);
    CHECK_FALSE(frame.buttons[0]);
    const auto released = Only<MouseEvent>(frame.events);
    REQUIRE(released.size() == 1);
    CHECK(released[0].type == MouseEventType::Release);
    CHECK(Only<KeyEvent>(frame.events).size() == 2);
    CHECK(Only<TextEvent>(frame.events).empty());
}

TEST_CASE("The ImGui bridge keeps each wheel delta, key and character in order with the modifiers held at the time",
          "[core][sakura][surface][bridge]") {
    SakuraTest::HeadlessUi ui;
    auto& io = ImGui::GetIO();
    io.ConfigInputTrickleEventQueue = false;
    io.ConfigMacOSXBehaviors = false;
    Sakura::InputFrame frame;

    io.AddMouseWheelEvent(0.0f, 1.0f);
    io.AddKeyEvent(ImGuiMod_Shift, true);
    io.AddMouseWheelEvent(0.0f, -1.0f);
    io.AddInputCharacter('a');
    io.AddKeyEvent(ImGuiKey_B, true);
    io.AddKeyEvent(ImGuiMod_Shift, false);
    io.AddKeyEvent(ImGuiKey_B, false);
    ui.frame([&] { Sakura::Private::SampleInput(frame); });

    REQUIRE(frame.events.size() == 5);
    const auto* first = std::get_if<MouseEvent>(&frame.events[0]);
    const auto* second = std::get_if<MouseEvent>(&frame.events[1]);
    const auto* text = std::get_if<TextEvent>(&frame.events[2]);
    const auto* press = std::get_if<KeyEvent>(&frame.events[3]);
    const auto* release = std::get_if<KeyEvent>(&frame.events[4]);
    REQUIRE(first);
    REQUIRE(second);
    REQUIRE(text);
    REQUIRE(press);
    REQUIRE(release);
    CHECK(first->scroll == Extent2D<F32>{0.0f, 1.0f});
    CHECK_FALSE(first->modifiers.shift);
    CHECK(second->scroll == Extent2D<F32>{0.0f, -1.0f});
    CHECK(second->modifiers.shift);
    CHECK(text->codepoint == U'a');
    CHECK(text->modifiers.shift);
    CHECK(press->key == KeyCode::B);
    CHECK(press->type == KeyEventType::Press);
    CHECK(press->modifiers.shift);
    CHECK(release->type == KeyEventType::Release);
    CHECK_FALSE(release->modifiers.shift);
    CHECK_FALSE(frame.modifiers.shift);
}

TEST_CASE("The ImGui bridge samples a frame once however often it is asked",
          "[core][sakura][surface][bridge]") {
    SakuraTest::HeadlessUi ui;
    auto& io = ImGui::GetIO();
    io.ConfigInputTrickleEventQueue = false;
    Sakura::InputFrame frame;

    io.AddInputCharacter('x');
    ui.frame([&] {
        Sakura::Private::SampleInput(frame);
        Sakura::Private::SampleInput(frame);
    });
    CHECK(frame.serial == 1);
    CHECK(Only<TextEvent>(frame.events).size() == 1);

    Sakura::InputRouter router;
    const auto routed = router.route(frame, {0.0f, 0.0f, 100.0f, 100.0f});
    CHECK(Only<TextEvent>(routed.events).size() == 1);
    CHECK(router.route(frame, {0.0f, 0.0f, 100.0f, 100.0f}).events.empty());

    ui.frame([&] { Sakura::Private::SampleInput(frame); });
    CHECK(frame.serial == 2);
    CHECK(frame.events.empty());
}

TEST_CASE("Queued pointer positions retain their order through sampling and routing",
          "[core][sakura][surface][bridge]") {
    SakuraTest::HeadlessUi ui;
    auto& io = ImGui::GetIO();
    io.ConfigInputTrickleEventQueue = false;
    Sakura::InputFrame frame;
    Sakura::InputRouter router;
    io.AddMousePosEvent(25.0f, 25.0f);
    io.AddMouseButtonEvent(0, true);
    io.AddKeyEvent(ImGuiKey_A, true);
    io.AddInputCharacter('x');
    io.AddMouseWheelEvent(0.0f, 1.0f);
    io.AddMousePosEvent(250.0f, 25.0f);
    io.AddMouseWheelEvent(0.0f, 2.0f);
    ui.frame([&] { Sakura::Private::SampleInput(frame); });
    REQUIRE(frame.events.size() == 7);
    CHECK(std::get<MouseEvent>(frame.events[0]).type == MouseEventType::Move);
    CHECK(std::get<MouseEvent>(frame.events[1]).position == Extent2D<F32>{25.0f, 25.0f});
    CHECK(std::holds_alternative<KeyEvent>(frame.events[2]));
    CHECK(std::holds_alternative<TextEvent>(frame.events[3]));
    CHECK(std::get<MouseEvent>(frame.events[4]).position == Extent2D<F32>{25.0f, 25.0f});
    CHECK(std::get<MouseEvent>(frame.events[5]).position == Extent2D<F32>{250.0f, 25.0f});
    CHECK(std::get<MouseEvent>(frame.events[6]).position == Extent2D<F32>{250.0f, 25.0f});

    const auto output = router.route(frame, {0.0f, 0.0f, 100.0f, 100.0f});
    CHECK(output.active);
    CHECK_FALSE(output.hovered);
    CHECK_FALSE(output.outsideClick);
    REQUIRE(output.events.size() == 6);
    CHECK(std::get<MouseEvent>(output.events[1]).type == MouseEventType::Click);
    CHECK(std::get<MouseEvent>(output.events[1]).position == Extent2D<F32>{0.25f, 0.25f});
    CHECK(std::holds_alternative<KeyEvent>(output.events[2]));
    CHECK(std::holds_alternative<TextEvent>(output.events[3]));
    CHECK(std::get<MouseEvent>(output.events[4]).type == MouseEventType::Scroll);
    CHECK(std::get<MouseEvent>(output.events[5]).position == Extent2D<F32>{2.5f, 0.25f});

    io.AddKeyEvent(ImGuiKey_A, false);
    io.AddMousePosEvent(300.0f, 50.0f);
    io.AddMouseButtonEvent(0, false);
    ui.frame([&] { Sakura::Private::SampleInput(frame); });
    const auto released = router.route(frame, {0.0f, 0.0f, 100.0f, 100.0f});
    CHECK_FALSE(released.active);
    const auto mouse = Only<MouseEvent>(released.events);
    REQUIRE(mouse.size() == 2);
    CHECK(mouse[0].type == MouseEventType::Move);
    CHECK(mouse[1].type == MouseEventType::Release);
    CHECK(mouse[1].position == Extent2D<F32>{3.0f, 0.5f});
    CHECK_FALSE(mouse[1].cancel);
}

TEST_CASE("Input routing validates event types and button indices before accessing capture",
          "[core][sakura][surface][bridge]") {
    Sakura::InputFrame frame;
    Sakura::InputRouter router;
    frame.pointer = Extent2D<F32>{25.0f, 25.0f};
    for (const auto type : {MouseEventType::Click, MouseEventType::Release, MouseEventType::Move,
                            MouseEventType::Scroll, MouseEventType::Enter, MouseEventType::Leave,
                            static_cast<MouseEventType>(255)}) {
        frame.advance(0.1f);
        frame.events = {MouseEvent{.type = type, .button = static_cast<MouseButton>(255),
                                  .position = {25.0f, 25.0f}}};
        const auto output = router.route(frame, {0.0f, 0.0f, 100.0f, 100.0f});
        CHECK_FALSE(output.active);
        CHECK_FALSE(output.middle);
        for (const auto& mouse : Only<MouseEvent>(output.events)) {
            CHECK(mouse.type != MouseEventType::Click);
            CHECK(mouse.type != MouseEventType::Release);
        }
    }
}

TEST_CASE("All cancellation boundaries balance three buttons and keys without unfocused actions",
          "[core][sakura][surface][bridge]") {
    const int boundary = GENERATE(0, 1, 2, 3, 4, 5, 6);
    Sakura::InputFrame frame;
    Sakura::InputRouter router;
    Jetstream::Rect rect{0.0f, 0.0f, 100.0f, 100.0f};
    frame.advance(0.1f);
    frame.pointer = Extent2D<F32>{25.0f, 25.0f};
    frame.buttons.fill(true);
    frame.events = {KeyEvent{KeyEventType::Press, KeyCode::A, {}},
                    MouseEvent{.type = MouseEventType::Click, .button = MouseButton::Left, .position = {25, 25}},
                    MouseEvent{.type = MouseEventType::Click, .button = MouseButton::Right, .position = {25, 25}},
                    MouseEvent{.type = MouseEventType::Click, .button = MouseButton::Middle, .position = {25, 25}}};
    const auto pressed = router.route(frame, rect);
    REQUIRE(pressed.active);
    REQUIRE(pressed.middle);
    frame.advance(0.1f);
    if (boundary == 0) frame.windowFocused = false;
    if (boundary == 1) frame.events.push_back(MouseEvent{.type = MouseEventType::Cancel});
    if (boundary == 2) rect.width = 0.0f;
    if (boundary == 3) rect.x = std::numeric_limits<F32>::quiet_NaN();
    if (boundary == 4) rect.height = std::numeric_limits<F32>::infinity();
    if (boundary == 5) frame.buttons[2] = false;
    if (boundary == 6) frame.events.push_back(FocusEvent{false});
    const auto cancelled = router.route(frame, rect);
    CHECK_FALSE(cancelled.active);
    CHECK_FALSE(cancelled.middle);
    CHECK_FALSE(cancelled.hovered);
    const auto keys = Only<KeyEvent>(cancelled.events);
    REQUIRE(keys.size() == 1);
    CHECK(keys[0].type == KeyEventType::Release);
    const auto mouse = Only<MouseEvent>(cancelled.events);
    REQUIRE(mouse.size() == 4);
    for (U64 index = 0; index < 3; ++index) {
        CHECK(mouse[index].type == MouseEventType::Release);
        CHECK(mouse[index].button == static_cast<MouseButton>(index));
        CHECK(mouse[index].cancel);
        REQUIRE(SurfaceMouseEvent(InputEvent{mouse[index]}));
        CHECK(SurfaceMouseEvent(InputEvent{mouse[index]})->type == MouseEventType::Cancel);
    }
    CHECK(mouse[3].type == MouseEventType::Cancel);
    frame.advance(0.1f);
    frame.windowFocused = false;
    frame.events = {MouseEvent{.type = MouseEventType::Click, .position = {25, 25}},
                    MouseEvent{.type = MouseEventType::Scroll, .position = {25, 25}, .scroll = {0, 1}},
                    KeyEvent{KeyEventType::Press, KeyCode::A, {}}, TextEvent{U'x', {}}};
    const auto unfocused = router.route(frame, {0, 0, 100, 100});
    CHECK(unfocused.events.empty());
    CHECK_FALSE(unfocused.active);
}

TEST_CASE("Input routing identifies manual frames and explicit source lifetimes independently of serial",
          "[core][sakura][surface][bridge]") {
    const bool explicitSource = GENERATE(false, true);
    Sakura::InputFrame first;
    Sakura::InputFrame second;
    if (explicitSource) {
        first.source = &first;
        second.source = &second;
    }
    first.advance(0.1f);
    second.advance(0.1f);
    first.pointer = Extent2D<F32>{25, 25};
    first.buttons.fill(true);
    first.events = {TextEvent{U'a', {}}, KeyEvent{KeyEventType::Press, KeyCode::A, {}},
                    MouseEvent{.type = MouseEventType::Click, .position = {25, 25}}};
    second.events = {TextEvent{U'b', {}}};
    Sakura::InputRouter router;
    const Jetstream::Rect rect{0, 0, 100, 100};
    CHECK(Only<TextEvent>(router.route(first, rect).events).size() == 1);
    CHECK(router.route(first, rect).events.empty());
    const auto transitioned = router.route(second, rect);
    REQUIRE(Only<TextEvent>(transitioned.events).size() == 1);
    CHECK(Only<TextEvent>(transitioned.events)[0].codepoint == U'b');
    CHECK_FALSE(transitioned.active);
    REQUIRE(Only<KeyEvent>(transitioned.events).size() == 1);
    CHECK(Only<KeyEvent>(transitioned.events)[0].type == KeyEventType::Release);
    REQUIRE(Only<MouseEvent>(transitioned.events).size() == 2);
    CHECK(Only<MouseEvent>(transitioned.events)[0].cancel);
    CHECK(router.route(second, rect).events.empty());
    ++second.sourceGeneration;
    CHECK(Only<TextEvent>(router.route(second, rect).events).size() == 1);
}

TEST_CASE("Ordered click counts survive batching and respect ImGui time and distance thresholds",
          "[core][sakura][surface][bridge]") {
    SakuraTest::HeadlessUi ui;
    auto& io = ImGui::GetIO();
    io.ConfigInputTrickleEventQueue = false;
    Sakura::InputFrame frame;
    const auto click = [&] {
        io.AddMouseButtonEvent(0, true);
        io.AddMouseButtonEvent(0, false);
    };
    const auto counts = [&] {
        std::vector<U8> result;
        for (const auto& mouse : Only<MouseEvent>(frame.events)) {
            if (mouse.type == MouseEventType::Click) result.push_back(mouse.clicks);
        }
        return result;
    };
    io.AddMousePosEvent(25, 25);
    click();
    click();
    ui.frame([&] { Sakura::Private::SampleInput(frame); });
    CHECK(counts() == std::vector<U8>{1, 2});
    ui.setMouse({25, 25}, false);
    click();
    ui.frame([&] { Sakura::Private::SampleInput(frame); });
    CHECK(counts() == std::vector<U8>{3});
    io.DeltaTime = io.MouseDoubleClickTime + 0.1f;
    click();
    ui.frame([&] { Sakura::Private::SampleInput(frame); });
    CHECK(counts() == std::vector<U8>{1});
    io.DeltaTime = 1.0f / 60.0f;
    io.AddMousePosEvent(25 + io.MouseDoubleClickMaxDist + 1, 25);
    click();
    ui.frame([&] { Sakura::Private::SampleInput(frame); });
    CHECK(counts() == std::vector<U8>{1});
    ui.setMouse({25 + io.MouseDoubleClickMaxDist + 1, 25}, false);
    for (int index = 0; index < 260; ++index) click();
    ui.frame([&] { Sakura::Private::SampleInput(frame); });
    REQUIRE(counts().size() == 260);
    CHECK(counts().back() == std::numeric_limits<U8>::max());
}

TEST_CASE("No-context sampling publishes cancellation and clipboard reads remain safe",
          "[core][sakura][surface][bridge]") {
    Sakura::InputFrame frame;
    Sakura::InputRouter router;
    {
        SakuraTest::HeadlessUi ui;
        ui.setMouse({25, 25}, true);
        ui.frame([&] { Sakura::Private::SampleInput(frame); });
        REQUIRE(router.route(frame, {0, 0, 100, 100}).active);
        auto& platform = ImGui::GetPlatformIO();
        platform.Platform_GetClipboardTextFn = nullptr;
        CHECK(Sakura::GetClipboardText().empty());
        platform.Platform_GetClipboardTextFn = [](ImGuiContext*) -> const char* { return nullptr; };
        CHECK(Sakura::GetClipboardText().empty());
        platform.Platform_GetClipboardTextFn = [](ImGuiContext*) -> const char* { return "sample"; };
        CHECK(Sakura::GetClipboardText() == "sample");
    }
    const U64 serial = frame.serial;
    REQUIRE(ImGui::GetCurrentContext() == nullptr);
    CHECK(Sakura::GetClipboardText().empty());
    Sakura::Private::SampleInput(frame);
    CHECK(frame.serial == serial + 1);
    CHECK_FALSE(frame.windowFocused);
    CHECK_FALSE(frame.pointer);
    CHECK(frame.buttons == std::array<bool, 3>{});
    CHECK(frame.source == nullptr);
    CHECK(frame.sourceGeneration == 0);
    CHECK(frame.sourceFrame == -1);
    REQUIRE(frame.events.size() == 1);
    CHECK(std::get<MouseEvent>(frame.events[0]).type == MouseEventType::Cancel);
    const auto cancelled = router.route(frame, {0, 0, 100, 100});
    CHECK_FALSE(cancelled.active);
    CHECK_FALSE(cancelled.hovered);
    REQUIRE(Only<MouseEvent>(cancelled.events).size() == 2);
    CHECK(Only<MouseEvent>(cancelled.events)[0].cancel);
}

TEST_CASE("Sampler caches belong to context lifetimes even when context storage is reused",
          "[core][sakura][surface][bridge]") {
    struct Allocator {
        ImGuiMemAllocFunc allocate;
        ImGuiMemFreeFunc release;
        void* user;
        void* storage;
        bool occupied = false;

        ~Allocator() {
            ImGui::SetAllocatorFunctions(allocate, release, user);
            release(storage, user);
        }
    } allocator;
    ImGui::GetAllocatorFunctions(&allocator.allocate, &allocator.release, &allocator.user);
    allocator.storage = allocator.allocate(sizeof(ImGuiContext), allocator.user);
    ImGui::SetAllocatorFunctions([](size_t size, void* user) -> void* {
        auto& a = *static_cast<Allocator*>(user);
        if (size == sizeof(ImGuiContext) && !a.occupied) {
            a.occupied = true;
            return a.storage;
        }
        return a.allocate(size, a.user);
    }, [](void* pointer, void* user) {
        auto& a = *static_cast<Allocator*>(user);
        if (pointer == a.storage) a.occupied = false;
        else a.release(pointer, a.user);
    }, &allocator);
    Sakura::InputFrame frame;
    U64 generation = 0;
    for (int lifetime = 0; lifetime < 2; ++lifetime) {
        SakuraTest::HeadlessUi ui;
        CHECK(ImGui::GetCurrentContext() == allocator.storage);
        ImGui::GetIO().AddInputCharacter(lifetime == 0 ? 'a' : 'b');
        ui.frame([&] { Sakura::Private::SampleInput(frame); });
        CHECK(frame.sourceGeneration != generation);
        generation = frame.sourceGeneration;
        const auto text = Only<TextEvent>(frame.events);
        REQUIRE(text.size() == 1);
        CHECK(text[0].codepoint == (lifetime == 0 ? U'a' : U'b'));
        Sakura::InputFrame other;
        Sakura::Private::SampleInput(other);
        CHECK(other.sourceGeneration == generation);
        CHECK(other.serial == frame.serial);
        CHECK(Only<TextEvent>(other.events).size() == 1);
    }
}

TEST_CASE("Legacy pointer reset cancels against the old recipient and never leaks an orphan release",
          "[core][sakura][surface][bridge]") {
    const int boundary = GENERATE(0, 1, 2, 3);
    std::vector<InputEvent> oldEvents;
    std::vector<InputEvent> newEvents;
    detail::SurfaceInputState state;
    {
        SakuraTest::HeadlessUi ui;
        ui.frame([&] {});
        ui.setMouse({25, 25}, true);
        ui.frame([&] {
            ImGui::InvisibleButton("reset-target", {100, 100});
            detail::ForwardSurfaceInputEvents({0, 0}, {100, 100}, state,
                [&](const InputEvent& event) { oldEvents.push_back(event); });
        });
        REQUIRE(state.leftPressed);
        oldEvents.clear();
        if (boundary == 0) ui.frame([&] {});
        if (boundary == 1) ui.frame([&] {
            detail::ForwardSurfaceInputEvents({0, 0}, {0, 100}, state,
                [&](const InputEvent& event) { newEvents.push_back(event); });
        });
        if (boundary == 2) state.reset([&](const InputEvent& event) { newEvents.push_back(event); });
        if (boundary != 3) {
            CHECK(newEvents.empty());
            ui.setMouse({25, 25}, false);
            ui.frame([&] {
                ImGui::InvisibleButton("reset-target", {100, 100});
                detail::ForwardSurfaceInputEvents({0, 0}, {100, 100}, state,
                    [&](const InputEvent& event) { newEvents.push_back(event); });
            });
        }
    }
    CHECK_FALSE(state.leftPressed);
    CHECK(state.lastPosition == Extent2D<F32>{0, 0});
    U64 releases = 0;
    for (const auto& event : oldEvents) {
        if (const auto* mouse = std::get_if<MouseEvent>(&event); mouse && mouse->type == MouseEventType::Release) {
            ++releases;
            CHECK(mouse->cancel);
        }
    }
    CHECK(releases == 1);
    {
        SakuraTest::HeadlessUi ui;
        ui.frame([&] {
            ImGui::InvisibleButton("reset-target", {100, 100});
            detail::ForwardSurfaceInputEvents({0, 0}, {100, 100}, state,
                [&](const InputEvent& event) { newEvents.push_back(event); });
        });
    }
    for (const auto& event : newEvents) {
        if (const auto* mouse = std::get_if<MouseEvent>(&event)) CHECK(mouse->type != MouseEventType::Release);
    }
}

TEST_CASE("Cancelled releases leave frame zoom and the waterfall camera unchanged",
          "[core][sakura][surface][bridge][consumer-cancel]") {
    FrameInputProbe frame;
    frame.view.zoom = 4.0f;
    frame.view.center = {0.4f, 0.6f};
    frame.processInputEvents({MouseEvent{.type = MouseEventType::Click, .button = MouseButton::Right,
                                          .position = {0.25f, 0.25f}}});
    REQUIRE(frame.view.selecting);
    frame.processInputEvents({MouseEvent{.type = MouseEventType::Release, .button = MouseButton::Right,
                                          .position = {0.25f, 0.25f}, .cancel = true}});
    CHECK_FALSE(frame.view.selecting);
    CHECK(frame.view.zoom == 4.0f);
    CHECK(frame.view.center == Extent2D<F32>{0.4f, 0.6f});

    SurfaceInteractionState interaction;
    interaction.zoom = 4.0f;
    interaction.offset = 0.2f;
    interaction.dragging = true;
    interaction = ProcessSurfaceInteraction(interaction, {}, {
        MouseEvent{.type = MouseEventType::Release, .button = MouseButton::Right, .cancel = true}});
    CHECK_FALSE(interaction.dragging);
    CHECK(interaction.zoom == 4.0f);
    CHECK(interaction.offset == 0.2f);

    Modules::SignalViewWaterfall3D waterfall;
    auto& camera = waterfall.*InputMember(WaterfallCameraMember{});
    camera.azimuth = 1.2f;
    camera.distance = 2.5f;
    camera.target = {0.2f, 0.3f, 0.4f};
    const auto before = camera;
    const auto process = InputMember(WaterfallInputMember{});
    (waterfall.*process)({MouseEvent{.type = MouseEventType::Click, .button = MouseButton::Right,
                                    .position = {0.25f, 0.25f}}});
    (waterfall.*process)({MouseEvent{.type = MouseEventType::Release, .button = MouseButton::Right,
                                    .position = {0.25f, 0.25f}, .cancel = true}});
    (waterfall.*process)({MouseEvent{.type = MouseEventType::Move, .position = {0.75f, 0.75f}}});
    CHECK(camera.azimuth == before.azimuth);
    CHECK(camera.elevation == before.elevation);
    CHECK(camera.distance == before.distance);
    CHECK(camera.target == before.target);

    Modules::detail::SignalViewSplitInteraction splitter;
    splitter.dragging = true;
    bool commit = true;
    const auto layout = Modules::detail::CalculateSignalViewPanels({1, 1}, {100, 100}, 0.5f);
    CHECK(splitter.process(MouseEvent{.type = MouseEventType::Release, .cancel = true},
                            layout, {100, 100}, 1.0f, true, commit));
    CHECK_FALSE(splitter.dragging);
    CHECK_FALSE(commit);
}

TEST_CASE("Shutdown cancellation can sample safely in either hook order with reused context storage",
          "[core][sakura][surface][bridge][shutdown-reentry]") {
    const bool samplerFirst = GENERATE(false, true);
    const bool initiallySampled = GENERATE(false, true);
    struct Allocator {
        ImGuiMemAllocFunc allocate;
        ImGuiMemFreeFunc release;
        void* user;
        void* storage;
        bool occupied = false;

        ~Allocator() {
            ImGui::SetAllocatorFunctions(allocate, release, user);
            release(storage, user);
        }
    } allocator;
    ImGui::GetAllocatorFunctions(&allocator.allocate, &allocator.release, &allocator.user);
    allocator.storage = allocator.allocate(sizeof(ImGuiContext), allocator.user);
    ImGui::SetAllocatorFunctions([](size_t size, void* user) -> void* {
        auto& a = *static_cast<Allocator*>(user);
        if (size == sizeof(ImGuiContext) && !a.occupied) {
            a.occupied = true;
            return a.storage;
        }
        return a.allocate(size, a.user);
    }, [](void* pointer, void* user) {
        auto& a = *static_cast<Allocator*>(user);
        if (pointer == a.storage) a.occupied = false;
        else a.release(pointer, a.user);
    }, &allocator);
    Sakura::InputFrame sampled;
    U64 generation = 0;
    for (int lifetime = 0; lifetime < 2; ++lifetime) {
        detail::SurfaceInputState state;
        Sakura::InputFrame duringShutdown;
        U64 cancellations = 0;
        bool destroying = false;
        auto ui = std::make_unique<SakuraTest::HeadlessUi>();
        CHECK(ImGui::GetCurrentContext() == allocator.storage);
        const auto emit = [&](const InputEvent& event) {
            if (!destroying) return;
            const auto* key = std::get_if<KeyEvent>(&event);
            const auto* mouse = std::get_if<MouseEvent>(&event);
            const auto* focus = std::get_if<FocusEvent>(&event);
            if ((key && key->type == KeyEventType::Release) || (mouse && mouse->cancel) ||
                (focus && !focus->focused)) {
                ++cancellations;
                auto* context = ImGui::GetCurrentContext();
                const int hooks = context->Hooks.Size;
                for (int call = 0; call < 2; ++call) {
                    Sakura::Private::SampleInput(duringShutdown);
                    CHECK_FALSE(duringShutdown.windowFocused);
                    CHECK_FALSE(duringShutdown.pointer);
                    CHECK(duringShutdown.buttons == std::array<bool, 3>{});
                    CHECK(duringShutdown.source == nullptr);
                    CHECK(duringShutdown.sourceGeneration == 0);
                    CHECK(duringShutdown.sourceFrame == -1);
                    CHECK(duringShutdown.events.size() == 1);
                    CHECK(std::get<MouseEvent>(duringShutdown.events[0]).type == MouseEventType::Cancel);
                    CHECK(context->Hooks.Size == hooks);
                }
            }
        };
        const auto draw = [&] {
            if (initiallySampled && samplerFirst) Sakura::Private::SampleInput(sampled);
            ImGui::SetCursorScreenPos({0, 0});
            ImGui::InvisibleButton("shutdown-owner", {100, 100});
            detail::ForwardSurfaceInputEvents({0, 0}, {100, 100}, state, emit);
            if (initiallySampled && !samplerFirst) Sakura::Private::SampleInput(sampled);
        };
        ui->setMouse({25, 25}, false);
        ui->frame(draw);
        ui->setMouse({25, 25}, true);
        ui->frame(draw);
        ImGui::GetIO().AddKeyEvent(ImGuiKey_B, true);
        ui->frame(draw);
        REQUIRE(state.leftPressed);
        if (initiallySampled) {
            CHECK(sampled.sourceGeneration != generation);
            generation = sampled.sourceGeneration;
        }
        destroying = true;
        ui.reset();
        CHECK(cancellations == 3);
        CHECK(duringShutdown.serial == 6);
        CHECK_FALSE(state.leftPressed);
    }
}

TEST_CASE("Default-trickle pointer leave and reentry preserve a subsequent click",
          "[core][sakura][surface][bridge][pointer-reentry]") {
    SakuraTest::HeadlessUi ui;
    Sakura::InputFrame frame;
    Sakura::InputRouter router;
    ui.setMouse({25, 25}, false);
    ui.frame([&] { Sakura::Private::SampleInput(frame); });
    router.route(frame, {0, 0, 100, 100});
    auto& io = ImGui::GetIO();
    REQUIRE(io.ConfigInputTrickleEventQueue);
    io.AddMousePosEvent(-std::numeric_limits<F32>::max(), -std::numeric_limits<F32>::max());
    io.AddMousePosEvent(25, 25);
    io.AddMouseButtonEvent(0, true);
    ui.frame([&] { Sakura::Private::SampleInput(frame); });
    REQUIRE(frame.events.size() == 3);
    CHECK(std::get<MouseEvent>(frame.events[0]).type == MouseEventType::Cancel);
    CHECK(std::get<MouseEvent>(frame.events[1]).type == MouseEventType::Move);
    CHECK(std::get<MouseEvent>(frame.events[2]).type == MouseEventType::Click);
    const auto output = router.route(frame, {0, 0, 100, 100});
    CHECK(output.active);
    CHECK(output.hovered);
    REQUIRE(output.events.size() == 3);
    CHECK(std::get<MouseEvent>(output.events[2]).type == MouseEventType::Click);
    CHECK(std::get<MouseEvent>(output.events[2]).position == Extent2D<F32>{0.25f, 0.25f});
}

TEST_CASE("Cancellation preserves later key text and pointer events in the same batch",
          "[core][sakura][surface][bridge][cancel-batch]") {
    Sakura::InputFrame frame;
    Sakura::InputRouter router;
    frame.advance(0.1f);
    frame.pointer = Extent2D<F32>{25, 25};
    frame.buttons[0] = true;
    frame.events = {KeyEvent{KeyEventType::Press, KeyCode::A, {}},
                    MouseEvent{.type = MouseEventType::Click, .position = {25, 25}}};
    REQUIRE(router.route(frame, {0, 0, 100, 100}).active);
    frame.advance(0.1f);
    frame.buttons[0] = false;
    frame.buttons[1] = true;
    frame.events = {MouseEvent{.type = MouseEventType::Cancel},
                    KeyEvent{KeyEventType::Press, KeyCode::B, {}}, TextEvent{U'x', {}},
                    MouseEvent{.type = MouseEventType::Move, .position = {25, 25}},
                    MouseEvent{.type = MouseEventType::Click, .button = MouseButton::Right, .position = {25, 25}}};
    const auto output = router.route(frame, {0, 0, 100, 100});
    CHECK(output.active);
    REQUIRE(output.events.size() == 7);
    CHECK(std::get<KeyEvent>(output.events[0]).type == KeyEventType::Release);
    CHECK(std::get<MouseEvent>(output.events[1]).cancel);
    CHECK(std::get<MouseEvent>(output.events[2]).type == MouseEventType::Cancel);
    CHECK(std::get<KeyEvent>(output.events[3]).key == KeyCode::B);
    CHECK(std::get<KeyEvent>(output.events[3]).type == KeyEventType::Press);
    CHECK(std::get<TextEvent>(output.events[4]).codepoint == U'x');
    CHECK(std::get<MouseEvent>(output.events[5]).type == MouseEventType::Move);
    CHECK(std::get<MouseEvent>(output.events[6]).button == MouseButton::Right);

    SakuraTest::HeadlessUi ui;
    Sakura::InputFrame sampled;
    Sakura::InputRouter sampledRouter;
    ui.setMouse({25, 25}, false);
    ui.frame([&] { Sakura::Private::SampleInput(sampled); });
    sampledRouter.route(sampled, {0, 0, 100, 100});
    auto& io = ImGui::GetIO();
    io.ConfigInputTrickleEventQueue = false;
    io.AddMousePosEvent(-std::numeric_limits<F32>::max(), -std::numeric_limits<F32>::max());
    io.AddKeyEvent(ImGuiKey_B, true);
    io.AddInputCharacter('x');
    ui.frame([&] { Sakura::Private::SampleInput(sampled); });
    const auto routed = sampledRouter.route(sampled, {0, 0, 100, 100});
    REQUIRE(routed.events.size() == 3);
    CHECK(std::get<MouseEvent>(routed.events[0]).type == MouseEventType::Cancel);
    CHECK(std::get<KeyEvent>(routed.events[1]).type == KeyEventType::Press);
    CHECK(std::get<TextEvent>(routed.events[2]).codepoint == U'x');
    io.AddKeyEvent(ImGuiKey_B, false);
    ui.setMouse({-std::numeric_limits<F32>::max(), -std::numeric_limits<F32>::max()}, false);
    ui.frame([&] { Sakura::Private::SampleInput(sampled); });
    REQUIRE(sampled.events.size() == 1);
    CHECK(std::get<KeyEvent>(sampled.events[0]).type == KeyEventType::Release);
    const auto released = sampledRouter.route(sampled, {0, 0, 100, 100});
    REQUIRE(Only<KeyEvent>(released.events).size() == 1);
    CHECK(Only<KeyEvent>(released.events)[0].type == KeyEventType::Release);
}

TEST_CASE("Legacy callback epochs keep normal delivery and cancellation with one recipient",
          "[core][sakura][surface][bridge][recipient]") {
    const bool pressedBeforeReplacement = GENERATE(false, true);
    const bool hidden = GENERATE(false, true);
    SakuraTest::HeadlessUi ui;
    std::vector<InputEvent> first;
    std::vector<InputEvent> second;
    Sakura::SurfaceView view;
    Sakura::SurfaceView::Config config;
    config.id = "recipient-epoch";
    config.texture = 1;
    config.size = {100, 100};
    config.onInput = [&](const InputEvent& event) { first.push_back(event); };
    const auto draw = [&] {
        view.update(config);
        ImGui::SetCursorScreenPos({0, 0});
        view.render(ui.sakura());
    };
    const auto count = [](const std::vector<InputEvent>& events, KeyCode code, KeyEventType type) {
        return std::count_if(events.begin(), events.end(), [&](const auto& event) {
            const auto* key = std::get_if<KeyEvent>(&event);
            return key && key->key == code && key->type == type;
        });
    };
    ui.setMouse({25, 25}, false);
    ui.frame(draw);
    ui.setMouse({25, 25}, true);
    ui.frame(draw);
    ui.setMouse({25, 25}, false);
    ui.frame(draw);
    if (pressedBeforeReplacement) {
        ImGui::GetIO().AddKeyEvent(ImGuiKey_B, true);
        ui.frame(draw);
    }
    config.onInput = [&](const InputEvent& event) { second.push_back(event); };
    if (!pressedBeforeReplacement) ImGui::GetIO().AddKeyEvent(ImGuiKey_B, true);
    ui.frame(draw);
    REQUIRE(count(first, KeyCode::B, KeyEventType::Press) == 1);
    CHECK(count(second, KeyCode::B, KeyEventType::Press) == 0);
    for (int frame = 0; frame < 40; ++frame) ui.frame(draw);
    CHECK(count(first, KeyCode::B, KeyEventType::Press) > 1);
    CHECK(count(first, KeyCode::B, KeyEventType::Release) == 0);
    CHECK(count(second, KeyCode::B, KeyEventType::Press) == 0);
    if (hidden) {
        ui.frame([] {});
        ImGui::GetIO().AddKeyEvent(ImGuiKey_B, false);
    } else {
        ImGui::GetIO().AddKeyEvent(ImGuiKey_B, false);
        ui.frame(draw);
        ui.frame([] {});
    }
    CHECK(count(first, KeyCode::B, KeyEventType::Release) == 1);
    CHECK(count(second, KeyCode::B, KeyEventType::Release) == 0);
    ui.frame(draw);
    ui.setMouse({25, 25}, true);
    ui.frame(draw);
    ui.setMouse({25, 25}, false);
    ui.frame(draw);
    ImGui::GetIO().AddKeyEvent(ImGuiKey_C, true);
    ui.frame(draw);
    REQUIRE(count(second, KeyCode::C, KeyEventType::Press) == 1);
    ui.frame([] {});
    CHECK(count(second, KeyCode::C, KeyEventType::Release) == 1);
    CHECK(count(first, KeyCode::C, KeyEventType::Press) == 0);
    CHECK(count(first, KeyCode::C, KeyEventType::Release) == 0);
}

TEST_CASE("Flattened missing and nonfinite pointers cancel every owned button",
          "[core][sakura][surface][bridge][flat-invalid]") {
    const F32 invalid = GENERATE(-std::numeric_limits<F32>::max(),
                                 std::numeric_limits<F32>::quiet_NaN(),
                                 std::numeric_limits<F32>::infinity());
    const bool vertical = GENERATE(false, true);
    SakuraTest::HeadlessUi ui;
    Sakura::InputFrame frame;
    Sakura::InputRouter router;
    ui.setMouse({25, 25}, true);
    auto& io = ImGui::GetIO();
    io.MouseDown[1] = true;
    io.MouseDown[2] = true;
    ui.frame([&] { Sakura::Private::SampleInput(frame); });
    const auto pressed = router.route(frame, {0, 0, 100, 100});
    REQUIRE(pressed.active);
    REQUIRE(pressed.middle);
    ui.frame([&] {
        io.MousePos = vertical ? ImVec2(25, invalid) : ImVec2(invalid, 25);
        Sakura::Private::SampleInput(frame);
        io.MousePos = {25, 25};
    });
    REQUIRE_FALSE(frame.pointer);
    const auto sampledMouse = Only<MouseEvent>(frame.events);
    REQUIRE(sampledMouse.size() == 1);
    CHECK(sampledMouse[0].type == MouseEventType::Cancel);
    CHECK(frame.buttons == std::array<bool, 3>{true, true, true});
    const auto cancelled = router.route(frame, {0, 0, 100, 100});
    CHECK_FALSE(cancelled.active);
    CHECK_FALSE(cancelled.middle);
    CHECK_FALSE(cancelled.hovered);
    const auto mouse = Only<MouseEvent>(cancelled.events);
    REQUIRE(mouse.size() == 4);
    for (U64 index = 0; index < 3; ++index) {
        CHECK(mouse[index].type == MouseEventType::Release);
        CHECK(mouse[index].button == static_cast<MouseButton>(index));
        CHECK(mouse[index].cancel);
    }
    CHECK(mouse[3].type == MouseEventType::Cancel);
    ui.frame([&] { Sakura::Private::SampleInput(frame); });
    const auto recovered = router.route(frame, {0, 0, 100, 100});
    CHECK_FALSE(recovered.active);
    CHECK_FALSE(recovered.middle);

    Sakura::InputFrame manual;
    Sakura::InputRouter manualRouter;
    manual.pointer = Extent2D<F32>{25, 25};
    manual.buttons[0] = true;
    manual.events = {MouseEvent{.type = MouseEventType::Click, .position = {25, 25}}};
    REQUIRE(manualRouter.route(manual, {0, 0, 100, 100}).active);
    manual.advance(0.1f);
    if (invalid == -std::numeric_limits<F32>::max()) manual.pointer.reset();
    else manual.pointer = vertical ? Extent2D<F32>{25, invalid} : Extent2D<F32>{invalid, 25};
    const auto manualCancelled = manualRouter.route(manual, {0, 0, 100, 100});
    CHECK_FALSE(manualCancelled.active);
    REQUIRE(Only<MouseEvent>(manualCancelled.events).size() == 2);
    CHECK(Only<MouseEvent>(manualCancelled.events)[0].cancel);
}

TEST_CASE("SignalView marker clearing followed by a cancelled left release preserves the clear",
          "[core][sakura][surface][signal-cancel]") {
    const bool enabled = GENERATE(false, true);
    SakuraTest::HeadlessUi ui;
    Modules::SignalView config;
    config.markers = {0.0f};
    Modules::SignalViewCanvas canvas;
    canvas.reset(config);
    canvas.axis = std::make_shared<Render::Components::Axis>(Render::Components::Axis::Config{});
    const Modules::SignalViewCanvas::Context context{config, {}, 256};
    U64 requests = 0;
    const Modules::SignalViewCanvas::Edits edits{
        .pending = [&] { return requests > 0; },
        .enabled = [&](const std::string&) { return enabled; },
        .request = [&](const Parser::Map&) { ++requests; return Result::SUCCESS; },
    };
    std::vector<InputEvent> events;
    detail::SurfaceInputState input;
    const auto draw = [&] {
        ImGui::SetCursorScreenPos({0, 0});
        ImGui::InvisibleButton("marker-input", {100, 100},
                               ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
        detail::ForwardSurfaceInputEvents({0, 0}, {100, 100}, input,
            [&](const InputEvent& event) { events.push_back(event); });
    };
    ui.setMouse({50, 25}, false);
    ui.frame(draw);
    events.clear();
    ui.setMouse({50, 25}, true);
    ui.frame(draw);
    canvas.processInputEvents(std::move(events), context, edits);
    REQUIRE(canvas.markerDrag.index == 0);
    auto& io = ImGui::GetIO();
    REQUIRE(io.ConfigInputTrickleEventQueue);
    io.AddKeyEvent(ImGuiMod_Shift, true);
    io.AddMouseButtonEvent(1, true);
    io.AddMouseButtonEvent(0, false);
    ui.setMouse({50, 25}, false);
    events.clear();
    ui.frame(draw);
    REQUIRE(events.size() == 3);
    REQUIRE(std::get<MouseEvent>(events[0]).type == MouseEventType::Click);
    REQUIRE(std::get<MouseEvent>(events[0]).button == MouseButton::Right);
    REQUIRE(std::get<MouseEvent>(events[0]).modifiers.shift);
    REQUIRE(std::get<MouseEvent>(events[1]).cancel);
    REQUIRE(std::get<MouseEvent>(events[2]).type == MouseEventType::Move);
    canvas.processInputEvents(std::move(events), context, edits);
    CHECK(canvas.markerPositions.empty());
    CHECK_FALSE(canvas.markerDrag.index);
    CHECK(config.markers == std::vector<F32>{0.0f});
    CHECK(requests == (enabled ? 1 : 0));
}

TEST_CASE("Legacy canvas cancellation stops retained scrollbar drags through every reset boundary",
          "[core][sakura][surface][retained-cancel][scroll-cancel]") {
    const int boundary = GENERATE(0, 1, 2, 3, 4);
    const bool horizontal = GENERATE(false, true);
    CAPTURE(boundary, horizontal);
    F32 value = 0.0f;
    U64 changes = 0;
    Sakura::Retained::ScrollView scroll;
    scroll.update({.id = "cancel-scroll", .contentWidth = horizontal ? 1000.0f : 200.0f,
                   .contentHeight = horizontal ? 200.0f : 1000.0f, .thickness = 10, .margin = 0,
                   .onScrollY = [&](F32 next) { value = next; ++changes; },
                   .onScrollX = [&](F32 next) { value = next; ++changes; }});
    RetainedInputRoot root(scroll);
    Sakura::Retained::Canvas canvas;
    canvas.mount(root);
    canvas.update({.id = "scroll-canvas", .size = {256, 256}});
    RetainedSurface surface(canvas, [&](const auto& ctx) { canvas.render(ctx); });
    const ImVec2 start = horizontal ? ImVec2(25, 215) : ImVec2(215, 25);
    const ImVec2 drag = horizontal ? ImVec2(70, 215) : ImVec2(215, 70);
    const ImVec2 later = horizontal ? ImVec2(180, 215) : ImVec2(215, 180);
    surface.frame(start, false);
    surface.frame(start, true);
    surface.frame(drag, true);
    REQUIRE(value > 0.0f);
    const F32 before = value;
    const U64 callbacks = changes;
    root.sink.setVisible(false);
    surface.cancel(boundary);
    CHECK(root.sink.cancellations == 1);
    CHECK(value == before);
    CHECK(changes == callbacks);
    surface.frame(later, false);
    CHECK(value == before);
    CHECK(changes == callbacks);
}

TEST_CASE("Legacy canvas cancellation ends retained text selection without editing or reselection",
          "[core][sakura][surface][retained-cancel][selection-cancel]") {
    const int boundary = GENERATE(0, 1, 2, 3, 4);
    U64 edits = 0;
    CAPTURE(boundary);
    U64 selections = 0;
    Sakura::Retained::TextGrid grid;
    grid.update({.id = "cancel-grid", .value = "abcdefghijklmnopqrstuvwxyz", .editable = true,
                 .fontSize = 15, .lineNumbers = false, .scrollbar = false,
                 .padding = Sakura::Padding{},
                 .onChange = [&](const auto&) { ++edits; },
                 .onSelect = [&](auto, auto) { ++selections; }});
    RetainedInputRoot root(grid);
    Sakura::Retained::Canvas canvas;
    canvas.mount(root);
    canvas.update({.id = "grid-canvas", .size = {256, 256}});
    RetainedSurface surface(canvas, [&](const auto& ctx) { canvas.render(ctx); });
    surface.frame({25, 30}, false);
    surface.frame({25, 30}, true);
    surface.frame({80, 30}, true);
    const auto cursor = grid.cursor();
    REQUIRE(cursor.column > 0);
    const U64 callbacks = selections;
    surface.cancel(boundary);
    CHECK(root.sink.cancellations == 1);
    CHECK(edits == 0);
    CHECK(selections == callbacks);
    surface.frame({170, 30}, false);
    CHECK(grid.cursor() == cursor);
    CHECK(edits == 0);
    CHECK(selections == callbacks);
}

TEST_CASE("Legacy canvas cancellation clears retained button presses without activation",
          "[core][sakura][surface][retained-cancel][button-cancel]") {
    const int boundary = GENERATE(0, 1, 2, 3, 4);
    U64 clicks = 0;
    CAPTURE(boundary);
    Sakura::Retained::Button button;
    button.update({.id = "cancel-button", .onClick = [&] { ++clicks; }});
    RetainedInputRoot root(button);
    Sakura::Retained::Canvas canvas;
    canvas.mount(root);
    canvas.update({.id = "button-canvas", .size = {256, 256}});
    RetainedSurface surface(canvas, [&](const auto& ctx) { canvas.render(ctx); });
    surface.frame({50, 50}, false);
    surface.frame({50, 50}, true);
    surface.cancel(boundary);
    CHECK(root.sink.cancellations == 1);
    CHECK(clicks == 0);
    surface.frame({60, 60}, false);
    CHECK(clicks == 0);
    CHECK_FALSE(SendRetained(canvas, MouseEvent{.type = MouseEventType::Release, .position = {0.2f, 0.2f}}));
    CHECK(clicks == 0);
    REQUIRE(SendRetained(canvas, MouseEvent{.type = MouseEventType::Click, .position = {0.2f, 0.2f}}));
    REQUIRE(SendRetained(canvas, MouseEvent{.type = MouseEventType::Release, .position = {0.2f, 0.2f}}));
    CHECK(clicks == 1);
}

TEST_CASE("Legacy code editor cancellation discards the console resize preview",
          "[core][sakura][surface][retained-cancel][resize-cancel]") {
    const int boundary = GENERATE(0, 1, 2, 3, 4);
    U64 changes = 0;
    CAPTURE(boundary);
    Sakura::Retained::CodeEditor editor;
    editor.update({.id = "cancel-editor", .size = {256, 256}, .consoleVisible = true,
                   .onChange = [&](const auto&) { ++changes; }});
    auto& canvas = (editor.*RetainedMember(CodeEditorCanvas{}))();
    RetainedSurface surface(canvas, [&](const auto& ctx) { editor.render(ctx); });
    surface.frame({20, 20}, false);
    REQUIRE(SendRetained(canvas, MouseEvent{.type = MouseEventType::Click, .position = {0.91f, 0.95f}}));
    surface.frame({20, 20}, false);
    const F32 headerY = (256.0f - Sakura::Retained::StatusBar::Height - 128.0f - 6.0f) / 256.0f;
    const ImVec2 start{128, headerY * 256.0f};
    surface.frame(start, true);
    REQUIRE(surface.consumed.back());
    surface.frame({128, start.y + 30}, true);
    REQUIRE(surface.consumed.back());
    surface.cancel(boundary);
    CHECK(changes == 0);
    CHECK_FALSE(SendRetained(canvas, MouseEvent{.type = MouseEventType::Move, .position = {0.5f, 0.05f}}));
    surface.frame({20, 20}, false);
    ImGui::SetMouseCursor(ImGuiMouseCursor_Arrow);
    REQUIRE(SendRetained(canvas, MouseEvent{.type = MouseEventType::Click, .position = {0.5f, headerY}}));
    REQUIRE(SendRetained(canvas, MouseEvent{.type = MouseEventType::Move, .position = {0.5f, 0.05f}}));
    CHECK(ImGui::GetMouseCursor() == ImGuiMouseCursor_ResizeNS);
    CHECK_FALSE(SendRetained(canvas, MouseEvent{.type = MouseEventType::Cancel,
                                               .position = {std::numeric_limits<F32>::quiet_NaN(), -1}}));
    CHECK_FALSE(SendRetained(canvas, MouseEvent{.type = MouseEventType::Move, .position = {0.5f, 0.05f}}));
    CHECK(changes == 0);
}
