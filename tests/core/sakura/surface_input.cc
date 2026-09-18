#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <jetstream/render/sakura/components/surface_view.hh>
#include <jetstream/render/sakura/components/window.hh>

#include "harness.hh"
#include "superluminal/surface_interaction.hh"

#include <algorithm>
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
