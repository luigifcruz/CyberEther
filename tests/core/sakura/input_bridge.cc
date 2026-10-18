#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <jetstream/render/sakura/clipboard.hh>
#include <jetstream/render/sakura/base.hh>
#include <imgui_internal.h>

#include "harness.hh"
#include "scene.hh"
#include "render/sakura/input.hh"
#include "render/sakura/detail/input_router.hh"
#include "domains/visualization/frame/module_impl.hh"
#include "domains/visualization/signal_view/waterfall_3d.hh"

#include <algorithm>
#include <limits>
#include <vector>

using namespace Jetstream;

namespace {

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

}  // namespace

TEST_CASE("The ImGui bridge keeps native Control and Command names and modifiers on every platform",
          "[core][sakura][input][bridge]") {
    const bool macOS = GENERATE(false, true);
    SakuraTest::HeadlessUi ui;
    auto& io = ImGui::GetIO();
    io.ConfigInputTrickleEventQueue = false;
    io.ConfigMacOSXBehaviors = macOS;
    Sakura::InputFrame frame;

    io.AddKeyEvent(ImGuiMod_Ctrl, true);
    io.AddKeyEvent(ImGuiKey_LeftCtrl, true);
    io.AddKeyEvent(ImGuiMod_Super, true);
    io.AddKeyEvent(ImGuiKey_RightSuper, true);
    io.AddKeyEvent(ImGuiMod_Alt, true);
    io.AddKeyEvent(ImGuiMod_Shift, true);
    ui.frame([&] { Sakura::Private::SampleInput(frame); });

    const auto keys = Only<KeyEvent>(frame.events);
    REQUIRE(keys.size() == 2);
    CHECK(keys[0].key == KeyCode::LeftCtrl);
    CHECK(keys[0].modifiers.control);
    CHECK_FALSE(keys[0].modifiers.super);
    CHECK(keys[1].key == KeyCode::RightSuper);
    CHECK(keys[1].modifiers.control);
    CHECK(keys[1].modifiers.super);
    CHECK(frame.modifiers.control);
    CHECK(frame.modifiers.shift);
    CHECK(frame.modifiers.alt);
    CHECK(frame.modifiers.super);

    io.AddMouseWheelEvent(-2.0f, 1.0f);
    ui.frame([&] { Sakura::Private::SampleInput(frame); });
    const auto mouse = Only<MouseEvent>(frame.events);
    REQUIRE(mouse.size() == 1);
    CHECK(mouse[0].type == MouseEventType::Scroll);
    CHECK(mouse[0].scroll == Extent2D<F32>{-2.0f, 1.0f});
    CHECK(mouse[0].modifiers.control);
    CHECK(mouse[0].modifiers.shift);
    CHECK(mouse[0].modifiers.alt);
    CHECK(mouse[0].modifiers.super);
}

TEST_CASE("The ImGui bridge samples ordered input with native modifiers once per frame",
          "[core][sakura][input][bridge]") {
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

TEST_CASE("The ImGui bridge keeps each wheel delta, key and character in order with the modifiers held at the time",
          "[core][sakura][input][bridge]") {
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

TEST_CASE("Shutdown sampling is safe before or after the input source hook with reused context storage",
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
    U64 generation = 0;
    for (int lifetime = 0; lifetime < 2; ++lifetime) {
        auto ui = std::make_unique<SakuraTest::HeadlessUi>();
        REQUIRE(ImGui::GetCurrentContext() == allocator.storage);
        Sakura::InputFrame sampled;
        Sakura::InputFrame duringShutdown;
        if (initiallySampled && samplerFirst) {
            ui->frame([&] { Sakura::Private::SampleInput(sampled); });
        }
        ImGuiContextHook hook;
        hook.Type = ImGuiContextHookType_Shutdown;
        hook.UserData = &duringShutdown;
        hook.Callback = [](ImGuiContext* context, ImGuiContextHook* hook) {
            auto& frame = *static_cast<Sakura::InputFrame*>(hook->UserData);
            const int hooks = context->Hooks.Size;
            for (int call = 0; call < 2; ++call) {
                Sakura::Private::SampleInput(frame);
                CHECK_FALSE(frame.windowFocused);
                CHECK_FALSE(frame.pointer);
                CHECK(frame.buttons == std::array<bool, 3>{});
                CHECK(frame.source == nullptr);
                CHECK(frame.sourceGeneration == 0);
                CHECK(frame.sourceFrame == -1);
                REQUIRE(frame.events.size() == 1);
                CHECK(std::get<MouseEvent>(frame.events[0]).type == MouseEventType::Cancel);
                CHECK(context->Hooks.Size == hooks);
            }
        };
        ImGui::AddContextHook(ImGui::GetCurrentContext(), &hook);
        if (initiallySampled && !samplerFirst) {
            ui->frame([&] { Sakura::Private::SampleInput(sampled); });
        }
        if (initiallySampled) {
            CHECK(sampled.sourceGeneration != generation);
            generation = sampled.sourceGeneration;
        }
        ui.reset();
        CHECK(duringShutdown.serial == 2);
        CHECK(ImGui::GetCurrentContext() == nullptr);
    }
}

TEST_CASE("SignalView marker clearing followed by a cancelled left release preserves the clear",
          "[core][sakura][surface][bridge][signal-cancel]") {
    const bool enabled = GENERATE(false, true);
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
    canvas.processInputEvents({MouseEvent{.type = MouseEventType::Click, .button = MouseButton::Left,
                                         .position = {0.5f, 0.25f}}}, context, edits);
    REQUIRE(canvas.markerDrag.index == 0);
    canvas.processInputEvents({
        MouseEvent{.type = MouseEventType::Click, .button = MouseButton::Right,
                   .position = {0.5f, 0.25f}, .modifiers = {.shift = true}},
        MouseEvent{.type = MouseEventType::Release, .button = MouseButton::Left,
                   .position = {0.5f, 0.25f}, .cancel = true},
        MouseEvent{.type = MouseEventType::Move, .position = {0.5f, 0.25f}},
    }, context, edits);
    CHECK(canvas.markerPositions.empty());
    CHECK_FALSE(canvas.markerDrag.index);
    CHECK(config.markers == std::vector<F32>{0.0f});
    CHECK(requests == (enabled ? 1 : 0));
}

TEST_CASE("Retained scrollbar cancellation stops thumb dragging without another scroll callback",
          "[core][sakura][input][widget-cancel]") {
    const bool horizontal = GENERATE(false, true);
    SakuraTest::Scene scene;
    F32 value = 0.0f;
    U64 changes = 0;
    Sakura::Scroller scroll;
    scroll.update({.contentWidth = horizontal ? 1000.0f : 200.0f,
                   .contentHeight = horizontal ? 200.0f : 1000.0f, .thickness = 10, .margin = 0,
                   .onScrollY = [&](F32 next) { value = next; ++changes; },
                   .onScrollX = [&](F32 next) { value = next; ++changes; }});
    SakuraTest::Host host(scroll);
    host.place(scene.ctx, {0, 0, 200, 200});
    const Extent2D<F32> start = horizontal ? Extent2D<F32>{25, 195} : Extent2D<F32>{195, 25};
    const Extent2D<F32> drag = horizontal ? Extent2D<F32>{70, 195} : Extent2D<F32>{195, 70};
    REQUIRE(host.send(MouseEvent{.type = MouseEventType::Click, .position = start}));
    REQUIRE(host.send(MouseEvent{.type = MouseEventType::Move, .position = drag}));
    REQUIRE(value > 0.0f);
    const F32 before = value;
    const U64 callbacks = changes;
    host.send(MouseEvent{.type = MouseEventType::Cancel});
    host.send(MouseEvent{.type = MouseEventType::Move, .position = {180, 180}});
    host.send(MouseEvent{.type = MouseEventType::Release, .position = {180, 180}});
    CHECK(value == before);
    CHECK(changes == callbacks);
}

TEST_CASE("Retained text selection cancellation does not edit or resume on later motion",
          "[core][sakura][input][widget-cancel]") {
    SakuraTest::Scene scene;
    U64 edits = 0;
    U64 selections = 0;
    Sakura::TextGrid grid;
    grid.update({.value = "abcdefghijklmnopqrstuvwxyz", .editable = true,
                 .fontSize = 15, .lineNumbers = false, .scrollbar = false, .padding = Sakura::Padding{},
                 .onChange = [&](const auto&) { ++edits; },
                 .onSelect = [&](auto, auto) { ++selections; }});
    SakuraTest::Host host(grid);
    host.place(scene.ctx, {0, 0, 256, 100});
    host.send(MouseEvent{.type = MouseEventType::Click, .position = {25, 30}});
    host.send(MouseEvent{.type = MouseEventType::Move, .position = {80, 30}});
    const auto cursor = grid.cursor();
    REQUIRE(cursor.column > 0);
    const U64 callbacks = selections;
    host.send(MouseEvent{.type = MouseEventType::Cancel});
    host.send(MouseEvent{.type = MouseEventType::Move, .position = {170, 30}});
    host.send(MouseEvent{.type = MouseEventType::Release, .position = {170, 30}});
    CHECK(grid.cursor() == cursor);
    CHECK(edits == 0);
    CHECK(selections == callbacks);
}

TEST_CASE("Retained console resize cancellation discards its preview and ignores later release",
          "[core][sakura][input][widget-cancel]") {
    SakuraTest::Scene scene;
    U64 changes = 0;
    Sakura::CodeEditor editor;
    editor.update({.height = 256, .consoleVisible = true, .onChange = [&](const auto&) { ++changes; }});
    SakuraTest::Host host(editor);
    const auto layout = [&] { host.place(scene.ctx, {0, 0, 256, 256}); };
    layout();
    const auto bars = SakuraTest::Find<Sakura::StatusBar>(editor);
    REQUIRE(bars.size() == 1);
    const auto bar = SakuraTest::Peek::Frame(*bars.front());
    const Extent2D<F32> toggle{bar.right() - 15.0f, bar.y + bar.height * 0.5f};
    REQUIRE(host.send(MouseEvent{.type = MouseEventType::Click, .position = toggle}));
    host.send(MouseEvent{.type = MouseEventType::Release, .position = toggle});
    layout();
    const auto consoles = SakuraTest::Find<Sakura::TextView>(editor);
    REQUIRE(consoles.size() == 1);
    const auto before = SakuraTest::Peek::Frame(*consoles.front());
    const Extent2D<F32> start{128, before.y - 6.0f};
    REQUIRE(host.send(MouseEvent{.type = MouseEventType::Click, .position = start}));
    REQUIRE(host.send(MouseEvent{.type = MouseEventType::Move, .position = {128, start.y + 30}}));
    layout();
    CHECK(SakuraTest::Peek::Frame(*consoles.front()) != before);
    host.send(MouseEvent{.type = MouseEventType::Cancel});
    layout();
    CHECK(SakuraTest::Peek::Frame(*consoles.front()) == before);
    host.send(MouseEvent{.type = MouseEventType::Release, .position = {128, start.y + 30}});
    layout();
    CHECK(SakuraTest::Peek::Frame(*consoles.front()) == before);
    CHECK(changes == 0);
}

TEST_CASE("The ImGui bridge samples a frame once however often it is asked",
          "[core][sakura][input][bridge]") {
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
