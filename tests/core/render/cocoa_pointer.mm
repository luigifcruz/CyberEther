#import <AppKit/AppKit.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_session.hpp>
#include <memory>

#include "harness.hh"

#define private public
#include "../../../src/viewport/glfw/tools/glfw_input.hh"
#undef private

#define GLFW_INCLUDE_NONE
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

namespace {

NSMutableArray* eventMonitors = nil;
void* inputUserPointer = nullptr;

void* TestWindowUserPointer(GLFWwindow*) {
    return inputUserPointer;
}

}  // namespace

@class PointerTestWindow;

@interface PointerTestEvent : NSObject {
 @public
    PointerTestWindow* eventWindow;
    NSEventPhase eventPhase;
    NSEventPhase eventMomentum;
    NSEventModifierFlags eventModifiers;
    CGFloat deltaX;
    CGFloat deltaY;
}
+ (id)addLocalMonitorForEventsMatchingMask:(NSEventMask)mask handler:(PointerTestEvent* (^)(PointerTestEvent*))handler;
+ (void)removeMonitor:(id)token;
- (PointerTestWindow*)window;
- (NSEventPhase)phase;
- (NSEventPhase)momentumPhase;
- (NSEventModifierFlags)modifierFlags;
- (CGFloat)scrollingDeltaX;
- (CGFloat)scrollingDeltaY;
@end

@implementation PointerTestEvent
+ (id)addLocalMonitorForEventsMatchingMask:(NSEventMask)mask handler:(PointerTestEvent* (^)(PointerTestEvent*))handler {
    (void)mask;
    id token = [handler copy];
    [eventMonitors addObject:token];
    return [token autorelease];
}
+ (void)removeMonitor:(id)token { [eventMonitors removeObjectIdenticalTo:token]; }
- (PointerTestWindow*)window { return eventWindow; }
- (NSEventPhase)phase { return eventPhase; }
- (NSEventPhase)momentumPhase { return eventMomentum; }
- (NSEventModifierFlags)modifierFlags { return eventModifiers; }
- (CGFloat)scrollingDeltaX { return deltaX; }
- (CGFloat)scrollingDeltaY { return deltaY; }
@end

@interface PointerTestView : NSObject {
 @public
    int areas;
}
- (void)addTrackingArea:(NSTrackingArea*)area;
- (void)removeTrackingArea:(NSTrackingArea*)area;
@end

@implementation PointerTestView
- (void)addTrackingArea:(NSTrackingArea*)area { (void)area; ++areas; }
- (void)removeTrackingArea:(NSTrackingArea*)area { (void)area; --areas; }
@end

@interface PointerTestWindow : NSObject {
 @public
    PointerTestView* view;
    BOOL key;
}
- (id)contentView;
- (BOOL)isKeyWindow;
@end

@implementation PointerTestWindow
- (id)contentView { return view; }
- (BOOL)isKeyWindow { return key; }
@end

namespace {

id TestCocoaWindow(GLFWwindow* handle) {
    return reinterpret_cast<id>(handle);
}

}  // namespace

#define NSEvent PointerTestEvent
#define NSWindow PointerTestWindow
#define glfwGetCocoaWindow TestCocoaWindow
#define JSTInactiveHover PointerTestInactiveHover
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcompare-distinct-pointer-types"
#pragma clang diagnostic ignored "-Wincompatible-pointer-types"
#include "../../../src/viewport/glfw/tools/cocoa_pointer.mm"
#pragma clang diagnostic pop
#undef JSTInactiveHover
#undef glfwGetCocoaWindow
#undef NSWindow
#undef NSEvent

#define glfwGetWindowUserPointer TestWindowUserPointer
#include "../../../src/viewport/glfw/tools/glfw_input.cc"
#undef glfwGetWindowUserPointer

namespace {

using Jetstream::ScrollPhase;
using Jetstream::KeyModifiers;
using Jetstream::Render::InputQueue;
using Jetstream::Render::InputState;
using Jetstream::Viewport::GlfwInput;

class PointerFixture {
 public:
    PointerFixture() {
        eventMonitors = [[NSMutableArray alloc] init];
        a = make();
        b = make();
        pointer = std::make_unique<Jetstream::Viewport::CocoaPointer>(reinterpret_cast<GLFWwindow*>(a),
                                                                      []() {});
    }

    ~PointerFixture() {
        pointer.reset();
        [a->view release];
        [b->view release];
        [a release];
        [b release];
        [eventMonitors release];
        eventMonitors = nil;
        [pool drain];
    }

    static PointerTestWindow* make() {
        auto* window = [[PointerTestWindow alloc] init];
        window->view = [[PointerTestView alloc] init];
        window->key = YES;
        return window;
    }

    void deliver(PointerTestWindow* window, NSEventPhase phase, NSEventPhase momentum, CGFloat dy,
                 NSEventModifierFlags modifiers = 0) {
        auto* event = [[[PointerTestEvent alloc] init] autorelease];
        event->eventWindow = window;
        event->eventPhase = phase;
        event->eventMomentum = momentum;
        event->eventModifiers = modifiers;
        event->deltaY = dy;
        NSArray* monitors = [eventMonitors copy];
        for (id token in monitors) {
            auto handler = (PointerTestEvent* (^)(PointerTestEvent*))token;
            handler(event);
        }
        [monitors release];
    }

    NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
    PointerTestWindow* a = nil;
    PointerTestWindow* b = nil;
    std::unique_ptr<Jetstream::Viewport::CocoaPointer> pointer;
};

class PointerInputFixture : public PointerFixture {
 public:
    PointerInputFixture() : input(reinterpret_cast<GLFWwindow*>(a), viewport) {
        pointer.reset();
        inputUserPointer = &input;
        input.pointer = std::make_unique<Jetstream::Viewport::CocoaPointer>(input.window, [this]() {
            GlfwInput::Impl::OnScroll(input.window, 0.0, 0.0);
        });
    }

    ~PointerInputFixture() {
        input.pointer.reset();
        inputUserPointer = nullptr;
    }

    RenderTest::Viewport viewport;
    GlfwInput::Impl input;
};

void CheckModifiers(const KeyModifiers& actual, const KeyModifiers& expected) {
    CHECK(actual.control == expected.control);
    CHECK(actual.shift == expected.shift);
    CHECK(actual.alt == expected.alt);
    CHECK(actual.super == expected.super);
}

}  // namespace

TEST_CASE("Cocoa pointers pair each scroll delta with its gesture phase once", "[core][render][cocoa]") {
    PointerFixture fixture;
    CHECK_FALSE(fixture.pointer->take().has_value());

    const struct {
        NSEventPhase phase;
        NSEventPhase momentum;
        ScrollPhase expected;
    } cases[] = {
        {NSEventPhaseMayBegin, NSEventPhaseNone, ScrollPhase::Began},
        {NSEventPhaseBegan, NSEventPhaseNone, ScrollPhase::Began},
        {NSEventPhaseChanged, NSEventPhaseNone, ScrollPhase::Changed},
        {NSEventPhaseEnded, NSEventPhaseNone, ScrollPhase::Ended},
        {NSEventPhaseCancelled, NSEventPhaseNone, ScrollPhase::Ended},
        {NSEventPhaseNone, NSEventPhaseBegan, ScrollPhase::Momentum},
        {NSEventPhaseNone, NSEventPhaseChanged, ScrollPhase::Momentum},
        {NSEventPhaseNone, NSEventPhaseEnded, ScrollPhase::MomentumEnded},
        {NSEventPhaseNone, NSEventPhaseCancelled, ScrollPhase::MomentumEnded},
        {NSEventPhaseNone, NSEventPhaseNone, ScrollPhase::None},
    };
    for (const auto& entry : cases) {
        CAPTURE(static_cast<int>(entry.expected));
        fixture.deliver(fixture.a, entry.phase, entry.momentum, 2.0);
        const auto scroll = fixture.pointer->take();
        REQUIRE(scroll.has_value());
        CHECK(scroll->phase == entry.expected);
        CHECK_FALSE(fixture.pointer->take().has_value());
    }

    fixture.deliver(fixture.a, NSEventPhaseBegan, NSEventPhaseNone, 1.0);
    fixture.deliver(fixture.a, NSEventPhaseChanged, NSEventPhaseNone, 3.0);
    const auto scroll = fixture.pointer->take();
    REQUIRE(scroll.has_value());
    CHECK(scroll->phase == ScrollPhase::Changed);
}

TEST_CASE("Cocoa pointers ignore scrolls aimed at other windows", "[core][render][cocoa]") {
    PointerFixture fixture;
    fixture.deliver(fixture.a, NSEventPhaseBegan, NSEventPhaseNone, 2.0, NSEventModifierFlagShift);
    fixture.deliver(fixture.b, NSEventPhaseChanged, NSEventPhaseNone, 2.0, NSEventModifierFlagControl);
    const auto scroll = fixture.pointer->take();
    REQUIRE(scroll.has_value());
    CHECK(scroll->phase == ScrollPhase::Began);
    CheckModifiers(scroll->modifiers, {.shift = true});
    fixture.deliver(fixture.b, NSEventPhaseEnded, NSEventPhaseNone, 2.0);
    CHECK_FALSE(fixture.pointer->take().has_value());
}

TEST_CASE("Cocoa pointers remove their monitor and tracking area on teardown", "[core][render][cocoa]") {
    PointerFixture fixture;
    CHECK([eventMonitors count] == 1);
    CHECK(fixture.a->view->areas == 1);
    CHECK(fixture.b->view->areas == 0);
    fixture.pointer.reset();
    CHECK([eventMonitors count] == 0);
    CHECK(fixture.a->view->areas == 0);
}

TEST_CASE("Native scrolls acquire event modifiers after focus loss and refocus", "[core][render][cocoa][scroll-modifiers]") {
    REQUIRE([NSThread isMainThread]);
    const struct {
        NSEventModifierFlags flags;
        KeyModifiers modifiers;
    } cases[] = {
        {NSEventModifierFlagShift | NSEventModifierFlagOption, {.shift = true, .alt = true}},
        {NSEventModifierFlagControl, {.control = true}},
        {NSEventModifierFlagCommand, {.super = true}},
        {0, {}},
    };
    for (const bool focused : {false, true}) {
        for (const bool precise : {false, true}) {
            CAPTURE(focused, precise);
            PointerInputFixture fixture;
            InputState state({.macOS = true});
            state.advance(0.0, {InputQueue::Pointer{Jetstream::Extent2D<Jetstream::F32>{20, 30}}});
            state.advance(1.0, {InputQueue::Modifiers{{.shift = true}},
                                InputQueue::Key{Jetstream::KeyCode::LeftShift, true}});
            GlfwInput::Impl::OnFocus(fixture.input.window, GLFW_FALSE);
            state.advance(2.0, fixture.viewport.input().drain());
            CheckModifiers(state.modifiers(), {});
            CHECK_FALSE(state.keyDown(Jetstream::KeyCode::LeftShift));
            if (focused) {
                state.advance(3.0, {InputQueue::Focus{true}});
            }
            fixture.a->key = focused;
            Jetstream::F64 now = 4.0;
            for (const auto& entry : cases) {
                CAPTURE(entry.flags);
                fixture.deliver(fixture.a, precise ? NSEventPhaseChanged : NSEventPhaseNone,
                                NSEventPhaseNone, 2.0, entry.flags);
                const Jetstream::F32 delta = precise ? 0.2f : 2.0f;
                GlfwInput::Impl::OnScroll(fixture.input.window, 0.0, delta);
                const auto queued = fixture.viewport.input().drain();
                REQUIRE(queued.size() == 2);
                const auto* modifiers = std::get_if<InputQueue::Modifiers>(&queued[0]);
                REQUIRE(modifiers);
                CheckModifiers(modifiers->modifiers, entry.modifiers);
                CHECK_FALSE(modifiers->remote);
                REQUIRE(std::holds_alternative<InputQueue::Wheel>(queued[1]));
                state.advance(now++, queued);
                CHECK(state.windowFocused() == focused);
                REQUIRE(state.events().size() == 1);
                const auto* scroll = std::get_if<Jetstream::MouseEvent>(&state.events()[0]);
                REQUIRE(scroll);
                CHECK(scroll->type == Jetstream::MouseEventType::Scroll);
                CHECK(scroll->phase == (precise ? ScrollPhase::Changed : ScrollPhase::None));
                CHECK(scroll->positioned);
                CHECK(scroll->position == Jetstream::Extent2D<Jetstream::F32>{20, 30});
                CHECK(scroll->scroll == Jetstream::Extent2D<Jetstream::F32>{0, delta});
                CheckModifiers(scroll->modifiers, entry.modifiers);
                CHECK_FALSE(state.keyDown(Jetstream::KeyCode::LeftShift));
                CHECK_FALSE(state.keyDown(Jetstream::KeyCode::LeftCtrl));
                CHECK_FALSE(state.keyDown(Jetstream::KeyCode::LeftAlt));
                CHECK_FALSE(state.keyDown(Jetstream::KeyCode::LeftSuper));
                CHECK_FALSE(state.hasPending());
                CHECK_FALSE(fixture.input.pointer->take().has_value());
            }
        }
    }
}

TEST_CASE("Inactive native phase-only scrolls preserve modifier ordering across a queued gesture",
          "[core][render][cocoa][scroll-modifiers]") {
    REQUIRE([NSThread isMainThread]);
    PointerInputFixture fixture;
    fixture.a->key = NO;
    InputState state({.macOS = true});
    state.advance(0.0, {InputQueue::Pointer{Jetstream::Extent2D<Jetstream::F32>{20, 30}},
                        InputQueue::Focus{false}});
    const auto shiftOption = NSEventModifierFlagShift | NSEventModifierFlagOption;
    const auto controlCommand = NSEventModifierFlagControl | NSEventModifierFlagCommand;
    fixture.deliver(fixture.a, NSEventPhaseBegan, NSEventPhaseNone, 0.0, shiftOption);
    fixture.deliver(fixture.a, NSEventPhaseChanged, NSEventPhaseNone, 2.0, shiftOption);
    GlfwInput::Impl::OnScroll(fixture.input.window, 0.0, 0.2);
    fixture.deliver(fixture.a, NSEventPhaseEnded, NSEventPhaseNone, 0.0, controlCommand);
    fixture.deliver(fixture.a, NSEventPhaseNone, NSEventPhaseEnded, 0.0);
    fixture.deliver(fixture.a, NSEventPhaseNone, NSEventPhaseNone, 0.0, shiftOption);
    fixture.deliver(fixture.b, NSEventPhaseEnded, NSEventPhaseNone, 0.0, shiftOption);
    const auto queued = fixture.viewport.input().drain();
    REQUIRE(queued.size() == 8);
    for (std::size_t i = 0; i < queued.size(); i += 2) {
        REQUIRE(std::holds_alternative<InputQueue::Modifiers>(queued[i]));
        REQUIRE(std::holds_alternative<InputQueue::Wheel>(queued[i + 1]));
    }
    const struct {
        ScrollPhase phase;
        KeyModifiers modifiers;
        Jetstream::F32 delta;
    } expected[] = {
        {ScrollPhase::Began, {.shift = true, .alt = true}, 0.0f},
        {ScrollPhase::Changed, {.shift = true, .alt = true}, 0.2f},
        {ScrollPhase::Ended, {.control = true, .super = true}, 0.0f},
        {ScrollPhase::MomentumEnded, {}, 0.0f},
    };
    state.advance(1.0, queued);
    std::size_t count = 0;
    for (int frame = 0; frame < 3; ++frame) {
        CAPTURE(frame);
        CHECK_FALSE(state.windowFocused());
        for (const auto& event : state.events()) {
            REQUIRE(count < std::size(expected));
            const auto* scroll = std::get_if<Jetstream::MouseEvent>(&event);
            REQUIRE(scroll);
            const auto& entry = expected[count++];
            CHECK(scroll->type == Jetstream::MouseEventType::Scroll);
            CHECK(scroll->phase == entry.phase);
            CHECK(scroll->scroll == Jetstream::Extent2D<Jetstream::F32>{0, entry.delta});
            CheckModifiers(scroll->modifiers, entry.modifiers);
        }
        state.advance(2.0 + frame, {});
    }
    CHECK(count == std::size(expected));
    CHECK(state.events().empty());
    CHECK_FALSE(state.hasPending());
    CHECK_FALSE(fixture.input.pointer->take().has_value());
    CHECK(fixture.viewport.input().drain().empty());
}

int main(int argc, char* argv[]) {
    return Catch::Session().run(argc, argv);
}
