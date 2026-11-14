#include "cocoa_pointer.hh"

#define GLFW_INCLUDE_NONE
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#import <AppKit/AppKit.h>

#include <utility>

namespace Jetstream::Viewport {

namespace {

ScrollPhase PhaseOf(NSEvent* event) {
    const NSEventPhase momentum = [event momentumPhase];
    if (momentum != NSEventPhaseNone) {
        return (momentum & (NSEventPhaseEnded | NSEventPhaseCancelled)) != 0 ? ScrollPhase::MomentumEnded
                                                                            : ScrollPhase::Momentum;
    }
    const NSEventPhase phase = [event phase];
    if ((phase & (NSEventPhaseMayBegin | NSEventPhaseBegan)) != 0) {
        return ScrollPhase::Began;
    }
    if ((phase & NSEventPhaseChanged) != 0) {
        return ScrollPhase::Changed;
    }
    if ((phase & (NSEventPhaseEnded | NSEventPhaseCancelled)) != 0) {
        return ScrollPhase::Ended;
    }
    return ScrollPhase::None;
}

}  // namespace

}  // namespace Jetstream::Viewport

@interface JSTInactiveHover : NSObject
@property(nonatomic, assign) NSWindow* window;
@end

@implementation JSTInactiveHover

- (void)forward:(SEL)selector event:(NSEvent*)event {
    NSWindow* window = self.window;
    if (!window || [window isKeyWindow]) {
        return;
    }
    NSView* view = [window contentView];
    if ([view respondsToSelector:selector]) {
        [view performSelector:selector withObject:event];
    }
}

- (void)mouseMoved:(NSEvent*)event {
    [self forward:@selector(mouseMoved:) event:event];
}

- (void)mouseEntered:(NSEvent*)event {
    [self forward:@selector(mouseEntered:) event:event];
}

- (void)mouseExited:(NSEvent*)event {
    [self forward:@selector(mouseExited:) event:event];
}

@end

namespace Jetstream::Viewport {

namespace {

KeyModifiers ModifiersOf(NSEventModifierFlags flags) {
    return {
        .control = (flags & NSEventModifierFlagControl) != 0,
        .shift = (flags & NSEventModifierFlagShift) != 0,
        .alt = (flags & NSEventModifierFlagOption) != 0,
        .super = (flags & NSEventModifierFlagCommand) != 0,
    };
}

}  // namespace

struct CocoaPointer::State {
    NSWindow* window = nil;
    id monitor = nil;
    JSTInactiveHover* hover = nil;
    NSTrackingArea* area = nil;
    std::optional<Scroll> pending;
    std::function<void()> onIdle;
    std::function<void(KeyModifiers)> onModifiers;

    void flagsEvent(NSEvent* event) {
        if ([event window] == window && onModifiers) {
            onModifiers(ModifiersOf([event modifierFlags]));
        }
    }

    void scrollEvent(NSEvent* event) {
        if ([event window] != window) {
            return;
        }
        pending = Scroll{
            .phase = PhaseOf(event),
            .modifiers = ModifiersOf([event modifierFlags]),
        };
        if ([event scrollingDeltaX] == 0.0 && [event scrollingDeltaY] == 0.0) {
            if (pending->phase != ScrollPhase::None && onIdle) {
                onIdle();
            }
            pending.reset();
        }
    }
};

CocoaPointer::CocoaPointer(GLFWwindow* handle, std::function<void()> onIdle,
                           std::function<void(KeyModifiers)> onModifiers)
    : state(std::make_shared<State>()) {
    state->onIdle = std::move(onIdle);
    state->onModifiers = std::move(onModifiers);
    @autoreleasepool {
        state->window = glfwGetCocoaWindow(handle);
        if (!state->window) {
            return;
        }
        std::weak_ptr<State> weak = state;
        state->monitor = [NSEvent addLocalMonitorForEventsMatchingMask:NSEventMaskScrollWheel | NSEventMaskFlagsChanged
                                                               handler:^NSEvent*(NSEvent* event) {
            if (auto shared = weak.lock()) {
                if ([event type] == NSEventTypeFlagsChanged) {
                    shared->flagsEvent(event);
                } else {
                    shared->scrollEvent(event);
                }
            }
            return event;
        }];
        NSView* view = [state->window contentView];
        state->hover = [[JSTInactiveHover alloc] init];
        state->hover.window = state->window;
        state->area = [[NSTrackingArea alloc] initWithRect:NSZeroRect
                                                   options:NSTrackingMouseMoved | NSTrackingMouseEnteredAndExited |
                                                           NSTrackingActiveAlways | NSTrackingInVisibleRect
                                                     owner:state->hover
                                                  userInfo:nil];
        [view addTrackingArea:state->area];
    }
}

CocoaPointer::~CocoaPointer() {
    if (state->monitor) {
        [NSEvent removeMonitor:state->monitor];
        state->monitor = nil;
    }
    if (state->area) {
        [[state->window contentView] removeTrackingArea:state->area];
        [state->area release];
        state->area = nil;
    }
    if (state->hover) {
        state->hover.window = nil;
        [state->hover release];
        state->hover = nil;
    }
    state->onIdle = nullptr;
    state->onModifiers = nullptr;
}

std::optional<CocoaPointer::Scroll> CocoaPointer::take() {
    return std::exchange(state->pending, std::nullopt);
}

}  // namespace Jetstream::Viewport
