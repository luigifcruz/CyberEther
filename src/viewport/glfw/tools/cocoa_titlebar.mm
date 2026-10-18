#include "cocoa_titlebar.hh"
#include "imgui_impl_glfw_internal.hh"

#define GLFW_INCLUDE_NONE
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#import <AppKit/AppKit.h>
#include <dispatch/dispatch.h>

#include <array>
#include <cmath>
#include <mutex>

namespace Jetstream::Viewport {

namespace {

constexpr CGFloat TitlebarHeight = 40.0;
constexpr CGFloat LeadingGap = 2.0;
constexpr CGFloat RowDrop = 1.0;

bool LeftButtonPressed() {
    return ([NSEvent pressedMouseButtons] & 1) != 0;
}

bool Fullscreen(NSWindow* window) {
    return ([window styleMask] & NSWindowStyleMaskFullScreen) != 0;
}

std::array<NSButton*, 3> WindowButtons(NSWindow* window) {
    return {
        [window standardWindowButton:NSWindowCloseButton],
        [window standardWindowButton:NSWindowMiniaturizeButton],
        [window standardWindowButton:NSWindowZoomButton],
    };
}

}  // namespace

struct CocoaTitlebar::State {
    GLFWwindow* handle = nullptr;
    NSWindow* window = nil;
    id monitor = nil;
    NSMutableArray* observers = nil;
    NSEvent* pressEvent = nil;
    std::function<void()> releaseLeftButton;
    std::mutex dragMutex;
    U64 generation = 0;
    bool pressed = false;
    bool requested = false;
    bool pending = false;
    bool dragging = false;
    bool layingOut = false;
    CGFloat spacing = 0.0;
    CGFloat leadingInset = 0.0;
    mutable std::mutex mutex;
    Render::TitlebarChrome chrome;
    NSWindowStyleMask originalStyleMask = 0;
    BOOL originalTransparent = NO;
    NSWindowTitleVisibility originalTitleVisibility = NSWindowTitleVisible;
    NSTitlebarSeparatorStyle originalSeparatorStyle = NSTitlebarSeparatorStyleAutomatic;
    BOOL originalMovable = NO;
    std::array<NSRect, 3> originalButtonFrames{};

    void finishDrag(U64 expected = 0) {
        bool release;
        {
            std::lock_guard lock(dragMutex);
            if (expected && generation != expected) {
                return;
            }
            release = pending || dragging;
            ++generation;
            pressed = false;
            pending = false;
            dragging = false;
        }
        if (window) {
            [window setMovable:NO];
        }
        if (release && releaseLeftButton) {
            releaseLeftButton();
        }
        releaseLeftButton = {};
        [pressEvent release];
        pressEvent = nil;
    }

    void mouseEvent(NSEvent* event) {
        finishDrag();
        if ([event type] != NSEventTypeLeftMouseDown || [event window] != window || !window) {
            return;
        }
        pressEvent = [event retain];
        releaseLeftButton = ImGui_ImplGlfw_MakeLeftButtonRelease(handle);
        std::lock_guard lock(dragMutex);
        pressed = true;
        requested = false;
    }

    void drag(U64 expected) {
        {
            std::lock_guard lock(dragMutex);
            if (generation != expected || !pending || !pressed || !window) {
                return;
            }
            pending = false;
            dragging = true;
        }

        NSWindow* native = [window retain];
        NSEvent* event = [pressEvent retain];
        @try {
            [native setMovable:YES];
            [native performWindowDragWithEvent:event];
        } @finally {
            finishDrag(expected);
            [event release];
            [native release];
        }
    }

    void layout() {
        if (!window || layingOut || Fullscreen(window)) {
            return;
        }

        const auto buttons = WindowButtons(window);
        for (NSButton* button : buttons) {
            if (!button || [button window] != window) {
                return;
            }
        }

        NSView* titlebar = [buttons[0] superview];
        NSView* container = [titlebar superview];
        NSView* theme = [container superview];
        if (!titlebar || !container || !theme) {
            return;
        }

        layingOut = true;

        if (spacing <= 0.0) {
            spacing = NSMinX([buttons[1] frame]) - NSMinX([buttons[0] frame]);
        }

        const NSRect bounds = [theme bounds];
        const NSRect containerFrame = NSMakeRect(0.0, NSMaxY(bounds) - TitlebarHeight, NSWidth(bounds), TitlebarHeight);
        if (!NSEqualRects([container frame], containerFrame)) {
            [container setFrame:containerFrame];
        }

        const NSRect titlebarFrame = NSMakeRect(0.0, 0.0, NSWidth(bounds), TitlebarHeight);
        if (!NSEqualRects([titlebar frame], titlebarFrame)) {
            [titlebar setFrame:titlebarFrame];
        }

        for (size_t i = 0; i < buttons.size(); ++i) {
            const NSSize size = [buttons[i] frame].size;
            const CGFloat margin = std::round((TitlebarHeight - size.height) / 2.0);
            const CGFloat drop = [[buttons[i] superview] isFlipped] ? RowDrop : -RowDrop;
            const NSPoint origin = NSMakePoint(margin + static_cast<CGFloat>(i) * spacing, margin + drop);
            if (!NSEqualPoints([buttons[i] frame].origin, origin)) {
                [buttons[i] setFrameOrigin:origin];
            }
            if (i + 1 == buttons.size()) {
                leadingInset = origin.x + size.width + LeadingGap;
            }
        }

        layingOut = false;
    }

    void measure() {
        Render::TitlebarChrome next;
        next.integrated = true;
        if (window && !Fullscreen(window)) {
            next.leadingInset = static_cast<F32>(leadingInset);
            next.height = static_cast<F32>(TitlebarHeight);
        }

        std::lock_guard lock(mutex);
        chrome = next;
    }

    void observe(const std::shared_ptr<State>& self) {
        observers = [[NSMutableArray alloc] init];

        NSMutableArray* targets = [NSMutableArray array];
        for (NSNotificationName name in @[NSWindowDidResizeNotification,
                                          NSWindowDidBecomeKeyNotification,
                                          NSWindowDidResignKeyNotification,
                                          NSWindowDidEnterFullScreenNotification,
                                          NSWindowDidExitFullScreenNotification,
                                          NSWindowDidChangeBackingPropertiesNotification]) {
            [targets addObject:@[name, window]];
        }

        const auto buttons = WindowButtons(window);
        NSView* titlebar = [buttons[0] superview];
        NSMutableArray* views = [NSMutableArray array];
        if (titlebar) {
            [views addObject:titlebar];
            if ([titlebar superview]) {
                [views addObject:[titlebar superview]];
            }
        }
        for (NSButton* button : buttons) {
            if (button) {
                [views addObject:button];
            }
        }
        for (NSView* view in views) {
            [view setPostsFrameChangedNotifications:YES];
            [targets addObject:@[NSViewFrameDidChangeNotification, view]];
        }

        std::weak_ptr<State> weak = self;
        for (NSArray* target in targets) {
            id token = [[NSNotificationCenter defaultCenter] addObserverForName:target[0]
                                                                         object:target[1]
                                                                          queue:nil
                                                                     usingBlock:^(NSNotification*) {
                if (auto shared = weak.lock()) {
                    if ([target[0] isEqualToString:NSWindowDidResignKeyNotification] ||
                        [target[0] isEqualToString:NSWindowDidEnterFullScreenNotification]) {
                        shared->finishDrag();
                    }
                    shared->layout();
                    shared->measure();
                }
            }];
            [observers addObject:token];
        }

        id token = [[NSNotificationCenter defaultCenter] addObserverForName:NSWindowWillCloseNotification
                                                                      object:window
                                                                       queue:nil
                                                                  usingBlock:^(NSNotification*) {
            if (auto shared = weak.lock()) {
                shared->finishDrag();
                shared->window = nil;
            }
        }];
        [observers addObject:token];
    }
};

CocoaTitlebar::CocoaTitlebar(GLFWwindow* handle) : state(std::make_shared<State>()) {
    @autoreleasepool {
        NSWindow* window = glfwGetCocoaWindow(handle);
        if (!window) {
            return;
        }

        state->originalStyleMask = [window styleMask];
        state->originalTransparent = [window titlebarAppearsTransparent];
        state->originalTitleVisibility = [window titleVisibility];
        state->originalSeparatorStyle = [window titlebarSeparatorStyle];
        state->originalMovable = [window isMovable];
        const auto buttons = WindowButtons(window);
        for (U64 index = 0; index < buttons.size(); ++index) {
            state->originalButtonFrames[index] = [buttons[index] frame];
        }

        [window setStyleMask:[window styleMask] | NSWindowStyleMaskFullSizeContentView];
        [window setTitlebarAppearsTransparent:YES];
        [window setTitleVisibility:NSWindowTitleHidden];
        [window setTitlebarSeparatorStyle:NSTitlebarSeparatorStyleNone];
        [window setMovable:NO];

        state->handle = handle;
        state->window = window;

        std::weak_ptr<State> weak = state;
        state->monitor = [NSEvent addLocalMonitorForEventsMatchingMask:NSEventMaskLeftMouseDown | NSEventMaskLeftMouseUp
                                                               handler:^NSEvent*(NSEvent* event) {
            if (auto shared = weak.lock()) {
                shared->mouseEvent(event);
            }
            return event;
        }];

        state->layout();
        state->observe(state);
        state->measure();
    }
}

CocoaTitlebar::~CocoaTitlebar() {
    state->finishDrag();
    if (state->monitor) {
        [NSEvent removeMonitor:state->monitor];
        state->monitor = nil;
    }
    if (state->observers) {
        for (id token in state->observers) {
            [[NSNotificationCenter defaultCenter] removeObserver:token];
        }
        [state->observers release];
        state->observers = nil;
    }
    if (state->window) {
        NSWindow* window = state->window;
        [window setStyleMask:([window styleMask] & ~NSWindowStyleMaskFullSizeContentView) |
                             (state->originalStyleMask & NSWindowStyleMaskFullSizeContentView)];
        [window setTitlebarAppearsTransparent:state->originalTransparent];
        [window setTitleVisibility:state->originalTitleVisibility];
        [window setTitlebarSeparatorStyle:state->originalSeparatorStyle];
        [window setMovable:state->originalMovable];
        const auto buttons = WindowButtons(window);
        for (U64 index = 0; index < buttons.size(); ++index) {
            [buttons[index] setFrame:state->originalButtonFrames[index]];
        }
    }
    state->window = nil;
}

void CocoaTitlebar::poll() {
    @autoreleasepool {
        if (!state->window) {
            return;
        }

        if (!LeftButtonPressed()) {
            state->finishDrag();
        }

        state->layout();
        state->measure();
    }
}

Render::TitlebarChrome CocoaTitlebar::chrome() const {
    std::lock_guard lock(state->mutex);
    return state->chrome;
}

void CocoaTitlebar::beginDrag() {
    auto shared = state;
    U64 generation;
    {
        std::lock_guard lock(shared->dragMutex);
        if (!shared->pressed || shared->requested) {
            return;
        }
        generation = shared->generation;
        shared->requested = true;
        shared->pending = true;
    }
    dispatch_async(dispatch_get_main_queue(), ^{
        if (!LeftButtonPressed()) {
            shared->finishDrag(generation);
            return;
        }
        shared->drag(generation);
    });
}

void CocoaTitlebar::zoom() {
    auto shared = state;
    dispatch_async(dispatch_get_main_queue(), ^{
        NSWindow* window = shared->window;
        if (!window) {
            return;
        }

        NSUserDefaults* defaults = [NSUserDefaults standardUserDefaults];
        NSString* action = [defaults stringForKey:@"AppleActionOnDoubleClick"];
        if (!action && [defaults boolForKey:@"AppleMiniaturizeOnDoubleClick"]) {
            action = @"Minimize";
        }

        if ([action isEqualToString:@"None"]) {
            return;
        }

        if ([action isEqualToString:@"Minimize"]) {
            [window miniaturize:nil];
            return;
        }

        [window zoom:nil];
    });
}

}  // namespace Jetstream::Viewport
