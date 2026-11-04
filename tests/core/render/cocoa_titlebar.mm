#import <AppKit/AppKit.h>
#include <dispatch/dispatch.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_session.hpp>
#include <functional>
#include <thread>
#include <vector>

#define GLFW_INCLUDE_NONE
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

namespace {

NSUInteger pressedButtons = 0;
NSMutableArray* eventMonitors = nil;
std::vector<dispatch_block_t> queuedBlocks;
NSString* doubleClickAction = nil;
bool legacyMiniaturize = false;
std::function<void()> nativeDrag;
std::function<id(GLFWwindow*)> nativeWindowLookup;

}  // namespace

@class TitlebarTestWindow;

@interface TitlebarTestEvent : NSObject {
 @public
    NSEventType eventType;
    TitlebarTestWindow* eventWindow;
}
+ (NSUInteger)pressedMouseButtons;
+ (id)addLocalMonitorForEventsMatchingMask:(NSEventMask)mask handler:(TitlebarTestEvent* (^)(TitlebarTestEvent*))handler;
+ (void)removeMonitor:(id)token;
- (NSEventType)type;
- (TitlebarTestWindow*)window;
@end

@implementation TitlebarTestEvent
+ (NSUInteger)pressedMouseButtons { return pressedButtons; }
+ (id)addLocalMonitorForEventsMatchingMask:(NSEventMask)mask handler:(TitlebarTestEvent* (^)(TitlebarTestEvent*))handler {
    (void)mask;
    id token = [handler copy];
    [eventMonitors addObject:token];
    return [token autorelease];
}
+ (void)removeMonitor:(id)token { [eventMonitors removeObjectIdenticalTo:token]; }
- (NSEventType)type { return eventType; }
- (TitlebarTestWindow*)window { return eventWindow; }
@end

@interface TitlebarTestDefaults : NSObject
@end

@implementation TitlebarTestDefaults
+ (TitlebarTestDefaults*)standardUserDefaults {
    static auto* defaults = [[TitlebarTestDefaults alloc] init];
    return defaults;
}
- (NSString*)stringForKey:(NSString*)key {
    REQUIRE([key isEqualToString:@"AppleActionOnDoubleClick"]);
    return doubleClickAction;
}
- (BOOL)boolForKey:(NSString*)key {
    REQUIRE([key isEqualToString:@"AppleMiniaturizeOnDoubleClick"]);
    return legacyMiniaturize;
}
@end

@interface TitlebarTestAppearance : NSObject {
    NSString* appearanceName;
}
+ (id)appearanceNamed:(NSString*)name;
- (NSString*)name;
@end

@implementation TitlebarTestAppearance
+ (id)appearanceNamed:(NSString*)name {
    auto* appearance = [[[TitlebarTestAppearance alloc] init] autorelease];
    appearance->appearanceName = [name copy];
    return appearance;
}
- (NSString*)name { return appearanceName; }
- (void)dealloc { [appearanceName release]; [super dealloc]; }
@end

@interface TitlebarTestWindow : NSObject {
 @public
    NSUInteger mask;
    bool movable;
    BOOL transparent;
    NSWindowTitleVisibility titleVisibility;
    NSTitlebarSeparatorStyle separatorStyle;
    int drags;
    int zooms;
    int minimizations;
    TitlebarTestEvent* lastDragEvent;
    id currentAppearance;
    int appearanceChanges;
}
- (id)appearance;
- (void)setAppearance:(id)value;
@end

@implementation TitlebarTestWindow
- (NSUInteger)styleMask { return mask; }
- (void)setStyleMask:(NSUInteger)value { mask = value; }
- (BOOL)titlebarAppearsTransparent { return transparent; }
- (void)setTitlebarAppearsTransparent:(BOOL)value { transparent = value; }
- (NSWindowTitleVisibility)titleVisibility { return titleVisibility; }
- (void)setTitleVisibility:(NSWindowTitleVisibility)value { titleVisibility = value; }
- (NSTitlebarSeparatorStyle)titlebarSeparatorStyle { return separatorStyle; }
- (void)setTitlebarSeparatorStyle:(NSTitlebarSeparatorStyle)value { separatorStyle = value; }
- (BOOL)isMovable { return movable; }
- (void)setMovable:(BOOL)value { movable = value; }
- (NSButton*)standardWindowButton:(NSWindowButton)button { (void)button; return nil; }
- (void)performWindowDragWithEvent:(TitlebarTestEvent*)event {
    ++drags;
    lastDragEvent = event;
    if (nativeDrag) {
        nativeDrag();
    }
}
- (void)zoom:(id)sender { (void)sender; ++zooms; }
- (void)miniaturize:(id)sender { (void)sender; ++minimizations; }
- (id)appearance { return currentAppearance; }
- (void)setAppearance:(id)value {
    CHECK([NSThread isMainThread]);
    [currentAppearance release];
    currentAppearance = [value retain];
    ++appearanceChanges;
}
- (void)dealloc { [currentAppearance release]; [super dealloc]; }
@end

namespace {

id TestCocoaWindow(GLFWwindow* handle) {
    return nativeWindowLookup ? nativeWindowLookup(handle) : reinterpret_cast<id>(handle);
}

void QueueBlock(dispatch_queue_t queue, dispatch_block_t block) {
    (void)queue;
    queuedBlocks.push_back([block copy]);
}

void DrainQueue() {
    while (!queuedBlocks.empty()) {
        auto block = queuedBlocks.front();
        queuedBlocks.erase(queuedBlocks.begin());
        block();
        [block release];
    }
}

}  // namespace

#define NSEvent TitlebarTestEvent
#define NSWindow TitlebarTestWindow
#define NSUserDefaults TitlebarTestDefaults
#define glfwGetCocoaWindow TestCocoaWindow
#define dispatch_async(queue, block) QueueBlock(queue, block)
#ifndef NSAppearance
#define NSAppearance TitlebarTestAppearance
#define JST_TEST_OWNS_APPEARANCE_MACRO
#endif
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcompare-distinct-pointer-types"
#include "../../../src/viewport/glfw/tools/cocoa_titlebar.mm"
#pragma clang diagnostic pop
#ifdef JST_TEST_OWNS_APPEARANCE_MACRO
#undef NSAppearance
#undef JST_TEST_OWNS_APPEARANCE_MACRO
#endif
#undef dispatch_async
#undef glfwGetCocoaWindow
#undef NSUserDefaults
#undef NSWindow
#undef NSEvent

namespace {

class TitlebarFixture {
 public:
    TitlebarFixture() {
        eventMonitors = [[NSMutableArray alloc] init];
        a = [[TitlebarTestWindow alloc] init];
        b = [[TitlebarTestWindow alloc] init];
        titlebar = make();
    }

    std::unique_ptr<Jetstream::Viewport::CocoaTitlebar> make() {
        return std::make_unique<Jetstream::Viewport::CocoaTitlebar>(handle(a), [this] { ++releases; });
    }

    ~TitlebarFixture() {
        titlebar.reset();
        DrainQueue();
        nativeDrag = {};
        nativeWindowLookup = {};
        doubleClickAction = nil;
        legacyMiniaturize = false;
        pressedButtons = 0;
        [a release];
        [b release];
        [eventMonitors release];
        eventMonitors = nil;
        [pool drain];
    }

    static GLFWwindow* handle(TitlebarTestWindow* window) {
        return reinterpret_cast<GLFWwindow*>(window);
    }

    TitlebarTestEvent* deliver(TitlebarTestWindow* window, NSEventType type) {
        pressedButtons = type == NSEventTypeLeftMouseDown ? 1 : 0;
        auto* event = [[[TitlebarTestEvent alloc] init] autorelease];
        event->eventType = type;
        event->eventWindow = window;
        NSArray* monitors = [eventMonitors copy];
        for (id token in monitors) {
            auto handler = (TitlebarTestEvent* (^)(TitlebarTestEvent*))token;
            handler(event);
        }
        [monitors release];
        return event;
    }

    void pressA() {
        deliver(a, NSEventTypeLeftMouseDown);
    }

    NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
    TitlebarTestWindow* a = nil;
    TitlebarTestWindow* b = nil;
    int releases = 0;
    std::unique_ptr<Jetstream::Viewport::CocoaTitlebar> titlebar;
};

}  // namespace

TEST_CASE("Cocoa appearance deduplicates native updates in main queue order", "[core][render][cocoa][appearance]") {
    REQUIRE([NSThread isMainThread]);
    for (const bool dark : {false, true}) {
        CAPTURE(dark);
        TitlebarFixture fixture;
        const auto handle = TitlebarFixture::handle(fixture.a);
        for (const bool value : {dark, dark, !dark, !dark, dark}) {
            REQUIRE(Jetstream::Viewport::CocoaTitlebar::SetAppearance(handle, value));
        }
        CHECK(fixture.a->appearanceChanges == 0);
        REQUIRE(queuedBlocks.size() == 5);
        DrainQueue();
        CHECK(fixture.a->appearanceChanges == 3);
        CHECK([[[fixture.a appearance] name] isEqualToString:dark ? NSAppearanceNameDarkAqua : NSAppearanceNameAqua]);
        REQUIRE(Jetstream::Viewport::CocoaTitlebar::SetAppearance(handle, dark));
        DrainQueue();
        CHECK(fixture.a->appearanceChanges == 3);
    }
}

TEST_CASE("Cocoa appearance follows a recreated native window behind the same handle", "[core][render][cocoa][appearance]") {
    REQUIRE([NSThread isMainThread]);
    for (const bool dark : {false, true}) {
        CAPTURE(dark);
        TitlebarFixture fixture;
        const auto handle = TitlebarFixture::handle(fixture.a);
        id target = fixture.a;
        nativeWindowLookup = [&](GLFWwindow*) { return target; };
        REQUIRE(Jetstream::Viewport::CocoaTitlebar::SetAppearance(handle, dark));
        DrainQueue();
        REQUIRE(fixture.a->appearanceChanges == 1);
        target = nil;
        CHECK_FALSE(Jetstream::Viewport::CocoaTitlebar::SetAppearance(handle, dark));
        CHECK(queuedBlocks.empty());
        target = fixture.b;
        REQUIRE(Jetstream::Viewport::CocoaTitlebar::SetAppearance(handle, dark));
        DrainQueue();
        CHECK(fixture.b->appearanceChanges == 1);
        CHECK([[[fixture.b appearance] name] isEqualToString:dark ? NSAppearanceNameDarkAqua : NSAppearanceNameAqua]);
        REQUIRE(Jetstream::Viewport::CocoaTitlebar::SetAppearance(handle, dark));
        DrainQueue();
        CHECK(fixture.a->appearanceChanges == 1);
        CHECK(fixture.b->appearanceChanges == 1);
        nativeWindowLookup = {};
    }
}

TEST_CASE("Cocoa pending drag cannot attach to another window press", "[core][render][cocoa]") {
    TitlebarFixture fixture;
    fixture.pressA();
    fixture.titlebar->beginDrag();
    fixture.deliver(fixture.a, NSEventTypeLeftMouseUp);
    fixture.deliver(fixture.b, NSEventTypeLeftMouseDown);
    DrainQueue();
    REQUIRE(fixture.a->drags == 0);
    REQUIRE_FALSE(fixture.a->movable);
    REQUIRE(fixture.releases == 1);
    fixture.titlebar->beginDrag();
    DrainQueue();
    REQUIRE(fixture.a->drags == 0);
}

TEST_CASE("Cocoa integrated titlebar teardown restores the native chrome and cancels pending actions",
          "[core][render][cocoa][review17]") {
    TitlebarFixture fixture;
    fixture.titlebar.reset();
    fixture.a->mask = NSWindowStyleMaskTitled;
    fixture.a->movable = true;
    fixture.a->transparent = NO;
    fixture.a->titleVisibility = NSWindowTitleVisible;
    fixture.a->separatorStyle = NSTitlebarSeparatorStyleLine;
    fixture.titlebar = fixture.make();
    REQUIRE_FALSE(fixture.a->movable);
    REQUIRE(fixture.a->transparent);
    REQUIRE(fixture.a->titleVisibility == NSWindowTitleHidden);
    REQUIRE((fixture.a->mask & NSWindowStyleMaskFullSizeContentView) != 0);
    fixture.a->mask |= NSWindowStyleMaskResizable;
    fixture.pressA();
    fixture.titlebar->beginDrag();
    fixture.titlebar->zoom();
    fixture.titlebar.reset();
    DrainQueue();
    CHECK(fixture.a->movable);
    CHECK_FALSE(fixture.a->transparent);
    CHECK(fixture.a->titleVisibility == NSWindowTitleVisible);
    CHECK(fixture.a->separatorStyle == NSTitlebarSeparatorStyleLine);
    CHECK(fixture.a->mask == (NSWindowStyleMaskTitled | NSWindowStyleMaskResizable));
    CHECK(fixture.a->drags == 0);
    CHECK(fixture.a->zooms == 0);
    CHECK(fixture.releases == 1);
    CHECK([eventMonitors count] == 0);

    fixture.titlebar = fixture.make();
    CHECK_FALSE(fixture.a->movable);
    fixture.titlebar.reset();
    CHECK(fixture.a->movable);
}

TEST_CASE("Cocoa drag completion releases the left button once", "[core][render][cocoa]") {
    TitlebarFixture fixture;
    fixture.pressA();
    auto* press = fixture.deliver(fixture.a, NSEventTypeLeftMouseDown);
    fixture.titlebar->beginDrag();
    nativeDrag = [] { pressedButtons = 0; };
    DrainQueue();
    REQUIRE(fixture.a->drags == 1);
    REQUIRE(fixture.a->lastDragEvent == press);
    REQUIRE_FALSE(fixture.a->movable);
    REQUIRE(fixture.releases == 1);
}

TEST_CASE("Cocoa drag requests are single use within one press generation", "[core][render][cocoa]") {
    TitlebarFixture fixture;
    fixture.pressA();
    fixture.titlebar->beginDrag();
    fixture.titlebar->beginDrag();
    fixture.titlebar->beginDrag();
    REQUIRE(queuedBlocks.size() == 1);
    DrainQueue();
    fixture.titlebar->beginDrag();
    DrainQueue();
    REQUIRE(fixture.a->drags == 1);
    fixture.deliver(fixture.a, NSEventTypeLeftMouseUp);
    fixture.pressA();
    fixture.titlebar->beginDrag();
    DrainQueue();
    REQUIRE(fixture.a->drags == 2);
}

TEST_CASE("Cocoa queued drag captures its generation before main queue execution", "[core][render][cocoa]") {
    TitlebarFixture fixture;
    fixture.pressA();
    std::thread requester([&] { fixture.titlebar->beginDrag(); });
    requester.join();
    fixture.deliver(fixture.a, NSEventTypeLeftMouseUp);
    auto* replacement = fixture.deliver(fixture.a, NSEventTypeLeftMouseDown);
    fixture.titlebar->beginDrag();
    DrainQueue();
    REQUIRE(fixture.a->drags == 1);
    REQUIRE(fixture.a->lastDragEvent == replacement);
}

TEST_CASE("Cocoa pending drags cancel on release polling focus loss and window close", "[core][render][cocoa]") {
    TitlebarFixture fixture;
    fixture.pressA();
    fixture.titlebar->beginDrag();
    SECTION("Release consumed outside the local monitor") {
        pressedButtons = 0;
        fixture.titlebar->poll();
    }
    SECTION("Focus cancellation") {
        [[NSNotificationCenter defaultCenter] postNotificationName:NSWindowDidResignKeyNotification object:fixture.a];
    }
    SECTION("Fullscreen cancellation") {
        [[NSNotificationCenter defaultCenter] postNotificationName:NSWindowDidEnterFullScreenNotification object:fixture.a];
    }
    SECTION("Native window teardown") {
        [[NSNotificationCenter defaultCenter] postNotificationName:NSWindowWillCloseNotification object:fixture.a];
    }
    DrainQueue();
    REQUIRE(fixture.a->drags == 0);
    REQUIRE_FALSE(fixture.a->movable);
    REQUIRE(fixture.releases == 1);
}

TEST_CASE("Cocoa titlebar teardown invalidates retained pending blocks", "[core][render][cocoa]") {
    TitlebarFixture fixture;
    fixture.pressA();
    fixture.titlebar->beginDrag();
    fixture.titlebar->zoom();
    int expectedDrags = 0;
    SECTION("Pending request") {
        fixture.titlebar.reset();
    }
    SECTION("Active native request") {
        expectedDrags = 1;
        nativeDrag = [&] { fixture.titlebar.reset(); };
    }
    DrainQueue();
    REQUIRE([eventMonitors count] == 0);
    [[NSNotificationCenter defaultCenter] postNotificationName:NSWindowDidResizeNotification object:fixture.a];
    REQUIRE(fixture.a->drags == expectedDrags);
    REQUIRE_FALSE(fixture.a->movable);
    REQUIRE(fixture.a->zooms == 0);
    REQUIRE(fixture.releases == 1);
}

TEST_CASE("Cocoa ordinary presses do not synthesize unrequested releases", "[core][render][cocoa]") {
    TitlebarFixture fixture;
    fixture.pressA();
    fixture.deliver(fixture.a, NSEventTypeLeftMouseUp);
    fixture.deliver(fixture.b, NSEventTypeLeftMouseDown);
    fixture.titlebar->beginDrag();
    fixture.titlebar.reset();
    DrainQueue();
    REQUIRE(fixture.a->drags == 0);
    REQUIRE(fixture.releases == 0);
}

TEST_CASE("Cocoa native drag nested cancellation cannot finish a newer press", "[core][render][cocoa]") {
    TitlebarFixture fixture;
    fixture.pressA();
    fixture.titlebar->beginDrag();
    nativeDrag = [&] {
        fixture.deliver(fixture.a, NSEventTypeLeftMouseUp);
        fixture.pressA();
    };
    DrainQueue();
    nativeDrag = {};
    fixture.titlebar->beginDrag();
    DrainQueue();
    REQUIRE(fixture.a->drags == 2);
    REQUIRE(fixture.releases == 2);
}

TEST_CASE("Cocoa zoom honors read only preferences and queued teardown", "[core][render][cocoa]") {
    TitlebarFixture fixture;
    doubleClickAction = @"None";
    fixture.titlebar->zoom();
    DrainQueue();
    REQUIRE(fixture.a->zooms == 0);
    REQUIRE(fixture.a->minimizations == 0);
    doubleClickAction = @"Minimize";
    fixture.titlebar->zoom();
    DrainQueue();
    REQUIRE(fixture.a->minimizations == 1);
    doubleClickAction = @"Maximize";
    fixture.titlebar->zoom();
    DrainQueue();
    REQUIRE(fixture.a->zooms == 1);
    doubleClickAction = nil;
    legacyMiniaturize = true;
    fixture.titlebar->zoom();
    DrainQueue();
    REQUIRE(fixture.a->minimizations == 2);
    legacyMiniaturize = false;
    fixture.titlebar->zoom();
    DrainQueue();
    REQUIRE(fixture.a->zooms == 2);
}

int main(int argc, char* argv[]) {
    return Catch::Session().run(argc, argv);
}
