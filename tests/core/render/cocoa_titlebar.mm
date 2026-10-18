#import <AppKit/AppKit.h>
#include <dispatch/dispatch.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_session.hpp>
#include <atomic>
#include <functional>
#include <thread>
#include <vector>

#include "../../../src/viewport/glfw/tools/imgui_impl_glfw.cpp"

namespace {

NSUInteger pressedButtons = 0;
NSMutableArray* eventMonitors = nil;
std::vector<dispatch_block_t> queuedBlocks;
NSString* doubleClickAction = nil;
bool legacyMiniaturize = false;
std::function<void()> nativeDrag;

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
}
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
@end

namespace {

id TestCocoaWindow(GLFWwindow* handle) {
    return reinterpret_cast<id>(handle);
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
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcompare-distinct-pointer-types"
#include "../../../src/viewport/glfw/tools/cocoa_titlebar.mm"
#pragma clang diagnostic pop
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
        contextA = ImGui::CreateContext();
        contextB = ImGui::CreateContext();
        ImGui_ImplGlfw_InputOwner_Init(contextA);
        ImGui_ImplGlfw_InputOwner_Init(contextB);
        ImGui_ImplGlfw_ContextMap_Add(handle(a), contextA);
        ImGui_ImplGlfw_ContextMap_Add(handle(b), contextB);
        titlebar = std::make_unique<Jetstream::Viewport::CocoaTitlebar>(handle(a));
    }

    ~TitlebarFixture() {
        titlebar.reset();
        DrainQueue();
        ImGui_ImplGlfw_ContextMap_Remove(handle(a));
        ImGui_ImplGlfw_ContextMap_Remove(handle(b));
        if (contextA) {
            ImGui_ImplGlfw_InputOwner_Shutdown(contextA);
            ImGui::DestroyContext(contextA);
        }
        ImGui_ImplGlfw_InputOwner_Shutdown(contextB);
        ImGui::DestroyContext(contextB);
        nativeDrag = {};
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
        contextA->IO.AddMouseButtonEvent(ImGuiMouseButton_Left, true);
    }

    int releases(ImGuiContext* context) const {
        int count = 0;
        for (const auto& event : context->InputEventsQueue) {
            if (event.Type == ImGuiInputEventType_MouseButton &&
                event.MouseButton.Button == ImGuiMouseButton_Left && !event.MouseButton.Down) {
                ++count;
            }
        }
        return count;
    }

    NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
    TitlebarTestWindow* a = nil;
    TitlebarTestWindow* b = nil;
    ImGuiContext* contextA = nullptr;
    ImGuiContext* contextB = nullptr;
    std::unique_ptr<Jetstream::Viewport::CocoaTitlebar> titlebar;
};

}  // namespace

TEST_CASE("Cocoa pending drag cannot attach to another window press", "[core][render][cocoa]") {
    TitlebarFixture fixture;
    fixture.pressA();
    fixture.titlebar->beginDrag();
    fixture.deliver(fixture.a, NSEventTypeLeftMouseUp);
    fixture.deliver(fixture.b, NSEventTypeLeftMouseDown);
    DrainQueue();
    REQUIRE(fixture.a->drags == 0);
    REQUIRE_FALSE(fixture.a->movable);
    REQUIRE(fixture.releases(fixture.contextA) == 1);
    REQUIRE(fixture.releases(fixture.contextB) == 0);
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
    fixture.titlebar = std::make_unique<Jetstream::Viewport::CocoaTitlebar>(TitlebarFixture::handle(fixture.a));
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
    CHECK(fixture.releases(fixture.contextA) == 1);
    CHECK([eventMonitors count] == 0);

    fixture.titlebar = std::make_unique<Jetstream::Viewport::CocoaTitlebar>(TitlebarFixture::handle(fixture.a));
    CHECK_FALSE(fixture.a->movable);
    fixture.titlebar.reset();
    CHECK(fixture.a->movable);
}

TEST_CASE("Cocoa drag completion releases only its living input owner", "[core][render][cocoa]") {
    TitlebarFixture fixture;
    fixture.pressA();
    auto* press = fixture.deliver(fixture.a, NSEventTypeLeftMouseDown);
    fixture.contextB->IO.AddMouseButtonEvent(ImGuiMouseButton_Left, true);
    fixture.titlebar->beginDrag();
    ImGui::SetCurrentContext(fixture.contextB);
    nativeDrag = [] { pressedButtons = 0; };
    DrainQueue();
    REQUIRE(fixture.a->drags == 1);
    REQUIRE(fixture.a->lastDragEvent == press);
    REQUIRE_FALSE(fixture.a->movable);
    REQUIRE(fixture.releases(fixture.contextA) == 1);
    REQUIRE(fixture.releases(fixture.contextB) == 0);
    REQUIRE(ImGui::GetCurrentContext() == fixture.contextB);
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
    REQUIRE(fixture.releases(fixture.contextA) == 1);
    REQUIRE(fixture.releases(fixture.contextB) == 0);
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
    REQUIRE(fixture.releases(fixture.contextA) == 1);
}

TEST_CASE("Cocoa queued releases tolerate input owner destruction or unregistration", "[core][render][cocoa]") {
    TitlebarFixture fixture;
    fixture.pressA();
    fixture.titlebar->beginDrag();
    nativeDrag = [] { pressedButtons = 0; };
    SECTION("Context shutdown hook invalidates the owner before queued execution") {
        ImGui::DestroyContext(fixture.contextA);
        fixture.contextA = nullptr;
    }
    SECTION("Context shutdown hook invalidates the owner during native execution") {
        nativeDrag = [&] {
            ImGui::DestroyContext(fixture.contextA);
            fixture.contextA = nullptr;
            pressedButtons = 0;
        };
    }
    SECTION("Window context routing removed before queued execution") {
        ImGui_ImplGlfw_ContextMap_Remove(TitlebarFixture::handle(fixture.a));
    }
    SECTION("Window context routing replaced before queued execution") {
        ImGui_ImplGlfw_ContextMap_Remove(TitlebarFixture::handle(fixture.a));
        ImGui_ImplGlfw_ContextMap_Add(TitlebarFixture::handle(fixture.a), fixture.contextB);
    }
    ImGui::SetCurrentContext(fixture.contextB);
    DrainQueue();
    REQUIRE(fixture.releases(fixture.contextB) == 0);
    REQUIRE(ImGui::GetCurrentContext() == fixture.contextB);
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
    REQUIRE(fixture.releases(fixture.contextA) == 0);
    REQUIRE(fixture.releases(fixture.contextB) == 0);
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
    REQUIRE(fixture.releases(fixture.contextA) == 2);
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

TEST_CASE("Cocoa native press ownership cannot mutate an executing frame hook range", "[core][render][cocoa][hooks]") {
    TitlebarFixture fixture;
    auto* context = fixture.contextA;
    struct Gate {
        std::atomic<bool> entered = false;
        std::atomic<bool> resume = false;
        std::atomic<int> completed = 0;
    } gate;
    ImGuiContextHook hook;
    hook.Type = ImGuiContextHookType_EndFramePost;
    hook.UserData = &gate;
    hook.Callback = [](ImGuiContext*, ImGuiContextHook* hook) {
        auto* gate = static_cast<Gate*>(hook->UserData);
        gate->entered.store(true, std::memory_order_release);
        while (!gate->resume.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
    };
    ImGui::AddContextHook(context, &hook);
    hook.Callback = [](ImGuiContext*, ImGuiContextHook* hook) {
        static_cast<Gate*>(hook->UserData)->completed.fetch_add(1, std::memory_order_relaxed);
    };
    while (context->Hooks.Size < context->Hooks.Capacity) {
        ImGui::AddContextHook(context, &hook);
    }
    auto* originalData = context->Hooks.Data;
    const auto originalSize = context->Hooks.Size;
    const auto originalCapacity = context->Hooks.Capacity;
    const std::vector<ImGuiContextHook> originalHooks(context->Hooks.begin(), context->Hooks.end());
    std::thread renderer([&] { ImGui::CallContextHooks(context, ImGuiContextHookType_EndFramePost); });
    while (!gate.entered.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    for (int i = 0; i < 64; ++i) {
        fixture.deliver(fixture.a, NSEventTypeLeftMouseDown);
        fixture.deliver(fixture.a, NSEventTypeLeftMouseUp);
    }
    gate.resume.store(true, std::memory_order_release);
    renderer.join();
    REQUIRE(context->Hooks.Data == originalData);
    REQUIRE(context->Hooks.Size == originalSize);
    REQUIRE(context->Hooks.Capacity == originalCapacity);
    REQUIRE(gate.completed.load(std::memory_order_relaxed) == originalSize - 2);
    for (int i = 0; i < originalSize; ++i) {
        REQUIRE(context->Hooks[i].HookId == originalHooks[i].HookId);
        REQUIRE(context->Hooks[i].Type == originalHooks[i].Type);
        REQUIRE(context->Hooks[i].Callback == originalHooks[i].Callback);
        REQUIRE(context->Hooks[i].UserData == originalHooks[i].UserData);
    }
    REQUIRE(context->InputEventsQueue.empty());
    REQUIRE(fixture.contextB->InputEventsQueue.empty());
}

TEST_CASE("Cocoa input callbacks cannot follow reused window or backend registrations", "[core][render][cocoa]") {
    TitlebarFixture fixture;
    fixture.contextA->IO.AddMouseButtonEvent(ImGuiMouseButton_Left, true);
    auto stale = ImGui_ImplGlfw_MakeLeftButtonRelease(TitlebarFixture::handle(fixture.a));
    ImGui_ImplGlfw_ContextMap_Remove(TitlebarFixture::handle(fixture.a));
    SECTION("Replacement registration in the same backend") {}
    SECTION("Replacement backend in the same context") {
        ImGui_ImplGlfw_InputOwner_Shutdown(fixture.contextA);
        ImGui_ImplGlfw_InputOwner_Init(fixture.contextA);
    }
    ImGui_ImplGlfw_ContextMap_Add(TitlebarFixture::handle(fixture.a), fixture.contextA);
    stale();
    REQUIRE(fixture.releases(fixture.contextA) == 0);
    REQUIRE(fixture.releases(fixture.contextB) == 0);
    auto replacement = ImGui_ImplGlfw_MakeLeftButtonRelease(TitlebarFixture::handle(fixture.a));
    REQUIRE(static_cast<bool>(replacement));
    replacement();
    REQUIRE(fixture.releases(fixture.contextA) == 1);
}

int main(int argc, char* argv[]) {
    return Catch::Session().run(argc, argv);
}
