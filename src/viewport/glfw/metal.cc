#include "jetstream/viewport/platforms/glfw/metal.hh"

#define GLFW_INCLUDE_NONE
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include "tools/cocoa_titlebar.hh"
#include "tools/glfw_input.hh"

namespace Jetstream::Viewport {

static void PrintGLFWError(int, const char* description) {
    JST_FATAL("[METAL] GLFW error: {}", description);
}

using Implementation = GLFW<DeviceType::Metal>;

Implementation::GLFW(const Config& config) : Adapter(config) {
    JST_DEBUG("[METAL] Creating GLFW viewport.");
}

Implementation::~GLFW() {
    JST_DEBUG("[METAL] Destroying GLFW viewport.");
}

Result Implementation::create() {
    // Initialize and configure GLFW.

    if (!glfwInit()) {
        JST_ERROR("[METAL] Failed to initialize GLFW.");
        return Result::ERROR;
    }

    glfwSetErrorCallback(&PrintGLFWError);

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_DOUBLEBUFFER, config.vsync);

    auto [width, height] = config.size;
    window = glfwCreateWindow(width, height,
        config.title.c_str(), nullptr, nullptr);

    if (!window) {
        glfwTerminate();
        JST_ERROR("[METAL] Failed to create window with GLFW.");
        return Result::ERROR;
    }

    attachPlatform(std::make_unique<GlfwInput>(window, *this));
    pumpPlatform();

    auto* device = Backend::State<DeviceType::Metal>()->getDevice();

    swapchain = CA::MetalLayer::layer()->retain();
    swapchain->setDevice(device);
    swapchain->setPixelFormat(MTL::PixelFormatBGRA8Unorm);
    swapchain->setFramebufferOnly(true);

    auto* native = static_cast<NS::Window*>(glfwGetCocoaWindow(window));
    native->contentView()->setLayer(swapchain);
    native->contentView()->setWantsLayer(true);

    setIntegratedTitlebar(config.integratedTitlebar);

    return Result::SUCCESS;
}

Result Implementation::destroy() {
    titlebar.reset();
    detachPlatform();

    glfwDestroyWindow(window);
    window = nullptr;
    glfwTerminate();

    swapchain->release();

    return Result::SUCCESS;
}

void Implementation::setIntegratedTitlebar(bool enabled) {
    Generic::setIntegratedTitlebar(enabled);
    if (enabled && window && !titlebar) {
        titlebar = std::make_shared<CocoaTitlebar>(window, [this]() {
            input().push(Render::InputQueue::Button{MouseButton::Left, false});
        });
    } else if (!enabled) {
        titlebar.reset();
    }
}

Extent2D<F32> Implementation::displaySize() const {
    int width = 0;
    int height = 0;
    glfwGetWindowSize(window, &width, &height);

    return {static_cast<F32>(width), static_cast<F32>(height)};
}

Extent2D<F32> Implementation::framebufferSize() const {
    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window, &width, &height);

    return {static_cast<F32>(width), static_cast<F32>(height)};
}

MTL::Texture* Implementation::nextDrawable(CA::MetalDrawable*& drawable) {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    swapchain->setDrawableSize({
        static_cast<CGFloat>(width),
        static_cast<CGFloat>(height)
    });

    drawable = swapchain->nextDrawable();
    return drawable ? drawable->texture() : nullptr;
}

Extent2D<U64> Implementation::getSwapchainExtent() const {
    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window, &width, &height);
    return {static_cast<U64>(width), static_cast<U64>(height)};
}

void Implementation::setCaptureEnabled(bool enabled) {
    swapchain->setFramebufferOnly(!enabled);
}

Result Implementation::waitEvents() {
    glfwWaitEventsTimeout(0.150);
    if (titlebar) {
        titlebar->poll();
    }
    pumpPlatform();
    return Result::SUCCESS;
}

Result Implementation::pollEvents() {
    glfwPollEvents();
    if (titlebar) {
        titlebar->poll();
    }
    pumpPlatform();
    return Result::SUCCESS;
}

bool Implementation::retainedPresent() const {
    return true;
}

bool Implementation::keepRunning() {
    return !glfwWindowShouldClose(window);
}

Render::TitlebarChrome Implementation::titlebarChrome() const {
    return titlebar ? titlebar->chrome() : Render::TitlebarChrome{};
}

void Implementation::beginWindowDrag() {
    if (titlebar) {
        titlebar->beginDrag();
    }
}

void Implementation::zoomWindow() {
    if (titlebar) {
        titlebar->zoom();
    }
}

bool Implementation::setDarkAppearance(bool dark) {
    return CocoaTitlebar::SetAppearance(window, dark);
}

}  // namespace Jetstream::Viewport
