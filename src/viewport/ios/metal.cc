#include "jetstream/viewport/platforms/ios/metal.hh"

namespace Jetstream::Viewport {
    
using Implementation = iOS<DeviceType::Metal>;

Implementation::iOS(const Config& config, CA::MetalLayer* layer) : Adapter(config) {
    JST_DEBUG("Creating iOS viewport.");
    swapchain = layer;
};

Implementation::~iOS() {
    JST_DEBUG("Destroying iOS viewport.");
}

Result Implementation::create() {
    auto* device = Backend::State<DeviceType::Metal>()->getDevice();
    
    swapchain->setDevice(device);
    swapchain->setPixelFormat(MTL::PixelFormatBGRA8Unorm);
    swapchain->setFramebufferOnly(true);

    lastTime = std::chrono::high_resolution_clock::now();
    
    return Result::SUCCESS;
}

Result Implementation::destroy() {
    swapchain->release();
    
    return Result::SUCCESS;
}

Result Implementation::createImgui() {

    return Result::SUCCESS;
}

Extent2D<F32> Implementation::displaySize() const {
    return {static_cast<F32>(swapchain->drawableSize().width / 2.0),
            static_cast<F32>(swapchain->drawableSize().height / 2.0)};
}

F32 Implementation::scale(const F32& scale) const {
    return scale;
}

Result Implementation::destroyImgui() {

    return Result::SUCCESS;
}

MTL::Texture* Implementation::nextDrawable(CA::MetalDrawable*& drawable) {
    auto currentTime = std::chrono::high_resolution_clock::now();
    float deltaTime = std::chrono::duration<float>(currentTime - lastTime).count();
    lastTime = currentTime;

    // Update ImGui state.

    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize.x = swapchain->drawableSize().width / 2.0;
    io.DisplaySize.y = swapchain->drawableSize().height / 2.0;
    io.DisplayFramebufferScale.x = 2.0;
    io.DisplayFramebufferScale.y = 2.0;
    io.DeltaTime = deltaTime;
    
    drawable = swapchain->nextDrawable();
    return drawable ? drawable->texture() : nullptr;
}

Extent2D<U64> Implementation::getSwapchainExtent() const {
    return {static_cast<U64>(swapchain->drawableSize().width),
            static_cast<U64>(swapchain->drawableSize().height)};
}

void Implementation::setCaptureEnabled(bool enabled) {
    swapchain->setFramebufferOnly(!enabled);
}

Result Implementation::waitEvents() {
    return Result::SUCCESS;
}

Result Implementation::pollEvents() {
    return Result::SUCCESS;
}

bool Implementation::keepRunning() {
    return true;
}

}  // namespace Jetstream::Viewport
