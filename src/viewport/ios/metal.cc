#include "jetstream/viewport/platforms/ios/metal.hh"

namespace Jetstream::Viewport {
    
using Implementation = iOS<Device::Metal>;

Implementation::iOS(const Config& config, CA::MetalLayer* layer) : Adapter(config) {
    JST_DEBUG("Creating iOS viewport.");
    swapchain = layer;
};

Implementation::~iOS() {
    JST_DEBUG("Destroying iOS viewport.");
}

Result Implementation::create() {
    auto* device = Backend::State<Device::Metal>()->getDevice();
    
    swapchain->setDevice(device);
    swapchain->setPixelFormat(MTL::PixelFormatBGRA8Unorm);
    swapchain->setFramebufferOnly(true);
    
    return Result::SUCCESS;
}

Result Implementation::destroy() {
    swapchain->release();
    
    return Result::SUCCESS;
}

Result Implementation::createImgui() {

    return Result::SUCCESS;
}

F32 Implementation::scale(const F32& scale) const {
    return scale;
}

Result Implementation::destroyImgui() {

    return Result::SUCCESS;
}

void* Implementation::nextDrawable() {
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize.x = swapchain->drawableSize().width / 2.0;
    io.DisplaySize.y = swapchain->drawableSize().height / 2.0;
    io.DisplayFramebufferScale.x = 2.0;
    io.DisplayFramebufferScale.y = 2.0;
    
    return static_cast<void*>(swapchain->nextDrawable());
}

Result Implementation::pollEvents() {
    return Result::SUCCESS;
}

bool Implementation::keepRunning() {
    return true;
}

}  // namespace Jetstream::Viewport
