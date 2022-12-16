#include "jetstream/viewport/ios.hh"

namespace Jetstream::Viewport {
    
iOS::iOS(const Config& config, CA::MetalLayer* layer) : Viewport::Generic(config) {
    JST_DEBUG("Creating iOS viewport.");
    swapchain = layer;
};

iOS::~iOS() {
    JST_DEBUG("Destroying iOS viewport.");
}

const Result iOS::create() {
    auto* device = Backend::State<Device::Metal>()->getDevice();
    
    swapchain->setDevice(device);
    swapchain->setPixelFormat(MTL::PixelFormatBGRA8Unorm);
    swapchain->setFramebufferOnly(true);
    
    return Result::SUCCESS;
}

const Result iOS::destroy() {
    swapchain->release();
    
    return Result::SUCCESS;
}

const Result iOS::createImgui() {

    return Result::SUCCESS;
}
const Result iOS::destroyImgui() {

    return Result::SUCCESS;
}

void* iOS::nextDrawable() {
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize.x = swapchain->drawableSize().width;
    io.DisplaySize.y = swapchain->drawableSize().height;
    
    return static_cast<void*>(swapchain->nextDrawable());
}

const Result iOS::pollEvents() {
    return Result::SUCCESS;
}

const bool iOS::keepRunning() {
    return true;
}

}  // namespace Jetstream::Viewport
