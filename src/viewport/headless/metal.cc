#include "jetstream/viewport/platforms/headless/metal.hh"

#include <thread>

namespace Jetstream::Viewport {

using Implementation = Headless<DeviceType::Metal>;

Implementation::Headless(const Config& config) : Adapter(config) {
    JST_DEBUG("[HEADLESS-METAL] Creating headless viewport.");
}

Implementation::~Headless() {
    JST_DEBUG("[HEADLESS-METAL] Destroying headless viewport.");
}

Result Implementation::create() {
    if (config.size.x == 0 || config.size.y == 0 || config.framerate == 0) {
        JST_ERROR("[HEADLESS-METAL] Viewport size and framerate must be greater than zero.");
        return Result::ERROR;
    }

    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    auto* descriptor = MTL::TextureDescriptor::texture2DDescriptor(
        MTL::PixelFormatBGRA8Unorm, config.size.x, config.size.y, false);
    descriptor->setStorageMode(MTL::StorageModePrivate);
    descriptor->setUsage(MTL::TextureUsageRenderTarget);
    texture = Backend::State<DeviceType::Metal>()->getDevice()->newTexture(descriptor);
    if (!texture) {
        JST_ERROR("[HEADLESS-METAL] Failed to create offscreen texture.");
        return Result::ERROR;
    }
    lastTime = std::chrono::steady_clock::now();
    return Result::SUCCESS;
}

Result Implementation::destroy() {
    if (texture) {
        texture->release();
        texture = nullptr;
    }
    return Result::SUCCESS;
}

Extent2D<F32> Implementation::framebufferSize() const {
    return {static_cast<F32>(config.size.x), static_cast<F32>(config.size.y)};
}

MTL::Texture* Implementation::nextDrawable(CA::MetalDrawable*& drawable) {
    const auto interval = std::chrono::duration<F64>(1.0 / config.framerate);
    std::this_thread::sleep_until(lastTime + interval);
    lastTime = std::chrono::steady_clock::now();

    drawable = nullptr;
    return texture;
}

Extent2D<U64> Implementation::getSwapchainExtent() const {
    return config.size;
}

Result Implementation::waitEvents() {
    std::this_thread::sleep_for(std::chrono::milliseconds(3));
    return Result::SUCCESS;
}

Result Implementation::pollEvents() {
    return Result::SUCCESS;
}

bool Implementation::keepRunning() {
    return texture != nullptr;
}

}  // namespace Jetstream::Viewport
