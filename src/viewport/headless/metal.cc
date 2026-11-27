#include "jetstream/viewport/platforms/headless/metal.hh"

#include <algorithm>
#include <cmath>
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

    extent = config.size;
    requestedExtent = config.size;
    JST_CHECK(createTexture(extent));
    running = true;
    lastTime = std::chrono::steady_clock::now();
    return Result::SUCCESS;
}

Result Implementation::createTexture(const Extent2D<U64>& size) {
    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    auto* descriptor = MTL::TextureDescriptor::texture2DDescriptor(
        MTL::PixelFormatBGRA8Unorm, size.x, size.y, false);
    descriptor->setStorageMode(MTL::StorageModePrivate);
    descriptor->setUsage(MTL::TextureUsageRenderTarget);
    auto* next = Backend::State<DeviceType::Metal>()->getDevice()->newTexture(descriptor);
    if (!next) {
        JST_ERROR("[HEADLESS-METAL] Failed to create offscreen texture.");
        return Result::ERROR;
    }
    if (texture) {
        texture->release();
    }
    texture = next;
    return Result::SUCCESS;
}

Result Implementation::resize(const Extent2D<U64>& size, F32 scale) {
    if (size.x == 0 || size.y == 0 || !(scale > 0.0f) || !std::isfinite(scale)) {
        return Result::ERROR;
    }

    {
        std::lock_guard<std::mutex> lock(extentMutex);
        requestedExtent = size;
        requestedScale = scale;
    }

    requestRedraw();
    return Result::SUCCESS;
}

Result Implementation::destroy() {
    running = false;
    if (texture) {
        texture->release();
        texture = nullptr;
    }
    return Result::SUCCESS;
}

Extent2D<F32> Implementation::displaySize() const {
    std::lock_guard<std::mutex> lock(extentMutex);
    return {static_cast<F32>(extent.x) / scale, static_cast<F32>(extent.y) / scale};
}

Extent2D<F32> Implementation::framebufferSize() const {
    std::lock_guard<std::mutex> lock(extentMutex);
    return {static_cast<F32>(extent.x), static_cast<F32>(extent.y)};
}

MTL::Texture* Implementation::nextDrawable(CA::MetalDrawable*& drawable) {
    using Clock = Jetstream::Platform::DeadlineWaiter::Clock;
    const auto period = std::chrono::duration_cast<Clock::duration>(std::chrono::duration<F64>(1.0 / config.framerate));
    lastTime = std::max(lastTime + period, Clock::now());
    waiter.waitUntil(lastTime, waiter.generation());

    Extent2D<U64> next;
    F32 nextScale = 1.0f;
    {
        std::lock_guard<std::mutex> lock(extentMutex);
        next = requestedExtent;
        nextScale = requestedScale;
    }
    if ((next.x != extent.x || next.y != extent.y) && createTexture(next) == Result::SUCCESS) {
        std::lock_guard<std::mutex> lock(extentMutex);
        extent = next;
    }
    {
        std::lock_guard<std::mutex> lock(extentMutex);
        scale = nextScale;
    }

    drawable = nullptr;
    return texture;
}

Extent2D<U64> Implementation::getSwapchainExtent() const {
    std::lock_guard<std::mutex> lock(extentMutex);
    return extent;
}

Result Implementation::waitEvents() {
    std::this_thread::sleep_for(std::chrono::milliseconds(3));
    return Result::SUCCESS;
}

Result Implementation::pollEvents() {
    return Result::SUCCESS;
}

bool Implementation::keepRunning() {
    return running;
}

bool Implementation::retainedPresent() const {
    return true;
}

}  // namespace Jetstream::Viewport
