#include <algorithm>

#include "jetstream/render/devices/metal/window.hh"
#include "jetstream/render/devices/metal/surface.hh"

#include "presenter.hh"

namespace Jetstream::Render {

using Implementation = WindowImp<DeviceType::Metal>;

Implementation::WindowImp(const Config& config,
                          const std::shared_ptr<Viewport::Adapter<DeviceType::Metal>>& viewport)
         : Window(config), viewport(viewport) {
}

Implementation::~WindowImp() = default;

Result Implementation::bindSurface(const std::shared_ptr<Surface>& surface) {
    auto _resource = std::dynamic_pointer_cast<SurfaceImp<DeviceType::Metal>>(surface);
    if (_resource->getConfig().composite) {
        surfaces.push_back(_resource);
    } else {
        const auto firstComposite = std::find_if(surfaces.begin(), surfaces.end(), [](const auto& other) {
            return other->getConfig().composite;
        });
        surfaces.insert(firstComposite, _resource);
    }
    return Result::SUCCESS;
}

Result Implementation::unbindSurface(const std::shared_ptr<Surface>& surface) {
    auto _resource = std::dynamic_pointer_cast<SurfaceImp<DeviceType::Metal>>(surface);
    surfaces.erase(std::remove(surfaces.begin(), surfaces.end(), _resource), surfaces.end());
    return Result::SUCCESS;
}

Result Implementation::underlyingCreate() {
    JST_DEBUG("[METAL] Creating window.");

    outerPool = NS::AutoreleasePool::alloc()->init();

    auto* dev = Backend::State<DeviceType::Metal>()->getDevice();

    commandQueue = dev->newCommandQueue();
    JST_ASSERT(commandQueue, "Failed to create command queue.");

    renderPassDescriptor = MTL::RenderPassDescriptor::alloc()->init();
    JST_ASSERT(renderPassDescriptor, "Failed to create render pass descriptor.");

    JST_CHECK(createPresenter());
    this->updateMetrics(*viewport);

    currentFrame = 0;

    return Result::SUCCESS;
}

Result Implementation::underlyingDestroy() {
    JST_DEBUG("[METAL] Destroying window.");

    Result result = underlyingSynchronize();
    underlyingCancel();

    if (presenter) {
        presenter->destroy();
        presenter.reset();
    }

    transferEncoder.destroy();

    if (renderPassDescriptor) {
        renderPassDescriptor->release();
        renderPassDescriptor = nullptr;
    }
    if (commandQueue) {
        commandQueue->release();
        commandQueue = nullptr;
    }

    if (outerPool) {
        outerPool->release();
        outerPool = nullptr;
    }

    return result;
}

Result Implementation::createPresenter() {
    presenter = std::make_unique<Presenter>();
    const Result result = presenter->create(MTL::PixelFormatBGRA8Unorm);
    if (result != Result::SUCCESS) {
        presenter.reset();
    }
    return result;
}

void Implementation::encodePresent(MTL::Texture* target) {
    auto colorAttachDescriptor = renderPassDescriptor->colorAttachments()->object(0)->init();
    colorAttachDescriptor->setTexture(target);
    colorAttachDescriptor->setLoadAction(MTL::LoadActionClear);
    colorAttachDescriptor->setStoreAction(MTL::StoreActionStore);

    const auto& background = frameClearColor();
    colorAttachDescriptor->setClearColor(MTL::ClearColor(background.r, background.g, background.b, background.a));
    auto renderCmdEncoder = commandBuffer->renderCommandEncoder(renderPassDescriptor);

    const Extent2D<U64> extent = {target->width(), target->height()};
    presenter->encode(renderCmdEncoder, framePresentList(), metrics(), extent);

    renderCmdEncoder->endEncoding();
}

Result Implementation::underlyingBegin() {
    if (inFlightCommandBuffers[currentFrame]) {
        inFlightCommandBuffers[currentFrame]->waitUntilCompleted();
        inFlightCommandBuffers[currentFrame]->release();
        inFlightCommandBuffers[currentFrame] = nullptr;
    }

    innerPool = NS::AutoreleasePool::alloc()->init();
    drawable = nullptr;

    commandBuffer = commandQueue->commandBuffer();
    if (!commandBuffer) {
        innerPool->release();
        innerPool = nullptr;
        return Result::ERROR;
    }

    for (auto& surface : surfaces) {
        const Result prepareResult = surface->prepare();
        if (prepareResult != Result::SUCCESS && prepareResult != Result::RELOAD) {
            innerPool->release();
            innerPool = nullptr;
            commandBuffer = nullptr;
            return prepareResult;
        }
    }

    this->updateMetrics(*viewport);
    this->updateInput(*viewport);

    return Result::SUCCESS;
}

Result Implementation::underlyingEnd() {
    Transfer::Batch transfers;

    const auto abortFrame = [&](const Result& result) {
        const Result recovery = underlyingCancel();
        return recovery == Result::SUCCESS ? result : recovery;
    };

    Result result = collectTransfers(transfers);
    if (result != Result::SUCCESS && result != Result::RELOAD) {
        return abortFrame(result);
    }
    if (!transfers.empty()) {
        result = transferEncoder.encode(transfers, commandBuffer, currentFrame);
        if (result != Result::SUCCESS && result != Result::RELOAD) {
            return abortFrame(result);
        }
    }

    for (auto& surface : surfaces) {
        const auto& framebuffer = *surface->getConfig().framebuffer;
        result = surface->draw(commandBuffer, textureVisibleForPresentation(framebuffer));
        if (result != Result::SUCCESS && result != Result::RELOAD) {
            return abortFrame(result);
        }
    }

    auto pass = beginPresent(transfers);
    MTL::Texture* target = nullptr;
    if (pass.presenting) {
        target = viewport->nextDrawable(drawable);
        acquired(pass, target ? Acquire::Ready : Acquire::Dropped);
    }

    if (pass.presenting) {
        encodePresent(target);
        result = viewport->encodeCapture(commandBuffer, target);
        if (result != Result::SUCCESS && result != Result::RELOAD) {
            return abortFrame(result);
        }
        if (drawable) {
            commandBuffer->presentDrawable(drawable);
        }
    }

    commandBuffer->retain();
    commandBuffer->commit();
    inFlightCommandBuffers[currentFrame] = commandBuffer;
    for (auto& surface : surfaces) {
        surface->commitDraw();
    }
    transfers.commit();

    drawable = nullptr;

    currentFrame = (currentFrame + 1) % FramesInFlight;

    innerPool->release();
    innerPool = nullptr;
    commandBuffer = nullptr;

    return finishPresent(pass);
}

Result Implementation::underlyingCancel() {
    commandBuffer = nullptr;
    drawable = nullptr;
    if (innerPool) {
        innerPool->release();
        innerPool = nullptr;
    }
    return Result::SUCCESS;
}

Result Implementation::underlyingSynchronize() {
    for (auto*& inFlight : inFlightCommandBuffers) {
        if (inFlight) {
            inFlight->waitUntilCompleted();
            inFlight->release();
            inFlight = nullptr;
        }
    }

    return Result::SUCCESS;
}

std::string Implementation::info() const {
    auto& backend = Backend::State<DeviceType::Metal>();

    return jst::fmt::format("{} ({:.0f} GB)", backend->getDeviceName(),
                                              (float)backend->getPhysicalMemory() / (1024*1024*1024));
}

}  // namespace Jetstream::Render
