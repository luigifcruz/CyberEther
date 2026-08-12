#include "jetstream/render/devices/metal/window.hh"
#include "jetstream/render/devices/metal/surface.hh"

#include "tools/imgui_impl_metal.h"

namespace Jetstream::Render {

using Implementation = WindowImp<DeviceType::Metal>;

Implementation::WindowImp(const Config& config,
                          const std::shared_ptr<Viewport::Adapter<DeviceType::Metal>>& viewport)
         : Window(config), viewport(viewport) {
}

Result Implementation::bindSurface(const std::shared_ptr<Surface>& surface) {
    auto _resource = std::dynamic_pointer_cast<SurfaceImp<DeviceType::Metal>>(surface);
    surfaces.push_back(_resource);
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

    dev = Backend::State<DeviceType::Metal>()->getDevice();

    commandQueue = dev->newCommandQueue();
    JST_ASSERT(commandQueue, "Failed to create command queue.");

    renderPassDescriptor = MTL::RenderPassDescriptor::alloc()->init();
    JST_ASSERT(renderPassDescriptor, "Failed to create render pass descriptor.");

    JST_CHECK(createImgui());

    // Reseting internal variables.

    statsData.droppedFrames = 0;
    statsData.recreatedFrames = 0;
    currentFrame = 0;

    return Result::SUCCESS;
}

Result Implementation::underlyingDestroy() {
    JST_DEBUG("[METAL] Destroying window.");

    Result result = underlyingSynchronize();
    underlyingCancel();

    if (imguiCreated) {
        const Result imguiResult = destroyImgui();
        if ((result == Result::SUCCESS || result == Result::RELOAD) &&
            imguiResult != Result::SUCCESS && imguiResult != Result::RELOAD) {
            result = imguiResult;
        }
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
    dev = nullptr;

    return result;
}

Result Implementation::createImgui() {
    JST_DEBUG("[METAL] Creating ImGui.");

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    io = &ImGui::GetIO();
    style = &ImGui::GetStyle();
    io->IniFilename = nullptr;

    io->ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    Result viewportResult;
    try {
        viewportResult = viewport->createImgui();
    } catch (...) {
        ImGui::DestroyContext();
        io = nullptr;
        style = nullptr;
        throw;
    }
    if (viewportResult != Result::SUCCESS && viewportResult != Result::RELOAD) {
        ImGui::DestroyContext();
        io = nullptr;
        style = nullptr;
        return viewportResult;
    }

    try {
        this->updateScalingFactor(*viewport);
        ImGui_ImplMetal_Init(dev);
    } catch (...) {
        viewport->destroyImgui();
        ImGui::DestroyContext();
        io = nullptr;
        style = nullptr;
        throw;
    }
    imguiCreated = true;

    return Result::SUCCESS;
}

Result Implementation::destroyImgui() {
    JST_DEBUG("[METAL] Destroying ImGui.");

    ImGui_ImplMetal_Shutdown();
    const Result result = viewport->destroyImgui();
    ImGui::DestroyContext();
    io = nullptr;
    style = nullptr;
    imguiCreated = false;

    return result;
}

Result Implementation::beginImgui() {
    ImGui_ImplMetal_NewFrame(renderPassDescriptor);

    this->updateScalingFactor(*viewport);

    ImGui::NewFrame();

    return Result::SUCCESS;
}

Result Implementation::endImgui() {
    auto renderCmdEncoder = commandBuffer->renderCommandEncoder(renderPassDescriptor);

    ImGui::Render();
    ImGui_ImplMetal_RenderDrawData(ImGui::GetDrawData(),
        commandBuffer, renderCmdEncoder);

    renderCmdEncoder->endEncoding();

    return Result::SUCCESS;
}

Result Implementation::underlyingBegin() {
    if (inFlightCommandBuffers[currentFrame]) {
        inFlightCommandBuffers[currentFrame]->waitUntilCompleted();
        inFlightCommandBuffers[currentFrame]->release();
        inFlightCommandBuffers[currentFrame] = nullptr;
    }

    innerPool = NS::AutoreleasePool::alloc()->init();

    drawable = static_cast<CA::MetalDrawable*>(viewport->nextDrawable());

    if (!drawable) {
        statsData.droppedFrames += 1;
        innerPool->release();
        innerPool = nullptr;
        return Result::SKIP;
    }

    auto colorAttachDescriptor = renderPassDescriptor->colorAttachments()->object(0)->init();
    colorAttachDescriptor->setTexture(drawable->texture());
    colorAttachDescriptor->setLoadAction(MTL::LoadActionClear);
    colorAttachDescriptor->setStoreAction(MTL::StoreActionStore);
    colorAttachDescriptor->setClearColor(MTL::ClearColor(0, 0, 0, 1.0));

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

    const Result result = beginImgui();
    if (result != Result::SUCCESS && result != Result::RELOAD) {
        innerPool->release();
        innerPool = nullptr;
        commandBuffer = nullptr;
        return result;
    }

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
        result = surface->draw(commandBuffer);
        if (result != Result::SUCCESS && result != Result::RELOAD) {
            return abortFrame(result);
        }
    }

    result = endImgui();
    if (result != Result::SUCCESS && result != Result::RELOAD) {
        return abortFrame(result);
    }

    commandBuffer->presentDrawable(drawable);
    commandBuffer->retain();
    commandBuffer->commit();
    inFlightCommandBuffers[currentFrame] = commandBuffer;
    for (auto& surface : surfaces) {
        surface->commitDraw();
    }
    transfers.commit();

    currentFrame = (currentFrame + 1) % FramesInFlight;

    innerPool->release();
    innerPool = nullptr;
    commandBuffer = nullptr;

    return Result::SUCCESS;
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

const Window::Stats& Implementation::stats() const {
    return statsData;
}

}  // namespace Jetstream::Render
