#include "jetstream/render/metal/window.hh"
#include "jetstream/render/metal/surface.hh"

namespace Jetstream::Render {

using Implementation = WindowImp<Device::Metal>;

Implementation::WindowImp(const Config& config,
                          std::shared_ptr<Viewport::Adapter<Device::Metal>>& viewport)
         : Window(config), viewport(viewport) {
}

Result Implementation::bind(const std::shared_ptr<Surface>& surface) {
    JST_DEBUG("Binding Metal surface to window.");

    surfaces.push_back(
        std::dynamic_pointer_cast<SurfaceImp<Device::Metal>>(surface)
    );

    return Result::SUCCESS;
}

Result Implementation::create() {
    JST_DEBUG("Creating Metal window.");

    outerPool = NS::AutoreleasePool::alloc()->init();

    dev = Backend::State<Device::Metal>()->getDevice();

    commandQueue = dev->newCommandQueue();
    JST_ASSERT(commandQueue);

    renderPassDescriptor = MTL::RenderPassDescriptor::alloc()->init();
    JST_ASSERT(renderPassDescriptor);

    for (auto& surface : surfaces) {
        JST_CHECK(surface->create());
    }

    if (config.imgui) {
        JST_CHECK(createImgui());
    }

    statsData.droppedFrames = 0;
    
    return Result::SUCCESS;
}

Result Implementation::destroy() {
    JST_DEBUG("Destroying Metal window.");

    for (auto& surface : surfaces) {
        JST_CHECK(surface->destroy());
    }

    if (config.imgui) {
        JST_CHECK(destroyImgui());
    } 

    renderPassDescriptor->release();
    commandQueue->release();

    outerPool->release();

    return Result::SUCCESS;
}

Result Implementation::createImgui() {
    JST_DEBUG("Creating Metal ImGui.");

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImNodes::CreateContext();

    io = &ImGui::GetIO();
    style = &ImGui::GetStyle();

#ifndef TARGET_OS_IPHONE
    io->ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
#endif
    io->ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    
    JST_CHECK(viewport->createImgui());

    const auto& scale = viewport->calculateScale(config.scale);
    ApplyImGuiTheme(scale);
    ApplyImNodesTheme(scale);

    ImGui_ImplMetal_Init(dev);

    return Result::SUCCESS;
}

Result Implementation::destroyImgui() {
    JST_DEBUG("Destroying Metal ImGui.");

    ImGui_ImplMetal_Shutdown();
    JST_CHECK(viewport->destroyImgui());
    ImNodes::DestroyContext();
    ImGui::DestroyContext();

    return Result::SUCCESS;
}

Result Implementation::beginImgui() {
    ImGui_ImplMetal_NewFrame(renderPassDescriptor);

    ApplyImGuiScale();
    ApplyImNodesScale();

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

Result Implementation::begin() {
    innerPool = NS::AutoreleasePool::alloc()->init();

    drawable = static_cast<CA::MetalDrawable*>(viewport->nextDrawable());

    if (!drawable) {
        statsData.droppedFrames += 1;
        return Result::SKIP;
    }

    auto colorAttachDescriptor = renderPassDescriptor->colorAttachments()->object(0)->init();
    colorAttachDescriptor->setTexture(drawable->texture());
    colorAttachDescriptor->setLoadAction(MTL::LoadActionClear);
    colorAttachDescriptor->setStoreAction(MTL::StoreActionStore);
    colorAttachDescriptor->setClearColor(MTL::ClearColor(0, 0, 0, 1.0));

    if (config.imgui) {
        JST_CHECK(beginImgui());
    }

    return Result::SUCCESS;
}

Result Implementation::end() {
    commandBuffer = commandQueue->commandBuffer();

    for (auto &surface : surfaces) {
        JST_CHECK(surface->draw(commandBuffer));
    }

    if (config.imgui) {
        JST_CHECK(endImgui());
    }

    commandBuffer->presentDrawable(drawable);
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();

    innerPool->release();

    return Result::SUCCESS;
}

void Implementation::drawDebugMessage() const {
    auto& backend = Backend::State<Device::Metal>();
    ImGuiIO& io = ImGui::GetIO();

    ImGui::TextFormatted("FPS: {:.1f} Hz", io.Framerate);
    ImGui::TextFormatted("Device Name: {}", backend->getDeviceName());
    ImGui::TextFormatted("Low Power Mode: {}", backend->getLowPowerStatus() ? "YES" : "NO");
    ImGui::TextFormatted("Has Unified Memory: {}", backend->hasUnifiedMemory() ? "YES" : "NO");
    ImGui::TextFormatted("Physical Memory: {:.0f} GB", (float)backend->getPhysicalMemory() / (1024*1024*1024));
    ImGui::TextFormatted("Thermal State: {}/3", backend->getThermalState());
    ImGui::TextFormatted("Processor Count: {}/{}", backend->getActiveProcessorCount(),
                                                   backend->getTotalProcessorCount());
}

const Window::Stats& Implementation::stats() const {
    return statsData;
}

}  // namespace Jetstream::Render
