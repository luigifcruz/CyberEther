#include "jetstream/render/metal/window.hh"
#include "jetstream/render/metal/surface.hh"

#include "tools/imgui_impl_metal.h"

namespace Jetstream::Render {

using Implementation = WindowImp<Device::Metal>;

Implementation::WindowImp(const Config& config,
                          std::shared_ptr<Viewport::Adapter<Device::Metal>>& viewport)
         : Window(config), viewport(viewport) {
}

Result Implementation::bindSurface(const std::shared_ptr<Surface>& surface) {
    JST_DEBUG("[Metal] Binding surface to window.");

    // Cast generic Surface.
    auto _surface = std::dynamic_pointer_cast<SurfaceImp<Device::Metal>>(surface);

    // Create the Surface.
    JST_CHECK(_surface->create());

    // Add Surface to window.
    surfaces.push_back(_surface);

    return Result::SUCCESS;
}

Result Implementation::unbindSurface(const std::shared_ptr<Surface>& surface) {
    JST_DEBUG("[Metal] Unbinding surface from window.");

    // Cast generic Surface.
    auto _surface = std::dynamic_pointer_cast<SurfaceImp<Device::Metal>>(surface);

    // Destroy the Surface.
    JST_CHECK(_surface->destroy());

    // Remove Surface from window.
    surfaces.erase(std::remove(surfaces.begin(), surfaces.end(), _surface), surfaces.end());

    return Result::SUCCESS;
}

Result Implementation::underlyingCreate() {
    JST_DEBUG("Creating Metal window.");

    outerPool = NS::AutoreleasePool::alloc()->init();

    dev = Backend::State<Device::Metal>()->getDevice();

    commandQueue = dev->newCommandQueue();
    JST_ASSERT(commandQueue);

    renderPassDescriptor = MTL::RenderPassDescriptor::alloc()->init();
    JST_ASSERT(renderPassDescriptor);

    if (config.imgui) {
        JST_CHECK(createImgui());
    }

    statsData.droppedFrames = 0;
    
    return Result::SUCCESS;
}

Result Implementation::underlyingDestroy() {
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

#ifndef JST_OS_IOS
    io->ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
#endif
    io->ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    
    JST_CHECK(viewport->createImgui());

    ScaleStyle(*viewport);

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

    ScaleStyle(*viewport);

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

Result Implementation::underlyingEnd() {
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

Result Implementation::underlyingSynchronize() {
    return Result::SUCCESS;
}

void Implementation::drawDebugMessage() const {
    auto& backend = Backend::State<Device::Metal>();

    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::Text("Device Name:");
    ImGui::TableSetColumnIndex(1);
    ImGui::SetNextItemWidth(-1);
    ImGui::TextFormatted("{}", backend->getDeviceName());

    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::Text("Low Power Mode:");
    ImGui::TableSetColumnIndex(1);
    ImGui::SetNextItemWidth(-1);
    ImGui::TextFormatted("{}", backend->getLowPowerStatus() ? "YES" : "NO");

    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::Text("Device Memory:");
    ImGui::TableSetColumnIndex(1);
    ImGui::SetNextItemWidth(-1);
    ImGui::TextFormatted("{:.0f} GB", (float)backend->getPhysicalMemory() / (1024*1024*1024));

    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::Text("Thermal State:");
    ImGui::TableSetColumnIndex(1);
    ImGui::SetNextItemWidth(-1);
    ImGui::TextFormatted("{}/3", backend->getThermalState());
}

const Window::Stats& Implementation::stats() const {
    return statsData;
}

}  // namespace Jetstream::Render
