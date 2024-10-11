#include "jetstream/render/devices/metal/window.hh"
#include "jetstream/render/devices/metal/surface.hh"
#include "jetstream/render/devices/metal/buffer.hh"
#include "jetstream/render/devices/metal/texture.hh"

#include "tools/imgui_impl_metal.h"

namespace Jetstream::Render {

using Implementation = WindowImp<Device::Metal>;

Implementation::WindowImp(const Config& config,
                          std::shared_ptr<Viewport::Adapter<Device::Metal>>& viewport)
         : Window(config), viewport(viewport) {
}

template<typename T>
Result Implementation::bindResource(const auto& resource, std::vector<std::shared_ptr<T>>& container) {
    // Cast generic resource.
    auto _resource = std::dynamic_pointer_cast<T>(resource);

    // Create the resource.
    JST_CHECK(_resource->create());

    // Add resource to container.
    container.push_back(_resource);

    return Result::SUCCESS;
}

template<typename T>
Result Implementation::unbindResource(const auto& resource, std::vector<std::shared_ptr<T>>& container) {
    // Cast generic resource.
    auto _resource = std::dynamic_pointer_cast<T>(resource);

    // Destroy the resource.
    JST_CHECK(_resource->destroy());

    // Remove resource from container.
    container.erase(std::remove(container.begin(), container.end(), _resource), container.end());

    return Result::SUCCESS;
}

Result Implementation::bindBuffer(const std::shared_ptr<Buffer>& buffer) {
    return bindResource<BufferImp<Device::Metal>>(buffer, buffers);
}

Result Implementation::unbindBuffer(const std::shared_ptr<Buffer>& buffer) {
    return unbindResource<BufferImp<Device::Metal>>(buffer, buffers);
}

Result Implementation::bindTexture(const std::shared_ptr<Texture>& texture) {
    return bindResource<TextureImp<Device::Metal>>(texture, textures);
}

Result Implementation::unbindTexture(const std::shared_ptr<Texture>& texture) {
    return unbindResource<TextureImp<Device::Metal>>(texture, textures);
}

Result Implementation::bindSurface(const std::shared_ptr<Surface>& surface) {
    return bindResource<SurfaceImp<Device::Metal>>(surface, surfaces);
}

Result Implementation::unbindSurface(const std::shared_ptr<Surface>& surface) {
    return unbindResource<SurfaceImp<Device::Metal>>(surface, surfaces);
}

Result Implementation::underlyingCreate() {
    JST_DEBUG("[METAL] Creating window.");

    outerPool = NS::AutoreleasePool::alloc()->init();

    dev = Backend::State<Device::Metal>()->getDevice();

    commandQueue = dev->newCommandQueue();
    JST_ASSERT(commandQueue);

    renderPassDescriptor = MTL::RenderPassDescriptor::alloc()->init();
    JST_ASSERT(renderPassDescriptor);

    JST_CHECK(createImgui());

    statsData.droppedFrames = 0;
    
    return Result::SUCCESS;
}

Result Implementation::underlyingDestroy() {
    JST_DEBUG("[METAL] Destroying window.");

    JST_CHECK(destroyImgui());

    if (!buffers.empty() || !textures.empty() || !surfaces.empty()) {
        JST_WARN("[METAL] Resources are still bounded to this window "
                 "(buffers={}, textures={}, surfaces={}).", 
                 buffers.size(), textures.size(), surfaces.size());
    }

    renderPassDescriptor->release();
    commandQueue->release();

    outerPool->release();

    return Result::SUCCESS;
}

Result Implementation::createImgui() {
    JST_DEBUG("[METAL] Creating ImGui.");

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    io = &ImGui::GetIO();
    style = &ImGui::GetStyle();

#ifndef JST_OS_IOS
    io->ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
#endif
    io->ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    
    JST_CHECK(viewport->createImgui());

    this->scaleStyle(*viewport);

    ImGui_ImplMetal_Init(dev);

    return Result::SUCCESS;
}

Result Implementation::destroyImgui() {
    JST_DEBUG("[METAL] Destroying ImGui.");

    ImGui_ImplMetal_Shutdown();
    JST_CHECK(viewport->destroyImgui());
    ImGui::DestroyContext();

    return Result::SUCCESS;
}

Result Implementation::beginImgui() {
    ImGui_ImplMetal_NewFrame(renderPassDescriptor);

    this->scaleStyle(*viewport);

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

    commandBuffer = commandQueue->commandBuffer();

    for (auto &surface : surfaces) {
        JST_CHECK(surface->draw(commandBuffer));
    }

    JST_CHECK(beginImgui());

    return Result::SUCCESS;
}

Result Implementation::underlyingEnd() {
    JST_CHECK(endImgui());

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
