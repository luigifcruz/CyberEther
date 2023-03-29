#include "jetstream/render/metal/window.hh"
#include "jetstream/render/metal/surface.hh"
#include "jetstream/render/tools/compressed_b612.hh"

namespace Jetstream::Render {

using Implementation = WindowImp<Device::Metal>;

Implementation::WindowImp(const Config& config,
                          std::shared_ptr<Viewport::Generic>& viewport)
         : Window(config, viewport) {
}

const Result Implementation::bind(const std::shared_ptr<Surface>& surface) {
    JST_DEBUG("Binding Metal surface to window.");

    surfaces.push_back(
        std::dynamic_pointer_cast<SurfaceImp<Device::Metal>>(surface)
    );

    return Result::SUCCESS;
}

const Result Implementation::create() {
    JST_DEBUG("Creating Metal window.");

    outerPool = NS::AutoreleasePool::alloc()->init();

    JST_CHECK(viewport->create());

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

const Result Implementation::destroy() {
    JST_DEBUG("Destroying Metal window.");

    for (auto& surface : surfaces) {
        JST_CHECK(surface->destroy());
    }

    if (config.imgui) {
        JST_CHECK(destroyImgui());
    } 

    renderPassDescriptor->release();
    commandQueue->release();

    JST_CHECK(viewport->destroy());

    outerPool->release();

    return Result::SUCCESS;
}

const Result Implementation::createImgui() {
    JST_DEBUG("Creating Metal ImGui.");

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    io = &ImGui::GetIO();
    style = &ImGui::GetStyle();

#ifndef TARGET_OS_IPHONE
    io->ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
#endif
    io->ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    style->ScaleAllSizes(config.scale);
    io->Fonts->AddFontFromMemoryCompressedTTF(
        B612_compressed_data, 
        B612_compressed_size,
        12.0f * config.scale, 
        nullptr, 
        nullptr);

    ImGui::StyleColorsDark();

    JST_CHECK(viewport->createImgui());

    ImGui_ImplMetal_Init(dev);

    return Result::SUCCESS;
}

const Result Implementation::destroyImgui() {
    JST_DEBUG("Destroying Metal ImGui.");

    ImGui_ImplMetal_Shutdown();
    JST_CHECK(viewport->destroyImgui());
    ImGui::DestroyContext();

    return Result::SUCCESS;
}

const Result Implementation::beginImgui() {
    ImGui_ImplMetal_NewFrame(renderPassDescriptor);
    ImGui::NewFrame();

    return Result::SUCCESS;
}

const Result Implementation::endImgui() {
    auto renderCmdEncoder = commandBuffer->renderCommandEncoder(renderPassDescriptor);

    ImGui::Render();
    ImGui_ImplMetal_RenderDrawData(ImGui::GetDrawData(),
        commandBuffer, renderCmdEncoder);

    renderCmdEncoder->endEncoding();

    return Result::SUCCESS;
}

const Result Implementation::begin() {
    innerPool = NS::AutoreleasePool::alloc()->init();

    drawable = static_cast<CA::MetalDrawable*>(viewport->nextDrawable());

    if (!drawable) {
        statsData.droppedFrames += 1;
        return Result::SKIP;
    }

    auto colorAttachDescriptor = renderPassDescriptor->colorAttachments()->object(0);
    colorAttachDescriptor->setTexture(drawable->texture());
    colorAttachDescriptor->setLoadAction(MTL::LoadActionClear);
    colorAttachDescriptor->setStoreAction(MTL::StoreActionStore);
    colorAttachDescriptor->setClearColor(MTL::ClearColor(0, 0, 0, 1.0));

    if (config.imgui) {
        JST_CHECK(beginImgui());

#if !defined(NDEBUG)
        ImGui::ShowMetricsWindow();
        ImGui::Begin("Render Info");
        ImGui::Text("Renderer Vendor: %s", "Apple");
        ImGui::Text("Renderer Name: %s", "Metal");

        auto& backend = Backend::State<Device::Metal>();
        ImGui::Text("Device Name: %s", backend->getDeviceName().c_str());
        ImGui::Text("Low Power Mode: %s", backend->getLowPowerStatus() ? "YES" : "NO");
        ImGui::Text("Has Unified Memory: %s", backend->hasUnifiedMemory() ? "YES" : "NO");
        ImGui::Text("Physical Memory: %.00f GB", (float)backend->physicalMemory() / 1e9);
        ImGui::Text("Thermal State: %llu/3", backend->getThermalState());
        ImGui::Text("Processor Count: %llu/%llu", backend->getActiveProcessorCount(),
                                                  backend->getTotalProcessorCount());

        ImGui::End();
#endif
    }

    return Result::SUCCESS;
}

const Result Implementation::end() {
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

const Window::Stats& Implementation::stats() const {
    return statsData;
}

}  // namespace Jetstream::Render
