#include "jetstream/render/metal/surface.hh"
#include "jetstream/render/metal/window.hh"

namespace Jetstream::Render {

using Implementation = WindowImp<Device::Metal>;

Implementation::WindowImp(const Config& config) : Window(config) {
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

    if (!glfwInit()) {
        return Result::ERROR;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_DOUBLEBUFFER, config.vsync);

    auto [width, height] = config.size;
    window = glfwCreateWindow(width, height, 
        config.title.c_str(), nullptr, nullptr);

    if (!window) {
        glfwTerminate();
        return Result::ERROR;
    }

    device = Backend::State<Device::Metal>()->getDevice();
    view = std::make_unique<View>(device, window);

    commandQueue = device->newCommandQueue();
    JST_ASSERT(commandQueue);

    renderPassDescriptor = MTL::RenderPassDescriptor::alloc()->init();
    JST_ASSERT(renderPassDescriptor);

    for (auto& surface : surfaces) {
        JST_CHECK(surface->create());
    }

    if (config.imgui) {
        JST_CHECK(createImgui());
    }

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

    glfwDestroyWindow(window);
    glfwTerminate();

    renderPassDescriptor->release();
    commandQueue->release();

    return Result::SUCCESS;
}

const Result Implementation::createImgui() {
    JST_DEBUG("Creating Metal ImGui.");

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    io = &ImGui::GetIO();
    style = &ImGui::GetStyle();

    io->ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io->ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    style->ScaleAllSizes(config.scale);
    io->Fonts->AddFontFromFileTTF("B612Mono-Regular.ttf",
        12.0f * config.scale, NULL, NULL);

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplMetal_Init(device);

    return Result::SUCCESS;
}

const Result Implementation::destroyImgui() {
    JST_DEBUG("Destroying Metal ImGui.");

    ImGui_ImplMetal_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    return Result::SUCCESS;
}

const Result Implementation::beginImgui() {
    ImGui_ImplMetal_NewFrame(renderPassDescriptor);
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    return Result::SUCCESS;
}

const Result Implementation::endImgui() {
    auto renderCmdEncoder = commandBuffer->renderCommandEncoder(renderPassDescriptor);

    ImGui::Render();
    ImGui_ImplMetal_RenderDrawData(ImGui::GetDrawData(),
        commandBuffer, renderCmdEncoder);

    renderCmdEncoder->endEncoding();
    renderCmdEncoder->release();

    return Result::SUCCESS;
}

const Result Implementation::begin() {
    commandBuffer = commandQueue->commandBuffer();
    drawable = view->draw();

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
            ImGui::Text("Renderer Name: %s", "Apple Metal");
            ImGui::Text("Renderer Vendor: %s", "Apple");
            ImGui::End();
#endif
    }

    return Result::SUCCESS;
}

const Result Implementation::end() {
    for (auto &surface : surfaces) {
        JST_CHECK(surface->draw(commandBuffer));
    }

    if (config.imgui) {
        JST_CHECK(endImgui());
    }

    glfwPollEvents();

    commandBuffer->presentDrawable(drawable);
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();

    commandBuffer->release();
    drawable->release();

    return Result::SUCCESS;
}

const Result Implementation::synchronize() {
    return Result::SUCCESS;
}

const bool Implementation::keepRunning() {
    return !glfwWindowShouldClose(window);
}

}  // namespace Jetstream::Render
