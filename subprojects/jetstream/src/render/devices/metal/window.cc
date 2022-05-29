#include "jetstream/render/metal/window.hh"

namespace Jetstream::Render {

using Metal = WindowImp<Device::Metal>;

Metal::WindowImp(const Config& config) : Window(config) {
    JST_INFO("Greetings from the Metal thingy.");
}

const Result Metal::bind(const std::shared_ptr<Surface>& surface) {
    surfaces.push_back(
        std::dynamic_pointer_cast<SurfaceImp<Device::Metal>>(surface)
    );

    return Result::SUCCESS;
}

const Result Metal::create() {
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

const Result Metal::destroy() {
    JST_DEBUG("Destroying Metal window.");

    for (auto& surface : surfaces) {
        JST_CHECK(surface->destroy());
    }

    for (config.imgui) {
        JST_CHECK(destroyImgui());
    } 

    glfwDestroyWindow(window);
    glfwTerminate();

    renderPassDescriptor->release();
    commandQueue->release();
}

const Result Metal::createImgui() {
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

const Result Metal::destroyImgui() {
    JST_DEBUG("Destroying Metal ImGui.");

    ImGui_ImplMetal_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    return Result::SUCCESS;
}

const Result Metal::beginImgui() {
    JST_DEBUG("Begin Metal ImGui.");

    ImGui_ImplMetal_NewFrame(renderPassDesc);
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    return Result::SUCCESS;
}

const Result Metal::endImgui() {
    auto& renderCmdEncoder = commandBuffer->
        renderCommandEncoder(renderPassDescriptor);

    ImGui::Render();
    ImGui_ImplMetal_RenderDrawData(ImGui::GetDrawData(),
        commandBuffer, renderCmdEncoder);

    renderCmdEncoder->endEncoding();
    renderCmdEncoder->release();

    return Result::SUCCESS;
}

const Result Metal::begin() {
    commandBuffer = commandQueue->commandBuffer();
    drawable = view->draw();

    auto colorAttachDesc = renderPassDescriptor->colorAttachments()->object(0);
    colorAttachDescriptor->setTexture(drawable->texture());
    colorAttachDescriptor->setLoadAction(MTL::LoadActionClear);
    colorAttachDescriptor->setStoreAction(MTL::StoreActionStore);
    colorAttachDescriptor->setClearColor(MTL::ClearColor(0, 0, 0, 1.0));

    if (config.imgui) {
        JST_CHECK(beginImgui());

        if (config.debug) {
            ImGui::ShowMetricsWindow();
            ImGui::Begin("Render Info");
            ImGui::Text("Renderer Name: %s", "Apple Metal");
            ImGui::Text("Renderer Vendor: %s", "Apple");
            ImGui::End();
        }
    }

    return Result::SUCCESS;
}

const Result Metal::end() {
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

const Result Metal::synchronize() {
    return Result::SUCCESS;
}

const bool Metal::keepRunning() {
    return !glfwWindowShouldClose(window);
}

const MTL::PixelFormat Metal::convertPixelFormat(const PixelFormat& pfmt,
                                                 const PixelType& ptype) {
    if (pfmt == PixelFormat::RED && ptype == PixelType::F32) {
        return MTL::PixelFormatR32Float;
    }

    if (pfmt == PixelFormat::RED && ptype == PixelType::UI8) {
        return MTL::PixelFormatR8Unorm;
    }

    if (pfmt == PixelFormat::RGBA && ptype == PixelType::F32) {
        return MTL::PixelFormatRGBA32Float;
    }

    if (pfmt == PixelFormat::RGBA && ptype == PixelType::UI8) {
        return MTL::PixelFormatRGBA8Unorm;
    }

    throw Result::ERROR;
}

const std::size_t Metal::getPixelByteSize(const MTL::PixelFormat& pfmt) {
    switch (pfmt) {
        case MTL::PixelFormatR32Float:
            return 4;
        case MTL::PixelFormatR8Unorm:
            return 1;
        case MTL::PixelFormatRGBA32Float:
            return 16;
        case MTL::PixelFormatRGBA8Unorm:
            return 4;
        default:
            throw "pixel format not implemented yet";
    }
}

}  // namespace Jetstream::Render
