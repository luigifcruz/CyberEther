#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include "render/metal/instance.hpp"
#include "render/metal/program.hpp"
#include "render/metal/surface.hpp"
#include "render/metal/texture.hpp"
#include "render/metal/vertex.hpp"
#include "render/metal/draw.hpp"

namespace Render {

Metal::Metal::Metal(const Config& config) : Render::Instance(config) {
}

Result Metal::create() {
    if (!glfwInit()) {
        return Result::FAILED_TO_OPEN_SCREEN;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_DOUBLEBUFFER, config.vsync);

    auto [width, height] = config.size;
    window = glfwCreateWindow(width, height, config.title.c_str(), nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        return Result::FAILED_TO_OPEN_SCREEN;
    }

    device = MTL::CreateSystemDefaultDevice();
    RENDER_ASSERT(device);

    metalWindow = std::make_unique<MetalWindow>(device, window);

    glfwMakeContextCurrent(window);

    if (config.scale == -1.0) {
        config.scale = 1.0;  // Let's leave it as is for macOS.
    }

    rendererString = "Apple Metal";
    versionString  = "2.0+";
    vendorString   = device->name()->utf8String();
    unifiedString  = device->hasUnifiedMemory() ? "YES" : "NO";
    shaderString   = "Metal Shading Language";

    commandQueue = device->newCommandQueue();
    RENDER_ASSERT(commandQueue);

    renderPassDesc = MTL::RenderPassDescriptor::alloc()->init();
    RENDER_ASSERT(renderPassDesc);

    for (auto &surface : surfaces) {
        CHECK(surface->create());
    }

    if (config.imgui) {
        CHECK(this->createImgui());
    }

    return Result::SUCCESS;
}

Result Metal::destroy() {
    for (auto &surface : surfaces) {
        CHECK(surface->destroy());
    }

    if (config.imgui) {
        CHECK(this->destroyImgui());
    }

    glfwDestroyWindow(window);
    glfwTerminate();

    renderPassDesc->release();
    commandQueue->release();
    device->release();

    return Result::SUCCESS;
}

Result Metal::createImgui() {
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

Result Metal::destroyImgui() {
    ImGui_ImplMetal_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    return Result::SUCCESS;
}

Result Metal::beginImgui() {
    ImGui_ImplMetal_NewFrame(renderPassDesc);
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    return Result::SUCCESS;
}

Result Metal::endImgui() {
    auto renderCmdEncoder = commandBuffer->renderCommandEncoder(renderPassDesc);

    ImGui::Render();
    ImGui_ImplMetal_RenderDrawData(ImGui::GetDrawData(),
        commandBuffer, renderCmdEncoder);

    renderCmdEncoder->endEncoding();
    renderCmdEncoder->release();

    return Result::SUCCESS;
}

Result Metal::begin() {
    commandBuffer = commandQueue->commandBuffer();
    drawable = metalWindow->draw();

    auto colorAttachDesc = renderPassDesc->colorAttachments()->object(0);
    colorAttachDesc->setTexture(drawable->texture());
    colorAttachDesc->setLoadAction(MTL::LoadActionClear);
    colorAttachDesc->setStoreAction(MTL::StoreActionStore);
    colorAttachDesc->setClearColor(MTL::ClearColor(0, 0, 0, 1.0));

    if (config.imgui) {
        CHECK(this->beginImgui());

        if (config.debug) {
            ImGui::ShowMetricsWindow();
            ImGui::Begin("Render Info");
            ImGui::Text("Renderer Name: %s", rendererString);
            ImGui::Text("Renderer Vendor: %s", vendorString);
            ImGui::Text("Renderer Version: %s", versionString);
            ImGui::Text("Unified Memory: %s", unifiedString);
            ImGui::Text("Shader Version: %s", shaderString);
            ImGui::End();
        }
    }

    return Result::SUCCESS;
}

Result Metal::end() {
    for (auto &surface : surfaces) {
        CHECK(surface->draw(commandBuffer));
    }

    if (config.imgui) {
        CHECK(this->endImgui());
    }

    glfwPollEvents();

    commandBuffer->presentDrawable(drawable);
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();

    commandBuffer->release();
    drawable->release();

    return Result::SUCCESS;
}

Result Metal::synchronize() {
    return Result::SUCCESS;
}

bool Metal::keepRunning() {
    return !glfwWindowShouldClose(window);
}

MTL::PixelFormat Metal::convertPixelFormat(const PixelFormat& pfmt,
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

std::size_t Metal::getPixelByteSize(const MTL::PixelFormat& pfmt) {
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

}  // namespace Render
