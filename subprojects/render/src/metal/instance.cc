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

Result Metal::create() {
    if (!glfwInit()) {
        return Result::FAILED_TO_OPEN_SCREEN;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_DOUBLEBUFFER, cfg.vsync);

    auto [width, height] = cfg.size;
    window = glfwCreateWindow(width, height, cfg.title.c_str(), nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        return Result::FAILED_TO_OPEN_SCREEN;
    }

    device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        return Result::RENDER_BACKEND_ERROR;
    }

    metalWindow = std::make_unique<MetalWindow>(device, window);

    glfwMakeContextCurrent(window);

    if (cfg.scale == -1.0) {
        // The macOS usually handles HiDPI well. Let's leave as is.
        cfg.scale = 1.0;
    }

    rendererString = "Apple Metal";
    versionString  = "2.0+";
    vendorString   = device->name()->utf8String();
    unifiedString  = device->hasUnifiedMemory() ? "YES" : "NO";
    shaderString   = "Metal Shading Language";

    commandQueue = device->newCommandQueue();
    assert(commandQueue);

    renderPassDesc = MTL::RenderPassDescriptor::alloc()->init();
    assert(renderPassDesc);

    for (auto &surface : surfaces) {
        CHECK(surface->create());
    }

    if (cfg.imgui) {
        CHECK(this->createImgui());
    }

    return Result::SUCCESS;
}

Result Metal::destroy() {
    for (auto &surface : surfaces) {
        CHECK(surface->destroy());
    }

    if (cfg.imgui) {
        CHECK(this->destroyImgui());
    }

    glfwDestroyWindow(window);
    glfwTerminate();

    return Result::SUCCESS;
}

Result Metal::createImgui() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    io = &ImGui::GetIO();
    style = &ImGui::GetStyle();

    io->ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io->ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    style->ScaleAllSizes(cfg.scale);

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplMetal_Init(device);

    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result Metal::destroyImgui() {
    ImGui_ImplMetal_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result Metal::beginImgui() {
    ImGui_ImplMetal_NewFrame(renderPassDesc);
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result Metal::endImgui() {
    auto renderCmdEncoder = commandBuffer->renderCommandEncoder(renderPassDesc);

    ImGui::Render();
    ImGui_ImplMetal_RenderDrawData(ImGui::GetDrawData(), commandBuffer, renderCmdEncoder);

    renderCmdEncoder->endEncoding();
    renderCmdEncoder->release();

    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result Metal::begin() {
    commandBuffer = commandQueue->commandBuffer();
    drawable = metalWindow->draw();

    auto colorAttachDesc = renderPassDesc->colorAttachments()->object(0);
    colorAttachDesc->setTexture(drawable->texture());
    colorAttachDesc->setLoadAction(MTL::LoadActionClear);
    colorAttachDesc->setStoreAction(MTL::StoreActionStore);
    colorAttachDesc->setClearColor(MTL::ClearColor(0, 0, 0, 0));

    if (cfg.imgui) {
        CHECK(this->beginImgui());

        if (cfg.debug) {
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

    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result Metal::end() {
    for (auto &surface : surfaces) {
        CHECK(surface->draw(commandBuffer));
    }

    if (cfg.imgui) {
        CHECK(this->endImgui());
    }

    glfwPollEvents();

    commandBuffer->presentDrawable(drawable);
    commandBuffer->commit();
    drawable->release();

    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result Metal::synchronize() {
    return Result::SUCCESS;
}

bool Metal::keepRunning() {
    return !glfwWindowShouldClose(window);
}

MTL::PixelFormat Metal::convertPixelFormat(const PixelFormat& pfmt, const PixelType& ptype) {
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

Result Metal::getError(std::string func, std::string file, int line) {
    // TODO: Implement this.
    /*
    int error = glGetError();
    if (error != GL_NO_ERROR) {
        std::cout << "[OPENGL] GL returned an error #" << error
                  << " inside function " << func << " @ "
                  << file << ":" << line << std::endl;
        return Result::RENDER_BACKEND_ERROR;
    }
    */
    return Result::SUCCESS;
}

std::shared_ptr<Render::Surface> Metal::createAndBind(const Render::Surface::Config& cfg) {
    auto surface = std::make_shared<Surface>(cfg, *this);
    surfaces.push_back(surface);
    return surface;
}

std::shared_ptr<Render::Program> Metal::create(const Render::Program::Config& cfg) {
    return std::make_shared<Program>(cfg, *this);
}

std::shared_ptr<Render::Texture> Metal::create(const Render::Texture::Config& cfg) {
    return std::make_shared<Texture>(cfg, *this);
}

std::shared_ptr<Render::Vertex> Metal::create(const Render::Vertex::Config& cfg) {
    return std::make_shared<Vertex>(cfg, *this);
}

std::shared_ptr<Render::Draw> Metal::create(const Render::Draw::Config& cfg) {
    return std::make_shared<Draw>(cfg, *this);
}

} // namespace Render
