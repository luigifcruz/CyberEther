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

    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

    device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        return Result::RENDER_BACKEND_ERROR;
    }

    metalWindow = std::make_unique<MetalWindow>(device, window);

    glfwMakeContextCurrent(window);

    if (cfg.scale == -1.0) {
        GLFWmonitor *monitor = glfwGetPrimaryMonitor();
        glfwGetMonitorContentScale(monitor, &cfg.scale, nullptr);
    }

    cached_vendor_str = device->name()->utf8String();
    cached_renderer_str = "Apple Metal";
    cached_version_str = "2.0+";
    cached_glsl_str = "Metal Shading Language";

    const auto shaders = NS::String::string(R"""(
        #include <metal_stdlib>
        using namespace metal;
        vertex float4 vertFunc(
            const device packed_float3* vertexArray [[buffer(0)]],
            unsigned int vID[[vertex_id]])
        {
            return float4(vertexArray[vID], 1.0);
        }
        fragment half4 fragFunc()
        {
            return half4(1.0, 0.0, 0.0, 1.0);
        }
    )""", NS::ASCIIStringEncoding);

    NS::Error* err;
    MTL::CompileOptions* opts = MTL::CompileOptions::alloc();
    auto library = device->newLibrary(shaders, opts, &err);
    if (!library) {
        fmt::print("Library error:\n{}\n", err->description()->utf8String());
        return Result::ERROR;
    }

    MTL::Function* vertFunc = library->newFunction(NS::String::string("vertFunc", NS::ASCIIStringEncoding));
    assert(vertFunc);

    MTL::Function* fragFunc = library->newFunction(NS::String::string("fragFunc", NS::ASCIIStringEncoding));
    assert(fragFunc);

    auto renderPipelineDesc = MTL::RenderPipelineDescriptor::alloc()->init();
    assert(renderPipelineDesc);
    renderPipelineDesc->setVertexFunction(vertFunc);
    renderPipelineDesc->setFragmentFunction(fragFunc);
    renderPipelineDesc->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatBGRA8Unorm);

    auto renderPipelineState = device->newRenderPipelineState(renderPipelineDesc, &err);
    assert(renderPipelineState);

    const float vertexData[] = {
         0.0f,  1.0f, 0.0f,
        -1.0f, -1.0f, 0.0f,
         1.0f, -1.0f, 0.0f,
    };

    auto vertexBuffer = device->newBuffer(vertexData, sizeof(vertexData), MTL::ResourceOptionCPUCacheModeDefault);
    assert(vertexBuffer);

    auto cmdQueue = device->newCommandQueue();
    assert(cmdQueue);

    auto renderPassDesc = MTL::RenderPassDescriptor::alloc()->init();
    assert(renderPassDesc);

    this->createImgui();

    bool show_demo_window = true;

    while(!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        metalWindow->draw([&](CA::MetalDrawable* drawable) {
            auto cmdBuffer = cmdQueue->commandBuffer();
            assert(cmdBuffer);

            auto colorAttachDesc = renderPassDesc->colorAttachments()->object(0);
            colorAttachDesc->setTexture(drawable->texture());
            colorAttachDesc->setLoadAction(MTL::LoadActionClear);
            colorAttachDesc->setStoreAction(MTL::StoreActionStore);
            colorAttachDesc->setClearColor(MTL::ClearColor(0, 0, 0, 0));

            ImGui_ImplMetal_NewFrame(renderPassDesc);
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            ImGui::ShowDemoWindow(&show_demo_window);

            auto renderCmdEncoder = cmdBuffer->renderCommandEncoder(renderPassDesc);
            renderCmdEncoder->setRenderPipelineState(renderPipelineState);
            renderCmdEncoder->setVertexBuffer(vertexBuffer, 0, 0);
            renderCmdEncoder->drawPrimitives(MTL::PrimitiveTypeTriangle, (NS::UInteger)0, 3);
            renderCmdEncoder->endEncoding();

            ImGui::Render();
            ImGui_ImplMetal_RenderDrawData(ImGui::GetDrawData(), cmdBuffer, renderCmdEncoder);

            cmdBuffer->presentDrawable(drawable);
            cmdBuffer->commit();
        });
    }

    pool->release();

    return Result::SUCCESS;
}

Result Metal::destroy() {
    for (auto &surface : surfaces) {
        CHECK(surface->destroy());
    }

    if (cfg.imgui) {
        this->destroyImgui();
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
    //io->Fonts->AddFontFromFileTTF("JetBrainsMono-Regular.ttf", 12.0f * cfg.scale, NULL, NULL);

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
   // ImGui_ImplMetal_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result Metal::endImgui() {
    ImGui::Render();
    //ImGui_ImplMetal_RenderDrawData(ImGui::GetDrawData());

    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result Metal::begin() {
    /*
    glLineWidth(cfg.scale);

    if (cfg.imgui) {
        this->beginImgui();

        if (cfg.debug) {
            ImGui::ShowMetricsWindow();
            ImGui::Begin("Render Info");
            ImGui::Text("Renderer Name: %s", this->renderer_str().c_str());
            ImGui::Text("Renderer Vendor: %s", this->vendor_str().c_str());
            ImGui::Text("Renderer Version: %s", this->version_str().c_str());
            ImGui::Text("Renderer GLSL Version: %s", this->glsl_str().c_str());
            ImGui::End();
        }
    }
    */

    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result Metal::end() {
    /*
    for (auto &surface : surfaces) {
        CHECK(surface->draw());
    }

    if (cfg.imgui) {
        this->endImgui();
    }

    glfwGetFramebufferSize(window, &cfg.size.width, &cfg.size.height);
    glfwSwapBuffers(window);
    glfwPollEvents();
    */

    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result Metal::synchronize() {
    //glFinish();
    return Result::SUCCESS;
}

bool Metal::keepRunning() {
    return !glfwWindowShouldClose(window);
}

std::string Metal::renderer_str() {
    return cached_renderer_str;
}

std::string Metal::version_str() {
    return cached_version_str;
}

std::string Metal::glsl_str() {
    return cached_glsl_str;
}

std::string Metal::vendor_str() {
    return cached_vendor_str;
}

uint Metal::convertPixelFormat(PixelFormat pfmt) {
    /*
    switch (pfmt) {
        case PixelFormat::RGB:
            return GL_RGB;
        case PixelFormat::RED:
            return GL_RED;
    }
    */
    throw Result::ERROR;
}

uint Metal::convertPixelType(PixelType ptype) {
    /*
    switch (ptype) {
        case PixelType::UI8:
            return GL_UNSIGNED_BYTE;
        case PixelType::F32:
            return GL_FLOAT;
    }
    */
    throw Result::ERROR;
}

uint Metal::convertDataFormat(DataFormat dfmt) {
    /*
    switch (dfmt) {
        case DataFormat::UI8:
            return GL_R8;
        case DataFormat::RGB:
            return GL_RGB;
        case DataFormat::F32:
            return GL_R32F;
    }
    */
    throw Result::ERROR;
}

Result Metal::getError(std::string func, std::string file, int line) {
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
