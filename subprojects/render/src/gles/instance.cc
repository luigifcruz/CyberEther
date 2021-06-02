#include "render/gles/instance.hpp"
#include "render/gles/program.hpp"
#include "render/gles/surface.hpp"
#include "render/gles/texture.hpp"
#include "render/gles/vertex.hpp"
#include "render/gles/draw.hpp"

namespace Render {

Result GLES::create() {
    if (!glfwInit()) {
        return Result::FAILED_TO_OPEN_SCREEN;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_RESIZABLE, cfg.resizable);
    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_DOUBLEBUFFER, cfg.enableVsync);

    window = glfwCreateWindow(cfg.width, cfg.height, cfg.title.c_str(), NULL, NULL);
    if (!window) {
        glfwTerminate();
        return Result::FAILED_TO_OPEN_SCREEN;
    }

    glfwMakeContextCurrent(window);

    if (cfg.scale == -1.0) {
#ifndef __EMSCRIPTEN__
        GLFWmonitor *monitor = glfwGetPrimaryMonitor();
        glfwGetMonitorContentScale(monitor, &cfg.scale, nullptr);
#else
        cfg.scale = 1.0;
#endif
    }

    cached_renderer_str = (const char*)glGetString(GL_RENDERER);
    cached_version_str = (const char*)glGetString(GL_SHADING_LANGUAGE_VERSION);
    cached_vendor_str = (const char*)glGetString(GL_VENDOR);
    cached_glsl_str = (const char*)glGetString(GL_VERSION);

    for (auto &surface : surfaces) {
        RENDER_ASSERT_SUCCESS(surface->create());
    }

    if (cfg.enableImgui) {
        this->createImgui();
    }

    return Result::SUCCESS;
}

Result GLES::destroy() {
    for (auto &surface : surfaces) {
        RENDER_ASSERT_SUCCESS(surface->destroy());
    }

    if (cfg.enableImgui) {
        this->destroyImgui();
    }

    glfwDestroyWindow(window);
    glfwTerminate();

    return Result::SUCCESS;
}

Result GLES::createImgui() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    io = &ImGui::GetIO();
    style = &ImGui::GetStyle();

    io->ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io->ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    style->ScaleAllSizes(cfg.scale);
    io->Fonts->AddFontFromFileTTF("roboto.ttf", 12.0f * cfg.scale, NULL, NULL);

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(nullptr);

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::destroyImgui() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::startImgui() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::endImgui() {
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::start() {
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glLineWidth(2.0 * cfg.scale);

    if (cfg.enableImgui) {
        this->startImgui();

        if (cfg.enableDebug) {
            ImGui::ShowMetricsWindow();
            ImGui::Begin("Render Info");
            ImGui::Text("Renderer Name: %s", this->renderer_str().c_str());
            ImGui::Text("Renderer Vendor: %s", this->vendor_str().c_str());
            ImGui::Text("Renderer Version: %s", this->version_str().c_str());
            ImGui::Text("Renderer GLSL Version: %s", this->glsl_str().c_str());
            ImGui::End();
        }
    }

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::end() {
    for (auto &surface : surfaces) {
        RENDER_ASSERT_SUCCESS(surface->draw());
    }

    if (cfg.enableImgui) {
        this->endImgui();
    }

    glfwGetFramebufferSize(window, &cfg.width, &cfg.height);
    glfwSwapBuffers(window);
    glfwPollEvents();

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

bool GLES::keepRunning() {
    return !glfwWindowShouldClose(window);
}

std::string GLES::renderer_str() {
    return cached_renderer_str;
}

std::string GLES::version_str() {
    return cached_version_str;
}

std::string GLES::glsl_str() {
    return cached_glsl_str;
}

std::string GLES::vendor_str() {
    return cached_vendor_str;
}

GLenum GLES::getPixelFormat(PixelFormat pfmt) {
    switch (pfmt) {
        case PixelFormat::RGB:
            return GL_RGB;
        case PixelFormat::RED:
            return GL_RED;
        case PixelFormat::UINT8:
            return GL_R8;
    }
}

Result GLES::getError(std::string func, std::string file, int line) {
    int error = glGetError();
    if (error != GL_NO_ERROR) {
        std::cout << "[OPENGL] GL returned an error #" << error
                  << " inside function " << func << " @ "
                  << file << ":" << line << std::endl;
        return Result::RENDER_BACKEND_ERROR;
    }
    return Result::SUCCESS;
}

std::shared_ptr<Render::Surface> GLES::createAndBind(Render::Surface::Config& cfg) {
    auto surface = std::make_shared<Surface>(cfg, *this);
    surfaces.push_back(surface);
    return surface;
}

Result GLES::unbind(std::shared_ptr<Render::Surface> surface) {
    if (std::remove(surfaces.begin(), surfaces.end(), surface) != surfaces.end()) {
        return Result::SUCCESS;
    }
    return Result::ERROR;
}

std::shared_ptr<Render::Program> GLES::create(Render::Program::Config& cfg) {
    return std::make_shared<Program>(cfg, *this);
}

std::shared_ptr<Render::Texture> GLES::create(Render::Texture::Config& cfg) {
    return std::make_shared<Texture>(cfg, *this);
}

std::shared_ptr<Render::Vertex> GLES::create(Render::Vertex::Config& cfg) {
    return std::make_shared<Vertex>(cfg, *this);
}

std::shared_ptr<Render::Draw> GLES::create(Render::Draw::Config& cfg) {
    return std::make_shared<Draw>(cfg, *this);
}

} // namespace Render
