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
    glfwWindowHint(GLFW_DOUBLEBUFFER, cfg.vsync);

    auto [width, height] = cfg.size;
    window = glfwCreateWindow(width, height, cfg.title.c_str(), NULL, NULL);
    if (!window) {
        glfwTerminate();
        return Result::FAILED_TO_OPEN_SCREEN;
    }

    glfwMakeContextCurrent(window);

    if (cfg.scale == -1.0) {
#if !defined(__EMSCRIPTEN__) && GLFW_VERSION_MINOR >= 3
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
        CHECK(surface->create());
    }

    if (cfg.imgui) {
        this->createImgui();
    }

    return Result::SUCCESS;
}

Result GLES::destroy() {
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
    glLineWidth(cfg.scale);

    if (cfg.imgui) {
        this->startImgui();

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

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::end() {
    for (auto &surface : surfaces) {
        CHECK(surface->draw());
    }

    if (cfg.imgui) {
        this->endImgui();
    }

    glfwGetFramebufferSize(window, &cfg.size.width, &cfg.size.height);
    glfwSwapBuffers(window);
    glfwPollEvents();

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::synchronize() {
    glFinish();
    return Result::SUCCESS;
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

uint GLES::convertPixelFormat(PixelFormat pfmt) {
    switch (pfmt) {
        case PixelFormat::RGB:
            return GL_RGB;
        case PixelFormat::RED:
            return GL_RED;
    }
}

uint GLES::convertPixelType(PixelType ptype) {
    switch (ptype) {
        case PixelType::UI8:
            return GL_UNSIGNED_BYTE;
        case PixelType::F32:
            return GL_FLOAT;
    }
}

uint GLES::convertDataFormat(DataFormat dfmt) {
    switch (dfmt) {
        case DataFormat::UI8:
            return GL_R8;
        case DataFormat::RGB:
            return GL_RGB;
        case DataFormat::F32:
            return GL_R32F;
    }
}

Result GLES::getError(std::string func, std::string file, int line) {
#ifdef RENDER_DEBUG
    int error = glGetError();
    if (error != GL_NO_ERROR) {
        std::cout << "[OPENGL] GL returned an error #" << error
                  << " inside function " << func << " @ "
                  << file << ":" << line << std::endl;
        return Result::RENDER_BACKEND_ERROR;
    }
#endif
    return Result::SUCCESS;
}

std::shared_ptr<Render::Surface> GLES::createAndBind(const Render::Surface::Config & cfg) {
    auto surface = std::make_shared<Surface>(cfg, *this);
    surfaces.push_back(surface);
    return surface;
}

std::shared_ptr<Render::Program> GLES::create(const Render::Program::Config & cfg) {
    return std::make_shared<Program>(cfg, *this);
}

std::shared_ptr<Render::Texture> GLES::create(const Render::Texture::Config & cfg) {
    return std::make_shared<Texture>(cfg, *this);
}

std::shared_ptr<Render::Vertex> GLES::create(const Render::Vertex::Config & cfg) {
    return std::make_shared<Vertex>(cfg, *this);
}

std::shared_ptr<Render::Draw> GLES::create(const Render::Draw::Config & cfg) {
    return std::make_shared<Draw>(cfg, *this);
}

} // namespace Render
