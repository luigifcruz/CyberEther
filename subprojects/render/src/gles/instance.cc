#include "render/gles/instance.hpp"
#include "render/gles/program.hpp"
#include "render/gles/surface.hpp"
#include "render/gles/texture.hpp"
#include "render/gles/vertex.hpp"
#include "render/gles/draw.hpp"
#include "render/gles/buffer.hpp"

namespace Render {

GLES::GLES(const Config& config) : Render::Instance(config) {
}

Result GLES::create() {
    if (!glfwInit()) {
        return Result::FAILED_TO_OPEN_SCREEN;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_RESIZABLE, config.resizable);
    glfwWindowHint(GLFW_DOUBLEBUFFER, config.vsync);

    auto [width, height] = config.size;
    window = glfwCreateWindow(width, height, config.title.c_str(), NULL, NULL);
    if (!window) {
        glfwTerminate();
        return Result::FAILED_TO_OPEN_SCREEN;
    }

    glfwMakeContextCurrent(window);

    if (config.scale == -1.0) {
#if !defined(__EMSCRIPTEN__) && GLFW_VERSION_MINOR >= 3
        GLFWmonitor *monitor = glfwGetPrimaryMonitor();
        glfwGetMonitorContentScale(monitor, &config.scale, nullptr);
#else
        config.scale = 1.0;
#endif
    }

    rendererString  = reinterpret_cast<const char*>(glGetString(GL_RENDERER));
    versionString   = reinterpret_cast<const char*>(glGetString(
        GL_SHADING_LANGUAGE_VERSION));
    vendorString    = reinterpret_cast<const char*>(glGetString(GL_VENDOR));
    shaderString    = reinterpret_cast<const char*>(glGetString(GL_VERSION));

    for (auto &surface : surfaces) {
        CHECK(surface->create());
    }

    if (config.imgui) {
        this->createImgui();
    }

    return Result::SUCCESS;
}

Result GLES::destroy() {
    for (auto &surface : surfaces) {
        CHECK(surface->destroy());
    }

    if (config.imgui) {
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

    style->ScaleAllSizes(config.scale);
    io->Fonts->AddFontFromFileTTF("B612Mono-Regular.ttf",
        12.0f * config.scale, NULL, NULL);

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

Result GLES::beginImgui() {
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

Result GLES::begin() {
    glLineWidth(config.scale);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    if (config.imgui) {
        this->beginImgui();

        if (config.debug) {
            ImGui::ShowMetricsWindow();
            ImGui::Begin("Render Info");
            ImGui::Text("Renderer Name: %s", rendererString);
            ImGui::Text("Renderer Vendor: %s", vendorString);
            ImGui::Text("Renderer Version: %s", versionString);
            ImGui::Text("Shader Version: %s", shaderString);
            ImGui::End();
        }
    }

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::end() {
    for (auto &surface : surfaces) {
        CHECK(surface->draw());
    }

    if (config.imgui) {
        this->endImgui();
    }

    glfwGetFramebufferSize(window, &config.size.width, &config.size.height);
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

uint GLES::convertPixelFormat(PixelFormat pfmt) {
    switch (pfmt) {
        case PixelFormat::RGBA:
            return GL_RGBA;
        case PixelFormat::RED:
            return GL_RED;
    }
    throw Result::ERROR;
}

uint GLES::convertPixelType(PixelType ptype) {
    switch (ptype) {
        case PixelType::UI8:
            return GL_UNSIGNED_BYTE;
        case PixelType::F32:
            return GL_FLOAT;
    }
    throw Result::ERROR;
}

uint GLES::convertDataFormat(DataFormat dfmt) {
    switch (dfmt) {
        case DataFormat::UI8:
            return GL_R8;
        case DataFormat::RGBA:
            return GL_RGBA;
        case DataFormat::F32:
            return GL_R32F;
    }
    throw Result::ERROR;
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

}  // namespace Render
