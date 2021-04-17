#include "render/gles/instance.hpp"
#include "render/gles/program.hpp"
#include "render/gles/surface.hpp"
#include "render/gles/texture.hpp"

namespace Render {

Result GLES::init() {
    if (!glfwInit()) {
        return Result::FAILED_TO_OPEN_SCREEN;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_RESIZABLE, cfg.resizable);

    state = (State*)malloc(sizeof(State));
    state->window = glfwCreateWindow(cfg.width, cfg.height, cfg.title.c_str(), NULL, NULL);
    if (!state->window) {
        glfwTerminate();
        return Result::FAILED_TO_OPEN_SCREEN;
    }

    glfwMakeContextCurrent(state->window);

    ASSERT_SUCCESS(this->createBuffers());

    for (auto &program : programs) {
        ASSERT_SUCCESS(program->create());
    }

    if (cfg.enableImgui) {
        this->createImgui();
    }

    return Result::SUCCESS;
}

Result GLES::terminate() {
    for (auto &program : programs) {
        ASSERT_SUCCESS(program->destroy());
    }

    this->destroyBuffers();

    if (cfg.enableImgui) {
        this->destroyImgui();
    }

    glfwDestroyWindow(state->window);
    glfwTerminate();
    return Result::SUCCESS;
}

Result GLES::createBuffers() {
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glGenBuffers(1, &vbo);
    glGenBuffers(1, &ebo);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(elements), elements, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::destroyBuffers() {
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
    glDeleteBuffers(1, &ebo);

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::createImgui() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    io = &ImGui::GetIO();
    style = &ImGui::GetStyle();

    io->ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io->ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    float xscale = 1.0, yscale = 1.0;
#ifndef __EMSCRIPTEN__
    GLFWmonitor *monitor = glfwGetPrimaryMonitor();
    glfwGetMonitorContentScale(monitor, &xscale, &yscale);
    style->ScaleAllSizes(xscale);
#endif
    io->Fonts->AddFontFromFileTTF("roboto.ttf", 12.0f * xscale, NULL, NULL);

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(state->window, true);
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

Result GLES::clear() {
    if (cfg.enableImgui) {
        this->startImgui();
    }

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::draw() {
    for (auto &program : programs) {
        ASSERT_SUCCESS(program->draw());
    }

    if (cfg.enableImgui) {
        this->endImgui();
    }

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::step() {
    glfwGetFramebufferSize(state->window, &cfg.width, &cfg.height);
    glfwSwapBuffers(state->window);
    glfwPollEvents();

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

bool GLES::keepRunning() {
    return !glfwWindowShouldClose(state->window);
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

} // namespace Render
