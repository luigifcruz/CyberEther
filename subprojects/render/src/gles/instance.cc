#include "gles/instance.hpp"

namespace Render {

Result GLES::Instance::init() {
    if (!glfwInit()) {
        return Result::FAILED_TO_OPEN_SCREEN;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_RESIZABLE, cfg.resizable);

    state.window = glfwCreateWindow(cfg.width, cfg.height, cfg.title.c_str(), NULL, NULL);
    if (!state.window) {
        glfwTerminate();
        return Result::FAILED_TO_OPEN_SCREEN;
    }

    glfwMakeContextCurrent(state.window);

    this->createBuffers();

    for (auto &program : programs) {
        ASSERT_SUCCESS(program->create());
    }

    if (cfg.enableImgui) {
        this->createImgui();
    }

    return Result::SUCCESS;
}

Result GLES::Instance::terminate() {
    for (auto &program : programs) {
        ASSERT_SUCCESS(program->destroy());
    }

    this->destroyBuffers();

    if (cfg.enableImgui) {
        this->destroyImgui();
    }

    glfwDestroyWindow(state.window);
    glfwTerminate();
    return Result::SUCCESS;
}

Result GLES::Instance::createBuffers() {
    glGenBuffers(1, &state.vbo);
    glGenBuffers(1, &state.ebo);

    glBindBuffer(GL_ARRAY_BUFFER, state.vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, state.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(elements), elements, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Instance::destroyBuffers() {
    glDeleteBuffers(1, &state.vbo);
    glDeleteBuffers(1, &state.ebo);

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Instance::createImgui() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    io = &ImGui::GetIO();

    io->ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io->ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    float xscale, yscale;
    GLFWmonitor *monitor = glfwGetPrimaryMonitor();
    glfwGetMonitorContentScale(monitor, &xscale, &yscale);

    ImGuiStyle &style = ImGui::GetStyle();
    style.ScaleAllSizes(xscale);
    io->Fonts->AddFontFromFileTTF("roboto.ttf", 12.0f * xscale, NULL, NULL);

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(state.window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Instance::destroyImgui() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Instance::startImgui() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Instance::endImgui() {
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Instance::clear() {
    if (cfg.enableImgui) {
        this->startImgui();
    }

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Instance::draw() {
    for (auto &program : programs) {
        ASSERT_SUCCESS(program->draw());
    }

    if (cfg.enableImgui) {
        this->endImgui();
    }

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Instance::step() {
    if (cfg.resizable) {
        glfwGetFramebufferSize(state.window, &cfg.width, &cfg.height);
        glViewport(0, 0, cfg.width, cfg.height);
    }

    glfwSwapBuffers(state.window);
    glfwPollEvents();

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

bool GLES::Instance::keepRunning() {
    return !glfwWindowShouldClose(state.window);
}

} // namespace Render
