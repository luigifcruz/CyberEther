#include "opengl/opengl.hpp"

namespace Render {

BackendId OpenGL::getBackendId() {
    return BackendId::OPENGL;
}

Result OpenGL::init(Config cfg) {
    if (!glfwInit()) {
        return Result::FAILED_TO_OPEN_SCREEN;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_RESIZABLE, cfg.resizable);

    window = glfwCreateWindow(cfg.width, cfg.height, cfg.title.c_str(), NULL, NULL);
    if (!window) {
        glfwTerminate();
        return Result::FAILED_TO_OPEN_SCREEN;
    }

    glfwMakeContextCurrent(window);

    this->createSurface();
    this->createShaders(cfg.vertexSource, cfg.fragmentSource);

    if (cfg.enableImgui) {
        this->createImgui();
    }

    config = cfg;

    return Result::SUCCESS;
}

Result OpenGL::terminate() {
    this->destroyShaders();
    this->destroySurface();

    if (config.enableImgui) {
        this->destroyImgui();
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return Result::SUCCESS;
}

Result OpenGL::getError(std::string func, std::string file, int line) {
    int error = glGetError();
    if (error != GL_NO_ERROR) {
        std::cout << "[OPENGL] GL returned an error #" << error
                  << " inside function " << func << " @ "
                  << file << ":" << line << std::endl;
        return Result::RENDER_BACKEND_ERROR;
    }
    return Result::SUCCESS;
}

Result OpenGL::checkShaderCompilation(uint shader) {
    int success;
    char infoLog[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        std::cout << "[OPENGL] Shader #" << shader << " compilation error:\n"
                  << infoLog << std::endl;
        return Result::RENDER_BACKEND_ERROR;
    }
    return Result::SUCCESS;
}

Result OpenGL::checkProgramCompilation(uint program) {
    int success;
    char infoLog[512];
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(program, 512, NULL, infoLog);
        std::cout << "[OPENGL] Program #" << program << " compilation error:\n"
                  << infoLog << std::endl;
        return Result::RENDER_BACKEND_ERROR;
    }
    return Result::SUCCESS;
}

Result OpenGL::createSurface() {
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

    glBindVertexArray(0);

    return this->getError(__FUNCTION__, __FILE__, __LINE__);
}

Result OpenGL::destroySurface() {
    glDeleteBuffers(1, &vbo);
    glDeleteBuffers(1, &ebo);

    return this->getError(__FUNCTION__, __FILE__, __LINE__);
}

Result OpenGL::createShaders(const char *vertexSource, const char *fragmentSource) {
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexSource, NULL);
    glCompileShader(vertexShader);

    if (checkShaderCompilation(vertexShader) != Result::SUCCESS) {
        return Result::RENDER_BACKEND_ERROR;
    }

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
    glCompileShader(fragmentShader);

    if (checkShaderCompilation(fragmentShader) != Result::SUCCESS) {
        return Result::RENDER_BACKEND_ERROR;
    }

    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    if (checkProgramCompilation(shaderProgram) != Result::SUCCESS) {
        return Result::RENDER_BACKEND_ERROR;
    }

    glUseProgram(shaderProgram);

    return this->getError(__FUNCTION__, __FILE__, __LINE__);
}

Result OpenGL::destroyShaders() {
    glDeleteProgram(shaderProgram);

    return this->getError(__FUNCTION__, __FILE__, __LINE__);
}

Result OpenGL::createImgui() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    float xscale, yscale;
    GLFWmonitor *monitor = glfwGetPrimaryMonitor();
    glfwGetMonitorContentScale(monitor, &xscale, &yscale);

    ImGuiStyle &style = ImGui::GetStyle();
    style.ScaleAllSizes(xscale);
    io.Fonts->AddFontFromFileTTF("roboto.ttf", 12.0f * xscale, NULL, NULL);

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    return this->getError(__FUNCTION__, __FILE__, __LINE__);
}

Result OpenGL::destroyImgui() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    return this->getError(__FUNCTION__, __FILE__, __LINE__);
}

Result OpenGL::startImgui() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    return this->getError(__FUNCTION__, __FILE__, __LINE__);
}

Result OpenGL::endImgui() {
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    return this->getError(__FUNCTION__, __FILE__, __LINE__);
}

Result OpenGL::clear() {
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    if (config.enableImgui) {
        this->startImgui();
    }

    return this->getError(__FUNCTION__, __FILE__, __LINE__);
}

Result OpenGL::draw() {
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

    if (config.enableImgui) {
        this->endImgui();
    }

    return this->getError(__FUNCTION__, __FILE__, __LINE__);
}

Result OpenGL::step() {
    if (config.resizable) {
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
    }

    glfwSwapBuffers(window);
    glfwPollEvents();

    return this->getError(__FUNCTION__, __FILE__, __LINE__);
}

bool OpenGL::keepRunning() {
    return !glfwWindowShouldClose(window);
}

} // namespace Render
