#include "gles/program.hpp"

namespace Render {

Result GLES::Program::checkShaderCompilation(uint shader) {
    int success;
    char infoLog[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
#ifdef RENDER_DEBUG
        std::cout << "[OPENGL] Shader #" << shader << " compilation error:\n"
                  << infoLog << std::endl;
#endif
        return Result::RENDER_BACKEND_ERROR;
    }
    return Result::SUCCESS;
}

Result GLES::Program::checkProgramCompilation(uint program) {
    int success;
    char infoLog[512];
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(program, 512, NULL, infoLog);
#ifdef RENDER_DEBUG
        std::cout << "[OPENGL] Program #" << program << " compilation error:\n"
                  << infoLog << std::endl;
#endif
        return Result::RENDER_BACKEND_ERROR;
    }
    return Result::SUCCESS;
}

Result GLES::Program::create() {
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, p.vertexSource, NULL);
    glCompileShader(vertexShader);

    if (checkShaderCompilation(vertexShader) != Result::SUCCESS) {
        return Result::RENDER_BACKEND_ERROR;
    }

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, p.fragmentSource, NULL);
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

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Program::destroy() {
    glDeleteProgram(shaderProgram);

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Program::draw() {
    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

} // namespace Render
