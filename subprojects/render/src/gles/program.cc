#include "gles/program.hpp"

namespace Render {

Result GLES::Program::checkShaderCompilation(uint shader) {
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

Result GLES::Program::checkProgramCompilation(uint program) {
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

Result GLES::Program::create() {
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, cfg.vertexSource, NULL);
    glCompileShader(vertexShader);

    ASSERT_SUCCESS(checkShaderCompilation(vertexShader));

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, cfg.fragmentSource, NULL);
    glCompileShader(fragmentShader);

    ASSERT_SUCCESS(checkShaderCompilation(fragmentShader))

    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    if (checkProgramCompilation(shaderProgram) != Result::SUCCESS) {
        return Result::RENDER_BACKEND_ERROR;
    }

    ASSERT_SUCCESS(cfg.surface->create());

    int i = 0;
    for (const auto& texture : cfg.textures) {
        ASSERT_SUCCESS(std::get<1>(texture)->fill());
        ASSERT_SUCCESS(this->setUniform(std::get<0>(texture),
                    std::vector<int>{i++}))
    }

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Program::setUniform(std::string name, const std::vector<int> & vars) {
    // optimize: this can be cached
    // optimize: are std::vector performant?
    glUseProgram(shaderProgram);
    int loc = glGetUniformLocation(shaderProgram, name.c_str());

    if (loc == 1) {
#ifdef RENDER_DEBUG
        std::cerr << "[RENDER:PROGRAM] Invalid program location." << std::endl;
#endif
        return Result::RENDER_BACKEND_ERROR;
    }

    switch(vars.size()) {
        case 1:
            glUniform1i(loc, vars.at(0));
            break;
        case 2:
            glUniform2i(loc, vars.at(0), vars.at(1));
            break;
        case 3:
            glUniform3i(loc, vars.at(0), vars.at(1), vars.at(2));
            break;
        case 4:
            glUniform4i(loc, vars.at(0), vars.at(1), vars.at(2), vars.at(3));
            break;
        default:
#ifdef RENDER_DEBUG
        std::cerr << "[RENDER:PROGRAM] Invalid number of uniforms (vars.size() > 4)." << std::endl;
#endif
            return Result::RENDER_BACKEND_ERROR;
    }

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Program::setUniform(std::string name, const std::vector<float> & vars) {
    // optimize: this can be cached
    // optimize: are std::vector performant?
    glUseProgram(shaderProgram);
    int loc = glGetUniformLocation(shaderProgram, name.c_str());

    if (loc == 1) {
#ifdef RENDER_DEBUG
        std::cerr << "[RENDER:PROGRAM] Invalid program location." << std::endl;
#endif
        return Result::RENDER_BACKEND_ERROR;
    }

    switch(vars.size()) {
        case 1:
            glUniform1f(loc, vars.at(0));
            break;
        case 2:
            glUniform2f(loc, vars.at(0), vars.at(1));
            break;
        case 3:
            glUniform3f(loc, vars.at(0), vars.at(1), vars.at(2));
            break;
        case 4:
            glUniform4f(loc, vars.at(0), vars.at(1), vars.at(2), vars.at(3));
            break;
        default:
#ifdef RENDER_DEBUG
        std::cerr << "[RENDER:PROGRAM] Invalid number of uniforms (vars.size() > 4)." << std::endl;
#endif
            return Result::RENDER_BACKEND_ERROR;
    }

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Program::destroy() {
    glDeleteProgram(shaderProgram);
    ASSERT_SUCCESS(cfg.surface->destroy());

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Program::draw() {
    ASSERT_SUCCESS(cfg.surface->start());

    int i = 0;
    for (const auto& texture : cfg.textures) {
        glActiveTexture(GL_TEXTURE0 + i++);
        ASSERT_SUCCESS(std::get<1>(texture)->start());
    }

    glUseProgram(shaderProgram);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

    i = 0;
    for (const auto& texture : cfg.textures) {
        glActiveTexture(GL_TEXTURE0 + i++);
        ASSERT_SUCCESS(std::get<1>(texture)->end());
    }

    ASSERT_SUCCESS(cfg.surface->end());

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

} // namespace Render
