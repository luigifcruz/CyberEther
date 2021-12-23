#include "render/gles/program.hpp"
#include "render/gles/texture.hpp"
#include "render/gles/draw.hpp"

namespace Render {

GLES::Program::Program(const Config& config, const GLES& instance)
         : Render::Program(config), instance(instance) {
    for (const auto& draw : config.draws) {
        draws.push_back(std::dynamic_pointer_cast<GLES::Draw>(draw));
    }

    for (const auto& texture : config.textures) {
        textures.push_back(std::dynamic_pointer_cast<GLES::Texture>(texture));
    }
}

Result GLES::Program::create() {
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, config.vertexSource, NULL);
    glCompileShader(vertexShader);

    CHECK(checkShaderCompilation(vertexShader));

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, config.fragmentSource, NULL);
    glCompileShader(fragmentShader);

    CHECK(checkShaderCompilation(fragmentShader));

    shader = glCreateProgram();
    glAttachShader(shader, vertexShader);
    glAttachShader(shader, fragmentShader);
    glLinkProgram(shader);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    if (checkProgramCompilation(shader) != Result::SUCCESS) {
        return Result::RENDER_BACKEND_ERROR;
    }

    i = 0;
    for (const auto& texture : textures) {
        CHECK(texture->create());
        CHECK(this->setUniform(texture->config.key,
                    std::vector<int>{i++}));
    }

    for (const auto& draw : draws) {
        CHECK(draw->create());
    }

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Program::destroy() {
    for (const auto& draw : draws) {
        CHECK(draw->destroy());
    }

    for (const auto& texture : textures) {
        CHECK(texture->destroy());
    }

    glDeleteProgram(shader);

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Program::draw() {
    i = 0;
    for (const auto& texture : textures) {
        glActiveTexture(GL_TEXTURE0 + i++);
        CHECK(texture->begin());
    }

    glUseProgram(shader);

    i = 0;
    for (const auto& draw : draws) {
        CHECK(this->setUniform("drawIndex", std::vector<int>{i++}));
        CHECK(draw->draw());
    }

    i = 0;
    for (const auto& texture : textures) {
        glActiveTexture(GL_TEXTURE0 + i++);
        CHECK(texture->end());
    }

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

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

}  // namespace Render
