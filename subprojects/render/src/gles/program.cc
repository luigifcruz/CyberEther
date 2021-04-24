#include "render/gles/program.hpp"
#include "render/gles/texture.hpp"
#include "render/gles/vertex.hpp"

namespace Render {

Result GLES::Program::create() {
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, cfg.vertexSource, NULL);
    glCompileShader(vertexShader);

    ASSERT_SUCCESS(checkShaderCompilation(vertexShader));

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, cfg.fragmentSource, NULL);
    glCompileShader(fragmentShader);

    ASSERT_SUCCESS(checkShaderCompilation(fragmentShader));

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
        ASSERT_SUCCESS(texture->create());
        ASSERT_SUCCESS(this->setUniform(texture->cfg.key,
                    std::vector<int>{i++}));
    }

    for (const auto& vertex : vertices) {
        ASSERT_SUCCESS(vertex->create());
    }

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Program::destroy() {
    for (const auto& vertex : vertices) {
        ASSERT_SUCCESS(vertex->destroy());
    }

    for (const auto& texture : textures) {
        ASSERT_SUCCESS(texture->destroy());
    }

    glDeleteProgram(shader);

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Program::draw() {
    i = 0;
    for (const auto& texture : textures) {
        glActiveTexture(GL_TEXTURE0 + i++);
        ASSERT_SUCCESS(texture->start());
    }

    glUseProgram(shader);

    i = 0;
    for (const auto& vertex : vertices) {
        ASSERT_SUCCESS(this->setUniform("vertexIdx", std::vector<int>{i++}));
        ASSERT_SUCCESS(vertex->draw());
    }

    i = 0;
    for (const auto& texture : textures) {
        glActiveTexture(GL_TEXTURE0 + i++);
        ASSERT_SUCCESS(texture->end());
    }

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Program::setUniform(std::string name, const std::vector<int> & vars) {
    // optimize: this can be cached
    // optimize: are std::vector performant?
    glUseProgram(shader);
    int loc = glGetUniformLocation(shader, name.c_str());

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
    glUseProgram(shader);
    int loc = glGetUniformLocation(shader, name.c_str());

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

std::shared_ptr<Texture> GLES::Program::bind(Render::Texture::Config& cfg) {
    auto texture = std::make_shared<GLES::Texture>(cfg, inst);
    textures.push_back(texture);
    return texture;
}

std::shared_ptr<Vertex> GLES::Program::bind(Render::Vertex::Config& cfg) {
    auto vertex = std::make_shared<GLES::Vertex>(cfg, inst);
    vertices.push_back(vertex);
    return vertex;
}

} // namespace Render
