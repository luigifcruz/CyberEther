#include "render/metal/program.hpp"
#include "render/metal/texture.hpp"
#include "render/metal/draw.hpp"

namespace Render {

Result Metal::Program::create() {
    /*
    for (const auto& draw : cfg.draws) {
        draws.push_back(std::dynamic_pointer_cast<Metal::Draw>(draw));
    }

    for (const auto& texture : cfg.textures) {
        textures.push_back(std::dynamic_pointer_cast<Metal::Texture>(texture));
    }

    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, cfg.vertexSource, NULL);
    glCompileShader(vertexShader);

    CHECK(checkShaderCompilation(vertexShader));

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, cfg.fragmentSource, NULL);
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
        CHECK(this->setUniform(texture->cfg.key,
                    std::vector<int>{i++}));
    }

    for (const auto& draw : draws) {
        CHECK(draw->create());
    }
    */

    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result Metal::Program::destroy() {
    for (const auto& draw : draws) {
        CHECK(draw->destroy());
    }

    for (const auto& texture : textures) {
        CHECK(texture->destroy());
    }

    //glDeleteProgram(shader);

    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result Metal::Program::draw() {
    /*
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
    */

    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result Metal::Program::setUniform(std::string name, const std::vector<int>& vars) {
    /*
    // optimize: this can be cached
    // optimize: are std::vector performant?
    glUseProgram(shader);
    int loc = glGetUniformLocation(shader, name.c_str());

    switch(vars.size()) {
        case 1: glUniform1i(loc, vars.at(0)); break; case 2:
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
    */

    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result Metal::Program::setUniform(std::string name, const std::vector<float>& vars) {
    /*
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
    */

    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result Metal::Program::checkShaderCompilation(uint shader) {
    /*
    int success;
    char infoLog[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        std::cout << "[OPENGL] Shader #" << shader << " compilation error:\n"
                  << infoLog << std::endl;
        return Result::RENDER_BACKEND_ERROR;
    }
    */

    return Result::SUCCESS;
}

Result Metal::Program::checkProgramCompilation(uint program) {
    /*
    int success;
    char infoLog[512];
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(program, 512, NULL, infoLog);
        std::cout << "[OPENGL] Program #" << program << " compilation error:\n"
                  << infoLog << std::endl;
        return Result::RENDER_BACKEND_ERROR;
    }
    */

    return Result::SUCCESS;
}

} // namespace Render
