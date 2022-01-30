#include "render/gles/program.hpp"
#include "render/gles/texture.hpp"
#include "render/gles/draw.hpp"
#include "render/gles/buffer.hpp"

namespace Render {

GLES::Program::Program(const Config& config, const GLES& instance)
         : Render::Program(config), instance(instance) {
    for (const auto& draw : config.draws) {
        draws.push_back(std::dynamic_pointer_cast<GLES::Draw>(draw));
    }

    for (const auto& texture : config.textures) {
        textures.push_back(std::dynamic_pointer_cast<GLES::Texture>(texture));
    }

    for (const auto& buffer : config.buffers) {
        buffers.push_back(std::dynamic_pointer_cast<GLES::Buffer>(buffer));
    }
}

Result GLES::Program::create() {
    const auto& vertexShaderSrc = config.shaders[instance.getBackendId()][0];
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSrc, NULL);
    glCompileShader(vertexShader);

    CHECK(checkShaderCompilation(vertexShader));

    const auto& fragmentShaderSrc = config.shaders[instance.getBackendId()][1];
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSrc, NULL);
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

    for (const auto& draw : draws) {
        CHECK(draw->create());
    }

    i = 0;
    for (const auto& texture : textures) {
        CHECK(texture->create());
        CHECK(this->setUniform(texture->config.key, std::vector{i++}));
    }

    for (std::size_t i = 0; i < buffers.size(); i++) {
        CHECK(buffers[i]->create());
        CHECK(buffers[i]->begin());
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, *buffers[i]->getHandle());
        CHECK(buffers[i]->end());
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

    for (const auto& buffer : buffers) {
        CHECK(buffer->destroy());
    }

    glDeleteProgram(shader);

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Program::draw() {
    // Attach textures.
    i = 0;
    for (const auto& texture : textures) {
        glActiveTexture(GL_TEXTURE0 + i++);
        CHECK(texture->begin());
    }

    glUseProgram(shader);

    i = 0;
    for (const auto& draw : draws) {
        CHECK(this->setUniform("drawIndex", std::vector{i++}));
        CHECK(draw->draw());
    }

    i = 0;
    for (const auto& texture : textures) {
        glActiveTexture(GL_TEXTURE0 + i++);
        CHECK(texture->end());
    }

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Program::setUniform(const std::string& key,
        const std::variant<std::vector<float>, std::vector<uint32_t>>& data) {
    glUseProgram(shader);
    int loc = glGetUniformLocation(shader, key.c_str());

    std::visit(overloaded {
        [&](std::vector<float> data) {
            switch (data.size()) {
            case 1:
                glUniform1f(loc, data[0]);
                break;
            case 2:
                glUniform2f(loc, data[0], data[1]);
                break;
            case 3:
                glUniform3f(loc, data[0], data[1], data[2]);
                break;
            case 4:
                glUniform4f(loc, data[0], data[1], data[2], data[3]);
                break;
            default:
                std::cerr << "[RENDER] Number of uniforms invalid." << std::endl;
                throw Result::ERROR;
            }
        },
        [&](std::vector<uint32_t> data) {
            switch (data.size()) {
            case 1:
                glUniform1i(loc, data[0]);
                break;
            case 2:
                glUniform2i(loc, data[0], data[1]);
                break;
            case 3:
                glUniform3i(loc, data[0], data[1], data[2]);
                break;
            case 4:
                glUniform4i(loc, data[0], data[1], data[2], data[3]);
                break;
            default:
                std::cerr << "[RENDER] Number of uniforms invalid." << std::endl;
                throw Result::ERROR;
            }
        },
    }, data);

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
