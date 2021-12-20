#ifndef RENDER_BASE_PROGRAM_H
#define RENDER_BASE_PROGRAM_H

#include "render/type.hpp"
#include "render/base/texture.hpp"
#include "render/base/vertex.hpp"
#include "render/base/draw.hpp"

namespace Render {

class Program {
public:
    struct Config {
        const char* const* vertexSource = nullptr;
        const char* const* fragmentSource = nullptr;

        const void* fragmentUniforms = nullptr;
        std::size_t fragmentUniformsSize;

        const void* vertexUniforms = nullptr;
        std::size_t vertexUniformsSize;

        std::vector<std::shared_ptr<Draw>> draws;
        std::vector<std::shared_ptr<Texture>> textures;
    };

    Program(const Config& c) : cfg(c) {};
    virtual ~Program() = default;

protected:
    Config cfg;

    struct {
        uint32_t drawIndex;
    } renderUniforms;
};

} // namespace Render

#endif
