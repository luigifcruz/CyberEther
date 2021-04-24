#ifndef RENDER_BASE_PROGRAM_H
#define RENDER_BASE_PROGRAM_H

#include "render/types.hpp"
#include "texture.hpp"
#include "vertex.hpp"

namespace Render {

class Program {
public:
    struct Config {
        const char* const* vertexSource = nullptr;
        const char* const* fragmentSource = nullptr;
    };

    Config& cfg;
    Program(Config& c) : cfg(c) {};
    virtual ~Program() = default;

    virtual std::shared_ptr<Render::Texture> bind(Render::Texture::Config&) = 0;
    virtual std::shared_ptr<Render::Vertex> bind(Render::Vertex::Config&) = 0;

    virtual Result setUniform(std::string, const std::vector<int>&) = 0;
    virtual Result setUniform(std::string, const std::vector<float>&) = 0;

protected:
    virtual Result create() = 0;
    virtual Result destroy() = 0;
    virtual Result draw() = 0;
};

} // namespace Render

#endif
