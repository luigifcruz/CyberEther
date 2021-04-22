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
        std::vector<std::shared_ptr<Vertex>> vertices;
        std::vector<std::shared_ptr<Texture>> textures;
    };

    Program(Config& c) : cfg(c) {};
    virtual ~Program() = default;

    virtual Result create() = 0;
    virtual Result destroy() = 0;
    virtual Result draw() = 0;

    Config& config() {
        return cfg;
    }

    virtual Result setUniform(std::string, const std::vector<int> &) = 0;
    virtual Result setUniform(std::string, const std::vector<float> &) = 0;

protected:
    Config& cfg;
};

} // namespace Render

#endif
