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
        std::vector<std::shared_ptr<Draw>> draws;
        std::vector<std::shared_ptr<Texture>> textures;
    };

    Program(const Config& c) : cfg(c) {};
    virtual ~Program() = default;

    virtual Result setUniform(std::string, const std::vector<int>&) = 0;
    virtual Result setUniform(std::string, const std::vector<float>&) = 0;

protected:
    Config cfg;
};

} // namespace Render

#endif
