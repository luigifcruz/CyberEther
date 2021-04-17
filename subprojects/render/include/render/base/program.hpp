#ifndef RENDER_BASE_PROGRAM_H
#define RENDER_BASE_PROGRAM_H

#include "render/types.hpp"
#include "surface.hpp"

namespace Render {

typedef std::vector<std::tuple<std::string,
        std::shared_ptr<Texture>>> TexturePlan;

class Program {
public:
    struct Config {
        const char* const* vertexSource = nullptr;
        const char* const* fragmentSource = nullptr;
        std::shared_ptr<Surface> surface;
        TexturePlan textures;
    };

    Program(Config& c) : cfg(c) {};
    virtual ~Program() = default;

    virtual Result create() = 0;
    virtual Result destroy() = 0;

    virtual Result setUniform(std::string, const std::vector<int> &) = 0;
    virtual Result setUniform(std::string, const std::vector<float> &) = 0;

    virtual Result draw() = 0;

protected:
    Config& cfg;
};

} // namespace Render

#endif
