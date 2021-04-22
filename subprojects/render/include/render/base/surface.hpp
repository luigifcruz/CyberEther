#ifndef RENDER_BASE_SURFACE_H
#define RENDER_BASE_SURFACE_H

#include "render/types.hpp"
#include "texture.hpp"
#include "program.hpp"

namespace Render {

class Surface {
public:
    struct Config {
        int* width;
        int* height;
        std::shared_ptr<Texture> texture;
        std::vector<std::shared_ptr<Program>> programs;
    };

    Surface(Config& c) : cfg(c) {};
    virtual ~Surface() = default;

    virtual Result create() = 0;
    virtual Result destroy() = 0;
    virtual Result draw() = 0;

    Config& config() {
        return cfg;
    }

protected:
    Config& cfg;
};

} // namespace Render

#endif
