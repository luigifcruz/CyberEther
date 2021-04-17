#ifndef RENDER_BASE_SURFACE_H
#define RENDER_BASE_SURFACE_H

#include "render/types.hpp"
#include "texture.hpp"

namespace Render {

class Surface {
public:
    struct Config {
        int* width;
        int* height;
        std::shared_ptr<Texture> texture;
    };

    Surface(Config& c) : cfg(c) {};
    virtual ~Surface() = default;

    virtual Result create() = 0;
    virtual Result destroy() = 0;
    virtual Result start() = 0;
    virtual Result end() = 0;

protected:
    Config& cfg;
};

} // namespace Render

#endif
