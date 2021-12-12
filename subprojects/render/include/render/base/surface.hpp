#ifndef RENDER_BASE_SURFACE_H
#define RENDER_BASE_SURFACE_H

#include "render/type.hpp"
#include "render/base/texture.hpp"
#include "render/base/program.hpp"

namespace Render {

class Surface {
public:
    struct Config {
        std::shared_ptr<Texture> framebuffer;
        std::vector<std::shared_ptr<Program>> programs;
    };

    Surface(const Config& c) : cfg(c) {};
    virtual ~Surface() = default;

    const Size2D<int> size() const {
        if (cfg.framebuffer) {
            return cfg.framebuffer->size();
        }
        return {-1, -1};
    }
    virtual Size2D<int> size(const Size2D<int>&) = 0;

protected:
    Config cfg;

    virtual Result create() = 0;
    virtual Result destroy() = 0;
    virtual Result draw() = 0;
};

} // namespace Render

#endif
