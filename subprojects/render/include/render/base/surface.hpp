#ifndef RENDER_BASE_SURFACE_H
#define RENDER_BASE_SURFACE_H

#include "render/types.hpp"
#include "texture.hpp"
#include "program.hpp"

namespace Render {

class Surface {
public:
    struct Config {
    };

    Config& cfg;
    Surface(Config& c) : cfg(c) {};
    virtual ~Surface() = default;

    virtual std::shared_ptr<Render::Texture> bind(Render::Texture::Config&) = 0;
    virtual std::shared_ptr<Render::Program> bind(Render::Program::Config&) = 0;

protected:
    virtual Result create() = 0;
    virtual Result destroy() = 0;
    virtual Result draw() = 0;
};

} // namespace Render

#endif
