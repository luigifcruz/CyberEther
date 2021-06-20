#ifndef RENDER_BASE_DRAW_H
#define RENDER_BASE_DRAW_H

#include "render/type.hpp"
#include "render/base/vertex.hpp"

namespace Render {

class Draw {
public:
    enum Mode {
        TriangleFan,
        Triangles,
        LineStrip,
        LineLoop,
        Points,
        Lines,
    };

    struct Config {
        std::shared_ptr<Vertex> buffer;
        Mode mode = Triangles;
    };

    Draw(Config& c) : cfg(c) {};
    virtual ~Draw() = default;

protected:
    Config& cfg;

    virtual Result create() = 0;
    virtual Result destroy() = 0;
    virtual Result draw() = 0;
};

} // namespace Render

#endif
