#ifndef RENDER_BASE_DRAW_H
#define RENDER_BASE_DRAW_H

#include "render/types.hpp"
#include "vertex.hpp"

namespace Render {

class Draw {
public:
    enum Mode {
        TriangleFan,
        Triangles,
        LineLoop,
        Points,
        Lines,
    };

    struct Config {
        std::shared_ptr<Vertex> buffer;
        Mode mode = Triangles;
    };

    Config& cfg;
    Draw(Config& c) : cfg(c) {};
    virtual ~Draw() = default;

protected:
    virtual Result create() = 0;
    virtual Result destroy() = 0;
    virtual Result draw() = 0;
};

} // namespace Render

#endif
