#ifndef RENDER_BASE_VERTEX_H
#define RENDER_BASE_VERTEX_H

#include "render/types.hpp"

namespace Render {

class Vertex {
public:
    struct Config {
        float *vertices;
        float *elements;
        int a;
        int b;
    };

    Vertex(Config& c) : cfg(c) {};
    virtual ~Vertex() = default;

    virtual Result create() = 0;
    virtual Result destroy() = 0;
    virtual Result start() = 0;
    virtual Result end() = 0;

protected:
    Config& cfg;
};

} // namespace Render

#endif

