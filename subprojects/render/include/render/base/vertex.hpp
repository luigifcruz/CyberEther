#ifndef RENDER_BASE_VERTEX_H
#define RENDER_BASE_VERTEX_H

#include "render/types.hpp"

namespace Render {

class Vertex {
public:
    struct Buffer {
        enum Usage {
            Dynamic,
            Static,
            Stream,
        };

        float* data;
        size_t size = 0;
        size_t stride = 0;
        Usage usage = Static;
        uint index;
    };

    struct Config {
        std::vector<Buffer> buffers;
        std::vector<uint> indices;
    };

    Config& cfg;
    Vertex(Config& c) : cfg(c) {};
    virtual ~Vertex() = default;

    virtual Result update() = 0;

protected:
    virtual Result create() = 0;
    virtual Result destroy() = 0;
    virtual Result start() = 0;
    virtual Result end() = 0;
};

} // namespace Render

#endif
