#ifndef RENDER_BASE_DRAW_H
#define RENDER_BASE_DRAW_H

#include <memory>

#include "render/type.hpp"
#include "render/base/vertex.hpp"

namespace Render {

class Draw {
 public:
    enum Mode {
        TriangleFan,
        Triangles,
        LineStrip,
        Lines,
        Points,
    };

    struct Config {
        std::shared_ptr<Vertex> buffer;
        Mode mode = Triangles;
    };

    explicit Draw(const Config& config) : config(config) {}
    virtual ~Draw() = default;

 protected:
    Config config;
};

}  // namespace Render

#endif
