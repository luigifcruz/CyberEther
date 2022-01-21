#ifndef RENDER_BASE_VERTEX_H
#define RENDER_BASE_VERTEX_H

#include <vector>

#include "render/type.hpp"
#include "render/base/buffer.hpp"

namespace Render {

class Vertex {
 public:
    struct Config {
        std::vector<std::pair<std::shared_ptr<Buffer>, uint32_t>> buffers;
        std::shared_ptr<Buffer> indices;
    };

    explicit Vertex(const Config& config) : config(config) {}
    virtual ~Vertex() = default;

 protected:
    Config config;
};

}  // namespace Render

#endif
