#ifndef RENDER_BASE_VERTEX_H
#define RENDER_BASE_VERTEX_H

#include <vector>

#include "render/type.hpp"

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
        bool cudaInterop = false;
        uint index = 0;

#ifdef RENDER_CUDA_AVAILABLE
        cudaGraphicsResource* _cuda_res = nullptr;
#endif
    };

    struct Config {
        std::vector<Buffer> buffers;
        std::vector<uint16_t> indices;
    };

    explicit Vertex(const Config& config) : config(config) {}
    virtual ~Vertex() = default;

    virtual Result update() = 0;

 protected:
    Config config;
};

}  // namespace Render

#endif
