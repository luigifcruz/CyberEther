#ifndef JETSTREAM_RENDER_VERTEX_HH
#define JETSTREAM_RENDER_VERTEX_HH

#include <memory>
#include <vector>

#include "jetstream/types.hh"
#include "jetstream/logger.hh"
#include "jetstream/render/buffer.hh"
#include "jetstream/render/types.hh"

namespace Jetstream::Render {

template<Device D> class VertexImp;

class Vertex {
 public:
    struct Config {
        std::vector<std::pair<std::shared_ptr<Buffer>, U32>> buffers;
        std::shared_ptr<Buffer> indices;
    };

    explicit Vertex(const Config& config) : config(config) {
        JST_DEBUG("Vertex initialized.");
    }
    virtual ~Vertex() = default;

    template<Device D> 
    static std::shared_ptr<Vertex> Factory(const Config& config) {
        return std::make_shared<VertexImp<D>>(config);
    }

 protected:
    Config config;
};

}  // namespace Jetstream::Render

#endif
