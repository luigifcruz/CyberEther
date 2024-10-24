#ifndef JETSTREAM_RENDER_BASE_VERTEX_HH
#define JETSTREAM_RENDER_BASE_VERTEX_HH

#include <memory>
#include <vector>

#include "jetstream/types.hh"
#include "jetstream/logger.hh"
#include "jetstream/render/base/buffer.hh"
#include "jetstream/render/types.hh"
#include "jetstream/render/base/implementations.hh"

namespace Jetstream::Render {

class Vertex {
 public:
    struct Config {
        std::vector<std::pair<std::shared_ptr<Buffer>, U32>> vertices;
        std::vector<std::pair<std::shared_ptr<Buffer>, U32>> instances;
        std::shared_ptr<Buffer> indices;
    };

    explicit Vertex(const Config& config) : config(config) {}
    virtual ~Vertex() = default;

    const Config& getConfig() const {
        return config;
    }

    template<Device D>
    static std::shared_ptr<Vertex> Factory(const Config& config) {
        return std::make_shared<VertexImp<D>>(config);
    }

 protected:
    Config config;
};

}  // namespace Jetstream::Render

#endif
