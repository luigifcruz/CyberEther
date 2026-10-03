#ifndef JETSTREAM_RENDER_BASE_DRAW_HH
#define JETSTREAM_RENDER_BASE_DRAW_HH

#include <algorithm>
#include <memory>
#include <vector>

#include "jetstream/types.hh"
#include "jetstream/logger.hh"
#include "jetstream/render/base/vertex.hh"
#include "jetstream/render/types.hh"
#include "jetstream/render/base/implementations.hh"

namespace Jetstream::Render {

class JETSTREAM_API Draw {
 public:
    enum class Mode : U64 {
        TRIANGLE_STRIP,
        TRIANGLES,
        LINE_STRIP,
        LINES,
        POINTS,
    };

    struct Config {
        Mode mode = Mode::TRIANGLES;
        U64 numberOfDraws = 1;
        U64 numberOfInstances = 1;
        std::shared_ptr<Vertex> buffer;
    };

    explicit Draw(const Config& config) : config(config), drawCount(config.numberOfDraws) {}
    virtual ~Draw() = default;

    const Config& getConfig() const {
        return config;
    }

    void setDrawCount(U64 count) {
        drawCount = std::min(count, config.numberOfDraws);
    }

    virtual Result updateVertexCount(U64 vertexCount) = 0;
    virtual Result updateInstanceCount(U64 instanceCount) = 0;

    template<DeviceType D>
    static std::shared_ptr<Draw> Factory(const Config& config) {
        return std::make_shared<DrawImp<D>>(config);
    }

 protected:
    Config config;
    U64 drawCount = 0;
    std::vector<std::shared_ptr<Buffer>> transferBuffers;

 private:
    friend class Surface;
};

}  // namespace Jetstream::Render

#endif
