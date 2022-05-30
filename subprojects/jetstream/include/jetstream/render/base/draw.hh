#ifndef JETSTREAM_RENDER_BASE_DRAW_HH
#define JETSTREAM_RENDER_BASE_DRAW_HH

#include <memory>

#include "jetstream/types.hh"
#include "jetstream/logger.hh"
#include "jetstream/render/base/vertex.hh"
#include "jetstream/render/types.hh"
#include "jetstream/render/base/implementations.hh"

namespace Jetstream::Render {

class Draw {
 public:
    enum class Mode : U64 {
        TRIANGLE_FAN,
        TRIANGLES, 
        LINE_STRIP,
        LINES,
        POINTS,
    };

    struct Config {
        Mode mode = Mode::TRIANGLES;
        std::shared_ptr<Vertex> buffer;
    };

    explicit Draw(const Config& config) : config(config) {
        JST_DEBUG("Draw initialized.");
    }
    virtual ~Draw() = default;

    template<Device D> 
    static std::shared_ptr<Draw> Factory(const Config& config) {
        return std::make_shared<DrawImp<D>>(config);
    }

 protected:
    Config config;
};

}  // namespace Jetstream::Render

#endif
