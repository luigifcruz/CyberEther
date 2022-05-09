#ifndef JETSTREAM_RENDER_DRAW_HH
#define JETSTREAM_RENDER_DRAW_HH

#include <memory>

#include "jetstream/types.hh"
#include "jetstream/logger.hh"
#include "jetstream/render/vertex.hh"
#include "jetstream/render/types.hh"

namespace Jetstream::Render {

template<Device D> class DrawImp;

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
