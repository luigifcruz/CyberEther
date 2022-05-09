#ifndef JETSTREAM_RENDER_INSTANCE_HH
#define JETSTREAM_RENDER_INSTANCE_HH

#include <memory>

#include "jetstream/types.hh"
#include "jetstream/logger.hh"
#include "jetstream/render/surface.hh"
#include "jetstream/render/types.hh"

namespace Jetstream::Render {

template<Device D> class InstanceImp;

class Instance {
 public:
    struct Config {
        Size2D<U64> size = {1280, 720};
        std::string title = "Render";
        bool resizable = false;
        bool imgui = false;
        bool vsync = true;
        float scale = -1.0;
    };

    explicit Instance(const Config& config) : config(config) {
        JST_DEBUG("Instance initialized.");
    }
    virtual ~Instance() = default;

    virtual const Result create() = 0;
    virtual const Result destroy() = 0;
    virtual const Result begin() = 0;
    virtual const Result end() = 0;

    virtual const Result synchronize() = 0;
    virtual const bool keepRunning() = 0;

    template<Device D> 
    static std::shared_ptr<Instance> Factory(const Config& config) {
        return std::make_shared<InstanceImp<D>>(config);
    }

 protected:
    Config config;
};

}  // namespace Jetstream::Render

#endif
