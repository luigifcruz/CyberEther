#ifndef JETSTREAM_RENDER_PROGRAM_HH
#define JETSTREAM_RENDER_PROGRAM_HH

#include <memory>
#include <vector>

#include "jetstream/types.hh"
#include "jetstream/logger.hh"
#include "jetstream/render/texture.hh"
#include "jetstream/render/buffer.hh"
#include "jetstream/render/draw.hh"
#include "jetstream/render/types.hh"

namespace Jetstream::Render {

template<Device D> class ProgramImp;

class Program {
 public:
    struct Config {
        std::vector<std::shared_ptr<Draw>> draws;
        std::vector<std::shared_ptr<Texture>> textures;
        std::vector<std::shared_ptr<Buffer>> buffers;
        std::map<Device, std::vector<const char*>> shaders;
    };

    explicit Program(const Config& config) : config(config) {
        JST_DEBUG("Program initialized.");
    }
    virtual ~Program() = default;

    template<Device D> 
    static std::shared_ptr<Program> Factory(const Config& config) {
        return std::make_shared<ProgramImp<D>>(config);
    }

 protected:
    Config config;

    uint32_t drawIndex = 0;
};

}  // namespace Jetstream::Render

#endif
