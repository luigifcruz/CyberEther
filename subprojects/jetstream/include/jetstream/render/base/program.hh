#ifndef JETSTREAM_RENDER_BASE_PROGRAM_HH
#define JETSTREAM_RENDER_BASE_PROGRAM_HH

#include <memory>
#include <vector>

#include "jetstream/types.hh"
#include "jetstream/logger.hh"
#include "jetstream/render/base/texture.hh"
#include "jetstream/render/base/buffer.hh"
#include "jetstream/render/base/draw.hh"
#include "jetstream/render/types.hh"
#include "jetstream/render/base/implementations.hh"

namespace Jetstream::Render {

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
