#ifndef JETSTREAM_RENDER_BASE_PROGRAM_HH
#define JETSTREAM_RENDER_BASE_PROGRAM_HH

#include <utility>
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
    enum class Target : uint8_t {
        VERTEX   = 1 << 0,
        FRAGMENT = 1 << 1,
    };

    struct Config {
        std::shared_ptr<Draw> draw;
        std::vector<std::shared_ptr<Texture>> textures;
        std::vector<std::pair<std::shared_ptr<Buffer>, Target>> buffers;
        std::unordered_map<Device, std::vector<std::vector<U8>>> shaders;
    };

    explicit Program(const Config& config) : config(config) {}
    virtual ~Program() = default;

    template<Device D> 
    static std::shared_ptr<Program> Factory(const Config& config) {
        return std::make_shared<ProgramImp<D>>(config);
    }

 protected:
    Config config;

    uint32_t drawIndex = 0;
};

inline constexpr Program::Target operator|(Program::Target lhs, Program::Target rhs) {
    return static_cast<Program::Target>(static_cast<uint8_t>(lhs) | static_cast<uint8_t>(rhs));
}

inline constexpr Program::Target operator&(Program::Target lhs, Program::Target rhs) {
    return static_cast<Program::Target>(static_cast<uint8_t>(lhs) & static_cast<uint8_t>(rhs));
}

}  // namespace Jetstream::Render

#endif
