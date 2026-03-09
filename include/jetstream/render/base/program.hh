#ifndef JETSTREAM_RENDER_BASE_PROGRAM_HH
#define JETSTREAM_RENDER_BASE_PROGRAM_HH

#include <optional>
#include <utility>
#include <memory>
#include <unordered_map>
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
    enum class Target : U8 {
        VERTEX   = 1 << 0,
        FRAGMENT = 1 << 1,
    };

    struct Config {
        std::vector<std::shared_ptr<Draw>> draws;
        std::vector<std::shared_ptr<Texture>> textures;
        std::vector<std::pair<std::shared_ptr<Buffer>, Target>> buffers;
        std::unordered_map<DeviceType, std::vector<std::vector<U8>>> shaders;
        bool enableAlphaBlending = false;
        std::optional<ScissorRect> scissorRect;
    };

    explicit Program(const Config& config) : config(config) {}
    virtual ~Program() = default;

    const Config& getConfig() const {
        return config;
    }

    void scissorRect(const std::optional<ScissorRect>& rect) {
        config.scissorRect = rect;
    }

    template<DeviceType D>
    static std::shared_ptr<Program> Factory(const Config& config) {
        return std::make_shared<ProgramImp<D>>(config);
    }

 protected:
    Config config;

    uint32_t drawIndex = 0;
};

inline constexpr Program::Target operator|(Program::Target lhs, Program::Target rhs) {
    return static_cast<Program::Target>(static_cast<U8>(lhs) | static_cast<U8>(rhs));
}

inline constexpr Program::Target operator&(Program::Target lhs, Program::Target rhs) {
    return static_cast<Program::Target>(static_cast<U8>(lhs) & static_cast<U8>(rhs));
}

}  // namespace Jetstream::Render

#endif
