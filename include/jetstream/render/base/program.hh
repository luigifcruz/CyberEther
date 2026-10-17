#ifndef JETSTREAM_RENDER_BASE_PROGRAM_HH
#define JETSTREAM_RENDER_BASE_PROGRAM_HH

#include <atomic>
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

class JETSTREAM_API Program {
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
        bool eraseBlending = false;
        bool borrowedTextures = false;
        std::optional<ScissorRect> scissorRect;
    };

    explicit Program(const Config& config) : config(config), boundAllocations(textureAllocations()) {}
    virtual ~Program() = default;

    const Config& getConfig() const {
        return config;
    }

    bool enabled() const {
        return enabledState;
    }

    void setEnabled(bool enabled) {
        enabledState = enabled;
    }

    void scissorRect(const std::optional<ScissorRect>& rect) {
        config.scissorRect = rect;
    }

    virtual Result rebindTextures() {
        return Result::SUCCESS;
    }

    Result refreshTextureBindings();
    std::vector<U64> textureAllocations() const;

    template<DeviceType D>
    static std::shared_ptr<Program> Factory(const Config& config) {
        return std::make_shared<ProgramImp<D>>(config);
    }

 protected:
    Config config;

    uint32_t drawIndex = 0;

 private:
    std::atomic_bool surfaceOwned = false;
    bool enabledState = true;
    std::vector<U64> boundAllocations;

    friend class Surface;
};

inline std::vector<U64> Program::textureAllocations() const {
    std::vector<U64> allocations;
    allocations.reserve(config.textures.size());
    for (const auto& texture : config.textures) {
        allocations.push_back(texture ? texture->allocation() : 0);
    }
    return allocations;
}

inline Result Program::refreshTextureBindings() {
    for (const auto& texture : config.textures) {
        if (!texture || !texture->allocated()) {
            return Result::ERROR;
        }
    }
    std::vector<U64> current = textureAllocations();
    if (current == boundAllocations) {
        return Result::SUCCESS;
    }
    JST_CHECK(rebindTextures());
    boundAllocations = std::move(current);
    return Result::SUCCESS;
}

inline constexpr Program::Target operator|(Program::Target lhs, Program::Target rhs) {
    return static_cast<Program::Target>(static_cast<U8>(lhs) | static_cast<U8>(rhs));
}

inline constexpr Program::Target operator&(Program::Target lhs, Program::Target rhs) {
    return static_cast<Program::Target>(static_cast<U8>(lhs) & static_cast<U8>(rhs));
}

}  // namespace Jetstream::Render

#endif
