#ifndef JETSTREAM_RENDER_BASE_TEXTURE_HH
#define JETSTREAM_RENDER_BASE_TEXTURE_HH

#include <memory>

#include "jetstream/types.hh"
#include "jetstream/logger.hh"
#include "jetstream/render/base/buffer.hh"
#include "jetstream/render/types.hh"
#include "jetstream/render/base/implementations.hh"

namespace Jetstream::Render {

class Texture {
 public:
    enum class PixelFormat : U64 {
        RGBA,
        RED,
    };

    enum class DataFormat : U64 {
        RGBA,
        UI8,
        F32,
    };

    enum class PixelType : U64 {
        UI8,
        F32,
    };

    struct Config {
        Extent2D<U64> size;
        const uint8_t* buffer = nullptr;
        DataFormat dfmt = DataFormat::RGBA;
        PixelFormat pfmt = PixelFormat::RGBA;
        PixelType ptype = PixelType::UI8;
        bool multisampled = false;
    };

    explicit Texture(const Config& config) : config(config) {}
    virtual ~Texture() = default;

    const Config& getConfig() const {
        return config;
    }

    virtual Result create() = 0;
    virtual Result destroy() = 0;

    constexpr const bool& multisampled() const {
        return config.multisampled;
    }

    constexpr const Extent2D<U64>& size() const {
        return config.size;
    }
    virtual bool size(const Extent2D<U64>& size) = 0;

    virtual void* raw() = 0;
    virtual Result fill() = 0;
    virtual Result fillRow(const U64& y, const U64& height) = 0;

    template<Device D> 
    static std::shared_ptr<Texture> Factory(const Config& config) {
        return std::make_shared<TextureImp<D>>(config);
    }

 protected:
    Config config;
};

}  // namespace Jetstream::Render

#endif
