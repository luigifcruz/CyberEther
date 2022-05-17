#ifndef JETSTREAM_RENDER_BASE_TEXTURE_HH
#define JETSTREAM_RENDER_BASE_TEXTURE_HH

#include <memory>

#include "jetstream/types.hh"
#include "jetstream/logger.hh"
#include "jetstream/render/base/buffer.hh"
#include "jetstream/render/types.hh"

namespace Jetstream::Render {

template<Device D> class TextureImp;

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
        std::string key;
        Size2D<U64> size;
        std::shared_ptr<Buffer> buffer;
        DataFormat dfmt = DataFormat::RGBA;
        PixelFormat pfmt = PixelFormat::RGBA;
        PixelType ptype = PixelType::UI8;
    };

    explicit Texture(const Config& config) : config(config) {
        JST_DEBUG("Texture initialized.");
    }
    virtual ~Texture() = default;

    constexpr const Size2D<U64> size() const {
        return config.size;
    }
    virtual const Size2D<U64> size(const Size2D<U64>&) = 0;

    virtual const void* raw() = 0;
    virtual const Result fill() = 0;
    virtual const Result fillRow(const U64& y, const U64& height) = 0;

    template<Device D> 
    static std::shared_ptr<Texture> Factory(const Config& config) {
        return std::make_shared<TextureImp<D>>(config);
    }

 protected:
    Config config;
};

}  // namespace Jetstream::Render

#endif
