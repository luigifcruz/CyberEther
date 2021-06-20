#ifndef RENDER_BASE_TEXTURE_H
#define RENDER_BASE_TEXTURE_H

#include "render/type.hpp"

namespace Render {

class Texture {
public:
    struct Config {
        std::string key;
        Size2D<int> size;
        uint8_t* buffer = nullptr;
        DataFormat dfmt = DataFormat::RGB;
        PixelFormat pfmt = PixelFormat::RGB;
        PixelType ptype = PixelType::UI8;
        bool cudaInterop = false;
    };

    Texture(const Config & c) : cfg(c) {};
    virtual ~Texture() = default;

    constexpr const Size2D<int> size() const {
        return cfg.size;
    }
    virtual Size2D<int> size(const Size2D<int> &) = 0;

    virtual uint raw() = 0;
    virtual Result pour() = 0;
    virtual Result fill() = 0;
    virtual Result fill(int, int, int, int) = 0;

protected:
    Config cfg;

    virtual Result create() = 0;
    virtual Result destroy() = 0;
    virtual Result start() = 0;
    virtual Result end() = 0;
};

} // namespace Render

#endif
