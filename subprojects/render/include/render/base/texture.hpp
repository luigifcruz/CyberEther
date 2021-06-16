#ifndef RENDER_BASE_TEXTURE_H
#define RENDER_BASE_TEXTURE_H

#include "render/type.hpp"

namespace Render {

class Texture {
public:
    struct Config {
        int width = 0;
        int height = 0;
        std::string key;
        uint8_t* buffer = nullptr;
        DataFormat dfmt = DataFormat::RGB;
        PixelFormat pfmt = PixelFormat::RGB;
        PixelType ptype = PixelType::UI8;
        bool cudaInterop = false;
    };

    Config& cfg;
    Texture(Config& c) : cfg(c) {};
    virtual ~Texture() = default;

    virtual uint raw() = 0;
    virtual Result pour() = 0;
    virtual Result fill() = 0;
    virtual Result fill(int, int, int, int) = 0;

protected:
    virtual Result create() = 0;
    virtual Result destroy() = 0;
    virtual Result start() = 0;
    virtual Result end() = 0;
};

} // namespace Render

#endif
