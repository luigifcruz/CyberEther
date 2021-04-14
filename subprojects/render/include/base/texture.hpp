#ifndef RENDER_BASE_TEXTURE_H
#define RENDER_BASE_TEXTURE_H

#include "types.hpp"

namespace Render {

class Texture {
public:
    struct Config {
        int width;
        int height;
    };

    Texture(Config& c) : cfg(c) {};
    virtual ~Texture() = default;

    virtual Result create() = 0;
    virtual Result destroy() = 0;
    virtual Result start() = 0;
    virtual Result end() = 0;

    virtual uint raw() = 0;
    virtual Result pour(const uint8_t*) = 0;
    virtual Result fill(const uint8_t*) = 0;

protected:
    Config& cfg;
};

} // namespace Render

#endif
