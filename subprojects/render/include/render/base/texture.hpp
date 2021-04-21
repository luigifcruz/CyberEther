#ifndef RENDER_BASE_TEXTURE_H
#define RENDER_BASE_TEXTURE_H

#include "render/types.hpp"

namespace Render {

class Texture {
public:
    struct Config {
        int width = 0;
        int height = 0;
        std::string key;
        uint8_t* buffer = nullptr;
    };

    Texture(Config& c) : cfg(c) {};
    virtual ~Texture() = default;

    virtual Result create() = 0;
    virtual Result destroy() = 0;
    virtual Result start() = 0;
    virtual Result end() = 0;

    Config& config() {
        return cfg;
    }

    virtual uint raw() = 0;
    virtual Result pour() = 0;
    virtual Result fill() = 0;
    virtual Result fill(int, int, int, int) = 0;

protected:
    Config& cfg;
};

} // namespace Render

#endif
