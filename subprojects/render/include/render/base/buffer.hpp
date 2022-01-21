#ifndef RENDER_BASE_BUFFER_H
#define RENDER_BASE_BUFFER_H

#include <string>

#include "render/type.hpp"

namespace Render {

class Buffer {
 public:
    struct Config {
        std::size_t size;
        std::size_t elementByteSize;
        void* buffer = nullptr;
    };

    explicit Buffer(const Config& config) : config(config) {}
    virtual ~Buffer() = default;

    constexpr const std::size_t size() const {
        return config.size;
    }

    virtual Result update() = 0;
    virtual Result update(const std::size_t& offset, const std::size_t& size) = 0;

 protected:
    Config config;
};

}  // namespace Render

#endif
