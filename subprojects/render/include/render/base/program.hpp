#ifndef RENDER_BASE_PROGRAM_H
#define RENDER_BASE_PROGRAM_H

#include <map>
#include <vector>
#include <memory>
#include <utility>
#include <string>

#include "render/type.hpp"
#include "render/base/texture.hpp"
#include "render/base/vertex.hpp"
#include "render/base/draw.hpp"
#include "render/base/buffer.hpp"

namespace Render {

class Program {
 public:
    struct Config {
        std::vector<std::shared_ptr<Draw>> draws;
        std::vector<std::shared_ptr<Texture>> textures;
        std::vector<std::shared_ptr<Buffer>> buffers;
        std::vector<std::pair<std::string, nonstd::span<uint8_t>>> uniforms;
        std::map<Backend, std::vector<const char*>> shaders;
    };

    explicit Program(const Config& config) : config(config) {}
    virtual ~Program() = default;

 protected:
    Config config;

    uint32_t drawIndex = 0;
};

}  // namespace Render

#endif
