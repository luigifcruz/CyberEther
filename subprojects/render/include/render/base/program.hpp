#ifndef RENDER_BASE_PROGRAM_H
#define RENDER_BASE_PROGRAM_H

#include <variant>
#include <vector>
#include <memory>
#include <utility>
#include <string>

#include "render/type.hpp"
#include "render/base/texture.hpp"
#include "render/base/vertex.hpp"
#include "render/base/draw.hpp"

namespace Render {

class Program {
 public:
    struct Config {
        const char* const* vertexSource = nullptr;
        const char* const* fragmentSource = nullptr;
        std::vector<std::shared_ptr<Draw>> draws;
        std::vector<std::shared_ptr<Texture>> textures;
        std::vector<std::pair<std::string,
            std::variant<std::vector<float>*, std::vector<uint32_t>*>>> uniforms;
    };

    explicit Program(const Config& config) : config(config) {}
    virtual ~Program() = default;

 protected:
    Config config;

    uint32_t drawIndex = 0;
};

}  // namespace Render

#endif
