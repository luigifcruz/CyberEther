#ifndef RENDER_BASE_SURFACE_H
#define RENDER_BASE_SURFACE_H

#include <vector>
#include <memory>

#include "render/type.hpp"
#include "render/base/texture.hpp"
#include "render/base/program.hpp"

namespace Render {

class Surface {
 public:
    struct Config {
        std::shared_ptr<Texture> framebuffer;
        std::vector<std::shared_ptr<Program>> programs;
    };

    explicit Surface(const Config& config) : config(config) {}
    virtual ~Surface() = default;

    const Size2D<int> size() const {
        if (config.framebuffer) {
            return config.framebuffer->size();
        }
        return {-1, -1};
    }
    virtual Size2D<int> size(const Size2D<int>&) = 0;

 protected:
    Config config;
};

}  // namespace Render

#endif
