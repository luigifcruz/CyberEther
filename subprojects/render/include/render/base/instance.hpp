#ifndef RENDER_BASE_INSTANCE_H
#define RENDER_BASE_INSTANCE_H

#include "render/type.hpp"
#include "render/base/program.hpp"
#include "render/base/surface.hpp"
#include "render/base/texture.hpp"
#include "render/base/vertex.hpp"
#include "render/base/draw.hpp"
#include "render/tools/imgui.h"

namespace Render {

class Instance {
public:
    struct Config {
        Size2D<int> size = {1280, 720};
        std::string title = "Render";
        bool resizable = false;
        float scale = -1.0;
        bool imgui = false;
        bool debug = false;
        bool vsync = true;
    };

    explicit Instance(const Config& config) : config(config) {}
    virtual ~Instance() = default;

    virtual Result create() = 0;
    virtual Result destroy() = 0;
    virtual Result begin() = 0;
    virtual Result end() = 0;

    virtual Result synchronize() = 0;
    virtual bool keepRunning() = 0;

    constexpr static bool cudaInteropSupported() {
#ifdef RENDER_CUDA_AVAILABLE
        return true;
#else
        return false;
#endif
    };

protected:
    Config config;
};

} // namespace Render

#endif
