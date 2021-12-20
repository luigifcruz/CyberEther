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
        float scale = -1.0;
        bool resizable = false;
        bool imgui = false;
        bool vsync = true;
        bool debug = false;
        std::string title = "Render";
    };

    Instance(const Config& c) : cfg(c) {};
    virtual ~Instance() = default;

    virtual Result create() = 0;
    virtual Result destroy() = 0;
    virtual Result begin() = 0;
    virtual Result end() = 0;

    virtual Result synchronize() = 0;
    virtual bool keepRunning() = 0;

    constexpr Size2D<int> size() const {
        return cfg.size;
    }
    constexpr float scale() const {
        return cfg.scale;
    }
    constexpr bool resizable() const {
        return cfg.resizable;
    }
    constexpr bool imgui() const {
        return cfg.imgui;
    }
    constexpr bool vsync() const {
        return cfg.vsync;
    }
    constexpr bool debug() const {
        return cfg.debug;
    }
    std::string title() const {
        return cfg.title;
    }

    virtual std::shared_ptr<Surface> createAndBind(const Surface::Config&) = 0;
    virtual std::shared_ptr<Program> create(const Program::Config&) = 0;
    virtual std::shared_ptr<Texture> create(const Texture::Config&) = 0;
    virtual std::shared_ptr<Vertex> create(const Vertex::Config&) = 0;
    virtual std::shared_ptr<Draw> create(const Draw::Config&) = 0;

    constexpr static bool cudaInteropSupported() {
#ifdef RENDER_CUDA_AVAILABLE
        return true;
#else
        return false;
#endif
    };

protected:
    Config cfg;

    virtual Result createImgui() = 0;
    virtual Result destroyImgui() = 0;
    virtual Result beginImgui() = 0;
    virtual Result endImgui() = 0;
};

} // namespace Render

#endif