#ifndef RENDER_BASE_INSTANCE_H
#define RENDER_BASE_INSTANCE_H

#include "render/type.hpp"
#include "render/base/program.hpp"
#include "render/base/surface.hpp"
#include "render/base/texture.hpp"
#include "render/base/vertex.hpp"
#include "render/base/draw.hpp"

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

    Instance(const Config & c) : cfg(c) {};
    virtual ~Instance() = default;

    virtual Result create() = 0;
    virtual Result destroy() = 0;
    virtual Result start() = 0;
    virtual Result end() = 0;

    virtual Result synchronize() = 0;
    virtual bool keepRunning() = 0;

    virtual std::string renderer_str() = 0;
    virtual std::string version_str() = 0;
    virtual std::string vendor_str() = 0;
    virtual std::string glsl_str() = 0;

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

    virtual std::shared_ptr<Surface> createAndBind(Surface::Config&) = 0;
    virtual Result unbind(std::shared_ptr<Surface>) = 0;

    virtual std::shared_ptr<Program> create(Program::Config&) = 0;
    virtual std::shared_ptr<Texture> create(Texture::Config&) = 0;
    virtual std::shared_ptr<Vertex> create(Vertex::Config&) = 0;
    virtual std::shared_ptr<Draw> create(Draw::Config&) = 0;

    static bool cudaInteropSupported() {
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
    virtual Result startImgui() = 0;
    virtual Result endImgui() = 0;

    std::string cached_renderer_str;
    std::string cached_version_str;
    std::string cached_vendor_str;
    std::string cached_glsl_str;

    static Result getError(std::string func, std::string file, int line);
};

} // namespace Render

#endif
