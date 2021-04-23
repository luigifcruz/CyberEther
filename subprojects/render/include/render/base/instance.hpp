#ifndef RENDER_BASE_INSTANCE_H
#define RENDER_BASE_INSTANCE_H

#include "render/types.hpp"
#include "program.hpp"
#include "surface.hpp"
#include "texture.hpp"
#include "vertex.hpp"

namespace Render {

class Instance {
public:
    struct Config {
        int width = 1280;
        int height = 720;
        float scale = -1.0;
        bool resizable = false;
        bool enableImgui = false;
        bool enableDebug = false;
        std::string title = "Render";
        std::vector<std::shared_ptr<Surface>> surfaces;
    };

    Instance(Config& c) : cfg(c) {};
    virtual ~Instance() = default;

    virtual Result create() = 0;
    virtual Result destroy() = 0;
    virtual Result start() = 0;
    virtual Result end() = 0;

    Config& config() {
        return cfg;
    }

    virtual bool keepRunning() = 0;

    virtual std::string renderer_str() = 0;
    virtual std::string version_str() = 0;
    virtual std::string vendor_str() = 0;
    virtual std::string glsl_str() = 0;

    virtual std::shared_ptr<Surface> create(Surface::Config&) = 0;
    virtual std::shared_ptr<Program> create(Program::Config&) = 0;
    virtual std::shared_ptr<Texture> create(Texture::Config&) = 0;
    virtual std::shared_ptr<Vertex> create(Vertex::Config&) = 0;

protected:
    Config& cfg;

    virtual Result createImgui() = 0;
    virtual Result destroyImgui() = 0;
    virtual Result startImgui() = 0;
    virtual Result endImgui() = 0;

    static Result getError(std::string func, std::string file, int line);

    std::string cached_renderer_str;
    std::string cached_version_str;
    std::string cached_vendor_str;
    std::string cached_glsl_str;
};

} // namespace Render

#endif
