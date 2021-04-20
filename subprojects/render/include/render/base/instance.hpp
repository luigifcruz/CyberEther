#ifndef RENDER_BASE_INSTANCE_H
#define RENDER_BASE_INSTANCE_H

#include "render/types.hpp"
#include "program.hpp"
#include "surface.hpp"
#include "texture.hpp"

namespace Render {

const float vertices[] = {
    +1.0f, +1.0f, 0.0f, +0.0f, +0.0f,
    +1.0f, -1.0f, 0.0f, +0.0f, +1.0f,
    -1.0f, -1.0f, 0.0f, +1.0f, +1.0f,
    -1.0f, +1.0f, 0.0f, +1.0f, +0.0f,
};

const uint elements[] = {
    0, 1, 2,
    2, 3, 0
};

class Instance {
public:
    struct Config {
        int width = 1280;
        int height = 720;
        bool resizable = false;
        bool enableImgui = false;
        std::string title = "Render";
    };

    Instance(Config& c) : cfg(c) {};
    virtual ~Instance() = default;

    virtual Result init() = 0;
    virtual Result terminate() = 0;

    virtual Result clear() = 0;
    virtual Result draw() = 0;
    virtual Result step() = 0;

    virtual bool keepRunning() = 0;

    virtual std::string renderer_str() = 0;
    virtual std::string version_str() = 0;
    virtual std::string vendor_str() = 0;
    virtual std::string glsl_str() = 0;

    template<class T> static std::shared_ptr<T> Create(Instance::Config&);
    template<class T> std::shared_ptr<Program> createProgram(Program::Config&);
    template<class T> std::shared_ptr<Surface> createSurface(Surface::Config&);
    template<class T> std::shared_ptr<Texture> createTexture(Texture::Config&);

protected:
    Config& cfg;

    virtual Result createBuffers() = 0;
    virtual Result destroyBuffers() = 0;

    virtual Result createImgui() = 0;
    virtual Result destroyImgui() = 0;
    virtual Result startImgui() = 0;
    virtual Result endImgui() = 0;

    static Result getError(std::string func, std::string file, int line);

    struct State* state;
    std::vector<std::shared_ptr<Program>> programs;
    std::vector<std::shared_ptr<Surface>> surfaces;
    std::vector<std::shared_ptr<Texture>> textures;
        
    std::string cached_renderer_str;
    std::string cached_version_str;
    std::string cached_vendor_str;
    std::string cached_glsl_str;
};

} // namespace Render

#endif
