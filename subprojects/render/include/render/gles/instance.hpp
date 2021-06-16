#ifndef RENDER_GLES_INSTANCE_H
#define RENDER_GLES_INSTANCE_H

#define GLFW_INCLUDE_ES3
#include <GLFW/glfw3.h>
#include "render/tools/imgui_impl_glfw.h"
#include "render/tools/imgui_impl_opengl3.h"
#include "render/base/instance.hpp"

namespace Render {

class GLES : public Render::Instance {
public:
    class Program;
	class Surface;
    class Texture;
    class Vertex;
    class Draw;

    GLES(Config& c) : Render::Instance(c) {};

    Result create();
    Result destroy();
    Result start();
    Result end();

    bool keepRunning();

    std::string renderer_str();
    std::string version_str();
    std::string vendor_str();
    std::string glsl_str();

    std::shared_ptr<Render::Surface> createAndBind(Render::Surface::Config&);
    Result unbind(std::shared_ptr<Render::Surface>);

    std::shared_ptr<Render::Program> create(Render::Program::Config&);
    std::shared_ptr<Render::Texture> create(Render::Texture::Config&);
    std::shared_ptr<Render::Vertex> create(Render::Vertex::Config&);
    std::shared_ptr<Render::Draw> create(Render::Draw::Config&);

protected:
    ImGuiIO* io;
    ImGuiStyle* style;
    GLFWwindow* window;
    std::vector<std::shared_ptr<GLES::Surface>> surfaces;

    static uint convertPixelFormat(PixelFormat);
    static uint convertPixelType(PixelType);
    static uint convertDataFormat(DataFormat);
    static Result getError(std::string, std::string, int);

    Result createImgui();
    Result destroyImgui();
    Result startImgui();
    Result endImgui();
};

} // namespace Render

#endif
