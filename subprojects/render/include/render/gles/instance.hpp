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

    GLES(const Config& c) : Render::Instance(c) {};

    Result create() final;
    Result destroy() final;
    Result begin() final;
    Result end() final;
    Result synchronize() final;

    bool keepRunning() final;

    std::shared_ptr<Render::Surface> createAndBind(const Render::Surface::Config&) final;
    std::shared_ptr<Render::Program> create(const Render::Program::Config&) final;
    std::shared_ptr<Render::Texture> create(const Render::Texture::Config&) final;
    std::shared_ptr<Render::Vertex> create(const Render::Vertex::Config&) final;
    std::shared_ptr<Render::Draw> create(const Render::Draw::Config&) final;

protected:
    ImGuiIO* io;
    ImGuiStyle* style;
    GLFWwindow* window;
    std::vector<std::shared_ptr<GLES::Surface>> surfaces;

    static uint convertPixelFormat(PixelFormat);
    static uint convertPixelType(PixelType);
    static uint convertDataFormat(DataFormat);
    static Result getError(std::string, std::string, int);

private:
    const char* rendererString;
    const char* versionString;
    const char* vendorString;
    const char* shaderString;

    Result createImgui();
    Result destroyImgui();
    Result beginImgui();
    Result endImgui();
};

} // namespace Render

#endif
