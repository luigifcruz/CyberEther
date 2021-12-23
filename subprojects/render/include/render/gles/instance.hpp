#ifndef RENDER_GLES_INSTANCE_H
#define RENDER_GLES_INSTANCE_H

#include <vector>
#include <memory>
#include <string>

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

    explicit GLES(const Config& config);

    Result create() final;
    Result destroy() final;
    Result begin() final;
    Result end() final;
    Result synchronize() final;

    bool keepRunning() final;

 protected:
    std::vector<std::shared_ptr<GLES::Surface>> surfaces;

    static uint convertPixelFormat(PixelFormat);
    static uint convertPixelType(PixelType);
    static uint convertDataFormat(DataFormat);
    static Result getError(std::string, std::string, int);

 private:
    ImGuiIO* io = nullptr;
    ImGuiStyle* style = nullptr;
    GLFWwindow* window = nullptr;

    const char* rendererString = "N/A";
    const char* versionString = "N/A";
    const char* vendorString = "N/A";
    const char* shaderString = "N/A";

    Result createImgui();
    Result destroyImgui();
    Result beginImgui();
    Result endImgui();
};

}  // namespace Render

#endif
