#ifndef RENDER_METAL_INSTANCE_H
#define RENDER_METAL_INSTANCE_H

#include "render/tools/imgui_impl_glfw.h"
#include "render/tools/imgui_impl_metal.h"

#include "render/metal/window.hpp"
#include "render/base/instance.hpp"

namespace Render {

class Metal : public Render::Instance {
public:
    class Program;
	class Surface;
    class Texture;
    class Vertex;
    class Draw;

    explicit Metal(const Config& config);

    Result create() final;
    Result destroy() final;
    Result begin() final;
    Result end() final;
    Result synchronize() final;
    bool keepRunning() final;

protected:
    MTL::Device* device;

    std::vector<std::shared_ptr<Metal::Surface>> surfaces;

    static MTL::PixelFormat convertPixelFormat(const PixelFormat&, const PixelType&);
    static std::size_t getPixelByteSize(const MTL::PixelFormat&);

private:
    ImGuiIO* io;
    ImGuiStyle* style;

    GLFWwindow* window;
    std::unique_ptr<MetalWindow> metalWindow;

    CA::MetalDrawable* drawable;
    MTL::CommandQueue* commandQueue;
    MTL::CommandBuffer* commandBuffer;
    MTL::RenderPassDescriptor* renderPassDesc;

    const char* vendorString;
    const char* rendererString;
    const char* versionString;
    const char* unifiedString;
    const char* shaderString;

    Result createImgui();
    Result destroyImgui();
    Result beginImgui();
    Result endImgui();
};

} // namespace Render

#endif
