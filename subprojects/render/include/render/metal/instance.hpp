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

    Metal(const Config& c) : Render::Instance(c) {};

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
    MTL::Device* device;

    static MTL::PixelFormat convertPixelFormat(const PixelFormat&, const PixelType&);
    static Result getError(std::string, std::string, int);

private:
    ImGuiIO* io;
    ImGuiStyle* style;

    GLFWwindow* window;
    std::unique_ptr<MetalWindow> metalWindow;

    CA::MetalDrawable* drawable;
    MTL::CommandQueue* commandQueue;
    MTL::CommandBuffer* commandBuffer;
    MTL::RenderPassDescriptor* renderPassDesc;

    std::vector<std::shared_ptr<Metal::Surface>> surfaces;

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
