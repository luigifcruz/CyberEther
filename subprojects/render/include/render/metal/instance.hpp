#ifndef RENDER_METAL_INSTANCE_H
#define RENDER_METAL_INSTANCE_H

#include <vector>
#include <memory>

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

    const Backend getBackendId() const {
        return Backend::Metal;
    }

    const bool hasCudaInterop() const {
        return false;
    }

 protected:
    std::vector<std::shared_ptr<Metal::Surface>> surfaces;

    static MTL::PixelFormat convertPixelFormat(const PixelFormat&, const PixelType&);
    static std::size_t getPixelByteSize(const MTL::PixelFormat&);

    constexpr MTL::Device* getDevice() const {
        return this->device;
    }

 private:
    ImGuiIO* io = nullptr;
    ImGuiStyle* style = nullptr;
    GLFWwindow* window = nullptr;
    MTL::Device* device = nullptr;
    std::unique_ptr<MetalWindow> metalWindow;

    CA::MetalDrawable* drawable = nullptr;
    MTL::CommandQueue* commandQueue = nullptr;
    MTL::CommandBuffer* commandBuffer = nullptr;
    MTL::RenderPassDescriptor* renderPassDesc = nullptr;

    const char* vendorString = "N/A";
    const char* rendererString = "N/A";
    const char* versionString = "N/A";
    const char* unifiedString = "N/A";
    const char* shaderString = "N/A";

    Result createImgui();
    Result destroyImgui();
    Result beginImgui();
    Result endImgui();
};

}  // namespace Render

#endif
