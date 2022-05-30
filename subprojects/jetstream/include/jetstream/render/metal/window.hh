#ifndef JETSTREAM_RENDER_METAL_WINDOW_HH
#define JETSTREAM_RENDER_METAL_WINDOW_HH

#include "jetstream/render/tools/imgui_impl_glfw.h"
#include "jetstream/render/tools/imgui_impl_metal.h"

#include "jetstream/render/base/surface.hh"
#include "jetstream/render/base/window.hh"
#include "jetstream/render/metal/view.hh"
#include "jetstream/backend/base.hh"

namespace Jetstream::Render {

template<>
class WindowImp<Device::Metal> : public Window {
 public:
    explicit WindowImp(const Config& config);

    const Result create();
    const Result destroy();
    const Result begin();
    const Result end();

    const Result synchronize();
    const bool keepRunning();

    const Result bind(const std::shared_ptr<Surface>& surface);

    constexpr const Device implementation() const {
        return Device::Metal;
    };

 private:
    GLFWwindow* window;
    std::unique_ptr<View> view;
    ImGuiIO* io = nullptr;
    ImGuiStyle* style = nullptr;
    MTL::Device* device = nullptr;
    CA::MetalDrawable* drawable = nullptr;
    MTL::CommandQueue* commandQueue = nullptr;
    MTL::CommandBuffer* commandBuffer = nullptr;
    MTL::RenderPassDescriptor* renderPassDescriptor = nullptr;
    std::vector<std::shared_ptr<SurfaceImp<Device::Metal>>> surfaces;

    const Result createImgui();
    const Result destroyImgui();
    const Result beginImgui();
    const Result endImgui();
};

}  // namespace Jetstream::Render

#endif
