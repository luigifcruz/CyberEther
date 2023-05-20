#ifndef JETSTREAM_VIEWPORT_MACOS_HH
#define JETSTREAM_VIEWPORT_MACOS_HH

#define GLFW_INCLUDE_NONE
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include "jetstream/viewport/devices/metal.hh"
#include "jetstream/render/tools/imgui_impl_glfw.h"
#include "jetstream/backend/base.hh"

namespace Jetstream::Viewport {

class MacOS : public Provider<Device::Metal> {
 public:
    explicit MacOS(const Config& config);
    virtual ~MacOS();

    const std::string name() const {
        return "MacOS (GLFW)";
    }

    constexpr Device device() const {
        return Device::Metal;
    };

    Result create();
    Result destroy();

    Result createImgui();
    Result destroyImgui();

    void* nextDrawable();
    
    Result pollEvents();
    bool keepRunning();

    static std::shared_ptr<MacOS> Factory(const Config& config) {
        return std::make_shared<MacOS>(config);
    }

 private:
    GLFWwindow* window = nullptr;
    CA::MetalLayer* swapchain;
};

}  // namespace Jetstream::Viewport

#endif
