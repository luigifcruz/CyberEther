#ifndef JETSTREAM_VIEWPORT_MACOS_HH
#define JETSTREAM_VIEWPORT_MACOS_HH

#define GLFW_INCLUDE_NONE
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include "jetstream/viewport/generic.hh"
#include "jetstream/backend/base.hh"

#include "jetstream/viewport/tools/imgui_impl_glfw.h"

namespace Jetstream::Viewport {

class MacOS : public Generic {
 public:
    explicit MacOS(const Config& config);
    virtual ~MacOS();

    const std::string name() const {
        return "MacOS (GLFW)";
    }

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

}

#endif
