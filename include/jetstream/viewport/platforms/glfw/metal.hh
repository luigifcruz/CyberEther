#ifndef JETSTREAM_VIEWPORT_PLATFORM_GLFW_METAL_HH
#define JETSTREAM_VIEWPORT_PLATFORM_GLFW_METAL_HH

#include "jetstream/viewport/adapters/metal.hh"
#include "jetstream/viewport/platforms/glfw/generic.hh"

#define GLFW_INCLUDE_NONE
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

namespace Jetstream::Viewport {

template<>
class GLFW<Device::Metal> : public Adapter<Device::Metal> {
 public:
    explicit GLFW(const Config& config);
    virtual ~GLFW();

    std::string_view prettyName() const {
        return "GLFW (Metal)";
    }

    std::string_view name() const {
        return "glfw";
    }

    constexpr Device device() const {
        return Device::Metal;
    };

    Result create();
    Result destroy();

    Result createImgui();
    Result destroyImgui();
    F32 calculateScale(const F32& scale);

    void* nextDrawable();
    
    Result pollEvents();
    bool keepRunning();

 private:
    GLFWwindow* window = nullptr;
    CA::MetalLayer* swapchain;
};

}  // namespace Jetstream::Viewport 

#endif
