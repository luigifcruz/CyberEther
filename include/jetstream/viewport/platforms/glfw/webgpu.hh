#ifndef JETSTREAM_VIEWPORT_PLATFORM_GLFW_WEBGPU_HH
#define JETSTREAM_VIEWPORT_PLATFORM_GLFW_WEBGPU_HH

#include "jetstream/viewport/adapters/webgpu.hh"
#include "jetstream/viewport/platforms/glfw/generic.hh"

struct GLFWwindow;

namespace Jetstream::Viewport {

template<>
class GLFW<Device::WebGPU> : public Adapter<Device::WebGPU> {
 public:
    explicit GLFW(const Config& config);
    virtual ~GLFW();

    std::string id() const {
        return "glfw";
    }

    std::string name() const {
        return "GLFW (WebGPU)";
    }

    constexpr Device device() const {
        return Device::WebGPU;
    };

    Result create();
    Result destroy();

    Result createImgui();
    Result destroyImgui();
    F32 scale(const F32& scale) const;

    Result createSwapchain();
    Result destroySwapchain();

    Result nextDrawable();
    Result commitDrawable(WGPUTextureView* framebufferTexture);

    Result waitEvents();
    Result pollEvents();
    bool keepRunning();

 private:
    GLFWwindow* window;
    WGPUSurface surface;
    WGPUInstance instance;
    Extent2D<U64> swapchainSize;

    static void framebufferResizeCallback(GLFWwindow *window, int width, int height);
};

}  // namespace Jetstream::Viewport

#endif
