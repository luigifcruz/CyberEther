#ifndef JETSTREAM_VIEWPORT_PLATFORM_GLFW_METAL_HH
#define JETSTREAM_VIEWPORT_PLATFORM_GLFW_METAL_HH

#include "jetstream/viewport/adapters/metal.hh"
#include "jetstream/viewport/platforms/glfw/generic.hh"

struct GLFWwindow;

namespace Jetstream::Viewport {

template<>
class GLFW<Device::Metal> : public Adapter<Device::Metal> {
 public:
    explicit GLFW(const Config& config);
    virtual ~GLFW();

    std::string id() const {
        return "glfw";
    }

    std::string name() const {
        return "GLFW (Metal)";
    }

    constexpr Device device() const {
        return Device::Metal;
    };

    Result create();
    Result destroy();

    Result createImgui();
    Result destroyImgui();
    F32 scale(const F32& scale) const;

    void* nextDrawable();
    
    Result pollEvents();
    bool keepRunning();

 private:
    GLFWwindow* window = nullptr;
    CA::MetalLayer* swapchain;
};

}  // namespace Jetstream::Viewport 

#endif
