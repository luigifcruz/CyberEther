#ifndef JETSTREAM_VIEWPORT_PLATFORM_GLFW_METAL_HH
#define JETSTREAM_VIEWPORT_PLATFORM_GLFW_METAL_HH

#include <memory>

#include "jetstream/viewport/adapters/metal.hh"
#include "jetstream/viewport/platforms/glfw/generic.hh"

struct GLFWwindow;

namespace Jetstream::Viewport {

template<>
class GLFW<DeviceType::Metal> : public Adapter<DeviceType::Metal> {
 public:
    explicit GLFW(const Config& config);
    virtual ~GLFW();

    std::string id() const {
        return "glfw";
    }

    std::string name() const {
        return "GLFW (Metal)";
    }

    constexpr DeviceType device() const {
        return DeviceType::Metal;
    };

    Result create();
    Result destroy();

    Extent2D<F32> displaySize() const;
    Extent2D<F32> framebufferSize() const;

    void* nextDrawable();
    
    Result waitEvents();
    Result pollEvents();
    bool keepRunning();
    bool retainedPresent() const;

    Render::TitlebarChrome titlebarChrome() const;
    void setIntegratedTitlebar(bool enabled);
    void beginWindowDrag();
    void zoomWindow();

 private:
    GLFWwindow* window = nullptr;
    std::shared_ptr<CocoaTitlebar> titlebar;
    CA::MetalLayer* swapchain;
};

}  // namespace Jetstream::Viewport 

#endif
