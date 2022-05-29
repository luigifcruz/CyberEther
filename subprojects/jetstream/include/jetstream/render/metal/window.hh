#ifndef JETSTREAM_RENDER_METAL_WINDOW_HH
#define JETSTREAM_RENDER_METAL_WINDOW_HH

#include <memory>

#include "jetstream/render/base/surface.hh"
#include "jetstream/render/metal/view.hh"
#include "jetstream/render/base/window.hh"
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

    constexpr const Device device() const {
        return Device::Metal;
    };

 protected:
    static const MTL::PixelFormat convertPixelFormat(const PixelFormat&, 
                                                     const PixelType&);
    static const std::size_t getPixelByteSize(const MTL::PixelFormat&);

 private:
    GLFWwindow* window;
    ImGuiIO* io = nullptr;
    std::unique_ptr<View> view;
    ImGuiStyle* style = nullptr;
    MTL::Device* device = nullptr;
    MTL::CommandQueue* commandQueue = nullptr;
    MTL::RenderPassDescriptor* renderPassDescriptor = nullptr;
    std::vector<SurfaceImp<Device::Metal>> surfaces;

    const Result createImgui();
    const Result destroyImgui();
    const Result beginImgui();
    const Result endImgui();
};

}  // namespace Jetstream::Render

#endif
