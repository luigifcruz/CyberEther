#ifndef JETSTREAM_VIEWPORT_IOS_HH
#define JETSTREAM_VIEWPORT_IOS_HH

#include "jetstream/viewport/generic.hh"
#include "jetstream/backend/base.hh"

namespace Jetstream::Viewport {

class iOS : public Generic {
 public:
    explicit iOS(const Config& config, CA::MetalLayer* layer);
    virtual ~iOS();

    const Result create();
    const Result destroy();

    const Result createImgui();
    const Result destroyImgui();

    void* nextDrawable();
    
    const Result pollEvents();
    const bool keepRunning();

    static std::shared_ptr<iOS> Factory(const Config& config, 
                                        CA::MetalLayer* layer) {
        return std::make_shared<iOS>(config, layer);
    }

 private:
    CA::MetalLayer* swapchain;
};

}

#endif
