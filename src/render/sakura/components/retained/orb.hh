#ifndef JETSTREAM_RENDER_SAKURA_RETAINED_ORB_HH
#define JETSTREAM_RENDER_SAKURA_RETAINED_ORB_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>
#include <string>

namespace Jetstream::Sakura::Retained {

struct OrbView : public Component {
    struct Config {
        std::string id;
        Rect rect;
        Extent2D<F32> orbCenter = {0.0f, 0.0f};
        F32 orbRadius = 0.0f;
        bool visible = true;
        F32 time = 0.0f;
        F32 busy = 0.0f;
        F32 activity = 0.0f;
        F32 level = 0.0f;
        ColorRGBA<F32> listeningColor;
        ColorRGBA<F32> busyColor;
        ColorRGBA<F32> listeningAccent;
        ColorRGBA<F32> busyAccent;
    };

    OrbView();
    ~OrbView() override;

    bool update(Config config);

 protected:
    Result build(Context& ctx) override;
    Result paint() override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura::Retained

#endif  // JETSTREAM_RENDER_SAKURA_RETAINED_ORB_HH
