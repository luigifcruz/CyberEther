#ifndef JETSTREAM_RENDER_SAKURA_NODE_SURFACE_HH
#define JETSTREAM_RENDER_SAKURA_NODE_SURFACE_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/surface.hh>
#include <jetstream/types.hh>

#include <memory>
#include <string>

namespace Jetstream::Render {
class Texture;
}  // namespace Jetstream::Render

namespace Jetstream::Sakura {

struct NodeSurface : public Component {
    struct Config {
        std::string id;
        std::shared_ptr<const Render::Texture> texture;
        F32 minimumHeight = 150.0f;
        Callback<void(const SurfaceResize&)> onSize;
        Callback<void(InputEvent)> onInput;
        Callback<SurfaceCursor()> resolveCursor;

        bool operator==(const Config&) const = default;
    };

    NodeSurface();
    ~NodeSurface();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::shared_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_NODE_SURFACE_HH
