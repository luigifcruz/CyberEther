#ifndef JETSTREAM_RENDER_SAKURA_BACKGROUND_LAYER_HH
#define JETSTREAM_RENDER_SAKURA_BACKGROUND_LAYER_HH

#include <jetstream/render/sakura/components/layer.hh>
#include <jetstream/types.hh>

#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct BackgroundLayer : public Layer {
    struct Config {
        bool particles = true;
        U64 particleCount = 60;
        F32 topOffset = 0.0f;
        std::string backgroundColorKey = "background";
        std::string particleColorKey = "workspace_particle";

        bool operator==(const Config&) const = default;
    };

    BackgroundLayer();
    ~BackgroundLayer();

    void update(Config config);
    bool covers(const Extent2D<F32>& point) const override;

 protected:
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_BACKGROUND_LAYER_HH
