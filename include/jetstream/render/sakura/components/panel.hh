#ifndef JETSTREAM_RENDER_SAKURA_PANEL_HH
#define JETSTREAM_RENDER_SAKURA_PANEL_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct Panel : public Component {
    struct Config {
        std::string id;
        Component* content = nullptr;
        Extent2D<F32> size = {Fill, 0.0f};
        F32 maxWidth = 0.0f;
        bool transparent = false;
        std::string backgroundColorKey = "background";

        bool operator==(const Config&) const = default;
    };

    Panel();
    ~Panel();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_PANEL_HH
