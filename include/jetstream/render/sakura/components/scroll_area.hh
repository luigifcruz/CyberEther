#ifndef JETSTREAM_RENDER_SAKURA_SCROLL_AREA_HH
#define JETSTREAM_RENDER_SAKURA_SCROLL_AREA_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>

namespace Jetstream::Sakura {

struct ScrollArea : public Component {
    struct Config {
        Component* content = nullptr;
        F32 height = 240.0f;
        bool alwaysScrollbar = true;
        bool border = true;

        bool operator==(const Config&) const = default;
    };

    ScrollArea();
    ~ScrollArea();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_SCROLL_AREA_HH
