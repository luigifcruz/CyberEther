#ifndef JETSTREAM_RENDER_SAKURA_OVERLAY_HH
#define JETSTREAM_RENDER_SAKURA_OVERLAY_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>
#include <vector>

namespace Jetstream::Sakura {

enum class Anchor {
    Fill,
    TopLeft,
    TopCenter,
    TopRight,
    Center,
    BottomLeft,
    BottomCenter,
    BottomRight,
};

struct Overlay : public Component {
    struct Item {
        Component* child = nullptr;
        Anchor anchor = Anchor::Fill;
        Extent2D<F32> offset = {0.0f, 0.0f};
        Extent2D<F32> size = {0.0f, 0.0f};

        bool operator==(const Item&) const = default;
    };

    struct Config {
        std::vector<Item> items;

        bool operator==(const Config&) const = default;
    };

    Overlay();
    ~Overlay();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_OVERLAY_HH
