#ifndef JETSTREAM_RENDER_SAKURA_GRID_HH
#define JETSTREAM_RENDER_SAKURA_GRID_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>
#include <vector>

namespace Jetstream::Sakura {

struct Grid : public Component {
    struct Config {
        U64 columns = 2;
        Extent2D<F32> size = {Fill, 0.0f};
        Extent2D<F32> cellPadding = {4.0f, 2.0f};
        bool outerPadding = true;
        std::vector<Component*> items;

        bool operator==(const Config&) const = default;
    };

    Grid();
    ~Grid();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_GRID_HH
