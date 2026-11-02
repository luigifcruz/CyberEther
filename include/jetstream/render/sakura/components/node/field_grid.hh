#ifndef JETSTREAM_RENDER_SAKURA_NODE_FIELD_GRID_HH
#define JETSTREAM_RENDER_SAKURA_NODE_FIELD_GRID_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct NodeFieldGrid : public Component {
    struct Item {
        Component* child = nullptr;
        bool fullWidth = false;

        bool operator==(const Item&) const = default;
    };

    struct Config {
        std::string id;
        std::vector<Item> items;

        bool operator==(const Config&) const = default;
    };

    NodeFieldGrid();
    ~NodeFieldGrid();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_NODE_FIELD_GRID_HH
