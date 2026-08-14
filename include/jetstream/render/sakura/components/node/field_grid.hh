#ifndef JETSTREAM_RENDER_SAKURA_NODE_FIELD_GRID_HH
#define JETSTREAM_RENDER_SAKURA_NODE_FIELD_GRID_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct NodeFieldGrid {
    using Child = std::function<void(const Context&)>;

    struct Item {
        Child child;
        bool fullWidth = false;
    };

    struct Config {
        std::string id;
        F32 minColumnWidth = 140.0f;
    };

    NodeFieldGrid();
    ~NodeFieldGrid();

    NodeFieldGrid(NodeFieldGrid&&) noexcept;
    NodeFieldGrid& operator=(NodeFieldGrid&&) noexcept;

    NodeFieldGrid(const NodeFieldGrid&) = delete;
    NodeFieldGrid& operator=(const NodeFieldGrid&) = delete;

    bool update(Config config);
    void render(const Context& ctx, const std::vector<Item>& items) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_NODE_FIELD_GRID_HH
