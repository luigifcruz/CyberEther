#ifndef JETSTREAM_RENDER_SAKURA_NODE_NODE_HH
#define JETSTREAM_RENDER_SAKURA_NODE_NODE_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace Jetstream::Sakura {

struct Node {
    using Child = std::function<void(const Context&)>;

    enum class State {
        Normal,
        Error,
        Pending,
        Loading,
    };

    enum class ResizeAxes : U8 {
        None = 0,
        X = 1 << 0,
        Y = 1 << 1,
        XY = 3,
    };

    struct Config {
        std::string id;
        State state = State::Normal;
        ResizeAxes resize = ResizeAxes::X;
        Extent2D<F32> dimensions = {0.0f, 0.0f};
        Extent2D<F32> minimumDimensions = {120.0f, 100.0f};
        std::optional<Extent2D<F32>> gridPosition;
        std::function<void()> onContextMenu;
        std::function<void(Extent2D<F32>, Extent2D<F32>, Extent2D<F32>, Extent2D<F32>)> onGeometryChange;
    };

    Node();
    ~Node();

    Node(Node&&) noexcept;
    Node& operator=(Node&&) noexcept;

    Node(const Node&) = delete;
    Node& operator=(const Node&) = delete;

    bool update(Config config);
    void render(const Context& ctx, Child child) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_NODE_NODE_HH
