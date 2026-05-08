#pragma once

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/context.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct NodeRuntimeOverlay : public Component {
    struct Geometry {
        Extent2D<F32> nodePos = {0.0f, 0.0f};
        Extent2D<F32> nodeSize = {0.0f, 0.0f};
    };

    struct Config {
        Extent2D<F32> nodePos = {0.0f, 0.0f};
        Extent2D<F32> nodeSize = {0.0f, 0.0f};
        std::vector<std::string> lines;
        F32 offset = 4.0f;
        std::function<std::optional<Geometry>()> onResolveGeometry;
    };

    NodeRuntimeOverlay();
    ~NodeRuntimeOverlay();

    NodeRuntimeOverlay(NodeRuntimeOverlay&&) noexcept;
    NodeRuntimeOverlay& operator=(NodeRuntimeOverlay&&) noexcept;

    NodeRuntimeOverlay(const NodeRuntimeOverlay&) = delete;
    NodeRuntimeOverlay& operator=(const NodeRuntimeOverlay&) = delete;

    bool update(Config config);
    void render(const Context& ctx) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura
