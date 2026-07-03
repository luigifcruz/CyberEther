#ifndef JETSTREAM_RENDER_SAKURA_NODE_RUNTIME_OVERLAY_HH
#define JETSTREAM_RENDER_SAKURA_NODE_RUNTIME_OVERLAY_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct NodeRuntimeOverlay {
    struct Geometry {
        Extent2D<F32> nodePos = {0.0f, 0.0f};
        Extent2D<F32> nodeSize = {0.0f, 0.0f};
    };

    struct Config {
        Extent2D<F32> nodePos = {0.0f, 0.0f};
        Extent2D<F32> nodeSize = {0.0f, 0.0f};
        std::vector<std::string> lines;
        F32 offset = 4.0f;
        std::string primaryColorKey = "text_primary";
        std::string secondaryColorKey = "text_secondary";
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

#endif  // JETSTREAM_RENDER_SAKURA_NODE_RUNTIME_OVERLAY_HH
