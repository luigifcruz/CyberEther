#ifndef JETSTREAM_RENDER_SAKURA_NODE_SUBTITLE_HH
#define JETSTREAM_RENDER_SAKURA_NODE_SUBTITLE_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct NodeSubtitle {
    struct Config {
        std::string text;
        F32 fontScale = 0.75f;
    };

    NodeSubtitle();
    ~NodeSubtitle();

    NodeSubtitle(NodeSubtitle&&) noexcept;
    NodeSubtitle& operator=(NodeSubtitle&&) noexcept;

    NodeSubtitle(const NodeSubtitle&) = delete;
    NodeSubtitle& operator=(const NodeSubtitle&) = delete;

    bool update(Config config);
    void render(const Context& ctx) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_NODE_SUBTITLE_HH
