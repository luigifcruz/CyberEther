#ifndef JETSTREAM_RENDER_SAKURA_NODE_TITLE_HH
#define JETSTREAM_RENDER_SAKURA_NODE_TITLE_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/components/node/node.hh>
#include <jetstream/types.hh>

#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct NodeTitle {
    struct Diagnostic {
        Node::State state = Node::State::Normal;
        std::string message;
    };

    struct Config {
        std::string title;
        Diagnostic diagnostic;
        F32 titleScale = 1.15f;
    };

    NodeTitle();
    ~NodeTitle();

    NodeTitle(NodeTitle&&) noexcept;
    NodeTitle& operator=(NodeTitle&&) noexcept;

    NodeTitle(const NodeTitle&) = delete;
    NodeTitle& operator=(const NodeTitle&) = delete;

    bool update(Config config);
    void render(const Context& ctx) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_NODE_TITLE_HH
