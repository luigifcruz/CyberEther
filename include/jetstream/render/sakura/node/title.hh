#pragma once

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/context.hh>
#include <jetstream/render/sakura/node/node.hh>
#include <jetstream/types.hh>

#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct NodeTitle : public Component {
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
