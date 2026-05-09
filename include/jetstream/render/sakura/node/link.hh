#pragma once

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/context.hh>

#include <functional>
#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct NodeLink : public Component {
    using Child = std::function<void(const Context&)>;

    struct Endpoint {
        std::string nodeId;
        std::string pinId;
        bool isInput = false;
    };

    struct Config {
        std::string id;
        Endpoint start;
        Endpoint end;
        bool unresolved = false;
        std::function<void(bool)> onHover;
    };

    NodeLink();
    ~NodeLink();

    NodeLink(NodeLink&&) noexcept;
    NodeLink& operator=(NodeLink&&) noexcept;

    NodeLink(const NodeLink&) = delete;
    NodeLink& operator=(const NodeLink&) = delete;

    bool update(Config config);
    void render(const Context& ctx, Child tooltip) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura
