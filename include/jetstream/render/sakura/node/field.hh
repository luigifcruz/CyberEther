#pragma once

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/context.hh>

#include <functional>
#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct NodeField : public Component {
    using Child = std::function<void(const Context&)>;

    struct Config {
        std::string id;
        std::string label;
        std::string help;
        bool title = true;
        bool background = true;
        bool divider = false;
    };

    NodeField();
    ~NodeField();

    NodeField(NodeField&&) noexcept;
    NodeField& operator=(NodeField&&) noexcept;

    NodeField(const NodeField&) = delete;
    NodeField& operator=(const NodeField&) = delete;

    bool update(Config config);
    void render(const Context& ctx, Child child) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura
