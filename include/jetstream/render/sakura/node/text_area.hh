#pragma once

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/context.hh>

#include <functional>
#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct NodeTextArea : public Component {
    struct Config {
        std::string id;
        std::string value;
        std::function<void(const std::string&)> onChange;
    };

    NodeTextArea();
    ~NodeTextArea();

    NodeTextArea(NodeTextArea&&) noexcept;
    NodeTextArea& operator=(NodeTextArea&&) noexcept;

    NodeTextArea(const NodeTextArea&) = delete;
    NodeTextArea& operator=(const NodeTextArea&) = delete;

    bool update(Config config);
    void render(const Context& ctx) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura
