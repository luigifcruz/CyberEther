#pragma once

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/context.hh>

#include <functional>
#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct NodePathInput : public Component {
    struct Config {
        std::string id;
        std::string value;
        std::function<void(const std::string&)> onChange;
        std::function<void()> onBrowse;
    };

    NodePathInput();
    ~NodePathInput();

    NodePathInput(NodePathInput&&) noexcept;
    NodePathInput& operator=(NodePathInput&&) noexcept;

    NodePathInput(const NodePathInput&) = delete;
    NodePathInput& operator=(const NodePathInput&) = delete;

    bool update(Config config);
    void render(const Context& ctx) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura
