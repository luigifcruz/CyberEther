#pragma once

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/context.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct NodeIntInput : public Component {
    struct Config {
        std::string id;
        U64 value = 0;
        std::string unit;
        std::function<void(U64)> onChange;
    };

    NodeIntInput();
    ~NodeIntInput();

    NodeIntInput(NodeIntInput&&) noexcept;
    NodeIntInput& operator=(NodeIntInput&&) noexcept;

    NodeIntInput(const NodeIntInput&) = delete;
    NodeIntInput& operator=(const NodeIntInput&) = delete;

    bool update(Config config);
    void render(const Context& ctx) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura
