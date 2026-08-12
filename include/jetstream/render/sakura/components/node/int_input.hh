#ifndef JETSTREAM_RENDER_SAKURA_NODE_INT_INPUT_HH
#define JETSTREAM_RENDER_SAKURA_NODE_INT_INPUT_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct NodeIntInput {
    struct Config {
        std::string id;
        I64 value = 0;
        std::string unit;
        std::function<void(I64)> onChange;
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

#endif  // JETSTREAM_RENDER_SAKURA_NODE_INT_INPUT_HH
