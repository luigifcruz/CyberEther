#ifndef JETSTREAM_RENDER_SAKURA_NODE_UINT_INPUT_HH
#define JETSTREAM_RENDER_SAKURA_NODE_UINT_INPUT_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct NodeUIntInput {
    struct Config {
        std::string id;
        U64 value = 0;
        std::string unit;
        std::function<void(U64)> onChange;
    };

    NodeUIntInput();
    ~NodeUIntInput();

    NodeUIntInput(NodeUIntInput&&) noexcept;
    NodeUIntInput& operator=(NodeUIntInput&&) noexcept;

    NodeUIntInput(const NodeUIntInput&) = delete;
    NodeUIntInput& operator=(const NodeUIntInput&) = delete;

    bool update(Config config);
    void render(const Context& ctx) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_NODE_UINT_INPUT_HH
