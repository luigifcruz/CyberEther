#ifndef JETSTREAM_RENDER_SAKURA_NODE_BOOL_INPUT_HH
#define JETSTREAM_RENDER_SAKURA_NODE_BOOL_INPUT_HH

#include <jetstream/render/sakura/component.hh>

#include <functional>
#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct NodeBoolInput {
    struct Config {
        std::string id;
        bool value = false;
        std::function<void(bool)> onChange;
    };

    NodeBoolInput();
    ~NodeBoolInput();

    NodeBoolInput(NodeBoolInput&&) noexcept;
    NodeBoolInput& operator=(NodeBoolInput&&) noexcept;

    NodeBoolInput(const NodeBoolInput&) = delete;
    NodeBoolInput& operator=(const NodeBoolInput&) = delete;

    bool update(Config config);
    void render(const Context& ctx) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_NODE_BOOL_INPUT_HH
