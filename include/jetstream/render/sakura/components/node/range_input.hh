#ifndef JETSTREAM_RENDER_SAKURA_NODE_RANGE_INPUT_HH
#define JETSTREAM_RENDER_SAKURA_NODE_RANGE_INPUT_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct NodeRangeInput {
    struct Config {
        std::string id;
        F32 min = 0.0f;
        F32 max = 1.0f;
        F32 value = 0.0f;
        bool integer = false;
        std::string unit;
        std::function<void(F32)> onChange;
    };

    NodeRangeInput();
    ~NodeRangeInput();

    NodeRangeInput(NodeRangeInput&&) noexcept;
    NodeRangeInput& operator=(NodeRangeInput&&) noexcept;

    NodeRangeInput(const NodeRangeInput&) = delete;
    NodeRangeInput& operator=(const NodeRangeInput&) = delete;

    bool update(Config config);
    void render(const Context& ctx) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_NODE_RANGE_INPUT_HH
