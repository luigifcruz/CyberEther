#ifndef JETSTREAM_RENDER_SAKURA_NODE_FLOAT_INPUT_HH
#define JETSTREAM_RENDER_SAKURA_NODE_FLOAT_INPUT_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace Jetstream::Sakura {

struct NodeFloatInput {
    struct Config {
        std::string id;
        F32 value = 0.0f;
        std::string unit;
        int precision = 2;
        std::optional<F32> step;
        std::function<void(F32)> onChange;
        std::function<void(F32)> onStepChange;
    };

    NodeFloatInput();
    ~NodeFloatInput();

    NodeFloatInput(NodeFloatInput&&) noexcept;
    NodeFloatInput& operator=(NodeFloatInput&&) noexcept;

    NodeFloatInput(const NodeFloatInput&) = delete;
    NodeFloatInput& operator=(const NodeFloatInput&) = delete;

    bool update(Config config);
    void render(const Context& ctx) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_NODE_FLOAT_INPUT_HH
