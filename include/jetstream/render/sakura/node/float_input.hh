#pragma once

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/context.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace Jetstream::Sakura {

struct NodeFloatInput : public Component {
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
