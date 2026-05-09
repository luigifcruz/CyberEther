#pragma once

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/context.hh>
#include <jetstream/render/sakura/text_input.hh>

#include <functional>
#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct NodeTextInput : public Component {
    struct Config {
        std::string id;
        std::string value;
        std::string unit;
        TextInput::Submit submit = TextInput::Submit::OnEnter;
        std::function<void(const std::string&)> onChange;
    };

    NodeTextInput();
    ~NodeTextInput();

    NodeTextInput(NodeTextInput&&) noexcept;
    NodeTextInput& operator=(NodeTextInput&&) noexcept;

    NodeTextInput(const NodeTextInput&) = delete;
    NodeTextInput& operator=(const NodeTextInput&) = delete;

    bool update(Config config);
    void render(const Context& ctx) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura
