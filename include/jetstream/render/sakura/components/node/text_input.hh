#ifndef JETSTREAM_RENDER_SAKURA_NODE_TEXT_INPUT_HH
#define JETSTREAM_RENDER_SAKURA_NODE_TEXT_INPUT_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/components/text_input.hh>

#include <functional>
#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct NodeTextInput {
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

#endif  // JETSTREAM_RENDER_SAKURA_NODE_TEXT_INPUT_HH
