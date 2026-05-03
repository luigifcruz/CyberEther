#pragma once

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/context.hh>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct KeyboardInput : public Component {
    enum class Key {
        Down,
        Up,
        Submit,
        N,
        O,
        S,
        W,
        I,
        Comma,
    };

    enum class Modifier {
        None,
        CommandOrControl,
    };

    struct Binding {
        Key key = Key::Submit;
        Modifier modifier = Modifier::None;
        std::function<void()> onPressed;
    };

    struct Config {
        std::string id;
        std::vector<Binding> bindings;
        bool repeat = false;
    };

    KeyboardInput();
    ~KeyboardInput();

    KeyboardInput(KeyboardInput&&) noexcept;
    KeyboardInput& operator=(KeyboardInput&&) noexcept;

    KeyboardInput(const KeyboardInput&) = delete;
    KeyboardInput& operator=(const KeyboardInput&) = delete;

    bool update(Config config);
    void render(const Context& ctx) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura
