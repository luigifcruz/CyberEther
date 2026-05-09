#pragma once

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/context.hh>

#include <functional>
#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct TextInput : public Component {
    enum class Submit {
        OnEdit,
        OnEnter,
        OnCommit,
    };

    struct Config {
        std::string id;
        std::string value;
        std::string hint;
        Submit submit = Submit::OnEnter;
        bool focus = false;
        bool focusOutline = true;
        std::function<void(const std::string&)> onChange;
    };

    TextInput();
    ~TextInput();

    TextInput(TextInput&&) noexcept;
    TextInput& operator=(TextInput&&) noexcept;

    TextInput(const TextInput&) = delete;
    TextInput& operator=(const TextInput&) = delete;

    bool update(Config config);
    void render(const Context& ctx) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura
