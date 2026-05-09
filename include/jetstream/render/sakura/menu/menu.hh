#pragma once

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/context.hh>
#include <jetstream/render/sakura/text.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct Menu : public Component {
    using Child = std::function<void(const Context&)>;

    struct Config {
        std::string id;
        std::string label;
        bool enabled = true;
        Text::Font font = Text::Font::Current;
        F32 scale = 1.0f;
        std::string colorKey = "text_primary";
    };

    Menu();
    ~Menu();

    Menu(Menu&&) noexcept;
    Menu& operator=(Menu&&) noexcept;

    Menu(const Menu&) = delete;
    Menu& operator=(const Menu&) = delete;

    bool update(Config config);
    void render(const Context& ctx, Child child) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura
