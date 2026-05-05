#pragma once

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/context.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct Button : public Component {
    enum class Variant {
        Default,
        Action,
        Destructive,
    };

    struct Config {
        std::string id;
        std::string str;
        Extent2D<F32> size = {0.0f, 0.0f};
        Variant variant = Variant::Default;
        bool disabled = false;
        std::string colorKey;
        std::string hoveredColorKey;
        std::string activeColorKey;
        std::string borderColorKey;
        std::string textColorKey;
        F32 textScale = 1.0f;
        std::function<void()> onClick;
    };

    Button();
    ~Button();

    Button(Button&&) noexcept;
    Button& operator=(Button&&) noexcept;

    Button(const Button&) = delete;
    Button& operator=(const Button&) = delete;

    bool update(Config config);
    void render(const Context& ctx) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura
