#pragma once

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/context.hh>

#include <functional>
#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct NavigationItem : public Component {
    struct Config {
        std::string id;
        std::string label;
        bool selected = false;
        std::string colorKey = "card";
        std::string hoveredColorKey = "header_hovered";
        std::string activeColorKey = "header_active";
        std::string selectedColorKey = "header_active";
        std::string textColorKey = "text_secondary";
        std::string selectedTextColorKey = "text_primary";
        std::string borderColorKey = "border";
        std::string selectedBorderColorKey = "accent_color";
        std::function<void()> onSelect;
    };

    NavigationItem();
    ~NavigationItem();

    NavigationItem(NavigationItem&&) noexcept;
    NavigationItem& operator=(NavigationItem&&) noexcept;

    NavigationItem(const NavigationItem&) = delete;
    NavigationItem& operator=(const NavigationItem&) = delete;

    bool update(Config config);
    void render(const Context& ctx) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura
