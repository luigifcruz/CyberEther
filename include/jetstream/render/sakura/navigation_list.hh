#pragma once

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/context.hh>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct NavigationList : public Component {
    struct Item {
        std::string label;
        bool selected = false;
        std::function<void()> onSelect;
    };

    struct Config {
        std::string id;
        std::string title;
        std::vector<Item> items;
    };

    NavigationList();
    ~NavigationList();

    NavigationList(NavigationList&&) noexcept;
    NavigationList& operator=(NavigationList&&) noexcept;

    NavigationList(const NavigationList&) = delete;
    NavigationList& operator=(const NavigationList&) = delete;

    bool update(Config config);
    void render(const Context& ctx) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura
