#ifndef JETSTREAM_RENDER_SAKURA_NAVIGATION_LIST_HH
#define JETSTREAM_RENDER_SAKURA_NAVIGATION_LIST_HH

#include <jetstream/render/sakura/component.hh>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct NavigationList {
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

#endif  // JETSTREAM_RENDER_SAKURA_NAVIGATION_LIST_HH
