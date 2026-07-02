#ifndef JETSTREAM_RENDER_SAKURA_MENU_MENU_ITEM_HH
#define JETSTREAM_RENDER_SAKURA_MENU_MENU_ITEM_HH

#include <jetstream/render/sakura/component.hh>

#include <functional>
#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct MenuItem {
    struct Config {
        std::string id;
        std::string label;
        std::string shortcut;
        bool selected = false;
        bool enabled = true;
        std::function<void()> onClick;
    };

    MenuItem();
    ~MenuItem();

    MenuItem(MenuItem&&) noexcept;
    MenuItem& operator=(MenuItem&&) noexcept;

    MenuItem(const MenuItem&) = delete;
    MenuItem& operator=(const MenuItem&) = delete;

    bool update(Config config);
    void render(const Context& ctx) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_MENU_MENU_ITEM_HH
