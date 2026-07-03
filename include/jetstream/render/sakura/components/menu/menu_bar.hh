#ifndef JETSTREAM_RENDER_SAKURA_MENU_MENU_BAR_HH
#define JETSTREAM_RENDER_SAKURA_MENU_MENU_BAR_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct MenuBar {
    using Child = std::function<void(const Context&)>;

    struct Config {
        std::string id;
        F32 heightScale = 1.0f;
        std::function<void(F32)> onHeight;
    };

    MenuBar();
    ~MenuBar();

    MenuBar(MenuBar&&) noexcept;
    MenuBar& operator=(MenuBar&&) noexcept;

    MenuBar(const MenuBar&) = delete;
    MenuBar& operator=(const MenuBar&) = delete;

    bool update(Config config);
    void render(const Context& ctx, Child child) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_MENU_MENU_BAR_HH
