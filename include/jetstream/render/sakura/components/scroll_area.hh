#ifndef JETSTREAM_RENDER_SAKURA_SCROLL_AREA_HH
#define JETSTREAM_RENDER_SAKURA_SCROLL_AREA_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct ScrollArea {
    using Child = std::function<void(const Context&)>;

    struct Config {
        std::string id;
        F32 height = 240.0f;
    };

    ScrollArea();
    ~ScrollArea();

    ScrollArea(ScrollArea&&) noexcept;
    ScrollArea& operator=(ScrollArea&&) noexcept;

    ScrollArea(const ScrollArea&) = delete;
    ScrollArea& operator=(const ScrollArea&) = delete;

    bool update(Config config);
    void render(const Context& ctx, Child child) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_SCROLL_AREA_HH
