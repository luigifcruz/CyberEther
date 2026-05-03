#pragma once

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/context.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct ScrollArea : public Component {
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
