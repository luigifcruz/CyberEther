#pragma once

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/context.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct Tooltip : public Component {
    using Child = std::function<void(const Context&)>;

    struct Config {
        std::string id;
        F32 wrapWidth = 420.0f;
        bool delayed = true;
        bool visible = false;
    };

    Tooltip();
    ~Tooltip();

    Tooltip(Tooltip&&) noexcept;
    Tooltip& operator=(Tooltip&&) noexcept;

    Tooltip(const Tooltip&) = delete;
    Tooltip& operator=(const Tooltip&) = delete;

    bool update(Config config);
    void render(const Context& ctx, Child child) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura
