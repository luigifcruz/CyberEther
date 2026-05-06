#pragma once

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/context.hh>
#include <jetstream/types.hh>

#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct Divider : public Component {
    struct Config {
        std::string id;
        F32 spacing = -1.0f;
        bool separator = true;
    };

    Divider();
    ~Divider();

    Divider(Divider&&) noexcept;
    Divider& operator=(Divider&&) noexcept;

    Divider(const Divider&) = delete;
    Divider& operator=(const Divider&) = delete;

    bool update(Config config);
    void render(const Context& ctx) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura
