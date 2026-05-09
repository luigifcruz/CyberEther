#pragma once

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/context.hh>
#include <jetstream/types.hh>

#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct Spacing : public Component {
    struct Config {
        std::string id;
        U64 lines = 1;
    };

    Spacing();
    ~Spacing();

    Spacing(Spacing&&) noexcept;
    Spacing& operator=(Spacing&&) noexcept;

    Spacing(const Spacing&) = delete;
    Spacing& operator=(const Spacing&) = delete;

    bool update(Config config);
    void render(const Context& ctx) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura
