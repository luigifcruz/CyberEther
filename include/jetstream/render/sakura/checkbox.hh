#pragma once

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/context.hh>

#include <functional>
#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct Checkbox : public Component {
    struct Config {
        std::string id;
        std::string label;
        bool value = false;
        std::function<void(bool)> onChange;
    };

    Checkbox();
    ~Checkbox();

    Checkbox(Checkbox&&) noexcept;
    Checkbox& operator=(Checkbox&&) noexcept;

    Checkbox(const Checkbox&) = delete;
    Checkbox& operator=(const Checkbox&) = delete;

    bool update(Config config);
    void render(const Context& ctx) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura
