#pragma once

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/context.hh>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct Combo : public Component {
    struct Config {
        std::string id;
        std::vector<std::string> options;
        std::string value;
        std::function<void(const std::string&)> onChange;
    };

    Combo();
    ~Combo();

    Combo(Combo&&) noexcept;
    Combo& operator=(Combo&&) noexcept;

    Combo(const Combo&) = delete;
    Combo& operator=(const Combo&) = delete;

    bool update(Config config);
    void render(const Context& ctx) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura
