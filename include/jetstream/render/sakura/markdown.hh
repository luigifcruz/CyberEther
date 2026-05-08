#pragma once

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/context.hh>

#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct Markdown : public Component {
    struct Config {
        std::string id;
        std::string value;
    };

    Markdown();
    ~Markdown();

    Markdown(Markdown&&) noexcept;
    Markdown& operator=(Markdown&&) noexcept;

    Markdown(const Markdown&) = delete;
    Markdown& operator=(const Markdown&) = delete;

    bool update(Config config);
    void render(const Context& ctx) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura
