#pragma once

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/context.hh>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct TabBar : public Component {
    using Child = std::function<void(const Context&)>;
    using Children = std::vector<Child>;

    struct Config {
        std::string id;
        std::vector<std::string> labels;
    };

    TabBar();
    ~TabBar();

    TabBar(TabBar&&) noexcept;
    TabBar& operator=(TabBar&&) noexcept;

    TabBar(const TabBar&) = delete;
    TabBar& operator=(const TabBar&) = delete;

    bool update(Config config);
    void render(const Context& ctx, Children children) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura
