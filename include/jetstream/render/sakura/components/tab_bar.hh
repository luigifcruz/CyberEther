#ifndef JETSTREAM_RENDER_SAKURA_TAB_BAR_HH
#define JETSTREAM_RENDER_SAKURA_TAB_BAR_HH

#include <jetstream/render/sakura/component.hh>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct TabBar {
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

#endif  // JETSTREAM_RENDER_SAKURA_TAB_BAR_HH
