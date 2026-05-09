#pragma once

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/context.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct Grid : public Component {
    using Child = std::function<void(const Context&)>;
    using Children = std::vector<Child>;

    struct Config {
        std::string id;
        U64 columns = 2;
        Extent2D<F32> size = {0.0f, 0.0f};
        Extent2D<F32> cellPadding = {4.0f, 2.0f};
    };

    Grid();
    ~Grid();

    Grid(Grid&&) noexcept;
    Grid& operator=(Grid&&) noexcept;

    Grid(const Grid&) = delete;
    Grid& operator=(const Grid&) = delete;

    bool update(Config config);
    void render(const Context& ctx, Children children) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura
