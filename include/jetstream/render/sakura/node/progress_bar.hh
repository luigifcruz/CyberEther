#pragma once

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/context.hh>
#include <jetstream/types.hh>

#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct NodeProgressBar : public Component {
    struct Config {
        std::string id;
        F32 value = 0.0f;
        std::string overlay;
        Extent2D<F32> size = {-1.0f, 0.0f};
    };

    NodeProgressBar();
    ~NodeProgressBar();

    NodeProgressBar(NodeProgressBar&&) noexcept;
    NodeProgressBar& operator=(NodeProgressBar&&) noexcept;

    NodeProgressBar(const NodeProgressBar&) = delete;
    NodeProgressBar& operator=(const NodeProgressBar&) = delete;

    bool update(Config config);
    void render(const Context& ctx) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura
