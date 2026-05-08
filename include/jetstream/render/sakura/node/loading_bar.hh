#pragma once

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/context.hh>
#include <jetstream/types.hh>

#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct NodeLoadingBar : public Component {
    struct Config {
        std::string id;
        F32 height = 4.0f;
    };

    NodeLoadingBar();
    ~NodeLoadingBar();

    NodeLoadingBar(NodeLoadingBar&&) noexcept;
    NodeLoadingBar& operator=(NodeLoadingBar&&) noexcept;

    NodeLoadingBar(const NodeLoadingBar&) = delete;
    NodeLoadingBar& operator=(const NodeLoadingBar&) = delete;

    bool update(Config config);
    void render(const Context& ctx) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura
