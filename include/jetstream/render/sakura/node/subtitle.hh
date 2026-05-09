#pragma once

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/context.hh>
#include <jetstream/types.hh>

#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct NodeSubtitle : public Component {
    struct Config {
        std::string text;
        F32 fontScale = 0.75f;
    };

    NodeSubtitle();
    ~NodeSubtitle();

    NodeSubtitle(NodeSubtitle&&) noexcept;
    NodeSubtitle& operator=(NodeSubtitle&&) noexcept;

    NodeSubtitle(const NodeSubtitle&) = delete;
    NodeSubtitle& operator=(const NodeSubtitle&) = delete;

    bool update(Config config);
    void render(const Context& ctx) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura
