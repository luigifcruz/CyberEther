#pragma once

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/context.hh>

#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct NodeMarkdown : public Component {
    struct Config {
        std::string id;
        std::string value;
    };

    NodeMarkdown();
    ~NodeMarkdown();

    NodeMarkdown(NodeMarkdown&&) noexcept;
    NodeMarkdown& operator=(NodeMarkdown&&) noexcept;

    NodeMarkdown(const NodeMarkdown&) = delete;
    NodeMarkdown& operator=(const NodeMarkdown&) = delete;

    bool update(Config config);
    void render(const Context& ctx) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura
