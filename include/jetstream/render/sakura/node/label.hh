#pragma once

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/context.hh>
#include <jetstream/render/sakura/text.hh>
#include <jetstream/types.hh>

#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct NodeLabel : public Component {
    struct Config {
        std::string id;
        std::string str;
        Text::Font font = Text::Font::Current;
        Text::Tone tone = Text::Tone::Primary;
        Text::Align align = Text::Align::Left;
        bool wrapped = false;
        F32 scale = 1.0f;
    };

    NodeLabel();
    ~NodeLabel();

    NodeLabel(NodeLabel&&) noexcept;
    NodeLabel& operator=(NodeLabel&&) noexcept;

    NodeLabel(const NodeLabel&) = delete;
    NodeLabel& operator=(const NodeLabel&) = delete;

    bool update(Config config);
    void render(const Context& ctx) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura
