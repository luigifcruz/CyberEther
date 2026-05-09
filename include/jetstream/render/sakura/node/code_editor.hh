#pragma once

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/context.hh>

#include <functional>
#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct NodeCodeEditor : public Component {
    struct Config {
        std::string id;
        std::string value;
        bool collapsible = false;
        std::function<void(std::string)> onChange;
    };

    NodeCodeEditor();
    ~NodeCodeEditor();

    NodeCodeEditor(NodeCodeEditor&&) noexcept;
    NodeCodeEditor& operator=(NodeCodeEditor&&) noexcept;

    NodeCodeEditor(const NodeCodeEditor&) = delete;
    NodeCodeEditor& operator=(const NodeCodeEditor&) = delete;

    bool update(Config config);
    void render(const Context& ctx) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura
