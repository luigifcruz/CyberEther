#ifndef JETSTREAM_RENDER_SAKURA_NODE_TEXT_AREA_HH
#define JETSTREAM_RENDER_SAKURA_NODE_TEXT_AREA_HH

#include <jetstream/render/sakura/component.hh>

#include <functional>
#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct NodeTextArea {
    struct Config {
        std::string id;
        std::string value;
        std::function<void(const std::string&)> onChange;
    };

    NodeTextArea();
    ~NodeTextArea();

    NodeTextArea(NodeTextArea&&) noexcept;
    NodeTextArea& operator=(NodeTextArea&&) noexcept;

    NodeTextArea(const NodeTextArea&) = delete;
    NodeTextArea& operator=(const NodeTextArea&) = delete;

    bool update(Config config);
    void render(const Context& ctx) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_NODE_TEXT_AREA_HH
