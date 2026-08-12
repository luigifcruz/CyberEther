#ifndef JETSTREAM_RENDER_SAKURA_NODE_FIELD_HH
#define JETSTREAM_RENDER_SAKURA_NODE_FIELD_HH

#include <jetstream/render/sakura/component.hh>

#include <functional>
#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct NodeField {
    using Child = std::function<void(const Context&)>;

    struct Config {
        std::string id;
        std::string label;
        std::string help;
        bool title = true;
        bool background = true;
        bool divider = false;
    };

    NodeField();
    ~NodeField();

    NodeField(NodeField&&) noexcept;
    NodeField& operator=(NodeField&&) noexcept;

    NodeField(const NodeField&) = delete;
    NodeField& operator=(const NodeField&) = delete;

    bool update(Config config);
    void render(const Context& ctx, Child child) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_NODE_FIELD_HH
