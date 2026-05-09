#pragma once

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/context.hh>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct NodeCombo : public Component {
    struct Config {
        std::string id;
        std::vector<std::string> options;
        std::string value;
        std::function<void(const std::string&)> onChange;
    };

    NodeCombo();
    ~NodeCombo();

    NodeCombo(NodeCombo&&) noexcept;
    NodeCombo& operator=(NodeCombo&&) noexcept;

    NodeCombo(const NodeCombo&) = delete;
    NodeCombo& operator=(const NodeCombo&) = delete;

    bool update(Config config);
    void render(const Context& ctx) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura
