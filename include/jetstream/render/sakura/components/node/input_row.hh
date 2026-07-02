#ifndef JETSTREAM_RENDER_SAKURA_NODE_INPUT_ROW_HH
#define JETSTREAM_RENDER_SAKURA_NODE_INPUT_ROW_HH

#include <jetstream/render/sakura/component.hh>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct NodeInputRow {
    struct Combo {
        std::vector<std::string> options;
        std::string value;
        F32 width = 84.0f;
        std::function<void(const std::string&)> onChange;
    };

    struct Config {
        std::string id;
        std::string value;
        std::string hint;
        F32 minInputWidth = 56.0f;
        std::vector<Combo> combos;
        std::function<void(const std::string&)> onChange;
    };

    NodeInputRow();
    ~NodeInputRow();

    NodeInputRow(NodeInputRow&&) noexcept;
    NodeInputRow& operator=(NodeInputRow&&) noexcept;

    NodeInputRow(const NodeInputRow&) = delete;
    NodeInputRow& operator=(const NodeInputRow&) = delete;

    bool update(Config config);
    void render(const Context& ctx) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_NODE_INPUT_ROW_HH
