#pragma once

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/context.hh>
#include <jetstream/types.hh>

#include <memory>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct NodeTable : public Component {
    struct Config {
        std::string id;
        std::vector<std::string> columns;
        std::vector<std::vector<std::string>> rows;
        std::vector<F32> fixedColumnWidths;
        bool showHeaders = true;
    };

    NodeTable();
    ~NodeTable();

    NodeTable(NodeTable&&) noexcept;
    NodeTable& operator=(NodeTable&&) noexcept;

    NodeTable(const NodeTable&) = delete;
    NodeTable& operator=(const NodeTable&) = delete;

    bool update(Config config);
    void render(const Context& ctx) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura
