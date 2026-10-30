#ifndef JETSTREAM_RENDER_SAKURA_NODE_TABLE_HH
#define JETSTREAM_RENDER_SAKURA_NODE_TABLE_HH

#include <jetstream/render/sakura/component.hh>
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
        bool showHeaders = true;

        bool operator==(const Config&) const = default;
    };

    NodeTable();
    ~NodeTable();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_NODE_TABLE_HH
