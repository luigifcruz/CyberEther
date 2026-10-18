#ifndef JETSTREAM_RENDER_SAKURA_TABLE_HH
#define JETSTREAM_RENDER_SAKURA_TABLE_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct Table : public Component {
    struct Node {
        std::string id;
        std::string label;
        std::vector<std::string> cells;
        std::vector<Node> children;
        bool open = false;
        bool secondary = false;

        bool operator==(const Node&) const = default;
    };
    using Nodes = std::vector<Node>;

    struct Config {
        std::vector<std::string> columns;
        std::vector<std::vector<std::string>> rows;
        std::vector<std::vector<Component*>> cellRows;
        std::vector<Node> nodes;
        std::vector<F32> fixedColumnWidths;
        Extent2D<F32> size = {Fill, 0.0f};
        F32 maxHeight = 0.0f;
        bool showHeaders = true;
        bool wrapped = false;
        bool resizable = false;

        bool operator==(const Config&) const = default;
    };

    Table();
    ~Table();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;
    bool mouseEvent(const MouseEvent& event) override;
    std::optional<SurfaceCursor> cursorAt(const Extent2D<F32>& point) const override;

 private:
    void rebuild();

    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_TABLE_HH
