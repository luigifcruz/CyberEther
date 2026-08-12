#ifndef JETSTREAM_RENDER_SAKURA_TABLE_HH
#define JETSTREAM_RENDER_SAKURA_TABLE_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct Table {
    using Cell = std::function<void(const Context&)>;
    using Row = std::vector<Cell>;
    using Rows = std::vector<Row>;

    struct Config {
        std::string id;
        std::vector<std::string> columns;
        std::vector<std::vector<std::string>> rows;
        std::vector<F32> fixedColumnWidths;
        Extent2D<F32> size = {0.0f, 0.0f};
        bool showHeaders = true;
        bool wrapped = false;
    };

    Table();
    ~Table();

    Table(Table&&) noexcept;
    Table& operator=(Table&&) noexcept;

    Table(const Table&) = delete;
    Table& operator=(const Table&) = delete;

    bool update(Config config);
    void render(const Context& ctx) const;
    void render(const Context& ctx, Rows rows) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_TABLE_HH
