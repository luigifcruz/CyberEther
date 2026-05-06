#pragma once

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/context.hh>
#include <jetstream/types.hh>

#include <memory>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct Table : public Component {
    struct Config {
        std::string id;
        std::vector<std::string> columns;
        std::vector<std::vector<std::string>> rows;
        std::vector<F32> fixedColumnWidths;
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

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura
