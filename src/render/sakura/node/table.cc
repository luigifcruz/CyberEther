#include <jetstream/render/sakura/node/table.hh>

#include <jetstream/render/sakura/table.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct NodeTable::Impl {
    Config config;
    Table table;
};

NodeTable::NodeTable() {
    this->impl = std::make_unique<Impl>();
}

NodeTable::~NodeTable() = default;
NodeTable::NodeTable(NodeTable&&) noexcept = default;
NodeTable& NodeTable::operator=(NodeTable&&) noexcept = default;

bool NodeTable::update(Config config) {
    auto& impl = *this->impl;
    impl.config = std::move(config);
    impl.table.update({
        .id = impl.config.id,
        .columns = impl.config.columns,
        .rows = impl.config.rows,
        .fixedColumnWidths = impl.config.fixedColumnWidths,
        .showHeaders = impl.config.showHeaders,
    });
    return true;
}

void NodeTable::render(const Context& ctx) const {
    ImGui::PushStyleColor(ImGuiCol_TableHeaderBg, Private::ImColor(ctx, "table_header_bg"));
    ImGui::PushStyleColor(ImGuiCol_TableBorderStrong, Private::ImColor(ctx, "table_border_strong"));
    ImGui::PushStyleColor(ImGuiCol_TableBorderLight, Private::ImColor(ctx, "table_border_light"));
    ImGui::PushStyleColor(ImGuiCol_TableRowBg, Private::ImColor(ctx, "table_row_bg"));
    ImGui::PushStyleColor(ImGuiCol_TableRowBgAlt, Private::ImColor(ctx, "table_row_bg_alt"));
    this->impl->table.render(ctx);
    ImGui::PopStyleColor(5);
}

}  // namespace Jetstream::Sakura
