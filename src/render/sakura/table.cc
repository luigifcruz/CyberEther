#include <jetstream/render/sakura/table.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct Table::Impl {
    Config config;
    std::string tableId;
};

Table::Table() {
    this->impl = std::make_unique<Impl>();
}

Table::~Table() = default;
Table::Table(Table&&) noexcept = default;
Table& Table::operator=(Table&&) noexcept = default;

bool Table::update(Config config) {
    this->impl->config = std::move(config);
    this->impl->tableId = this->impl->config.id + "###" + this->impl->config.id;
    return true;
}

void Table::render(const Context& ctx) const {
    const auto& config = this->impl->config;

    if (config.columns.empty()) {
        return;
    }

    if (!ImGui::BeginTable(this->impl->tableId.c_str(),
                           config.columns.size(),
                           ImGuiTableFlags_SizingStretchProp |
                           ImGuiTableFlags_Borders |
                           ImGuiTableFlags_RowBg)) {
        return;
    }
    for (U64 column = 0; column < config.columns.size(); ++column) {
        if (column < config.fixedColumnWidths.size() && config.fixedColumnWidths[column] > 0.0f) {
            ImGui::TableSetupColumn(config.columns[column].c_str(),
                                    ImGuiTableColumnFlags_WidthFixed,
                                    Scale(ctx, config.fixedColumnWidths[column]));
        } else {
            ImGui::TableSetupColumn(config.columns[column].c_str(), ImGuiTableColumnFlags_WidthStretch);
        }
    }
    if (config.showHeaders) {
        ImGui::TableHeadersRow();
    }
    for (const auto& row : config.rows) {
        ImGui::TableNextRow();
        for (U64 column = 0; column < row.size() && column < config.columns.size(); ++column) {
            ImGui::TableSetColumnIndex(static_cast<int>(column));
            if (config.wrapped) {
                ImGui::TextWrapped("%s", row[column].c_str());
            } else {
                ImGui::TextUnformatted(row[column].c_str());
            }
        }
    }
    ImGui::EndTable();
}

}  // namespace Jetstream::Sakura
