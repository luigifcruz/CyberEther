#include <jetstream/render/sakura/components/table.hh>

#include "../helpers.hh"

namespace Jetstream::Sakura {

struct Table::Impl {
    Config config;
    std::string tableId;
    bool overflowing = false;

    bool scrollY() const {
        return config.size.y != 0.0f || (config.maxHeight > 0.0f && overflowing);
    }

    ImGuiTableFlags flags() const {
        ImGuiTableFlags flags = ImGuiTableFlags_SizingStretchProp |
                                ImGuiTableFlags_Borders |
                                ImGuiTableFlags_RowBg;
        if (scrollY()) {
            flags |= ImGuiTableFlags_ScrollY;
        }
        return flags;
    }

    void setupColumns(const Context& ctx) const {
        for (U64 column = 0; column < config.columns.size(); ++column) {
            if (column < config.fixedColumnWidths.size() && config.fixedColumnWidths[column] > 0.0f) {
                ImGui::TableSetupColumn(config.columns[column].c_str(),
                                        ImGuiTableColumnFlags_WidthFixed,
                                        Scale(ctx, config.fixedColumnWidths[column]));
            } else {
                ImGui::TableSetupColumn(config.columns[column].c_str(), ImGuiTableColumnFlags_WidthStretch);
            }
        }
    }
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

    Rows rows;
    rows.reserve(config.rows.size());
    for (const auto& row : config.rows) {
        Row cells;
        cells.reserve(row.size());
        for (const auto& cell : row) {
            cells.push_back([cell, wrapped = config.wrapped](const Context&) {
                if (wrapped) {
                    ImGui::TextWrapped("%s", cell.c_str());
                } else {
                    ImGui::TextUnformatted(cell.c_str());
                }
            });
        }
        rows.push_back(std::move(cells));
    }

    render(ctx, std::move(rows));
}

void Table::render(const Context& ctx, Rows rows) const {
    const auto& config = this->impl->config;
    if (config.columns.empty()) {
        return;
    }

    const bool capped = config.size.y == 0.0f && config.maxHeight > 0.0f;
    const F32 maxHeight = Scale(ctx, config.maxHeight);
    ImVec2 size = Private::ToImVec2(Scale(ctx, config.size));
    if (capped && this->impl->overflowing) {
        size.y = maxHeight;
    }
    if (!ImGui::BeginTable(this->impl->tableId.c_str(),
                            config.columns.size(),
                            this->impl->flags(),
                            size)) {
        return;
    }
    this->impl->setupColumns(ctx);
    if (config.showHeaders) {
        ImGui::TableHeadersRow();
    }
    for (const auto& row : rows) {
        ImGui::TableNextRow();
        for (U64 column = 0; column < config.columns.size(); ++column) {
            ImGui::TableSetColumnIndex(static_cast<int>(column));
            if (column < row.size() && row[column]) {
                row[column](ctx);
            }
        }
    }
    const bool innerOverflow = ImGui::GetScrollMaxY() > 0.0f;
    ImGui::EndTable();

    if (!capped) {
        return;
    }
    if (this->impl->overflowing) {
        this->impl->overflowing = innerOverflow;
    } else {
        this->impl->overflowing = ImGui::GetItemRectSize().y > maxHeight + Scale(ctx, 8.0f);
    }
}

}  // namespace Jetstream::Sakura
