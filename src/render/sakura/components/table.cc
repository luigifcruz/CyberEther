#include <jetstream/render/sakura/components/table.hh>

#include "../helpers.hh"

#include <functional>
#include <string>

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
        if (config.resizable) {
            flags |= ImGuiTableFlags_Resizable;
        }
        return flags;
    }

    void renderCell(const std::string& cell) const {
        if (config.wrapped) {
            ImGui::TextWrapped("%s", cell.c_str());
        } else {
            ImGui::TextUnformatted(cell.c_str());
        }
    }

    void renderNodes(const Context& ctx, const Nodes& nodes) const {
        for (const auto& node : nodes) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);

            ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_SpanAllColumns;
            if (node.children.empty()) {
                flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
            } else if (node.open) {
                flags |= ImGuiTreeNodeFlags_DefaultOpen;
            }
            const std::string& identity = node.id.empty() ? node.label : node.id;
            ImGui::PushID(static_cast<int>(std::hash<std::string>{}(identity)));
            const bool expanded = ImGui::TreeNodeEx("node", flags, "%s", node.label.c_str());

            if (node.secondary) {
                ImGui::PushStyleColor(ImGuiCol_Text, Private::ImColor(ctx, "text_secondary"));
            }
            for (U64 column = 1; column < config.columns.size(); ++column) {
                ImGui::TableSetColumnIndex(static_cast<int>(column));
                if (column - 1 < node.cells.size()) {
                    renderCell(node.cells[column - 1]);
                }
            }
            if (node.secondary) {
                ImGui::PopStyleColor();
            }

            if (!node.children.empty() && expanded) {
                renderNodes(ctx, node.children);
                ImGui::TreePop();
            }
            ImGui::PopID();
        }
    }

    bool begin(const Context& ctx) const;
    void end(const Context& ctx);

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

bool Table::Impl::begin(const Context& ctx) const {
    if (config.columns.empty()) {
        return false;
    }

    const bool capped = config.size.y == 0.0f && config.maxHeight > 0.0f;
    ImVec2 size = Private::ToImVec2(Scale(ctx, config.size));
    if (capped && overflowing) {
        size.y = Scale(ctx, config.maxHeight);
    }
    if (!ImGui::BeginTable(tableId.c_str(), config.columns.size(), flags(), size)) {
        return false;
    }
    setupColumns(ctx);
    if (config.showHeaders) {
        ImGui::TableHeadersRow();
    }
    return true;
}

void Table::Impl::end(const Context& ctx) {
    const bool innerOverflow = ImGui::GetScrollMaxY() > 0.0f;
    ImGui::EndTable();

    const bool capped = config.size.y == 0.0f && config.maxHeight > 0.0f;
    if (!capped) {
        return;
    }
    const F32 maxHeight = Scale(ctx, config.maxHeight);
    if (overflowing) {
        overflowing = innerOverflow;
    } else {
        overflowing = ImGui::GetItemRectSize().y > maxHeight + Scale(ctx, 8.0f);
    }
}

void Table::render(const Context& ctx, Rows rows) const {
    const auto& config = this->impl->config;
    if (!this->impl->begin(ctx)) {
        return;
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
    this->impl->end(ctx);
}

void Table::render(const Context& ctx, const Nodes& nodes) const {
    if (!this->impl->begin(ctx)) {
        return;
    }
    this->impl->renderNodes(ctx, nodes);
    this->impl->end(ctx);
}

}  // namespace Jetstream::Sakura
