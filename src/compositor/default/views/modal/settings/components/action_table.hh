#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_COMPONENTS_ACTION_TABLE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_COMPONENTS_ACTION_TABLE_HH

#include "jetstream/logger.hh"
#include "jetstream/render/sakura/base.hh"

#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct ActionTable {
    struct Cell {
        std::string str;
        Sakura::Text::Tone tone = Sakura::Text::Tone::Primary;
        bool wrapped = true;
    };

    struct Action {
        std::string str;
        std::function<void()> onClick;
    };

    struct Row {
        std::vector<Cell> cells;
        std::vector<Action> actions;
    };

    struct Config {
        std::string id;
        std::vector<std::string> columns;
        std::vector<F32> fixedColumnWidths;
        std::string empty;
        std::vector<Row> rows;
    };

    void update(Config config) {
        table.update({
            .id = config.id,
            .columns = std::move(config.columns),
            .fixedColumnWidths = std::move(config.fixedColumnWidths),
            .wrapped = true,
        });
        emptyText.update({
            .id = config.id + "EmptyText",
            .str = std::move(config.empty),
            .tone = Sakura::Text::Tone::Disabled,
        });

        rows.resize(config.rows.size());
        for (U64 i = 0; i < config.rows.size(); ++i) {
            auto& source = config.rows[i];
            auto& row = rows[i];
            row.texts.resize(source.cells.size());
            for (U64 j = 0; j < source.cells.size(); ++j) {
                row.texts[j].update({
                    .id = jst::fmt::format("{}Cell{}:{}", config.id, i, j),
                    .str = std::move(source.cells[j].str),
                    .tone = source.cells[j].tone,
                    .wrapped = source.cells[j].wrapped,
                });
            }
            row.actions.update({
                .id = jst::fmt::format("{}Actions{}", config.id, i),
                .spacing = 8.0f,
            });
            row.buttons.resize(source.actions.size());
            for (U64 j = 0; j < source.actions.size(); ++j) {
                row.buttons[j].update({
                    .id = jst::fmt::format("{}Action{}:{}", config.id, i, j),
                    .str = std::move(source.actions[j].str),
                    .variant = Sakura::Button::Variant::Text,
                    .onClick = std::move(source.actions[j].onClick),
                });
            }
        }
    }

    void render(const Sakura::Context& ctx) const {
        Sakura::Table::Rows cells;
        if (rows.empty()) {
            Sakura::Table::Row row;
            row.push_back([this](const Sakura::Context& ctx) {
                emptyText.render(ctx);
            });
            cells.push_back(std::move(row));
            table.render(ctx, std::move(cells));
            return;
        }

        cells.reserve(rows.size());
        for (const auto& source : rows) {
            Sakura::Table::Row row;
            for (const auto& text : source.texts) {
                row.push_back([&text](const Sakura::Context& ctx) {
                    text.render(ctx);
                });
            }
            if (!source.buttons.empty()) {
                row.push_back([&source](const Sakura::Context& ctx) {
                    Sakura::HStack::Children buttons;
                    buttons.reserve(source.buttons.size());
                    for (const auto& button : source.buttons) {
                        buttons.push_back([&button](const Sakura::Context& ctx) {
                            button.render(ctx);
                        });
                    }
                    source.actions.render(ctx, std::move(buttons));
                });
            }
            cells.push_back(std::move(row));
        }
        table.render(ctx, std::move(cells));
    }

 private:
    struct RowState {
        std::vector<Sakura::Text> texts;
        Sakura::HStack actions;
        std::vector<Sakura::Button> buttons;
    };

    Sakura::Table table;
    Sakura::Text emptyText;
    std::vector<RowState> rows;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_COMPONENTS_ACTION_TABLE_HH
