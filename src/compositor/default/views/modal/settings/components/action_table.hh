#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_COMPONENTS_ACTION_TABLE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_COMPONENTS_ACTION_TABLE_HH

#include "jetstream/render/sakura/base.hh"

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct ActionTable : public Sakura::Table {
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
        std::vector<std::string> columns;
        std::vector<F32> fixedColumnWidths;
        std::string empty;
        std::vector<Row> rows;
    };

    void update(Config config) {
        emptyText.update({
            .str = std::move(config.empty),
            .tone = Sakura::Text::Tone::Disabled,
        });

        while (rows.size() < config.rows.size()) {
            rows.push_back(std::make_unique<RowCells>());
        }
        std::vector<std::vector<Sakura::Component*>> cellRows;
        if (config.rows.empty()) {
            cellRows.push_back({&emptyText});
        }
        for (U64 i = 0; i < config.rows.size(); ++i) {
            auto& source = config.rows[i];
            auto& row = *rows[i];
            while (row.texts.size() < source.cells.size()) {
                row.texts.push_back(std::make_unique<Sakura::Text>());
            }
            while (row.buttons.size() < source.actions.size()) {
                row.buttons.push_back(std::make_unique<Sakura::Button>());
            }
            std::vector<Sakura::Component*> cells;
            for (U64 j = 0; j < source.cells.size(); ++j) {
                row.texts[j]->update({
                    .str = std::move(source.cells[j].str),
                    .tone = source.cells[j].tone,
                    .wrapped = source.cells[j].wrapped,
                });
                cells.push_back(row.texts[j].get());
            }
            if (!source.actions.empty()) {
                std::vector<Sakura::Stack::Item> buttons;
                for (U64 j = 0; j < source.actions.size(); ++j) {
                    row.buttons[j]->update({
                        .str = std::move(source.actions[j].str),
                        .variant = Sakura::Button::Variant::Text,
                        .onClick = std::move(source.actions[j].onClick),
                    });
                    buttons.push_back({.child = row.buttons[j].get()});
                }
                row.actions.update({
                    .axis = Sakura::Stack::Axis::Horizontal,
                    .items = std::move(buttons),
                });
                cells.push_back(&row.actions);
            }
            cellRows.push_back(std::move(cells));
        }

        Table::update({
            .columns = std::move(config.columns),
            .cellRows = std::move(cellRows),
            .fixedColumnWidths = std::move(config.fixedColumnWidths),
            .wrapped = true,
        });
    }

 private:
    struct RowCells {
        std::vector<std::unique_ptr<Sakura::Text>> texts;
        std::vector<std::unique_ptr<Sakura::Button>> buttons;
        Sakura::Stack actions;
    };

    Sakura::Text emptyText;
    std::vector<std::unique_ptr<RowCells>> rows;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_COMPONENTS_ACTION_TABLE_HH
