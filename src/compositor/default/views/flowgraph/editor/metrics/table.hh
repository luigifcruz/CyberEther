#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_METRICS_TABLE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_METRICS_TABLE_HH

#include "error.hh"
#include "types.hh"

#include "../field_tooltip.hh"

namespace Jetstream {

struct FlowgraphMetricTable {
    using Config = FlowgraphMetricConfig;

    void update(Config config) {
        this->config = std::move(config);
        parseValue();
        help.update(this->config.id, this->config.help);
        frame.update({
            .id = this->config.id,
            .label = this->config.label,
            .tooltip = help.overlay(),
            .background = false,
            .content = errorText.empty() ? static_cast<Sakura::Component*>(&table) : &error.component(),
        });
        table.update({
            .id = this->config.id + "Table",
            .columns = columns,
            .rows = rows,
            .showHeaders = true,
        });
        error.update({
            .id = this->config.id + "Error",
            .str = errorText,
        });
    }

    Sakura::Component& component() {
        return frame;
    }

 private:
    void parseValue() {
        if (!config.value.has_value()) {
            errorText = "No metric.";
            columns.clear();
            rows.clear();
            parsedValue.reset();
            return;
        }

        const auto* value = std::any_cast<Parser::Map>(&config.value);
        if (!value) {
            errorText = "Invalid metric type.";
            columns.clear();
            rows.clear();
            parsedValue.reset();
            return;
        }

        if (parsedValue && *value == *parsedValue) {
            return;
        }

        parsedValue = *value;
        errorText.clear();
        columns.clear();
        rows.clear();
        if (Parser::Deserialize(*value, "columns", columns) != Result::SUCCESS ||
            Parser::Deserialize(*value, "rows", rows) != Result::SUCCESS) {
            errorText = "Invalid table data.";
            return;
        }
        if (std::any_of(rows.begin(), rows.end(), [&](const auto& row) { return row.size() != columns.size(); })) {
            errorText = "Invalid table row size.";
            return;
        }

        if (columns.empty()) {
            errorText = "No data.";
        }
    }

    Config config;
    std::optional<Parser::Map> parsedValue;
    std::vector<std::string> columns;
    std::vector<std::vector<std::string>> rows;
    std::string errorText;
    FlowgraphFieldTooltip help;
    Sakura::NodeField frame;
    FlowgraphMetricError error;
    Sakura::NodeTable table;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_METRICS_TABLE_HH
