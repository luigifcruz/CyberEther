#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_DROPDOWN_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_DROPDOWN_HH

#include "types.hh"

namespace Jetstream {

struct FlowgraphConfigDropdownField {
    using Config = FlowgraphConfigFieldConfig;

    Result update(Config config) {
        this->config = std::move(config);
        if (this->config.format != parsedFormat) {
            parseFormat();
        }
        std::string selected;
        if (this->config.values.contains(this->config.name)) {
            JST_CHECK(Parser::TypedToString(this->config.values.at(this->config.name), selected));
        }
        currentIndex = -1;
        for (U64 i = 0; i < values.size(); ++i) {
            if (values[i] == selected) {
                currentIndex = static_cast<int>(i);
                break;
            }
        }
        frame.update({
            .id = this->config.id,
            .label = this->config.label,
            .help = this->config.help,
        });
        combo.update({
            .id = this->config.id + "Combo",
            .options = labels,
            .value = currentLabel(),
            .selectedIndex = currentIndex >= 0 ? std::optional<U64>(currentIndex) : std::nullopt,
            .onSelect = [this](U64 index) {
                Parser::Map patch;
                patch[this->config.name] = values.at(index);
                if (this->config.onApply) {
                    this->config.onApply(std::move(patch), false);
                }
            },
        });
        return Result::SUCCESS;
    }

    void render(const Sakura::Context& ctx) const {
        frame.render(ctx, [this](const Sakura::Context& ctx) {
            combo.render(ctx);
        });
    }

 private:
    std::string currentLabel() const {
        if (currentIndex < 0 || static_cast<U64>(currentIndex) >= labels.size()) {
            return "Selection unavailable";
        }
        return labels[static_cast<U64>(currentIndex)];
    }

    void parseFormat() {
        parsedFormat = config.format;
        values.clear();
        labels.clear();
        for (const auto& option : Parser::Get<std::vector<Parser::Map>>(config.format, "options")) {
            labels.push_back(Parser::Get<std::string>(option, "label"));
            values.push_back(Parser::Get<std::string>(option, "value"));
        }
    }

    Config config;
    Parser::Map parsedFormat;
    std::vector<std::string> values;
    std::vector<std::string> labels;
    int currentIndex = -1;
    Sakura::NodeField frame;
    Sakura::NodeCombo combo;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_DROPDOWN_HH
