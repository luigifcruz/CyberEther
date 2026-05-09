#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_DROPDOWN_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_DROPDOWN_HH

#include "types.hh"

// TODO: Cleanup parsing.

namespace Jetstream {

struct FlowgraphConfigDropdownField : public Sakura::Component {
    using Config = FlowgraphConfigFieldConfig;

    void update(Config config) {
        this->config = std::move(config);
        if (this->config.format != parsedFormat) {
            parseFormat();
        }
        if (this->config.encoded != parsedEncoded) {
            currentIndex = 0;
            for (U64 i = 0; i < keys.size(); ++i) {
                if (keys[i] == this->config.encoded) {
                    currentIndex = static_cast<int>(i);
                    break;
                }
            }
            parsedEncoded = this->config.encoded;
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
            .onChange = [this](const std::string& label) {
                for (U64 i = 0; i < labels.size(); ++i) {
                    if (labels[i] == label) {
                        auto values = this->config.values;
                        values[this->config.name] = keys[i];
                        if (this->config.onApply) {
                            this->config.onApply(std::move(values), false);
                        }
                        return;
                    }
                }
            },
        });
    }

    void render(const Sakura::Context& ctx) const {
        frame.render(ctx, [this](const Sakura::Context& ctx) {
            combo.render(ctx);
        });
    }

 private:
    std::string currentLabel() const {
        if (currentIndex < 0 || static_cast<U64>(currentIndex) >= labels.size()) {
            return {};
        }
        return labels[static_cast<U64>(currentIndex)];
    }

    void parseFormat() {
        parsedFormat = config.format;
        keys.clear();
        labels.clear();

        const auto parts = Parser::SplitString(config.format, ":");
        const std::string options = (parts.size() > 1) ? parts[1] : "";
        for (auto token : Parser::SplitString(options, ",")) {
            if (token.empty()) continue;

            const auto open = token.find('(');
            const auto close = token.rfind(')');
            if (open != std::string::npos && close != std::string::npos && close > open) {
                keys.push_back(token.substr(0, open));
                labels.push_back(token.substr(open + 1, close - open - 1));
            } else {
                keys.push_back(token);
                labels.push_back(token);
            }
        }
    }

    Config config;
    std::string parsedFormat;
    std::string parsedEncoded;
    std::vector<std::string> keys;
    std::vector<std::string> labels;
    int currentIndex = 0;
    Sakura::NodeField frame;
    Sakura::NodeCombo combo;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_DROPDOWN_HH
