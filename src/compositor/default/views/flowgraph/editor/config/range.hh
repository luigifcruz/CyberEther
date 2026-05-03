#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_RANGE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_RANGE_HH

#include "types.hh"

#include <cmath>

// TODO: Cleanup parsing.

namespace Jetstream {

struct FlowgraphConfigRangeField : public Sakura::Component {
    using Config = FlowgraphConfigFieldConfig;

    void update(Config config) {
        this->config = std::move(config);
        if (this->config.format != parsedFormat) {
            parseFormat();
        }
        if (this->config.encoded != parsedEncoded) {
            value = minValue;
            if (!this->config.encoded.empty()) {
                if (integer) {
                    U64 intValue = 0;
                    Parser::StringToTyped(this->config.encoded, intValue);
                    value = static_cast<F32>(intValue);
                } else {
                    Parser::StringToTyped(this->config.encoded, value);
                }
            }
            parsedEncoded = this->config.encoded;
        }
        frame.update({
            .id = this->config.id,
            .label = this->config.label,
            .help = this->config.help,
        });
        input.update({
            .id = this->config.id + "Input",
            .min = minValue,
            .max = maxValue,
            .value = value,
            .integer = integer,
            .unit = unit,
            .onChange = [this](F32 nextValue) {
                auto values = this->config.values;
                if (integer) {
                    values[this->config.name] = static_cast<U64>(std::round(nextValue));
                } else {
                    values[this->config.name] = nextValue;
                }
                if (this->config.onApply) {
                    this->config.onApply(std::move(values), true);
                }
            },
        });
    }

    void render(const Sakura::Context& ctx) const {
        frame.render(ctx, [this](const Sakura::Context& ctx) {
            input.render(ctx);
        });
    }

 private:
    void parseFormat() {
        parsedFormat = config.format;
        const auto parts = Parser::SplitString(config.format, ":");
        unit = (parts.size() > 3) ? parts[3] : "";
        const std::string type = (parts.size() > 4) ? parts[4] : "";
        integer = type == "int";
        if (integer) {
            minValue = static_cast<F32>((parts.size() > 1 && !parts[1].empty()) ? std::stoull(parts[1]) : 0);
            maxValue = static_cast<F32>((parts.size() > 2 && !parts[2].empty()) ? std::stoull(parts[2]) : 100);
        } else {
            minValue = (parts.size() > 1 && !parts[1].empty()) ? std::stof(parts[1]) : 0.0f;
            maxValue = (parts.size() > 2 && !parts[2].empty()) ? std::stof(parts[2]) : 1.0f;
        }
    }

    Config config;
    std::string parsedFormat;
    std::string parsedEncoded;
    std::string unit;
    F32 minValue = 0.0f;
    F32 maxValue = 1.0f;
    F32 value = 0.0f;
    bool integer = false;
    Sakura::NodeField frame;
    Sakura::NodeRangeInput input;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_RANGE_HH
