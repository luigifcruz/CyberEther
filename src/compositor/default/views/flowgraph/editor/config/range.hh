#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_RANGE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_RANGE_HH

#include "types.hh"

#include <cmath>
#include <limits>

namespace Jetstream {

struct FlowgraphConfigRangeField {
    using Config = FlowgraphConfigFieldConfig;

    Result update(Config config) {
        this->config = std::move(config);
        if (this->config.format != parsedFormat) {
            parseFormat();
        }
        value = minValue;
        JST_CHECK(Parser::Deserialize(this->config.values, this->config.name, value));
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
            .integer = unsignedInteger,
            .unit = unit,
            .onChange = [this](F32 nextValue) {
                const F32 unsignedLimit = std::ldexp(1.0f, std::numeric_limits<U64>::digits);
                if (!std::isfinite(nextValue) ||
                    (unsignedInteger &&
                     (nextValue < 0.0f || std::round(nextValue) >= unsignedLimit))) {
                    if (this->config.onError) {
                        this->config.onError(
                            Result::ERROR,
                            jst::fmt::format("{}: {}",
                                             this->config.label,
                                             unsignedInteger
                                                 ? "Value must be finite, non-negative, and fit in an unsigned 64-bit integer."
                                                 : "Value must be finite."));
                    }
                    return;
                }

                Parser::Map patch;
                if (unsignedInteger) {
                    patch[this->config.name] = static_cast<U64>(std::round(nextValue));
                } else {
                    patch[this->config.name] = nextValue;
                }
                if (this->config.onApply) {
                    this->config.onApply(std::move(patch), true);
                }
            },
        });
        return Result::SUCCESS;
    }

    void render(const Sakura::Context& ctx) const {
        frame.render(ctx, [this](const Sakura::Context& ctx) {
            input.render(ctx);
        });
    }

 private:
    void parseFormat() {
        parsedFormat = config.format;
        unit = Parser::Get<std::string>(config.format, "unit");
        unsignedInteger = Parser::Get<std::string>(config.format, "value_type", "float") == "uint";
        minValue = Parser::Get<F32>(config.format, "min", 0.0f);
        maxValue = Parser::Get<F32>(config.format, "max", unsignedInteger ? 100.0f : 1.0f);
    }

    Config config;
    Parser::Map parsedFormat;
    std::string unit;
    F32 minValue = 0.0f;
    F32 maxValue = 1.0f;
    F32 value = 0.0f;
    bool unsignedInteger = false;
    Sakura::NodeField frame;
    Sakura::NodeRangeInput input;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_RANGE_HH
