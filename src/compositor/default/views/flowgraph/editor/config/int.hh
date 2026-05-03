#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_INT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_INT_HH

#include "types.hh"

// TODO: Cleanup parsing.

namespace Jetstream {

struct FlowgraphConfigIntField : public Sakura::Component {
    using Config = FlowgraphConfigFieldConfig;

    void update(Config config) {
        this->config = std::move(config);
        if (this->config.format != parsedFormat) {
            const auto parts = Parser::SplitString(this->config.format, ":");
            unit = (parts.size() > 1) ? parts[1] : "";
            parsedFormat = this->config.format;
        }
        if (this->config.encoded != parsedEncoded) {
            value = 0;
            if (!this->config.encoded.empty()) {
                Parser::StringToTyped(this->config.encoded, value);
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
            .value = value,
            .unit = unit,
            .onChange = [this](U64 nextValue) {
                auto values = this->config.values;
                values[this->config.name] = nextValue;
                if (this->config.onApply) {
                    this->config.onApply(std::move(values), false);
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
    Config config;
    std::string parsedFormat;
    std::string parsedEncoded;
    std::string unit;
    U64 value = 0;
    Sakura::NodeField frame;
    Sakura::NodeIntInput input;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_INT_HH
