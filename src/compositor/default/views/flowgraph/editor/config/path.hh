#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_PATH_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_PATH_HH

#include "types.hh"

// TODO: Cleanup parsing.

namespace Jetstream {

struct FlowgraphConfigPathField : public Sakura::Component {
    using Config = FlowgraphConfigFieldConfig;

    enum class Mode {
        Open,
        Save,
    };

    void update(Config config) {
        this->config = std::move(config);
        if (this->config.format != parsedFormat) {
            parseFormat();
        }
        if (this->config.encoded != parsedEncoded) {
            value = this->config.encoded;
            parsedEncoded = this->config.encoded;
        }
        frame.update({
            .id = this->config.id,
            .label = this->config.label,
            .help = this->config.help,
        });
        input.update({
            .id = this->config.id + "Path",
            .value = value,
            .onChange = [this](const std::string& nextValue) {
                auto values = this->config.values;
                values[this->config.name] = nextValue;
                if (this->config.onApply) {
                    this->config.onApply(std::move(values), false);
                }
            },
            .onBrowse = [this]() {
                browse();
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
        extensions.clear();
        const auto parts = Parser::SplitString(config.format, ":");
        mode = parts.empty() || parts[0] != "filesave" ? Mode::Open : Mode::Save;
        if (parts.size() > 1 && !parts[1].empty()) {
            extensions = Parser::SplitString(parts[1], ",");
        }
    }

    void browse() const {
        auto applyPath = [values = config.values,
                          name = config.name,
                          onApply = config.onApply](std::string path) mutable {
            values[name] = std::move(path);
            if (onApply) {
                onApply(std::move(values), false);
            }
        };

        if (config.onBrowsePath) {
            config.onBrowsePath(mode == Mode::Save, extensions, std::move(applyPath));
            return;
        }
    }

    Config config;
    std::string parsedFormat;
    std::string parsedEncoded;
    std::vector<std::string> extensions;
    std::string value;
    Mode mode = Mode::Open;
    Sakura::NodeField frame;
    Sakura::NodePathInput input;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_PATH_HH
