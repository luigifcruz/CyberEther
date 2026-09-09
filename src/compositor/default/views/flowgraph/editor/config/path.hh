#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_PATH_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_PATH_HH

#include "types.hh"

namespace Jetstream {

struct FlowgraphConfigPathField {
    using Config = FlowgraphConfigFieldConfig;

    enum class Mode {
        Open,
        Save,
    };

    Result update(Config config) {
        this->config = std::move(config);
        if (this->config.format != parsedFormat) {
            parseFormat();
        }
        value.clear();
        JST_CHECK(Parser::Deserialize(this->config.values, this->config.name, value));
        frame.update({
            .id = this->config.id,
            .label = this->config.label,
            .help = this->config.help,
        });
        input.update({
            .id = this->config.id + "Path",
            .value = value,
            .onChange = [this](const std::string& nextValue) {
                Parser::Map patch;
                patch[this->config.name] = nextValue;
                if (this->config.onApply) {
                    this->config.onApply(std::move(patch), false);
                }
            },
            .onBrowse = [this]() {
                browse();
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
        mode = Parser::Get<std::string>(config.format, "type") == "filesave" ? Mode::Save : Mode::Open;
        extensions = Parser::Get<std::vector<std::string>>(config.format, "extensions");
    }

    void browse() const {
        auto applyPath = [name = config.name,
                          onApply = config.onApply](std::string path) mutable {
            Parser::Map patch;
            patch[name] = std::move(path);
            if (onApply) {
                onApply(std::move(patch), false);
            }
        };

        if (config.onBrowsePath) {
            config.onBrowsePath(mode == Mode::Save, extensions, std::move(applyPath));
            return;
        }
    }

    Config config;
    Parser::Map parsedFormat;
    std::vector<std::string> extensions;
    std::string value;
    Mode mode = Mode::Open;
    Sakura::NodeField frame;
    Sakura::NodePathInput input;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_PATH_HH
