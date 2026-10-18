#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_CONSOLE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_CONSOLE_HH

#include "types.hh"

namespace Jetstream {

struct FlowgraphConfigPythonConsoleField {
    using Config = FlowgraphConfigFieldConfig;

    FlowgraphNodeHeightSpec heightSpec() const {
        return {
            .policy = FlowgraphNodeHeightPolicy::FillRemaining,
            .minimum = 100.0f,
        };
    }

    Result update(Config config) {
        this->config = std::move(config);
        loaded = true;
        const auto fileKey = Parser::Get<std::string>(this->config.format, "file");
        if (!fileKey.empty()) {
            std::string path;
            if (Parser::Deserialize(this->config.values, fileKey, path) != Result::SUCCESS) {
                path.clear();
            }
            loaded = !path.empty();
        }
        frame.update({
            .id = this->config.id,
            .label = this->config.label,
            .title = false,
            .background = false,
            .content = &console,
        });
        if (!loaded) {
            console.update({
                .id = this->config.id + "Console",
                .status = "No file loaded.",
                .emptyText = "Choose a Python file to run.",
            });
        } else {
            console.update({
                .id = this->config.id + "Console",
                .output = this->config.consoleOutput,
                .status = this->config.status.empty() ? "Not running." : this->config.status,
                .statusTone = this->config.statusTone,
            });
        }
        return Result::SUCCESS;
    }

    Sakura::Component& component() {
        return frame;
    }

 private:
    Config config;
    bool loaded = false;
    Sakura::NodeField frame;
    Sakura::Console console;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_CONSOLE_HH
