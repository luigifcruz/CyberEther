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
            .help = this->config.help,
            .title = false,
            .background = false,
        });
        updateConsole();
        return Result::SUCCESS;
    }

    void setAllocatedHeight(std::optional<F32> height) {
        allocatedHeight = height;
        updateConsole();
    }

    void render(const Sakura::Context& ctx) const {
        frame.render(ctx, [this](const Sakura::Context& ctx) {
            console.render(ctx);
        });
    }

 private:
    void updateConsole() {
        if (!loaded) {
            console.update({
                .id = this->config.id + "Console",
                .status = "No file loaded.",
                .emptyText = "Choose a Python file to run.",
                .height = allocatedHeight,
            });
            return;
        }
        console.update({
            .id = this->config.id + "Console",
            .output = this->config.consoleOutput,
            .status = this->config.status.empty() ? "Not running." : this->config.status,
            .statusTone = this->config.statusTone,
            .height = allocatedHeight,
        });
    }

    Config config;
    std::optional<F32> allocatedHeight;
    bool loaded = false;
    Sakura::NodeField frame;
    Sakura::NodeConsole console;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_CONSOLE_HH
