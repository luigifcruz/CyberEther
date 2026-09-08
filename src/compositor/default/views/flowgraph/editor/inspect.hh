#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_INSPECT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_INSPECT_HH

#include "jetstream/parser.hh"
#include "jetstream/render/sakura/base.hh"

#include <functional>
#include <string>
#include <utility>

namespace Jetstream {

struct FlowgraphNodeInspector {
    struct Config {
        std::string id;
        std::string name;
        Parser::Map value;
        std::function<void(Parser::Map)> onApply;
        std::function<void()> onClose;
    };

    void update(Config config) {
        this->config = std::move(config);

        window.update({
            .id = this->config.id,
            .title = "Inspect & Edit (" + this->config.name + ")",
            .size = {640.0f, 520.0f},
            .padding = Extent2D<F32>{4.0f, 4.0f},
            .borderSize = 0.5f,
            .onClose = this->config.onClose,
        });
        editor.update({
            .id = this->config.id + ":yaml",
            .value = buffer,
            .status = status,
            .size = {0.0f, 0.0f},
            .statusTone = statusTone,
            .onChange = [this](std::string value) {
                buffer = std::move(value);
                status = canApply()
                    ? "Edited. Press Cmd+Enter to apply."
                    : "Block configuration YAML.";
                statusTone = Sakura::Retained::CodeEditor::StatusTone::Info;
            },
            .onSubmit = [this](std::string value) {
                buffer = std::move(value);
                apply();
            },
        });
    }

    void open() {
        std::string yaml;
        if (Parser::YamlEncode(config.value, yaml) != Result::SUCCESS) {
            buffer.clear();
            appliedBuffer.clear();
            status = "Unable to serialize this block configuration.";
            statusTone = Sakura::Retained::CodeEditor::StatusTone::Error;
            return;
        }

        buffer = std::move(yaml);
        appliedBuffer = buffer;
        status = "Block configuration YAML.";
        statusTone = Sakura::Retained::CodeEditor::StatusTone::Info;
    }

    void render(const Sakura::Context& ctx) {
        window.render(ctx, [this](const Sakura::Context& ctx) {
            editor.render(ctx);
        });
    }

 private:
    bool canApply() const {
        return buffer != appliedBuffer;
    }

    void apply() {
        if (!canApply() || !config.onApply) {
            return;
        }

        Parser::Map value;
        if (Parser::YamlDecode(buffer, value) != Result::SUCCESS) {
            status = "Invalid YAML. Fix the document before applying changes.";
            statusTone = Sakura::Retained::CodeEditor::StatusTone::Error;
            return;
        }

        appliedBuffer = buffer;
        status = "Changes applied.";
        statusTone = Sakura::Retained::CodeEditor::StatusTone::Success;
        config.onApply(std::move(value));
    }

    Config config;
    std::string buffer;
    std::string appliedBuffer;
    std::string status;
    Sakura::Retained::CodeEditor::StatusTone statusTone =
        Sakura::Retained::CodeEditor::StatusTone::Info;
    Sakura::Window window;
    Sakura::Retained::CodeEditor editor;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_INSPECT_HH
