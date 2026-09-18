#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_CONFIG_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_CONFIG_HH

#include "../../views/flowgraph/editor/config/types.hh"

#include <jetstream/flowgraph_view.hh>
#include <jetstream/runtime_context.hh>

namespace Jetstream {

inline std::vector<FlowgraphConfigFieldConfig> BuildFlowgraphConfigFields(
    const std::string& viewId, const Flowgraph::View::BlockData& block) {
    std::vector<FlowgraphConfigFieldConfig> fields;
    fields.reserve(block.interfaceConfigs.size());
    for (const auto& entry : block.interfaceConfigs) {
        FlowgraphConfigFieldConfig field{
            .id = viewId + ":config:" + entry.name,
            .name = entry.name,
            .label = entry.label.empty() ? entry.name : entry.label,
            .help = entry.help,
            .format = entry.format,
            .values = block.config,
        };
        const auto type = Parser::Get<std::string>(entry.format, "type");
        if (type == "python" || type == "python-console") {
            for (const auto& metric : block.metrics) {
                if (Parser::Get<std::string>(metric.format, "type") != "python-diagnostic" ||
                    Parser::Get<std::string>(metric.format, "visibility") != "internal") {
                    continue;
                }
                if (const auto* diagnostic = std::any_cast<Runtime::Context::Diagnostic>(&metric.value)) {
                    field.status = diagnostic->status;
                    field.statusTone = diagnostic->healthy ? Sakura::NodeCodeEditor::StatusTone::Success
                                                          : Sakura::NodeCodeEditor::StatusTone::Error;
                    field.consoleOutput = diagnostic->console;
                    field.consoleVisible = !field.consoleOutput.empty();
                }
                break;
            }
            if (field.status.empty() && !block.diagnostic.empty()) {
                field.status = "Not running.";
                field.statusTone = Sakura::NodeCodeEditor::StatusTone::Error;
                field.consoleOutput = {block.diagnostic};
                field.consoleVisible = true;
            }
        }
        fields.push_back(std::move(field));
    }
    return fields;
}

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_CONFIG_HH
