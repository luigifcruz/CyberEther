#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_CONFIG_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_CONFIG_HH

#include "../../model/messages.hh"
#include "../../model/state.hh"
#include "../../views/flowgraph/editor/config/types.hh"

#include <jetstream/flowgraph_view.hh>
#include <jetstream/runtime_context.hh>

#include <functional>
#include <string>
#include <utility>
#include <vector>

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

inline void ApplyLiveMarkdown(std::vector<FlowgraphConfigFieldConfig>& fields,
                              const DefaultCompositorState::FlowgraphState& flowgraphs,
                              const std::string& flowgraphId,
                              const std::string& blockName) {
    for (auto& field : fields) {
        if (Parser::Get<std::string>(field.format, "type") != "markdown") {
            continue;
        }
        const auto preview = flowgraphs.liveMarkdown.find(
            DefaultCompositorState::FlowgraphState::LiveMarkdownKey(flowgraphId, blockName, field.name));
        if (preview != flowgraphs.liveMarkdown.end()) {
            field.preview = preview->second;
        }
    }
}

inline std::vector<FlowgraphConfigFieldConfig> BuildFlowgraphDetachedConfigFields(
    const std::function<void(Mail&&)>& enqueue,
    const DefaultCompositorState::FlowgraphState& flowgraphs,
    const std::string& flowgraphId,
    const std::string& blockName,
    const std::string& viewId,
    const Flowgraph::View::BlockData& block) {
    auto fields = BuildFlowgraphConfigFields(viewId, block);
    ApplyLiveMarkdown(fields, flowgraphs, flowgraphId, blockName);
    for (auto& field : fields) {
        field.onApply = [enqueue, flowgraphId, blockName](Parser::Map patch, const bool silent) {
            enqueue(MailReconfigureBlock{flowgraphId,
                                         blockName,
                                         std::move(patch),
                                         silent});
        };
        field.onError = [enqueue](const Result result, const std::string& message) {
            enqueue(MailNotifyResult{.result = result, .message = message});
        };
        field.onBrowsePath = [enqueue](const bool save,
                                       std::vector<std::string> extensions,
                                       std::function<void(std::string)> onSelect) {
            enqueue(MailBrowseConfigPath{
                .path = "",
                .save = save,
                .extensions = std::move(extensions),
                .onSelect = std::move(onSelect),
            });
        };
    }
    return fields;
}

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_CONFIG_HH
