#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_CONFIG_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_CONFIG_HH

#include "../../model/messages.hh"
#include "../../model/state.hh"
#include "../../views/flowgraph/editor/config/types.hh"
#include "../../views/flowgraph/editor/metrics/types.hh"

#include <jetstream/flowgraph_view.hh>
#include <jetstream/runtime_context.hh>

#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

inline std::vector<FlowgraphMetricConfig> BuildFlowgraphMetrics(
    const std::string& viewId, const Flowgraph::View::BlockData& block) {
    std::vector<FlowgraphMetricConfig> metrics;
    for (const auto& metric : block.metrics) {
        if (Parser::Get<std::string>(metric.format, "visibility") == "internal") {
            continue;
        }

        metrics.push_back({
            .id = jst::fmt::format("{}:metric:{}", viewId, metric.name),
            .label = metric.label,
            .help = metric.help,
            .format = metric.format,
            .value = metric.value,
        });
    }
    return metrics;
}

inline std::vector<FlowgraphConfigFieldConfig> BuildFlowgraphConfigFields(
    const std::string& viewId, const Flowgraph::View::BlockData& block) {
    std::vector<FlowgraphConfigFieldConfig> fields;
    fields.reserve(block.interfaceConfigs.size());
    for (const auto& entry : block.interfaceConfigs) {
        FlowgraphConfigFieldConfig field{
            .id = jst::fmt::format("{}:config:{}", viewId, entry.name),
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
                    field.statusTone = diagnostic->healthy ? Sakura::StatusTone::Success
                                                          : Sakura::StatusTone::Error;
                    field.consoleOutput = diagnostic->console;
                    field.consoleVisible = !field.consoleOutput.empty();
                }
                break;
            }
            if (field.status.empty() && !block.diagnostic.empty()) {
                field.status = "Not running.";
                field.statusTone = Sakura::StatusTone::Error;
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
            DefaultCompositorState::FlowgraphState::FieldKey(flowgraphId, blockName, field.name));
        if (preview != flowgraphs.liveMarkdown.end()) {
            field.preview = preview->second;
        }
    }
}

inline void ApplyConfigDrafts(std::vector<FlowgraphConfigFieldConfig>& fields,
                              const Sakura::Callback<void(Mail&&)>& enqueue,
                              const DefaultCompositorState::FlowgraphState& flowgraphs,
                              const std::string& flowgraphId,
                              const std::string& blockName) {
    for (auto& field : fields) {
        const auto type = Parser::Get<std::string>(field.format, "type");
        if (type != "python" && type != "multiline" && type != "markdown") {
            continue;
        }
        const auto draft = flowgraphs.configDrafts.find(
            DefaultCompositorState::FlowgraphState::FieldKey(flowgraphId, blockName, field.name));
        if (draft != flowgraphs.configDrafts.end()) {
            std::string committed;
            if (Parser::Deserialize(field.values, field.name, committed) == Result::SUCCESS &&
                committed == draft->second.base) {
                field.draft = draft->second.text;
            }
        }
        field.onDraft = [enqueue, flowgraphId, blockName, name = field.name](std::optional<std::string> text,
                                                                              std::string base) {
            enqueue(MailSetConfigDraft{
                .flowgraph = flowgraphId,
                .block = blockName,
                .field = name,
                .base = std::move(base),
                .text = std::move(text),
            });
        };
    }
}

inline std::vector<FlowgraphConfigFieldConfig> BuildFlowgraphDetachedConfigFields(
    const Sakura::Callback<void(Mail&&)>& enqueue,
    const DefaultCompositorState::FlowgraphState& flowgraphs,
    const std::string& flowgraphId,
    const std::string& blockName,
    const std::string& viewId,
    const Flowgraph::View::BlockData& block) {
    auto fields = BuildFlowgraphConfigFields(viewId, block);
    ApplyLiveMarkdown(fields, flowgraphs, flowgraphId, blockName);
    ApplyConfigDrafts(fields, enqueue, flowgraphs, flowgraphId, blockName);
    for (auto& field : fields) {
        field.onApply = [enqueue, flowgraphId, blockName](Parser::Map patch,
                                                          const bool silent,
                                                          Sakura::Callback<void(Result)> onComplete) {
            enqueue(MailReconfigureBlock{flowgraphId,
                                         blockName,
                                         std::move(patch),
                                         silent,
                                         std::move(onComplete)});
        };
        field.onError = [enqueue](const Result result, const std::string& message) {
            enqueue(MailNotifyResult{.result = result, .message = message});
        };
        field.onBrowsePath = [enqueue](const bool save,
                                       std::vector<std::string> extensions,
                                       Sakura::Callback<void(std::string)> onSelect) {
            enqueue(MailBrowseConfigPath{
                .path = "",
                .save = save,
                .extensions = std::move(extensions),
                .onSelect = std::move(onSelect),
            });
        };
        field.onOpenUrl = [enqueue](const std::string& url) {
            enqueue(MailOpenUrl{.url = url, .notifyResult = true});
        };
    }
    return fields;
}

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_CONFIG_HH
