#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_MODALS_INFO_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_MODALS_INFO_HH

#include "target.hh"

#include "../../../model/messages.hh"
#include "../../../views/flowgraph/modals/info.hh"

#include <functional>
#include <optional>
#include <string>
#include <utility>

namespace Jetstream {

struct FlowgraphInfoModalPresenter {
    const PresenterContext& context;

    explicit FlowgraphInfoModalPresenter(const PresenterContext& context) : context(context) {}

    std::optional<FlowgraphInfoView::Config> build() const {
        const auto targetFlowgraph = BuildTargetFlowgraphId(context);
        if (!targetFlowgraph.has_value() || !context.state.flowgraph.items.contains(targetFlowgraph.value())) {
            return std::nullopt;
        }

        const auto enqueue = context.callbacks.enqueueMail;
        const std::string flowgraphId = targetFlowgraph.value();
        const auto& flowgraph = context.state.flowgraph.items.at(flowgraphId);
        return FlowgraphInfoView::Config{
            .flowgraphId = flowgraphId,
            .title = flowgraph->title(),
            .summary = flowgraph->summary(),
            .author = flowgraph->author(),
            .license = flowgraph->license(),
            .description = flowgraph->description(),
            .path = flowgraph->path(),
            .onTitleChange = [enqueue, flowgraphId](const std::string& value) {
                enqueue(MailSetFlowgraphInfo{.flowgraph = flowgraphId, .title = value});
            },
            .onSummaryChange = [enqueue, flowgraphId](const std::string& value) {
                enqueue(MailSetFlowgraphInfo{.flowgraph = flowgraphId, .summary = value});
            },
            .onAuthorChange = [enqueue, flowgraphId](const std::string& value) {
                enqueue(MailSetFlowgraphInfo{.flowgraph = flowgraphId, .author = value});
            },
            .onLicenseChange = [enqueue, flowgraphId](const std::string& value) {
                enqueue(MailSetFlowgraphInfo{.flowgraph = flowgraphId, .license = value});
            },
            .onDescriptionChange = [enqueue, flowgraphId](const std::string& value) {
                enqueue(MailSetFlowgraphInfo{.flowgraph = flowgraphId, .description = value});
            },
            .onBrowse = [enqueue](const std::string& currentPath, std::function<void(std::string)> onSelect) {
                enqueue(MailBrowseConfigPath{
                    .path = currentPath,
                    .save = true,
                    .extensions = {"yaml", "yml"},
                    .onSelect = std::move(onSelect),
                });
            },
            .onSave = [enqueue, flowgraphId](const std::string& filename) {
                enqueue(MailSaveFlowgraphPath{.flowgraph = flowgraphId, .path = filename});
            },
        };
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_MODALS_INFO_HH
