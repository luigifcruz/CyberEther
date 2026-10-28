#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_PANE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_PANE_HH

#include "detached_config.hh"
#include "editor/base.hh"
#include "labels.hh"
#include "layout.hh"
#include "surface.hh"

#include "../context.hh"

#include "../../model/messages.hh"
#include "../../views/components/callout.hh"
#include "../../views/flowgraph/pane.hh"

#include "jetstream/flowgraph.hh"
#include "jetstream/flowgraph_view.hh"

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace Jetstream {

struct FlowgraphPanePresenter {
    const PresenterContext& context;
    FlowgraphEditorPresenter editor;
    LayoutPresenter layout;
    FlowgraphDetachedSurfacePresenter surfaces;
    FlowgraphDetachedConfigPresenter configs;

    explicit FlowgraphPanePresenter(const PresenterContext& context) : context(context),
                                                                        editor(context),
                                                                        layout(context),
                                                                        surfaces(context),
                                                                        configs(context) {}

    FlowgraphPane::Config build(const std::string& flowgraphId,
                                const std::shared_ptr<Flowgraph>& flowgraph) const {
        const auto enqueue = context.callbacks.enqueueMail;
        std::string title = MakeFlowgraphTitle(flowgraphId, flowgraph);
        return FlowgraphPane::Config{
            .id = flowgraphId,
            .title = std::move(title),
            .editor = editor.build(flowgraphId, flowgraph),
            .layoutGroup = MakeFlowgraphLayoutGroup(flowgraphId),
            .editorPane = MakeEditorPaneId(flowgraphId),
            .layout = layout.build(flowgraphId),
            .describe = DescribeTile,
            .detachedSurfaces = surfaces.build(flowgraphId, flowgraph),
            .detachedConfigs = configs.build(flowgraphId, flowgraph),
            .documentation = documentation(flowgraphId),
            .empty = flowgraph->view().empty(),
            .showTips = context.state.interface.showTips,
            .dependencyReviewAvailable = hasDependencyReview(flowgraph),
            .dependencyReviewTone = dependencyReviewTone(),
            .dependencyReviewMessage = dependencyReviewMessage(),
            .onFocus = [enqueue, flowgraphId]() {
                enqueue(MailFocusFlowgraph{flowgraphId});
            },
            .onClose = [enqueue, flowgraphId]() {
                enqueue(MailCloseFlowgraph{flowgraphId});
            },
            .onLayout = [enqueue, flowgraphId](std::optional<Sakura::DockTiles> next) {
                std::optional<TileMeta> meta;
                if (next.has_value()) {
                    meta = ToTileMeta(*next, DockToLayoutPanes(flowgraphId));
                }
                enqueue(MailSetFlowgraphLayout{.flowgraph = flowgraphId, .tiles = std::move(meta)});
            },
            .onReviewDependencies = [enqueue, flowgraphId]() {
                enqueue(MailOpenModal{
                    .content = ModalContent::Dependencies,
                    .flowgraph = flowgraphId,
                });
            },
            .onOpenExamples = [enqueue]() {
                enqueue(MailOpenStartTab{});
                enqueue(MailSetWelcomeSection{.section = WelcomeSection::Flowgraphs});
            },
            .onMeasure = [layout = context.callbacks.layout, flowgraphId](FlowgraphEditor::Measured measured) {
                if (!layout) {
                    return;
                }
                Agent::Tools::Layout converted;
                for (auto& [block, node] : measured.nodes) {
                    converted.nodes[block] = {node.bounds, node.size, node.content};
                }
                converted.viewport = Agent::Tools::Layout::Viewport{.visible = measured.viewport};
                layout(flowgraphId, std::move(converted));
            },
        };
    }

 private:
    std::optional<FlowgraphDocumentation::Config> documentation(const std::string& flowgraphId) const {
        const auto entry = context.state.flowgraph.documentation.find(flowgraphId);
        if (entry == context.state.flowgraph.documentation.end()) {
            return std::nullopt;
        }
        const auto enqueue = context.callbacks.enqueueMail;
        const auto& documentation = entry->second;
        return FlowgraphDocumentation::Config{
            .id = kLayoutDocumentationPane,
            .title = MakeDocumentationTileTitle(documentation.block, documentation.title),
            .markdown = documentation.markdown,
            .onClose = [enqueue, flowgraphId]() {
                enqueue(MailCloseDocumentation{flowgraphId});
            },
        };
    }

    Callout::Tone dependencyReviewTone() const {
        switch (context.state.runtime.dependencyRequest.state) {
            case PythonDependencyRequestState::Failed:
                return Callout::Tone::Error;
            case PythonDependencyRequestState::Denied:
                return Callout::Tone::Warning;
            default:
                return Callout::Tone::Info;
        }
    }

    std::string dependencyReviewMessage() const {
        switch (context.state.runtime.dependencyRequest.state) {
            case PythonDependencyRequestState::Installing:
                return "Installing Python dependencies.";
            case PythonDependencyRequestState::Failed:
                return "Python dependency installation failed.";
            case PythonDependencyRequestState::Denied:
                return "Python dependencies are blocked by policy.";
            default:
                return "Python dependencies need approval.";
        }
    }

    bool hasDependencyReview(const std::shared_ptr<Flowgraph>& currentFlowgraph) const {
        const auto& request = context.state.runtime.dependencyRequest;
        if (request.state != PythonDependencyRequestState::ApprovalRequired &&
            request.state != PythonDependencyRequestState::Installing &&
            request.state != PythonDependencyRequestState::Failed &&
            request.state != PythonDependencyRequestState::Denied) {
            return false;
        }

        const auto* currentView = &currentFlowgraph->view();
        return !request.dependencies.empty() &&
               std::ranges::any_of(
                   request.dependencies,
                   [currentView](const auto& dependency) {
                       const auto view = dependency.view.lock();
                       return !view || view.get() == currentView;
                   });
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_PANE_HH
