#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_WORKBENCH_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_WORKBENCH_HH

#include "persist.hh"
#include "../model/callbacks.hh"
#include "../model/messages.hh"
#include "../model/state.hh"
#include "render/sakura/themes.hh"

#include "jetstream/settings.hh"
#include "jetstream/platform.hh"

#include <optional>
#include <string>
#include <tuple>

namespace Jetstream {

struct WorkbenchActions {
    using Filter = std::tuple<MailApplyTheme,
                              MailOpenModal,
                              MailCloseModal,
                              MailNotify,
                              MailNotifyResult,
                              MailOpenUrl,
                              MailCopyText,
                              MailSetFlowgraphMetadataVisible,
                              MailSetFlowgraphMetadataSearch,
                              MailSetFlowgraphEnvironmentVisible,
                              MailSetFlowgraphEnvironmentSearch,
                              MailQuit,
                              MailOpenStartTab,
                              MailCloseStartTab,
                              MailFocusStartTab,
                              MailOpenSettings,
                              MailCloseSettings,
                              MailFocusSettings>;

    DefaultCompositorState& state;
    DefaultCompositorCallbacks& callbacks;

    WorkbenchActions(DefaultCompositorState& state,
                     DefaultCompositorCallbacks& callbacks) :
        state(state),
        callbacks(callbacks) {}

    static void Focus(DefaultCompositorState& state, std::optional<WorkspaceTab> tab) {
        state.interface.focusedTab = tab;
    }

    static void Open(DefaultCompositorState& state, WorkspaceTab tab) {
        auto& ui = state.interface;
        switch (tab) {
            case WorkspaceTab::Start:
                if (!ui.startTab) {
                    ui.startTab = true;
                    ui.startTabBaseline = state.flowgraph.items.size();
                }
                ++ui.startTabRequests;
                break;
            case WorkspaceTab::Settings:
                ui.settingsTab = true;
                ++ui.settingsTabRequests;
                break;
        }
        Focus(state, tab);
    }

    static void Close(DefaultCompositorState& state, WorkspaceTab tab) {
        auto& ui = state.interface;
        switch (tab) {
            case WorkspaceTab::Start:
                ui.startTab = false;
                break;
            case WorkspaceTab::Settings:
                ui.settingsTab = false;
                state.settings.registryPage = RegistryPage::Overview;
                state.settings.registryPlugin.clear();
                state.settings.intelligencePage = IntelligencePage::Overview;
                break;
        }
        if (ui.focusedTab == tab && !state.tabShown(tab)) {
            Focus(state, std::nullopt);
        }
    }

    Result handle(const MailOpenStartTab&) {
        Open(state, WorkspaceTab::Start);
        return Result::SUCCESS;
    }

    Result handle(const MailCloseStartTab&) {
        Close(state, WorkspaceTab::Start);
        return Result::SUCCESS;
    }

    Result handle(const MailFocusStartTab&) {
        focusShown(WorkspaceTab::Start);
        return Result::SUCCESS;
    }

    Result handle(const MailOpenSettings& msg) {
        state.settings.section = msg.section;
        state.settings.registryPage = RegistryPage::Overview;
        state.settings.registryPlugin.clear();
        state.settings.intelligencePage = IntelligencePage::Overview;
        Open(state, WorkspaceTab::Settings);
        return Result::SUCCESS;
    }

    Result handle(const MailCloseSettings&) {
        Close(state, WorkspaceTab::Settings);
        return Result::SUCCESS;
    }

    Result handle(const MailFocusSettings&) {
        focusShown(WorkspaceTab::Settings);
        return Result::SUCCESS;
    }

    void settlePendingFocus() {
        auto& ui = state.interface;
        if (!ui.pendingFocusedFlowgraph || !state.flowgraph.items.contains(*ui.pendingFocusedFlowgraph)) {
            return;
        }
        ui.focusedFlowgraph = ui.pendingFocusedFlowgraph;
        ui.pendingFocusedFlowgraph.reset();
        Focus(state, std::nullopt);
    }

    void reconcileFlowgraphs() {
        if (state.flowgraph.items.empty()) {
            state.interface.focusedFlowgraph.reset();
            state.interface.flowgraphMetadataVisible = false;
            state.interface.flowgraphMetadataSearch.clear();
            state.interface.flowgraphEnvironmentVisible = false;
            state.interface.flowgraphEnvironmentSearch.clear();
        } else if (state.interface.focusedFlowgraph.has_value() &&
                   !state.flowgraph.items.contains(state.interface.focusedFlowgraph.value())) {
            state.interface.focusedFlowgraph.reset();
            state.interface.flowgraphMetadataVisible = false;
            state.interface.flowgraphMetadataSearch.clear();
            state.interface.flowgraphEnvironmentVisible = false;
            state.interface.flowgraphEnvironmentSearch.clear();
        }

        if (!state.interface.focusedFlowgraph.has_value() &&
            !state.interface.pendingFocusedFlowgraph.has_value() &&
            state.flowgraph.items.size() == 1) {
            state.interface.focusedFlowgraph = state.flowgraph.items.begin()->first;
        }
    }

    void reconcileModal() {
        if (!state.modal.content.has_value()) {
            state.modal.flowgraph.reset();
            return;
        }

        if (state.modal.content == ModalContent::FlowgraphInfo ||
            state.modal.content == ModalContent::FlowgraphClose) {
            const std::optional<std::string> targetFlowgraph = state.modal.flowgraph.has_value()
                ? state.modal.flowgraph
                : state.interface.focusedFlowgraph;
            if (!targetFlowgraph.has_value() || !state.flowgraph.items.contains(targetFlowgraph.value())) {
                state.modal.content.reset();
                state.modal.flowgraph.reset();
            }
            return;
        }

        if (state.modal.content == ModalContent::RenameBlock) {
            const std::optional<std::string> targetFlowgraph = state.modal.flowgraph.has_value()
                ? state.modal.flowgraph
                : state.interface.focusedFlowgraph;
            if (!targetFlowgraph.has_value() ||
                !state.flowgraph.items.contains(targetFlowgraph.value()) ||
                !state.modal.renameBlockOldName.has_value() ||
                !state.flowgraph.items.at(targetFlowgraph.value())->view().has(state.modal.renameBlockOldName.value())) {
                state.modal.content.reset();
                state.modal.flowgraph.reset();
                state.modal.renameBlockOldName.reset();
            }
        }
    }

    void settleStartTab() {
        auto& ui = state.interface;
        if (!ui.startTab) {
            return;
        }
        const U64 count = state.flowgraph.items.size();
        if (count > ui.startTabBaseline) {
            Close(state, WorkspaceTab::Start);
            return;
        }
        ui.startTabBaseline = count;
    }

    Result handle(const MailApplyTheme& msg) {
        if (!Sakura::Themes.contains(msg.themeKey)) {
            callbacks.notify(Sakura::ToastType::Error,
                             5000,
                             "Cannot apply theme because it doesn't exist.");
            return Result::SUCCESS;
        }

        state.sakura.themeKey = msg.themeKey;
        state.sakura.colorMap = Sakura::Themes.at(state.sakura.themeKey);
        state.sakura.runtime.update({
            .palette = &state.sakura.colorMap,
            .render = state.system.render.get(),
        });
        RethemeSurfaces(state.flowgraph.surfaces, state.sakura.colorMap);
        if (state.system.render) {
            state.system.render->setDarkAppearance(state.sakura.dark());
        }

        JST_CHECK(PersistSettings([&](Settings& settings) {
            settings.interface.themeKey = state.sakura.themeKey;
        }));

        return Result::SUCCESS;
    }

    Result handle(const MailOpenModal& msg) {
        state.modal.content = msg.content;
        state.modal.flowgraph = msg.flowgraph;
        if (msg.content == ModalContent::Dependencies) {
            state.modal.dependencyRequest = state.runtime.dependencyRequest;
        } else {
            state.modal.dependencyRequest.reset();
        }
        if (msg.content != ModalContent::RenameBlock) {
            state.modal.renameBlockOldName.reset();
        }
        return Result::SUCCESS;
    }

    Result handle(const MailCloseModal&) {
        state.modal.content.reset();
        state.modal.flowgraph.reset();
        state.modal.renameBlockOldName.reset();
        state.modal.dependencyRequest.reset();
        return Result::SUCCESS;
    }

    Result handle(const MailNotify& msg) {
        callbacks.notify(msg.type, msg.durationMs, msg.message);
        return Result::SUCCESS;
    }

    Result handle(const MailNotifyResult& msg) {
        callbacks.notifyResult(msg.result, msg.message);
        return Result::SUCCESS;
    }

    Result handle(const MailOpenUrl& msg) {
        const Result result = Platform::OpenUrl(msg.url);
        if (msg.notifyResult) {
            callbacks.notifyResult(result, "");
        }
        return Result::SUCCESS;
    }

    Result handle(const MailCopyText& msg) {
        callbacks.setClipboardText(msg.value);
        const std::string notification = msg.label + " copied to clipboard.";
        callbacks.notify(Sakura::ToastType::Info, 3000, notification);
        return Result::SUCCESS;
    }

    Result handle(const MailSetFlowgraphMetadataVisible& msg) {
        state.interface.flowgraphMetadataVisible = msg.value;
        if (!msg.value) {
            state.interface.flowgraphMetadataSearch.clear();
        }
        return Result::SUCCESS;
    }

    Result handle(const MailSetFlowgraphMetadataSearch& msg) {
        state.interface.flowgraphMetadataSearch = msg.value;
        return Result::SUCCESS;
    }

    Result handle(const MailSetFlowgraphEnvironmentVisible& msg) {
        state.interface.flowgraphEnvironmentVisible = msg.value;
        if (!msg.value) {
            state.interface.flowgraphEnvironmentSearch.clear();
        }
        return Result::SUCCESS;
    }

    Result handle(const MailSetFlowgraphEnvironmentSearch& msg) {
        state.interface.flowgraphEnvironmentSearch = msg.value;
        return Result::SUCCESS;
    }

    Result handle(const MailQuit&) {
        return state.system.instance->requestStop();
    }

 private:
    void focusShown(WorkspaceTab tab) {
        if (state.tabShown(tab)) {
            Focus(state, tab);
        }
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_WORKBENCH_HH
