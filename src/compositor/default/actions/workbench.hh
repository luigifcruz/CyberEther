#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_WORKBENCH_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_WORKBENCH_HH

#include "persist.hh"
#include "../model/callbacks.hh"
#include "../model/messages.hh"
#include "../model/state.hh"
#include "render/sakura/themes.hh"

#include "jetstream/settings.hh"
#include "jetstream/platform.hh"

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
                              MailFocusStartTab>;

    DefaultCompositorState& state;
    DefaultCompositorCallbacks& callbacks;

    WorkbenchActions(DefaultCompositorState& state,
                     DefaultCompositorCallbacks& callbacks) :
        state(state),
        callbacks(callbacks) {}

    Result handle(const MailOpenStartTab&) {
        if (!state.interface.startTab) {
            state.interface.startTab = true;
            state.interface.startTabBaseline = state.flowgraph.items.size();
        }
        ++state.interface.startTabRequests;
        state.interface.startTabFocused = true;
        return Result::SUCCESS;
    }

    Result handle(const MailCloseStartTab&) {
        state.interface.startTab = false;
        state.interface.startTabFocused = false;
        return Result::SUCCESS;
    }

    Result handle(const MailFocusStartTab&) {
        state.interface.startTabFocused = state.interface.startTab;
        return Result::SUCCESS;
    }

    void settleStartTab() {
        auto& ui = state.interface;
        if (!ui.startTab) {
            return;
        }
        const U64 count = state.flowgraph.items.size();
        if (count > ui.startTabBaseline) {
            ui.startTab = false;
            ui.startTabFocused = false;
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
        if (msg.settings.has_value()) {
            state.settings.section = msg.settings.value();
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
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_WORKBENCH_HH
