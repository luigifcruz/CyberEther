#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_WORKBENCH_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_WORKBENCH_HH

#include "../model/callbacks.hh"
#include "../model/messages.hh"
#include "../model/state.hh"
#include "../themes.hh"

#include "jetstream/render/sakura/sakura.hh"
#include "jetstream/settings.hh"
#include "jetstream/platform.hh"

#include <cstdlib>
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
                              MailQuit>;

    DefaultCompositorState& state;
    DefaultCompositorCallbacks& callbacks;

    WorkbenchActions(DefaultCompositorState& state,
                     DefaultCompositorCallbacks& callbacks) :
        state(state),
        callbacks(callbacks) {}

    Result handle(const MailApplyTheme& msg) {
        if (!themes.contains(msg.themeKey)) {
            Sakura::Notify(Sakura::NotificationType::Error,
                           5000,
                           "Cannot apply theme because it doesn't exist.");
            return Result::SUCCESS;
        }

        state.sakura.themeKey = msg.themeKey;
        state.sakura.colorMap = themes.at(state.sakura.themeKey);
        state.sakura.runtime.update({
            .palette = &state.sakura.colorMap,
            .render = state.system.render.get(),
        });

        Settings settings;
        JST_CHECK(Settings::Get(settings));
        settings.interface.themeKey = state.sakura.themeKey;
        JST_CHECK(Settings::Set(settings));

        return Result::SUCCESS;
    }

    Result handle(const MailOpenModal& msg) {
        state.modal.content = msg.content;
        state.modal.flowgraph = msg.flowgraph;
        if (msg.settings.has_value()) {
            state.settings.section = msg.settings.value();
        }
        if (msg.content != DefaultCompositorState::ModalState::Content::RenameBlock) {
            state.modal.renameBlockOldName.reset();
        }
        return Result::SUCCESS;
    }

    Result handle(const MailCloseModal&) {
        state.modal.content.reset();
        state.modal.flowgraph.reset();
        state.modal.renameBlockOldName.reset();
        return Result::SUCCESS;
    }

    Result handle(const MailNotify& msg) {
        Sakura::Notify(msg.type, msg.durationMs, msg.message);
        return Result::SUCCESS;
    }

    Result handle(const MailNotifyResult& msg) {
        Sakura::NotifyResultClean(msg.result, msg.message);
        return Result::SUCCESS;
    }

    Result handle(const MailOpenUrl& msg) {
        const Result result = Platform::OpenUrl(msg.url);
        if (msg.notifyResult) {
            Sakura::NotifyResultClean(result);
        }
        return Result::SUCCESS;
    }

    Result handle(const MailCopyText& msg) {
        ImGui::SetClipboardText(msg.value.c_str());
        const std::string notification = msg.label + " copied to clipboard.";
        Sakura::Notify(Sakura::NotificationType::Info, 3000, notification);
        return Result::SUCCESS;
    }

    Result handle(const MailQuit&) {
        std::exit(0);
        return Result::SUCCESS;
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_WORKBENCH_HH
