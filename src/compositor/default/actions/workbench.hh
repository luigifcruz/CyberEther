#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_WORKBENCH_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_WORKBENCH_HH

#include "gallery.hh"
#include "persist.hh"
#include "settings.hh"
#include "../support/docs.hh"
#include "../model/callbacks.hh"
#include "../model/messages.hh"
#include "../model/state.hh"
#include "render/sakura/themes.hh"

#include "jetstream/settings.hh"
#include "jetstream/platform.hh"

#include <algorithm>
#include <filesystem>
#include <optional>
#include <string>
#include <system_error>
#include <tuple>
#include <unordered_set>

namespace Jetstream {

struct WorkbenchActions {
    using Filter = std::tuple<MailApplyTheme,
                              MailOpenModal,
                              MailCloseModal,
                              MailNotify,
                              MailNotifyResult,
                              MailOpenUrl,
                              MailCopyText,
                              MailRevealPath,
                              MailQuit,
                              MailOpenStartTab,
                              MailCloseStartTab,
                              MailFocusStartTab,
                              MailSetWelcomeSection,
                              MailSetWelcomeSidebar,
                              MailSetWelcomeQuery,
                              MailSetWelcomeCategory,
                              MailSetWelcomePage,
                              MailOpenSettings,
                              MailCloseSettings,
                              MailFocusSettings,
                              MailOpenDocs,
                              MailCloseDocs,
                              MailSetDocsQuery,
                              MailFocusDocs>;

    DefaultCompositorState& state;
    DefaultCompositorCallbacks& callbacks;

    WorkbenchActions(DefaultCompositorState& state,
                     DefaultCompositorCallbacks& callbacks);
    static void Focus(DefaultCompositorState& state, std::optional<WorkspaceTab> tab);
    static void Open(DefaultCompositorState& state, WorkspaceTab tab);
    static void Close(DefaultCompositorState& state, WorkspaceTab tab);
    static void OpenSettings(DefaultCompositorState& state, SettingsSection section);
    Result handle(const MailOpenStartTab&);
    Result handle(const MailCloseStartTab&);
    Result handle(const MailSetWelcomeSection& msg);
    Result handle(const MailSetWelcomeSidebar& msg);
    Result handle(const MailSetWelcomeQuery& msg);
    Result handle(const MailSetWelcomeCategory& msg);
    Result handle(const MailSetWelcomePage& msg);
    Result handle(const MailFocusStartTab&);
    Result handle(const MailOpenSettings& msg);
    Result handle(const MailCloseSettings&);
    Result handle(const MailFocusSettings&);
    Result handle(const MailOpenDocs& msg);
    Result handle(const MailCloseDocs&);
    Result handle(const MailSetDocsQuery& msg);
    Result handle(const MailFocusDocs&);
    void settlePendingFocus();
    void reconcileFlowgraphs();
    static void Forget(DefaultCompositorState& state, const std::string& flowgraph);
    void reconcileModal();
    void settleStartTab();
    Result handle(const MailApplyTheme& msg);
    Result handle(const MailOpenModal& msg);
    Result handle(const MailCloseModal&);
    Result handle(const MailNotify& msg);
    Result handle(const MailNotifyResult& msg);
    Result handle(const MailOpenUrl& msg);
    Result handle(const MailCopyText& msg);
    Result handle(const MailRevealPath& msg);
    Result handle(const MailQuit&);
    static Result Quit(DefaultCompositorState& state);

 private:
    void focusShown(WorkspaceTab tab);

    std::unordered_set<std::string> knownFlowgraphs;
    bool startTabWasShown = false;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_WORKBENCH_HH
