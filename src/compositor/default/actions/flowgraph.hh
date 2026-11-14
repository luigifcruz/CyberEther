#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_FLOWGRAPH_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_FLOWGRAPH_HH

#include "persist.hh"
#include "workbench.hh"
#include "../support/history.hh"
#include "../model/callbacks.hh"
#include "../support/clipboard.hh"
#include "../support/history.hh"
#include "../model/messages.hh"
#include "../model/state.hh"
#include "layout.hh"

#include "jetstream/block.hh"
#include "jetstream/flowgraph.hh"
#include "jetstream/flowgraph_metadata.hh"
#include "jetstream/flowgraph_view.hh"
#include "jetstream/instance.hh"
#include "jetstream/platform.hh"
#include "jetstream/registry.hh"
#include "tools/naming.hh"
#include "tools/text.hh"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <ranges>
#include <regex>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Jetstream {

struct FlowgraphActions {
    using Filter = std::tuple<MailNewFlowgraph,
                              MailOpenFlowgraph,
                              MailOpenFlowgraphPath,
                              MailToggleFavoriteFlowgraph,
                              MailForgetRecentFlowgraph,
                              MailRememberFlowgraph,
                              MailDropFlowgraphPath,
                              MailOpenFlowgraphBlob,
                              MailOpenFlowgraphExample,
                              MailFocusFlowgraph,
                              MailSetFlowgraphPaused,
                              MailSaveFlowgraph,
                              MailCloseFlowgraph,
                              MailCancelCloseFlowgraph,
                              MailSaveFlowgraphPath,
                              MailSetFlowgraphInfo,
                              MailCreateBlock,
                              MailOpenRenameBlock,
                              MailRenameBlock,
                              MailDeleteBlock,
                              MailToggleBlockFrozen,
                              MailReloadBlock,
                              MailChangeBlockDevice,
                              MailConnectBlock,
                              MailDisconnectBlock,
                              MailReconfigureBlock,
                              MailCopyBlock,
                              MailPasteBlock,
                              MailDuplicateBlock,
                              MailSelectBlocks,
                              MailSetNodeMeta,
                              MailSetEditorMeta,
                              MailSetConfigCollapsed,
                              MailSetMetricsCollapsed,
                              MailSurfaceInput,
                              MailResizeSurface>;

    DefaultCompositorState& state;
    DefaultCompositorCallbacks& callbacks;

    FlowgraphActions(DefaultCompositorState& state,
                     DefaultCompositorCallbacks& callbacks);
    Result handle(const MailNewFlowgraph&);
    Result handle(const MailOpenFlowgraph&);
    Result handle(const MailOpenFlowgraphPath& msg);
    Result handle(const MailDropFlowgraphPath& msg);
    Result handle(const MailOpenFlowgraphBlob& msg);
    std::optional<std::string> openBlob(std::vector<char> blob);
    Result handle(const MailOpenFlowgraphExample& msg);
    Result handle(const MailFocusFlowgraph& msg);
    Result handle(const MailSaveFlowgraph& msg);
    static std::string TitleFromFile(const std::string& path);
    static void Remember(DefaultCompositorState::InterfaceState& ui, const std::string& path, std::string title,
                         U64 accessed);
    static void ToggleFavorite(DefaultCompositorState::InterfaceState& ui, const std::string& path);
    Result handle(const MailRememberFlowgraph& msg);
    Result handle(const MailToggleFavoriteFlowgraph& msg);
    Result handle(const MailForgetRecentFlowgraph& msg);
    Result handle(const MailCloseFlowgraph& msg);
    Result handle(const MailCancelCloseFlowgraph&);
    Result handle(const MailSaveFlowgraphPath& msg);
    Result handle(const MailSetFlowgraphPaused& msg);
    Result handle(const MailSetFlowgraphInfo& msg);
    Result handle(const MailCreateBlock& msg);
    Result handle(const MailOpenRenameBlock& msg);
    Result handle(const MailRenameBlock& msg);
    Result handle(const MailDeleteBlock& msg);
    Result handle(const MailToggleBlockFrozen& msg);
    Result handle(const MailReloadBlock& msg);
    Result handle(const MailChangeBlockDevice& msg);
    Result handle(const MailConnectBlock& msg);
    Result handle(const MailDisconnectBlock& msg);
    Result handle(const MailReconfigureBlock& msg);
    Result handle(const MailCopyBlock& msg);
    Result handle(const MailPasteBlock& msg);
    Result handle(const MailDuplicateBlock& msg);
    Result handle(const MailSelectBlocks& msg);
    Result handle(const MailSetNodeMeta& msg);
    Result handle(const MailSetEditorMeta& msg);
    Result handle(const MailSetConfigCollapsed& msg);
    Result handle(const MailSetMetricsCollapsed& msg);
    Result handle(const MailSurfaceInput& msg);
    Result handle(const MailResizeSurface& msg);
    static std::string GenerateRandomFlowgraphName();

 private:
    std::shared_ptr<std::string> pendingCopy;

    void markUntracked(const std::string& flowgraph);
    static std::string ReconfigureMergeKey(const std::string& block, const Parser::Map& config);
    Sakura::Callback<void(Result)> selectOnComplete(const std::string& flowgraphId,
                                                    std::shared_ptr<std::vector<std::string>> created);
    Sakura::Callback<void(Result)> mailOnSuccess(Mail mail) const;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_FLOWGRAPH_HH
