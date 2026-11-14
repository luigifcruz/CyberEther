#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_LAYOUT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_LAYOUT_HH

#include "../model/messages.hh"
#include "../model/state.hh"
#include "../model/tiles.hh"

#include "jetstream/logger.hh"
#include "jetstream/flowgraph_metadata.hh"
#include "jetstream/flowgraph_view.hh"
#include "tools/text.hh"

#include <algorithm>
#include <functional>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

namespace Jetstream {

struct LayoutActions {
    using Filter = std::tuple<MailSetFlowgraphLayout,
                              MailSetSurfaceDetached,
                              MailSetSurfaceConfigCollapsed,
                              MailSetConfigDetached,
                              MailRemoveLayoutBlock,
                              MailRenameLayoutBlock,
                              MailOpenDocumentation,
                              MailCloseDocumentation,
                              MailCloseFlowgraphDescription,
                              MailSetFlowgraphEnvironmentVisible,
                              MailSetFlowgraphEnvironmentSearch>;
    using FlowgraphState = DefaultCompositorState::FlowgraphState;

    FlowgraphState& state;

    explicit LayoutActions(FlowgraphState& state);
    static void ReplacePane(FlowgraphState& state, const std::string& flowgraphId, const std::string& pane,
                            const TileMeta& replacement);
    static void KeepPanes(FlowgraphState& state, const std::string& flowgraphId,
                          const std::function<bool(const std::string&)>& keep);
    static std::optional<TileMeta> TakePanes(FlowgraphState& state, const std::string& flowgraphId,
                                             const std::function<bool(const std::string&)>& take);
    void restoreFromMetadata();
    Result prepareSave(const std::string& flowgraphId);
    Result handle(const MailSetFlowgraphLayout& msg);
    Result handle(const MailSetSurfaceDetached& msg);
    Result handle(const MailSetSurfaceConfigCollapsed& msg);
    Result handle(const MailSetConfigDetached& msg);
    Result handle(const MailRemoveLayoutBlock& msg);
    Result handle(const MailRenameLayoutBlock& msg);
    Result handle(const MailOpenDocumentation& msg);
    Result handle(const MailCloseDocumentation& msg);
    Result handle(const MailSetFlowgraphEnvironmentVisible& msg);
    Result handle(const MailCloseFlowgraphDescription& msg);
    Result handle(const MailSetFlowgraphEnvironmentSearch& msg);

 private:
    static std::optional<TileMeta> WithoutPane(const TileMeta& tiles, const std::string& pane);
    static std::optional<TileMeta> Persistent(const TileMeta& tiles);
    Result persistLayout(const std::string& flowgraphId);
    Result trackLayout(const std::string& flowgraphId);
    Result setSurfaceDetachedMeta(Flowgraph& flowgraph, const std::string& block, const std::string& surface,
                                  const bool detached);
    Result setConfigDetachedMeta(Flowgraph& flowgraph, const std::string& block, const bool detached);
    Result editLayout(const std::string& flowgraphId,
                      const std::function<std::optional<TileMeta>(const TileMeta&)>& edit);
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_LAYOUT_HH
