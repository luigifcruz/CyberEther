#ifndef JETSTREAM_COMPOSITOR_HH
#define JETSTREAM_COMPOSITOR_HH

#include <tuple>
#include <stack>
#include <future>
#include <chrono>  
#include <memory>
#include <vector>
#include <optional>
#include <filesystem>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>

#include "jetstream/state.hh"
#include "jetstream/types.hh"
#include "jetstream/module.hh"
#include "jetstream/bundle.hh"
#include "jetstream/parser.hh"
#include "jetstream/interface.hh"
#include "jetstream/compute/base.hh"

namespace Jetstream {

class Instance;

class JETSTREAM_API Compositor {
 public:
    Compositor(Instance& instance)
         : instance(instance),
           graphSpatiallyOrganized(false),
           rightClickMenuEnabled(false),
           sourceEditorEnabled(false),
           moduleStoreEnabled(true),
           infoPanelEnabled(true),
           flowgraphEnabled(true),
           globalModalContentId(0),
           nodeContextMenuNodeId(0) {
        stacks["Graph"] = {true, 0};
        JST_CHECK_THROW(refreshState());
    };

    Compositor& showStore(const bool& enabled) {
        moduleStoreEnabled = enabled;
        return *this;
    }

    Compositor& showFlowgraph(const bool& enabled) {
        flowgraphEnabled = enabled;
        return *this;
    }

    Result addModule(const Locale& locale, const std::shared_ptr<BlockState>& block);
    Result removeModule(const Locale& locale);
    Result destroy();

    Result draw();
    Result processInteractions();

 private:
    Instance& instance;

    typedef std::pair<std::string, Device> CreateModuleMail;
    typedef std::pair<Locale, Locale> LinkMail;
    typedef std::pair<Locale, Locale> UnlinkMail;
    typedef Locale DeleteModuleMail;
    typedef std::pair<Locale, std::string> RenameModuleMail;
    typedef std::pair<Locale, Device> ChangeModuleBackendMail;
    typedef std::pair<Locale, std::tuple<std::string, std::string, std::string>> ChangeModuleDataTypeMail;
    typedef std::pair<Locale, bool> ToggleModuleMail;

    typedef U64 LinkId;
    typedef U64 PinId;
    typedef U64 NodeId;

    struct NodeState {
        std::shared_ptr<BlockState> block;
        NodeId id;
        U64 clusterLevel;
        std::unordered_map<PinId, Locale> inputs;
        std::unordered_map<PinId, Locale> outputs;
        std::unordered_set<NodeId> edges;
    };

    void lock();
    void unlock();

    Result refreshState();
    Result updateAutoLayoutState();

    Result drawStatic();
    Result drawGraph();

    I32 nodeDragId;
    bool graphSpatiallyOrganized;
    bool rightClickMenuEnabled;
    bool sourceEditorEnabled;
    bool moduleStoreEnabled;
    bool infoPanelEnabled;
    bool flowgraphEnabled;
    U64 globalModalContentId;
    I32 nodeContextMenuNodeId;

    std::atomic_flag interfaceHalt{false};

    std::future<Result> openFlowgraphAsyncTask;

    std::unordered_map<Locale, NodeState, Locale::Hasher> nodeStates;
    std::unordered_map<Locale, std::vector<Locale>, Locale::Hasher> outputInputCache;
    std::vector<std::vector<std::vector<NodeId>>> nodeTopology;
    std::unordered_map<std::string, std::pair<bool, ImGuiID>> stacks;

    std::unordered_map<LinkId, std::pair<Locale, Locale>> linkLocaleMap;
    std::unordered_map<Locale, PinId, Locale::Hasher> inputLocalePinMap;
    std::unordered_map<Locale, PinId, Locale::Hasher> outputLocalePinMap;
    std::unordered_map<PinId, Locale> pinLocaleMap;
    std::unordered_map<NodeId, Locale> nodeLocaleMap;

    CreateModuleMail createModuleStagingMailbox;
    
    std::optional<LinkMail> linkMailbox;
    std::optional<UnlinkMail> unlinkMailbox;
    std::optional<CreateModuleMail> createModuleMailbox;
    std::optional<DeleteModuleMail> deleteModuleMailbox;
    std::optional<RenameModuleMail> renameModuleMailbox;
    std::optional<ChangeModuleBackendMail> changeModuleBackendMailbox;
    std::optional<ChangeModuleDataTypeMail> changeModuleDataTypeMailbox;
    std::optional<ToggleModuleMail> toggleModuleMailbox;
    std::optional<bool> resetFlowgraphMailbox;
    std::optional<bool> closeFlowgraphMailbox;
    std::optional<const char*> openFlowgraphUrlMailbox;
    std::optional<const char*> openFlowgraphPathMailbox;
    std::optional<const char*> openFlowgraphBlobMailbox;
    std::optional<bool> saveFlowgraphMailbox;
    std::optional<bool> newFlowgraphMailbox;

    ImGuiID mainNodeId;

    static const U32 CpuColor              = IM_COL32(224, 146,   0, 255);
    static const U32 CpuColorSelected      = IM_COL32(184, 119,   0, 255);
    static const U32 CudaColor             = IM_COL32(118, 201,   3, 255);
    static const U32 CudaColorSelected     = IM_COL32( 95, 161,   2, 255);
    static const U32 MetalColor            = IM_COL32( 98,  60, 234, 255);
    static const U32 MetalColorSelected    = IM_COL32( 76,  33, 232, 255);
    static const U32 VulkanColor           = IM_COL32(238,  27,  52, 255);
    static const U32 VulkanColorSelected   = IM_COL32(209,  16,  38, 255);
    static const U32 WebGPUColor           = IM_COL32( 59, 165, 147, 255);
    static const U32 WebGPUColorSelected   = IM_COL32( 49, 135, 121, 255);
    static const U32 DisabledColor         = IM_COL32( 75,  75,  75, 255);
    static const U32 DisabledColorSelected = IM_COL32( 75,  75,  75, 255);
    static const U32 DefaultColor          = IM_COL32(255, 255, 255, 255);
};

}  // namespace Jetstream

#endif
