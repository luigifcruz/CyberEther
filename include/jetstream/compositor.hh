#ifndef JETSTREAM_COMPOSITOR_HH
#define JETSTREAM_COMPOSITOR_HH

#include <tuple>
#include <stack>
#include <memory>
#include <vector>
#include <optional>
#include <filesystem>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>

#include "jetstream/types.hh"
#include "jetstream/module.hh"
#include "jetstream/block.hh"
#include "jetstream/parser.hh"
#include "jetstream/block.hh"
#include "jetstream/render/base.hh"
#include "jetstream/compute/base.hh"

namespace Jetstream {

class Instance;

class JETSTREAM_API Compositor {
 public:
    Compositor(Instance& instance);
    ~Compositor();

    Compositor& showStore(const bool& enabled) {
        moduleStoreEnabled = enabled;
        return *this;
    }

    Compositor& showFlowgraph(const bool& enabled) {
        flowgraphEnabled = enabled;
        return *this;
    }

    Result addBlock(const Locale& locale, 
                    const std::shared_ptr<Block>& block, 
                    const Parser::RecordMap& inputMap, 
                    const Parser::RecordMap& outputMap, 
                    const Parser::RecordMap& stateMap,
                    const Block::Fingerprint& fingerprint);
    Result removeBlock(const Locale& locale);
    Result destroy();

    Result draw();
    Result processInteractions();

 private:
    typedef std::pair<std::string, Device> CreateBlockMail;
    typedef std::pair<Locale, Locale> LinkMail;
    typedef std::pair<Locale, Locale> UnlinkMail;
    typedef Locale DeleteBlockMail;
    typedef Locale ReloadBlockMail;
    typedef std::pair<Locale, std::string> RenameBlockMail;
    typedef std::pair<Locale, Device> ChangeBlockBackendMail;
    typedef std::pair<Locale, std::tuple<std::string, std::string>> ChangeBlockDataTypeMail;
    typedef std::pair<Locale, bool> ToggleBlockMail;

    typedef U64 LinkId;
    typedef U64 PinId;
    typedef U64 NodeId;

    struct NodeState {
        std::shared_ptr<Block> block;

        Parser::RecordMap inputMap;
        Parser::RecordMap outputMap;
        Parser::RecordMap stateMap;
        Block::Fingerprint fingerprint;
        std::string title;

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
    Result checkAutoLayoutState();

    Result drawStatic();
    Result drawGraph();

    Instance& instance;

    bool running;
    I32 nodeDragId;
    bool graphSpatiallyOrganized;
    bool rightClickMenuEnabled;
    bool sourceEditorEnabled;
    bool moduleStoreEnabled;
    bool infoPanelEnabled;
    bool flowgraphEnabled;
    bool debugDemoEnabled;
    bool debugLatencyEnabled;
    bool debugEnableTrace;
    U64 globalModalContentId;
    I32 nodeContextMenuNodeId;
    bool benchmarkRunning;
    std::string filenameField;

    std::atomic_flag interfaceHalt{false};

    std::shared_ptr<Render::Texture> primaryBannerTexture;
    std::shared_ptr<Render::Texture> secondaryBannerTexture;
    Result loadImageAsset(const uint8_t* binary_data, 
                          const U64& binary_len, 
                          std::shared_ptr<Render::Texture>& texture);

    std::unordered_map<Locale, NodeState, Locale::Hasher> nodeStates;
    std::unordered_map<Locale, std::vector<Locale>, Locale::Hasher> outputInputCache;
    std::vector<std::vector<std::vector<NodeId>>> nodeTopology;
    std::unordered_map<std::string, std::pair<bool, ImGuiID>> stacks;

    std::unordered_map<LinkId, std::pair<Locale, Locale>> linkLocaleMap;
    std::unordered_map<Locale, PinId, Locale::Hasher> inputLocalePinMap;
    std::unordered_map<Locale, PinId, Locale::Hasher> outputLocalePinMap;
    std::unordered_map<PinId, Locale> pinLocaleMap;
    std::unordered_map<NodeId, Locale> nodeLocaleMap;

    CreateBlockMail createBlockStagingMailbox;
    
    std::optional<LinkMail> linkMailbox;
    std::optional<UnlinkMail> unlinkMailbox;
    std::optional<CreateBlockMail> createBlockMailbox;
    std::optional<DeleteBlockMail> deleteBlockMailbox;
    std::optional<ReloadBlockMail> reloadBlockMailbox;
    std::optional<RenameBlockMail> renameBlockMailbox;
    std::optional<ChangeBlockBackendMail> changeBlockBackendMailbox;
    std::optional<ChangeBlockDataTypeMail> changeBlockDataTypeMailbox;
    std::optional<ToggleBlockMail> toggleBlockMailbox;
    std::optional<bool> resetFlowgraphMailbox;
    std::optional<bool> closeFlowgraphMailbox;
    std::optional<std::string> openFlowgraphUrlMailbox;
    std::optional<std::string> openFlowgraphPathMailbox;
    std::optional<const char*> openFlowgraphBlobMailbox;
    std::optional<bool> updateFlowgraphBlobMailbox;
    std::optional<bool> saveFlowgraphMailbox;
    std::optional<bool> newFlowgraphMailbox;

    ImGuiID mainNodeId;
    bool globalModalToggle;

    Locale renameBlockLocale;
    std::string renameBlockNewId;

    static const U32 CpuColor              = IM_COL32(224, 146,   0, 255);
    static const U32 CpuColorSelected      = IM_COL32(184, 119,   0, 255);
    static const U32 CudaColor             = IM_COL32( 95, 161,   2, 255);
    static const U32 CudaColorSelected     = IM_COL32( 85, 140,   2, 255);
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
