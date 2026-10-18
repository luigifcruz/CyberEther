#ifndef JETSTREAM_RENDER_SAKURA_DETAIL_COMPONENT_HH
#define JETSTREAM_RENDER_SAKURA_DETAIL_COMPONENT_HH

#include <jetstream/render/sakura/component.hh>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct Component::Impl {
    struct LayoutKey {
        F32 pixelRatio = 0.0f;
        F32 fontSize = 0.0f;
        U64 fontRevision = 0;
        Extent2D<F32> translation = {0.0f, 0.0f};
        std::string backdropColorKey;
        std::string backdropTopColorKey;

        static LayoutKey Of(const Context& ctx);
        bool matches(const Context& ctx) const;
    };

    struct MeasureMemo {
        U64 epoch = 0;
        U64 generation = 0;
        Extent2D<F32> available = {0.0f, 0.0f};
        Rect frame;
        LayoutKey key;
        Extent2D<F32> result = {0.0f, 0.0f};
    };

    static constexpr U64 kMeasureMemos = 4;
    static U64 layoutRuns;
    static U64 layoutEpoch;
    static void advance(F64 time);

    struct FocusScope {
        Component* owner = nullptr;
        U64 generation = 0;
        std::vector<std::pair<Component*, std::weak_ptr<bool>>> offers;
    };

    Component* self = nullptr;
    Component* parent = nullptr;
    Canvas* canvas = nullptr;
    std::vector<Component*> children;

    Rect frame;
    Rect clip;
    Extent2D<F32> offset = {0.0f, 0.0f};

    Extent2D<F32> translation() const;
    bool visible = true;
    bool clipsChildren = false;

    bool treeDirty = true;
    bool resourceDirty = true;
    bool paintDirty = true;
    bool orderDirty = false;
    bool subtreeResourceDirty = false;
    bool subtreeTreeDirty = false;
    bool subtreeOrderDirty = false;
    bool layoutDirty = true;
    bool subtreeLayoutDirty = false;
    U64 layoutGeneration = 0;
    U64 laidEpoch = 0;
    std::optional<LayoutKey> layoutKey;
    std::vector<MeasureMemo> measureMemos;
    bool popupOwner = false;
    U64 nextMeasureMemo = 0;
    bool built = false;
    bool placed = false;
    bool prepared = false;
    bool concealed = false;
    U64 childGeneration = 0;
    std::shared_ptr<FocusScope> focusScope;
    std::weak_ptr<FocusScope> focusHandle;
    std::shared_ptr<bool> alive = std::make_shared<bool>(true);
    bool focusOffered = false;

    void invalidate(Component::Dirty dirty);
    void relayout();
    void setFrame(Rect frame);
    void setClip(Rect clip);
    Rect childClip() const;
    void invalidateUnplaced();
    void placeChild(const Context& ctx, Component& child, Rect childFrame, std::optional<Rect> bounds);
    bool laidOut(const Context& ctx, Rect childFrame, Rect childClip, Extent2D<F32> childOffset) const;
    void runLayout(const Context& ctx);
    void settleSubtree();
    Extent2D<F32> measureCached(const Context& ctx, Extent2D<F32> available);
    void invalidateLayoutTree();
    void attachTo(Canvas* canvas);
    void layoutChildren(const Context& ctx);
    Result buildTree(Context& ctx);
    Result paintTree();
    void invalidatePaintTree();
    void concealTree();
    void markAncestors(Component::Dirty dirty);
    bool treeResourceDirty() const;
    bool treeOrderDirty() const;
    bool treeStructureDirty() const;
    Result buildPending(Context& ctx);
    struct Hit {
        const Component* target = nullptr;
        SurfaceCursor cursor = SurfaceCursor::Default;
    };

    std::optional<Hit> hit(Extent2D<F32> point) const;
    Impl& root();
    const Impl& root() const;
    std::shared_ptr<FocusScope> scope();
    bool within(const Component* ancestor) const;
    bool reachable() const;
    void adoptFocus(Component& child);
    Component* focusTarget() const;
    template<typename Handler>
    bool bubble(Handler&& handler);
    bool dispatchKey(const KeyEvent& event);
    bool dispatchText(const TextEvent& event);
    bool dispatchShortcut(const KeyEvent& event);
    U64 focusGeneration() const;
    void clickFocus(std::optional<Extent2D<F32>> point, bool inside, U64 generation);
    bool settleFocus();
    Component* claimantAt(Extent2D<F32> point) const;
    bool configured() const {
        return built || placed || prepared;
    }
    bool deferred() const {
        return !visible && unconfigured();
    }
    bool unconfigured() const;
    bool awaitingLayout() const;
    bool hasUnbuilt() const;
    void markUnbuilt();
    void clearStructureDirty();
    bool isPaintDirty() const;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_DETAIL_COMPONENT_HH
