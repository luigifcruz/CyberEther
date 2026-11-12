#include <jetstream/render/sakura/components/dock.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/card.hh>
#include <jetstream/render/sakura/components/label.hh>
#include <jetstream/render/sakura/components/spinner.hh>
#include <jetstream/render/sakura/components/stack.hh>
#include <jetstream/render/sakura/components/text.hh>
#include <jetstream/render/sakura/typography.hh>
#include <jetstream/render/tools/icons_lucide.hh>

#include "../context.hh"
#include "../detail/access.hh"
#include "../detail/dock_drag.hh"
#include "../detail/helpers.hh"
#include "../detail/text_metrics.hh"
#include "../metrics.hh"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <set>
#include <unordered_map>
#include <utility>
#include <variant>

namespace Jetstream::Sakura {

namespace {

using Side = Dock::Side;

constexpr F32 kDragThreshold = 6.0f;
constexpr F32 kPlaceholderSpinner = 18.0f;
constexpr F32 kCardGap = Dock::ContentPadding;
constexpr F32 kColumnWidth = 420.0f;
constexpr F32 kFlexColumnWidth = 480.0f;
constexpr F32 kMinColumnWidth = 240.0f;
constexpr F32 kSizeFloor = 64.0f;
constexpr F32 kSplitMinimum = 48.0f;
constexpr F32 kSplitShare = 0.45f;
constexpr F32 kSizeEpsilon = 1.0e-3f;
constexpr F32 kPlanTolerance = 0.5f;
constexpr F32 kSplitterSlack = 3.0f;
constexpr F32 kHeaderTitleTop = 6.0f;
constexpr F32 kHeaderPadX = 12.0f;
constexpr F32 kHeaderButton = 20.0f;
constexpr F32 kCloseGlyph = 13.0f;
constexpr F32 kHeaderGlyph = 12.0f;
constexpr F32 kStripWidth = 32.0f;
constexpr F32 kGutterBand = 12.0f;
constexpr F32 kStripPad = 12.0f;
constexpr F32 kStripRotation = -90.0f;
constexpr F32 kStripTitleSize = 17.0f;
constexpr F32 kInkNudge = 0.5f;
constexpr F32 kIdleCloseOpacity = 0.35f;
constexpr F32 kUnsavedGlyph = 32.0f;
constexpr F32 kUnsavedDot = kUnsavedGlyph / 6.0f;
constexpr F32 kUnsavedGap = 5.0f;
constexpr F32 kHeaderButtonGap = 2.0f;
constexpr F32 kHeaderIconGap = 6.0f;
constexpr F32 kPillBarHeight = 38.0f;
constexpr F32 kPillGap = 6.0f;
constexpr F32 kPillCloseGap = 6.0f;
constexpr F32 kPillClose = 16.0f;
constexpr F32 kPillCloseMargin = 7.0f;
constexpr F32 kTrackHeight = 32.0f;
constexpr F32 kTrackDrop = 1.0f;
constexpr F32 kTrackInset = 2.0f;
constexpr F32 kTabMaxWidth = 240.0f;
constexpr F32 kTabGap = 4.0f;
constexpr F32 kTabPadX = 10.0f;
constexpr F32 kIconGap = 6.0f;
constexpr F32 kDividerHeight = 15.0f;
constexpr F32 kHeaderTitleSize = 16.0f;
constexpr F32 kFillBorder = 0.10f;
constexpr F32 kFillRim = 0.14f;
constexpr const char* kTabFont = Typography::DisplayFont;
constexpr const char* kHeaderFont = Typography::DisplaySemiboldFont;
constexpr F32 kBarMargin = 6.0f;
constexpr F32 kSplitterLine = 2.0f;
constexpr U64 kBoxCapacity = 128;
constexpr F64 kHideDuration = 0.25;

void CollectLeaves(const DockTiles& tiles, std::vector<std::string>& leaves) {
    if (tiles.children.empty()) {
        if (!tiles.pane.empty()) {
            leaves.push_back(tiles.pane);
        }
        return;
    }
    for (const auto& child : tiles.children) {
        CollectLeaves(child, leaves);
    }
}

std::vector<std::string> Leaves(const DockTiles& tiles) {
    std::vector<std::string> leaves;
    CollectLeaves(tiles, leaves);
    return leaves;
}

bool Contains(const DockTiles& tiles, const std::string& pane) {
    if (tiles.children.empty()) {
        return tiles.pane == pane;
    }
    return std::any_of(tiles.children.begin(), tiles.children.end(), [&](const auto& child) {
        return Contains(child, pane);
    });
}

bool Valid(const DockTiles& tiles) {
    if (tiles.children.empty()) {
        return !tiles.pane.empty() && tiles.sizes.empty();
    }
    if (tiles.sizes.size() != tiles.children.size()) {
        return false;
    }
    for (const F32 size : tiles.sizes) {
        if (!std::isfinite(size) || size < 0.0f) {
            return false;
        }
    }
    return std::all_of(tiles.children.begin(), tiles.children.end(), Valid);
}

DockTiles Leaf(const std::string& pane) {
    return DockTiles{.pane = pane};
}

DockTiles Bare(DockTiles tiles) {
    tiles.minimized = false;
    tiles.maximized = false;
    for (auto& child : tiles.children) {
        child = Bare(std::move(child));
    }
    return tiles;
}

std::vector<F32> Scaled(std::vector<F32> sizes, F32 room) {
    F32 sum = 0.0f;
    for (const F32 size : sizes) {
        sum += size;
    }
    for (auto& size : sizes) {
        size = sum > 0.0f && room > 0.0f ? size * room / sum : 0.0f;
    }
    return sizes;
}

DockTiles Normalize(DockTiles tiles) {
    if (tiles.children.empty()) {
        tiles.sizes.clear();
        return tiles;
    }
    tiles.sizes.resize(tiles.children.size(), 0.0f);
    std::vector<DockTiles> children;
    std::vector<F32> sizes;
    for (U64 index = 0; index < tiles.children.size(); ++index) {
        auto child = Normalize(std::move(tiles.children[index]));
        if (child.children.size() >= 2 && child.vertical == tiles.vertical) {
            const F32 gaps = kCardGap * static_cast<F32>(child.children.size() - 1);
            const auto inner = Scaled(std::move(child.sizes), tiles.sizes[index] - gaps);
            for (U64 slot = 0; slot < child.children.size(); ++slot) {
                children.push_back(std::move(child.children[slot]));
                sizes.push_back(inner[slot]);
            }
            continue;
        }
        children.push_back(std::move(child));
        sizes.push_back(tiles.sizes[index]);
    }
    if (children.size() == 1) {
        return std::move(children.front());
    }
    tiles.children = std::move(children);
    tiles.sizes = std::move(sizes);
    tiles.pane.clear();
    return tiles;
}

std::optional<DockTiles> FilterNode(const DockTiles& tiles, const std::function<bool(const std::string&)>& keep) {
    if (tiles.children.empty()) {
        if (tiles.pane.empty() || !keep(tiles.pane)) {
            return std::nullopt;
        }
        return tiles;
    }
    DockTiles result{.vertical = tiles.vertical};
    for (U64 index = 0; index < tiles.children.size(); ++index) {
        if (auto filtered = FilterNode(tiles.children[index], keep)) {
            result.children.push_back(std::move(*filtered));
            result.sizes.push_back(index < tiles.sizes.size() ? tiles.sizes[index] : 0.0f);
        }
    }
    if (result.children.empty()) {
        return std::nullopt;
    }
    return result;
}

std::optional<DockTiles> Remove(const std::optional<DockTiles>& tiles, const std::string& pane) {
    if (!tiles.has_value()) {
        return std::nullopt;
    }
    return FilterNode(*tiles, [&](const std::string& id) {
        return id != pane;
    });
}

DockTiles* At(DockTiles& tiles, const std::vector<U8>& path) {
    DockTiles* node = &tiles;
    for (const U8 index : path) {
        if (index >= node->children.size()) {
            return nullptr;
        }
        node = &node->children[index];
    }
    return node;
}

std::optional<std::vector<U8>> PathOf(const DockTiles& tiles, const std::string& pane) {
    if (tiles.children.empty()) {
        return tiles.pane == pane ? std::optional<std::vector<U8>>(std::vector<U8>{}) : std::nullopt;
    }
    for (U64 index = 0; index < tiles.children.size(); ++index) {
        if (auto inner = PathOf(tiles.children[index], pane)) {
            inner->insert(inner->begin(), static_cast<U8>(index));
            return inner;
        }
    }
    return std::nullopt;
}

std::vector<F32> FitSizes(std::vector<F32> sizes, F32 room, std::optional<U64> flex, F32 minimum) {
    F32 sum = 0.0f;
    for (const F32 size : sizes) {
        sum += size;
    }
    room = std::max(0.0f, room);
    if (sizes.empty() || sum <= 0.0f || std::abs(sum - room) <= kSizeEpsilon) {
        return sizes;
    }
    if (sum < room && flex.has_value() && *flex < sizes.size()) {
        sizes[*flex] += room - sum;
        return sizes;
    }
    const U64 count = sizes.size();
    std::vector<bool> pinned(count, false);
    F32 free = room;
    F32 flexible = sum;
    if (sum > room) {
        for (bool changed = true; changed;) {
            changed = false;
            free = room;
            flexible = 0.0f;
            for (U64 index = 0; index < count; ++index) {
                if (pinned[index]) {
                    free -= std::min(sizes[index], minimum);
                } else {
                    flexible += sizes[index];
                }
            }
            if (free <= 0.0f || flexible <= 0.0f) {
                break;
            }
            for (U64 index = 0; index < count; ++index) {
                if (!pinned[index] && sizes[index] * free / flexible < std::min(sizes[index], minimum)) {
                    pinned[index] = true;
                    changed = true;
                }
            }
        }
        if (free <= 0.0f || flexible <= 0.0f) {
            pinned.assign(count, false);
            free = room;
            flexible = sum;
        }
    }
    for (U64 index = 0; index < count; ++index) {
        sizes[index] = pinned[index] ? std::min(sizes[index], minimum) : sizes[index] * free / flexible;
    }
    return sizes;
}

F32 Along(const Rect& rect, bool vertical) {
    return vertical ? rect.height : rect.width;
}

F32 StartOf(const Rect& rect, bool vertical) {
    return vertical ? rect.y : rect.x;
}

bool Vertical(Side side) {
    return side == Side::Top || side == Side::Bottom;
}

Rect Slice(const Rect& rect, Side side, F32 thickness) {
    const F32 size = std::min(thickness, Along(rect, Vertical(side)));
    switch (side) {
        case Side::Left:
            return {rect.x, rect.y, size, rect.height};
        case Side::Right:
            return {rect.right() - size, rect.y, size, rect.height};
        case Side::Top:
            return {rect.x, rect.y, rect.width, size};
        case Side::Bottom:
            return {rect.x, rect.bottom() - size, rect.width, size};
    }
    return rect;
}

bool Near(const Rect& a, const Rect& b, F32 tolerance) {
    return std::abs(a.x - b.x) <= tolerance && std::abs(a.y - b.y) <= tolerance &&
           std::abs(a.right() - b.right()) <= tolerance && std::abs(a.bottom() - b.bottom()) <= tolerance;
}

std::optional<Side> EdgeAt(const Rect& rect, const Extent2D<F32>& point) {
    if (rect.width <= 0.0f || rect.height <= 0.0f) {
        return std::nullopt;
    }
    const F32 u = (point.x - rect.x) / rect.width;
    const F32 v = (point.y - rect.y) / rect.height;
    const F32 left = u;
    const F32 right = 1.0f - u;
    const F32 top = v;
    const F32 bottom = 1.0f - v;
    const F32 nearest = std::min({left, right, top, bottom});
    if (nearest == left) {
        return Side::Left;
    }
    if (nearest == right) {
        return Side::Right;
    }
    if (nearest == top) {
        return Side::Top;
    }
    return Side::Bottom;
}

std::string Fit(TextMetrics& metrics, const std::string& text, F32 size, F32 width,
                const char* fontName = Typography::MonoFont) {
    return metrics.ellipsize(fontName, text, size, width);
}

}  // namespace

struct Dock::Impl {
    struct Caption {
        Caption() {
            spine->update({.fontName = kHeaderFont});
        }

        std::string title;
        std::string icon;
        std::unique_ptr<Text> text = std::make_unique<Text>();
        std::unique_ptr<Text> glyph = std::make_unique<Text>();
        std::unique_ptr<Text> close = std::make_unique<Text>();
        std::unique_ptr<Text> marker = std::make_unique<Text>();
        std::unique_ptr<Text> maximize = std::make_unique<Text>();
        std::unique_ptr<Text> minimize = std::make_unique<Text>();
        std::unique_ptr<Label> spine = std::make_unique<Label>();
        std::vector<std::unique_ptr<Text>> actions;
    };

    struct LeafLayout {
        std::string pane;
        Rect area;
        Rect tile;
        Rect header;
        Rect close;
        Rect maximize;
        Rect minimize;
        std::vector<Rect> actions;
        Rect body;
        Rect content;
        F32 slide = 0.0f;
        bool minimized = false;
        bool strip = false;
    };

    struct SplitterLayout {
        std::vector<U8> path;
        U64 left = 0;
        U64 right = 0;
        bool vertical = false;
        Rect rect;
    };

    struct SplitLayout {
        std::vector<U8> path;
        bool vertical = false;
        Rect area;
        std::vector<F32> sizes;
        std::vector<Rect> rects;
        std::vector<Rect> tiles;
        std::vector<F32> shown;
        std::vector<bool> open;
        std::optional<U64> flex;
    };

    enum class Press {
        None,
        Tab,
        TabClose,
        Header,
        HeaderClose,
        HeaderMaximize,
        HeaderMinimize,
        HeaderAction,
        Splitter,
        Handled,
    };

    enum class Collapse : U8 {
        Open,
        Strip,
        Header,
    };

    struct Model {
        std::optional<DockTiles> tree;
        std::set<std::string> minimized;
        std::string maximized;

        bool operator==(const Model&) const = default;
    };

    struct Arrangement {
        std::vector<LeafLayout> leaves;
        std::vector<SplitterLayout> splitters;
        std::vector<SplitLayout> splits;

        const LeafLayout* leaf(const std::string& pane) const {
            for (const auto& leaf : leaves) {
                if (leaf.pane == pane) {
                    return &leaf;
                }
            }
            return nullptr;
        }

        const SplitLayout* split(const std::vector<U8>& path) const {
            for (const auto& split : splits) {
                if (split.path == path) {
                    return &split;
                }
            }
            return nullptr;
        }
    };

    struct Origin {
        F32 width = 0.0f;
        F32 height = 0.0f;
        F32 column = 0.0f;
        bool minimized = false;
    };

    struct TileSplit {
        std::string pane;
        Side side = Side::Left;
    };

    struct ColumnInsert {
        std::string before;
        std::string after;
    };

    using Target = std::variant<TileSplit, ColumnInsert>;

    struct Gap {
        std::optional<Target> target;
        F32 distance = 0.0f;
    };

    struct Removal {
        Model shape;
        Arrangement base;
        std::optional<Origin> from;
    };

    struct Plan {
        Model model;
        Rect rect;
    };

    struct Level {
        std::vector<U8> path;
        bool vertical = false;
        bool row = false;
        Rect span;
        std::vector<F32> sizes;
        std::vector<bool> present;
        std::vector<bool> open;
        std::optional<U64> flex;
        U64 at = 0;
        std::vector<U64> payers;
        std::vector<std::string> paying;
    };

    struct Column {
        const DockTiles* node = nullptr;
        Rect tile;
    };

    struct State {
        std::vector<std::string> order;
        U64 active = 0;
        std::string focusedPane;
        Model model;
        std::vector<std::string> pending;

        bool operator==(const State&) const = default;
    };

    Config config;
    std::vector<std::string> order;
    U64 active = 0;
    std::optional<DockTiles> synced;
    std::string focusedPane;
    std::string pendingFocus;
    Model model;
    std::vector<std::string> pending;
    std::set<std::string> introduced;

    State state() const {
        return {.order = order, .active = active, .focusedPane = focusedPane, .model = model, .pending = pending};
    }

    Rect viewport;
    std::vector<SplitLayout> splits;
    SplitLayout pressSplit;
    std::vector<F32> pressStored;

    Box backgrounds;
    Box separators;
    Box tabBoxes;
    Box track;
    Box activeTab;
    Box tabRim;
    Box dividers;
    Card tileBase;
    Box tileFills;
    Box splitterBoxes;
    Box buttonHover;
    Card tileFrame;
    struct FillFrame {
        ColorRGBA<F32> fill;
        Palette palette;
        Box border;
        Box rim;
    };
    std::vector<std::unique_ptr<FillFrame>> fillFrames;
    TextMetrics metrics;
    std::unordered_map<std::string, Caption> tabCaptions;
    std::unordered_map<std::string, Caption> headerCaptions;
    struct Placeholder {
        Spinner spinner;
        Text text;
        Stack stack;
    };
    std::unordered_map<std::string, std::unique_ptr<Placeholder>> placeholders;

    Rect bounds;
    Rect tabBar;
    Rect contentArea;
    Rect leadingRect;
    Rect trailingRect;
    Rect endRect;
    std::vector<Component*> mountedChrome;
    Component* parkedAside = nullptr;
    std::vector<Component*> ordered;
    std::vector<Component*> managed;
    U64 contentCount = 0;

    void collectManaged() {
        managed.clear();
        for (const auto& item : config.panes) {
            if (item.content) {
                managed.push_back(item.content);
            }
        }
        contentCount = managed.size();
        const auto captionParts = [&](Caption& caption) {
            managed.insert(managed.end(), {caption.text.get(), caption.glyph.get(), caption.close.get(), caption.marker.get(),
                                            caption.maximize.get(), caption.minimize.get(), caption.spine.get()});
            for (auto& action : caption.actions) {
                managed.push_back(action.get());
            }
        };
        for (const auto& item : config.panes) {
            if (const auto it = tabCaptions.find(item.id); it != tabCaptions.end()) {
                captionParts(it->second);
            }
        }
        std::vector<std::string> headers;
        for (const auto& item : config.panes) {
            headers.push_back(item.id);
        }
        for (const auto& id : placed()) {
            if (std::find(headers.begin(), headers.end(), id) == headers.end()) {
                headers.push_back(id);
            }
        }
        for (const auto& id : headers) {
            if (const auto it = headerCaptions.find(id); it != headerCaptions.end()) {
                captionParts(it->second);
            }
            if (const auto it = placeholders.find(id); it != placeholders.end()) {
                managed.push_back(&it->second->stack);
            }
        }
        managed.insert(managed.end(), mountedChrome.begin(), mountedChrome.end());
    }

    std::vector<Component*> childOrder() {
        std::vector<Component*> order = {&backgrounds, &separators, &track, &tabBoxes, &activeTab,
                                         &tabRim, &dividers, &tileBase, &tileFills, &splitterBoxes};
        const auto split = managed.begin() + static_cast<I64>(std::min<U64>(contentCount, managed.size()));
        order.insert(order.end(), managed.begin(), split);
        order.push_back(&buttonHover);
        order.insert(order.end(), split, managed.end());
        order.push_back(&tileFrame);
        for (auto& frame : fillFrames) {
            order.push_back(&frame->border);
            order.push_back(&frame->rim);
        }
        return order;
    }
    std::vector<Rect> tabRects;
    std::vector<Rect> tabCloseRects;
    std::vector<std::string> tabPanes;
    std::vector<LeafLayout> leaves;
    std::vector<SplitterLayout> splitters;
    bool showHeaders = false;

    Press press = Press::None;
    U64 pressTab = 0;
    std::string pressPane;
    SplitterLayout pressLayout;
    Extent2D<F32> pressPoint = {0.0f, 0.0f};
    bool dragging = false;
    Extent2D<F32> dragPoint = {0.0f, 0.0f};
    DockDrag* drag = nullptr;
    std::weak_ptr<bool> dragAlive;
    F32 pixelRatio = 1.0f;

    std::optional<U64> hoveredTab;
    bool hoveredTabClose = false;
    std::string hoveredHeaderClose;
    std::string hoveredHeaderMaximize;
    std::string hoveredHeaderMinimize;
    std::string restored;
    std::optional<std::pair<std::string, U64>> hoveredHeaderAction;
    U64 pressAction = 0;
    std::optional<U64> hoveredSplitter;

    Rect hoveredHeaderButton() const {
        for (const auto& leaf : leaves) {
            if (hoveredHeaderClose == leaf.pane) {
                return leaf.close;
            }
            if (hoveredHeaderMaximize == leaf.pane) {
                return leaf.maximize;
            }
            if (hoveredHeaderMinimize == leaf.pane) {
                return leaf.minimize;
            }
            if (hoveredHeaderAction.has_value() && hoveredHeaderAction->first == leaf.pane &&
                hoveredHeaderAction->second < leaf.actions.size()) {
                return leaf.actions[hoveredHeaderAction->second];
            }
        }
        return {};
    }

    const Pane* pane(const std::string& id) const {
        for (const auto& item : config.panes) {
            if (item.id == id) {
                return &item;
            }
        }
        return nullptr;
    }

    bool displays(const std::string& id) const {
        if (config.tabs) {
            return active < order.size() && order[active] == id;
        }
        if (!model.maximized.empty()) {
            return model.maximized == id;
        }
        if (model.minimized.contains(id)) {
            return false;
        }
        const auto tiles = placed();
        return std::find(tiles.begin(), tiles.end(), id) != tiles.end();
    }

    bool known(const std::string& id) const {
        return pane(id) != nullptr;
    }

    bool columnMode() const {
        return !config.tabs && config.columns;
    }

    bool columnRow(const std::vector<U8>& path, bool vertical) const {
        return columnMode() && path.empty() && !vertical;
    }

    std::unordered_map<std::string, F32> shownProgress;
    std::optional<F64> shownSeen;
    bool shownMoving = false;
    std::optional<F32> reachProgress;
    std::optional<bool> reachTabsHidden;
    F32 reach = 0.0f;

    void advanceShown(F64 time) {
        const F64 elapsed = shownSeen.has_value() ? std::max(0.0, time - *shownSeen) : 0.0;
        const F32 step = static_cast<F32>(elapsed / kHideDuration);
        std::unordered_map<std::string, F32> next;
        shownMoving = false;
        for (const auto& item : config.panes) {
            const F32 target = item.hidden ? 0.0f : 1.0f;
            const auto it = shownProgress.find(item.id);
            F32 progress = it == shownProgress.end() ? target : it->second;
            progress = target > progress ? std::min(target, progress + step) : std::max(target, progress - step);
            shownMoving = shownMoving || progress != target;
            next[item.id] = progress;
        }
        shownProgress = std::move(next);
        const bool reaching = config.hideTabs && config.leadingReach > 0.0f;
        const F32 reachTarget = reaching ? 1.0f : 0.0f;
        if (reaching) {
            reach = config.leadingReach;
        }
        if (reachTabsHidden != config.hideTabs) {
            reachTabsHidden = config.hideTabs;
            reachProgress = reachTarget;
        }
        F32 progress = reachProgress.value_or(reachTarget);
        progress = reachTarget > progress ? std::min(reachTarget, progress + step)
                                          : std::max(reachTarget, progress - step);
        reachProgress = progress;
        shownMoving = shownMoving || progress != reachTarget;
        shownSeen = shownMoving ? std::optional<F64>(time) : std::nullopt;
    }

    F32 shownFactor(const std::string& id) const {
        const auto it = shownProgress.find(id);
        return it == shownProgress.end() ? 1.0f : Smoothstep(it->second);
    }

    F32 shownFactor(const DockTiles& node) const {
        F32 factor = 0.0f;
        for (const auto& id : Leaves(node)) {
            factor = std::max(factor, shownFactor(id));
        }
        return factor;
    }

    bool hiddenPane(const std::string& id) const {
        const auto* item = pane(id);
        return item && item->hidden;
    }

    bool collapsed(const Model& shape, const DockTiles& node) const {
        const auto panes = Leaves(node);
        const auto minimizedPane = [&shape](const std::string& id) {
            return shape.minimized.contains(id);
        };
        return std::any_of(panes.begin(), panes.end(), minimizedPane) &&
               std::all_of(panes.begin(), panes.end(), [&](const std::string& id) {
                   return minimizedPane(id) || hiddenPane(id);
               });
    }

    std::optional<U64> flexIndex(const Model& shape, const DockTiles& node) const {
        if (config.flexPane.empty()) {
            return std::nullopt;
        }
        for (U64 index = 0; index < node.children.size(); ++index) {
            const auto& child = node.children[index];
            if (Contains(child, config.flexPane) && !collapsed(shape, child) && shownFactor(child) >= 1.0f) {
                return index;
            }
        }
        return std::nullopt;
    }

    F32 defaultSize(const DockTiles& node, bool row, U64 index) const {
        if (row) {
            return !config.flexPane.empty() && Contains(node.children[index], config.flexPane) ? kFlexColumnWidth
                                                                                                 : kColumnWidth;
        }
        F32 sum = 0.0f;
        U64 sized = 0;
        for (const F32 size : node.sizes) {
            if (size > 0.0f) {
                sum += size;
                ++sized;
            }
        }
        return sized > 0 ? sum / static_cast<F32>(sized) : kColumnWidth;
    }

    std::vector<F32> resolved(const DockTiles& node, bool row) const {
        std::vector<F32> sizes(node.children.size(), 0.0f);
        for (U64 index = 0; index < sizes.size(); ++index) {
            const F32 size = index < node.sizes.size() ? node.sizes[index] : 0.0f;
            sizes[index] = size <= 0.0f ? defaultSize(node, row, index) : row ? std::max(size, kSizeFloor) : size;
        }
        return sizes;
    }

    void fillNested(DockTiles& node, bool root) const {
        if (node.children.empty()) {
            return;
        }
        node.sizes.resize(node.children.size(), 0.0f);
        const auto sizes = resolved(node, columnMode() && root && !node.vertical);
        for (U64 index = 0; index < node.children.size(); ++index) {
            auto& child = node.children[index];
            auto* inner = &child;
            while (inner->children.size() == 1) {
                inner = &inner->children.front();
            }
            if (inner->children.size() >= 2 && inner->vertical == node.vertical) {
                node.sizes[index] = sizes[index];
                inner->sizes = resolved(*inner, false);
            }
            fillNested(child, false);
        }
    }

    void settle(Model& shape) const {
        if (shape.tree.has_value()) {
            fillNested(*shape.tree, true);
            shape.tree = Normalize(std::move(*shape.tree));
        }
    }

    bool canMinimize(const std::string& id) const {
        if (!columnMode() || !model.maximized.empty()) {
            return false;
        }
        if (model.minimized.contains(id)) {
            return true;
        }
        U64 open = 0;
        for (const auto& pane : placed()) {
            open += model.minimized.contains(pane) || hiddenPane(pane) ? 0 : 1;
        }
        return open > 1;
    }

    void toggleMinimized(const std::string& id) {
        if (model.minimized.erase(id) > 0) {
            focus(id);
            commitView();
            return;
        }
        if (canMinimize(id)) {
            model.minimized.insert(id);
            commitView();
        }
    }

    void readView(const DockTiles& tiles) {
        if (tiles.children.empty()) {
            if (tiles.minimized) {
                model.minimized.insert(tiles.pane);
            }
            if (tiles.maximized && model.maximized.empty()) {
                model.maximized = tiles.pane;
            }
        }
        for (const auto& child : tiles.children) {
            readView(child);
        }
    }

    void writeView(DockTiles& tiles) const {
        if (tiles.children.empty()) {
            tiles.minimized = model.minimized.contains(tiles.pane);
            tiles.maximized = model.maximized == tiles.pane;
        }
        for (auto& child : tiles.children) {
            writeView(child);
        }
    }

    void commitView() {
        if (config.persistView) {
            commit();
        }
    }

    void settleMinimized() {
        if (!columnMode() || !config.headers) {
            model.minimized.clear();
            return;
        }
        const auto panes = placed();
        std::erase_if(model.minimized, [&panes](const std::string& id) {
            return std::find(panes.begin(), panes.end(), id) == panes.end();
        });
        const bool covered = std::all_of(panes.begin(), panes.end(), [this](const std::string& id) {
            return model.minimized.contains(id) || hiddenPane(id);
        });
        if (covered) {
            model.minimized.clear();
        }
    }

    void settleMaximized() {
        if (model.maximized.empty()) {
            return;
        }
        const auto panes = placed();
        if (config.tabs || !config.headers || std::find(panes.begin(), panes.end(), model.maximized) == panes.end() ||
            panes.size() < 2 || (columnMode() && hiddenPane(model.maximized))) {
            model.maximized.clear();
        }
    }

    std::vector<std::string> placed() const {
        if (config.tabs) {
            return order;
        }
        return model.tree.has_value() ? Leaves(*model.tree) : std::vector<std::string>{};
    }

    void append(Model& shape, const Pane& item) const {
        if (!shape.tree.has_value()) {
            shape.tree = Leaf(item.id);
            return;
        }
        auto& root = *shape.tree;
        if (root.children.empty() || root.vertical) {
            DockTiles row{.sizes = {0.0f}};
            row.children.push_back(std::move(root));
            root = std::move(row);
        }
        const auto at = item.leftmost ? 0 : static_cast<I64>(root.children.size());
        root.children.insert(root.children.begin() + at, Leaf(item.id));
        root.sizes.insert(root.sizes.begin() + at, std::max(0.0f, item.width));
        settle(shape);
    }

    void commit() {
        if (config.tabs) {
            return;
        }
        auto tiles = model.tree;
        if (config.persistView && tiles) {
            writeView(*tiles);
        }
        synced = tiles;
        if (config.onTiles) {
            config.onTiles(synced);
        }
    }

    void syncTiles() {
        if (config.tiles != synced) {
            synced = config.tiles;
            model.tree.reset();
            if (config.persistView) {
                model.minimized.clear();
                model.maximized.clear();
            }
            if (config.tiles.has_value() && Valid(*config.tiles)) {
                if (config.persistView) {
                    readView(*config.tiles);
                }
                model.tree = Bare(*config.tiles);
                settle(model);
            }
        }
        std::erase_if(introduced, [this](const std::string& id) {
            return !known(id);
        });
        std::erase_if(pending, [this](const std::string& id) {
            const auto* item = pane(id);
            return !item || !item->placement.has_value() || (model.tree && Contains(*model.tree, id));
        });
        bool added = false;
        bool addedVisible = false;
        for (const auto& item : config.panes) {
            const bool present = model.tree && Contains(*model.tree, item.id);
            if (!present && std::find(pending.begin(), pending.end(), item.id) == pending.end()) {
                if (item.placement.has_value()) {
                    pending.push_back(item.id);
                } else {
                    append(model, item);
                    added = true;
                }
                addedVisible = addedVisible || !item.startMinimized;
            }
            if (introduced.insert(item.id).second) {
                const bool saved = config.persistView && config.tiles && Contains(*config.tiles, item.id);
                addedVisible = addedVisible || (!saved && !item.startMinimized);
                if (item.startMinimized && !saved) {
                    model.minimized.insert(item.id);
                }
            }
        }
        if (addedVisible && !model.maximized.empty()) {
            model.maximized.clear();
            added = true;
        }
        if (added) {
            commit();
        }
    }

    void syncTabs() {
        const std::string activePane = active < order.size() ? order[active] : std::string();
        std::erase_if(order, [this](const std::string& id) {
            return !known(id);
        });
        bool added = false;
        for (const auto& item : config.panes) {
            if (std::find(order.begin(), order.end(), item.id) == order.end()) {
                order.push_back(item.id);
                added = true;
            }
        }
        const auto indexOf = [this](const std::string& id) -> std::optional<U64> {
            const auto it = std::find(order.begin(), order.end(), id);
            if (it == order.end()) {
                return std::nullopt;
            }
            return static_cast<U64>(it - order.begin());
        };
        bool activated = false;
        if (added) {
            active = order.size() - 1;
        } else if (const auto index = indexOf(activePane)) {
            active = *index;
        } else {
            activated = !activePane.empty();
        }
        if (config.activate.has_value()) {
            if (const auto index = indexOf(*config.activate)) {
                active = *index;
                activated = true;
            }
        }
        if (active >= order.size()) {
            active = order.empty() ? 0 : order.size() - 1;
        }
        if (order.empty()) {
            return;
        }
        if (activated) {
            focus(order[active]);
        } else if (added) {
            focusedPane = order[active];
            pendingFocus.clear();
        }
    }

    Rect device(const Rect& area) const {
        const F32 x0 = std::round(viewport.x + area.x * pixelRatio);
        const F32 y0 = std::round(viewport.y + area.y * pixelRatio);
        const F32 x1 = std::round(viewport.x + area.right() * pixelRatio);
        const F32 y1 = std::round(viewport.y + area.bottom() * pixelRatio);
        return {x0, y0, x1 - x0, y1 - y0};
    }

    Rect logicalArea() const {
        return {0.0f, 0.0f, std::max(0.0f, viewport.width / pixelRatio), std::max(0.0f, viewport.height / pixelRatio)};
    }

    Arrangement arrange(const Model& shape) const {
        Arrangement out;
        if (!shape.tree.has_value()) {
            return out;
        }
        const Rect area = logicalArea();
        if (!shape.maximized.empty() && Contains(*shape.tree, shape.maximized) && !hiddenPane(shape.maximized)) {
            out.leaves.push_back({.pane = shape.maximized, .area = area, .tile = viewport});
            return out;
        }
        std::vector<U8> path;
        arrangeNode(shape, out, *shape.tree, area, viewport, path, 0.0f, Collapse::Open);
        return out;
    }

    void arrangeNode(const Model& shape, Arrangement& out, const DockTiles& node, const Rect& area, const Rect& tile,
                     std::vector<U8>& path, F32 slide, Collapse style) const {
        if (node.children.empty()) {
            if (shownFactor(node.pane) <= 0.0f) {
                return;
            }
            out.leaves.push_back({.pane = node.pane, .area = area, .tile = tile, .slide = slide,
                                  .minimized = style != Collapse::Open || shape.minimized.contains(node.pane),
                                  .strip = style == Collapse::Strip});
            return;
        }
        const bool open = style == Collapse::Open;
        const bool vertical = open ? node.vertical : style == Collapse::Strip;
        const bool row = open && columnRow(path, vertical);
        const U64 count = node.children.size();
        SplitLayout split{.path = path,
                          .vertical = vertical,
                          .area = area,
                          .sizes = std::vector<F32>(count, 0.0f),
                          .rects = std::vector<Rect>(count),
                          .tiles = std::vector<Rect>(count),
                          .shown = std::vector<F32>(count, 0.0f),
                          .open = std::vector<bool>(count, false),
                          .flex = row ? flexIndex(shape, node) : std::nullopt};
        const std::vector<F32> stored = resolved(node, row);
        std::vector<F32> leads(count, 0.0f);
        std::vector<F32> fulls(count, 0.0f);
        std::vector<bool> folded(count, false);
        bool settled = true;
        F32 earlier = 0.0f;
        F32 room = Along(area, vertical);
        for (U64 index = 0; index < count; ++index) {
            const F32 shown = shownFactor(node.children[index]);
            split.shown[index] = shown;
            settled = settled && (shown <= 0.0f || shown >= 1.0f);
            if (index > 0) {
                leads[index] = kCardGap * std::min(shown, earlier);
                room -= leads[index];
            }
            earlier = std::max(earlier, shown);
            folded[index] = open && shown > 0.0f && collapsed(shape, node.children[index]);
        }
        if (open) {
            const F32 minimum = row ? kMinColumnWidth : 0.0f;
            const auto fit = [&](F32 space, const std::function<bool(U64)>& include) {
                std::vector<F32> picked;
                std::optional<U64> flex;
                for (U64 index = 0; index < count; ++index) {
                    if (include(index)) {
                        if (split.flex == index) {
                            flex = picked.size();
                        }
                        picked.push_back(stored[index]);
                    }
                }
                picked = FitSizes(std::move(picked), space, flex, minimum);
                std::vector<F32> fitted(count, 0.0f);
                for (U64 index = 0, slot = 0; index < count; ++index) {
                    if (include(index)) {
                        fitted[index] = picked[slot++];
                    }
                }
                return fitted;
            };
            F32 fullRoom = room;
            for (U64 index = 0; index < count; ++index) {
                const F32 shown = split.shown[index];
                if (folded[index]) {
                    fulls[index] = kStripWidth;
                    split.sizes[index] = kStripWidth * shown;
                    room -= split.sizes[index];
                    fullRoom -= split.sizes[index];
                } else if (shown > 0.0f && shown < 1.0f) {
                    fullRoom += kCardGap - leads[index];
                }
            }
            const auto natural = fit(fullRoom, [&](U64 index) {
                return !folded[index] && split.shown[index] > 0.0f;
            });
            for (U64 index = 0; index < count; ++index) {
                const F32 shown = split.shown[index];
                if (!folded[index] && shown > 0.0f && shown < 1.0f) {
                    fulls[index] = natural[index];
                    split.sizes[index] = natural[index] * shown;
                    room -= split.sizes[index];
                }
            }
            const auto fitted = fit(room, [&](U64 index) {
                return !folded[index] && split.shown[index] >= 1.0f;
            });
            for (U64 index = 0; index < count; ++index) {
                if (!folded[index] && split.shown[index] >= 1.0f) {
                    split.sizes[index] = fitted[index];
                    fulls[index] = fitted[index];
                    split.open[index] = true;
                }
            }
        } else {
            F32 total = 0.0f;
            for (U64 index = 0; index < count; ++index) {
                total += stored[index] * split.shown[index];
            }
            for (U64 index = 0; index < count; ++index) {
                split.sizes[index] = total > 0.0f ? std::max(0.0f, room) * stored[index] * split.shown[index] / total
                                                  : 0.0f;
                fulls[index] = split.sizes[index];
            }
        }
        F32 cursor = StartOf(area, vertical);
        for (U64 index = 0; index < count; ++index) {
            cursor += leads[index];
            const F32 size = split.sizes[index];
            split.rects[index] = vertical ? Rect{area.x, cursor, area.width, size} : Rect{cursor, area.y, size, area.height};
            const Rect snapped = device(split.rects[index]);
            split.tiles[index] = vertical ? Rect{tile.x, snapped.y, tile.width, snapped.height}
                                          : Rect{snapped.x, tile.y, snapped.width, tile.height};
            cursor += size;
        }
        if (open && settled) {
            for (U64 boundary = 1; boundary < count; ++boundary) {
                if (split.shown[boundary] <= 0.0f) {
                    continue;
                }
                std::optional<U64> upper;
                for (U64 scan = boundary; scan-- > 0;) {
                    if (!folded[scan] && split.shown[scan] > 0.0f) {
                        upper = scan;
                        break;
                    }
                }
                std::optional<U64> lower;
                for (U64 scan = boundary; scan < count; ++scan) {
                    if (!folded[scan] && split.shown[scan] > 0.0f) {
                        lower = scan;
                        break;
                    }
                }
                if (!upper.has_value() || !lower.has_value()) {
                    continue;
                }
                std::optional<U64> previous;
                for (U64 scan = boundary; scan-- > 0;) {
                    if (split.shown[scan] > 0.0f) {
                        previous = scan;
                        break;
                    }
                }
                if (!previous.has_value()) {
                    continue;
                }
                const Rect& at = split.tiles[boundary];
                const Rect& prior = split.tiles[*previous];
                const Rect gap = vertical ? Rect{tile.x, prior.bottom(), tile.width, at.y - prior.bottom()}
                                          : Rect{prior.right(), tile.y, at.x - prior.right(), tile.height};
                out.splitters.push_back(
                    {.path = path, .left = *upper, .right = *lower, .vertical = vertical, .rect = gap});
            }
        }
        for (U64 index = 0; index < count; ++index) {
            const F32 shown = split.shown[index];
            if (shown <= 0.0f) {
                continue;
            }
            const Collapse next = !open        ? style
                                  : folded[index] ? (vertical ? Collapse::Header : Collapse::Strip)
                                                  : Collapse::Open;
            const F32 direction = split.flex.has_value() && index > *split.flex ? -1.0f : 1.0f;
            const F32 own = !vertical && shown < 1.0f ? direction * (fulls[index] - split.sizes[index]) * pixelRatio
                                                      : slide;
            path.push_back(static_cast<U8>(index));
            arrangeNode(shape, out, node.children[index], split.rects[index], split.tiles[index], path, own, next);
            path.pop_back();
        }
        out.splits.push_back(std::move(split));
    }

    void apply(Arrangement arranged) {
        leaves = std::move(arranged.leaves);
        splitters = std::move(arranged.splitters);
        splits = std::move(arranged.splits);
    }

    const SplitLayout* splitAt(const std::vector<U8>& path) const {
        for (const auto& split : splits) {
            if (split.path == path) {
                return &split;
            }
        }
        return nullptr;
    }

    std::vector<Column> columns() const {
        std::vector<Column> result;
        if (!model.tree.has_value()) {
            return result;
        }
        const auto& root = *model.tree;
        if (!root.children.empty() && !root.vertical) {
            const auto* split = splitAt({});
            if (!split) {
                return result;
            }
            for (U64 index = 0; index < root.children.size() && index < split->rects.size(); ++index) {
                if (split->shown[index] > 0.0f) {
                    result.push_back({.node = &root.children[index], .tile = split->tiles[index]});
                }
            }
        } else if (shownFactor(root) > 0.0f) {
            result.push_back({.node = &root, .tile = viewport});
        }
        return result;
    }

    static const DockTiles* columnOf(const Model& shape, const std::string& pane) {
        if (pane.empty() || !shape.tree.has_value() || !Contains(*shape.tree, pane)) {
            return nullptr;
        }
        const auto& root = *shape.tree;
        if (root.children.empty() || root.vertical) {
            return &root;
        }
        for (const auto& child : root.children) {
            if (Contains(child, pane)) {
                return &child;
            }
        }
        return nullptr;
    }

    std::optional<Origin> origin(const std::string& id) const {
        if (!model.tree.has_value() || !Contains(*model.tree, id)) {
            return std::nullopt;
        }
        Origin result{.minimized = model.minimized.contains(id)};
        for (const auto& leaf : leaves) {
            if (leaf.pane == id && !leaf.minimized) {
                result.width = leaf.area.width;
                result.height = leaf.area.height;
            }
        }
        const auto& root = *model.tree;
        if (root.children.empty()) {
            result.column = result.width;
        } else if (!root.vertical) {
            for (U64 index = 0; index < root.children.size(); ++index) {
                if (root.children[index].children.empty() && root.children[index].pane == id) {
                    result.column = index < root.sizes.size() ? root.sizes[index] : 0.0f;
                }
            }
        }
        return result;
    }

    Removal removal(const std::string& id, std::optional<Origin> from) const {
        Removal result{.shape = model, .from = from};
        result.shape.tree = Remove(model.tree, id);
        settle(result.shape);
        result.shape.maximized.clear();
        if (!from || !from->minimized) {
            result.shape.minimized.erase(id);
        }
        result.base = arrange(result.shape);
        return result;
    }

    Removal removal(const std::string& id) const {
        return removal(id, origin(id));
    }

    std::string representative(const DockTiles& node, const std::string& dragged, bool last) const {
        auto panes = Leaves(node);
        if (last) {
            std::reverse(panes.begin(), panes.end());
        }
        for (const auto& pane : panes) {
            if (pane != dragged && shownFactor(pane) > 0.0f) {
                return pane;
            }
        }
        return {};
    }

    std::optional<Gap> gapAt(const Extent2D<F32>& point, const std::string& dragged, const Model& shape) const {
        const auto shown = columns();
        if (shown.empty() || !Inflate(viewport, kCardGap * pixelRatio).contains(point.x, point.y)) {
            return std::nullopt;
        }
        const F32 band = kGutterBand * pixelRatio;
        std::optional<Gap> best;
        const U64 count = shown.size();
        for (U64 index = 0; index <= count; ++index) {
            const F32 left = index == 0 ? viewport.x : shown[index - 1].tile.right();
            const F32 right = index == count ? viewport.right() : shown[index].tile.x;
            const F32 center = index == 0 ? viewport.x : index == count ? viewport.right() : (left + right) * 0.5f;
            const F32 distance = std::abs(point.x - center);
            if (point.x < left - band || point.x > right + band || (best && best->distance <= distance)) {
                continue;
            }
            const std::string before = index == 0 ? std::string() : representative(*shown[index - 1].node, dragged, true);
            const std::string after = index == count ? std::string() : representative(*shown[index].node, dragged, false);
            const auto* lead = columnOf(shape, before);
            const auto* trail = columnOf(shape, after);
            const bool leadOpen = lead && !collapsed(shape, *lead);
            const bool trailOpen = trail && !collapsed(shape, *trail);
            const auto whole = [](const DockTiles* column, bool open) {
                return column && open && column->children.empty() ? column->pane : std::string();
            };
            const std::string leadWhole = whole(lead, leadOpen);
            const std::string trailWhole = whole(trail, trailOpen);
            const bool blocked = (index > 0 && before.empty()) || (index < count && after.empty()) ||
                                 (!leadOpen && !trailOpen);
            best = Gap{.distance = distance};
            if (blocked) {
                continue;
            }
            if (!leadWhole.empty() || !trailWhole.empty()) {
                const bool useTrail = !trailWhole.empty() && (leadWhole.empty() || point.x >= center);
                best->target = TileSplit{.pane = useTrail ? trailWhole : leadWhole,
                                         .side = useTrail ? Side::Left : Side::Right};
                continue;
            }
            best->target = ColumnInsert{.before = before, .after = after};
        }
        return best;
    }

    std::optional<Target> aim(const Extent2D<F32>& point, const std::string& dragged, const Removal& removed) const {
        if (!model.tree.has_value() || leaves.empty()) {
            return ColumnInsert{};
        }
        if (!model.maximized.empty()) {
            return std::nullopt;
        }
        if (const auto gap = gapAt(point, dragged, removed.shape)) {
            return gap->target;
        }
        const auto* leaf = targetAt(point, kCardGap * pixelRatio * 0.5f + 1.0f);
        if (!leaf || leaf->pane == dragged || model.minimized.contains(leaf->pane) || !removed.shape.tree ||
            !Contains(*removed.shape.tree, leaf->pane)) {
            return std::nullopt;
        }
        if (const auto side = EdgeAt(leaf->tile, point)) {
            return TileSplit{.pane = leaf->pane, .side = *side};
        }
        return std::nullopt;
    }

    Level levelOf(const SplitLayout& split) const {
        Level level{.path = split.path,
                    .vertical = split.vertical,
                    .row = split.path.empty() && !split.vertical,
                    .span = split.area,
                    .sizes = split.sizes,
                    .flex = split.flex};
        for (U64 index = 0; index < split.sizes.size(); ++index) {
            level.present.push_back(split.shown[index] > 0.0f);
            level.open.push_back(split.open[index]);
        }
        return level;
    }

    std::optional<Level> splitLevel(Model& shape, const Arrangement& base, const TileSplit& target) const {
        if (!shape.tree.has_value()) {
            return std::nullopt;
        }
        const auto path = PathOf(*shape.tree, target.pane);
        const auto* leaf = base.leaf(target.pane);
        if (!path || !leaf) {
            return std::nullopt;
        }
        const bool vertical = Vertical(target.side);
        const U64 after = target.side == Side::Right || target.side == Side::Bottom ? 1 : 0;
        if (!path->empty()) {
            const std::vector<U8> parentPath(path->begin(), path->end() - 1);
            const auto* parent = At(*shape.tree, parentPath);
            const auto* split = base.split(parentPath);
            if (parent && split && parent->vertical == vertical) {
                auto level = levelOf(*split);
                level.at = path->back() + after;
                level.payers = {path->back()};
                return level;
            }
        }
        auto* node = At(*shape.tree, *path);
        const F32 size = Along(leaf->area, vertical);
        DockTiles wrapper{.vertical = vertical, .sizes = {size}};
        wrapper.children.push_back(std::move(*node));
        *node = std::move(wrapper);
        return Level{.path = *path,
                     .vertical = vertical,
                     .row = path->empty() && !vertical,
                     .span = leaf->area,
                     .sizes = {size},
                     .present = {true},
                     .open = {!leaf->minimized},
                     .at = after,
                     .payers = {0}};
    }

    std::optional<Level> insertLevel(Model& shape, const Arrangement& base, const ColumnInsert& target) const {
        if (!shape.tree.has_value()) {
            return std::nullopt;
        }
        auto& root = *shape.tree;
        std::optional<Level> level;
        if (!root.children.empty() && !root.vertical) {
            if (const auto* split = base.split({})) {
                level = levelOf(*split);
            }
        } else {
            const Rect area = logicalArea();
            const F32 shown = shownFactor(root);
            level = Level{.row = true,
                          .span = area,
                          .sizes = {area.width},
                          .present = {shown > 0.0f},
                          .open = {shown >= 1.0f && !collapsed(shape, root)}};
            DockTiles row{.sizes = {area.width}};
            row.children.push_back(std::move(root));
            root = std::move(row);
        }
        if (!level.has_value()) {
            return std::nullopt;
        }
        const auto indexOf = [&root](const std::string& pane) -> std::optional<U64> {
            for (U64 index = 0; index < root.children.size(); ++index) {
                if (Contains(root.children[index], pane)) {
                    return index;
                }
            }
            return std::nullopt;
        };
        const auto lead = target.before.empty() ? std::nullopt : indexOf(target.before);
        const auto trail = target.after.empty() ? std::nullopt : indexOf(target.after);
        if ((!target.before.empty() && !lead) || (!target.after.empty() && !trail)) {
            return std::nullopt;
        }
        level->at = trail ? *trail : lead ? *lead + 1 : root.children.size();
        for (const auto& spot : {lead, trail}) {
            if (spot && level->open[*spot]) {
                level->payers.push_back(*spot);
            }
        }
        return level;
    }

    std::optional<Rect> place(Level& level, Model& shape, const std::string& id, const std::optional<Origin>& from) const {
        auto* node = At(*shape.tree, level.path);
        if (!node) {
            return std::nullopt;
        }
        F32 room = 0.0f;
        for (const U64 payer : level.payers) {
            if (!level.open[payer]) {
                return std::nullopt;
            }
            room += level.sizes[payer];
            for (auto& pane : Leaves(node->children[payer])) {
                level.paying.push_back(std::move(pane));
            }
        }
        if (room <= 0.0f) {
            return std::nullopt;
        }
        const auto bases = resolved(*node, columnRow(level.path, level.vertical));
        node->sizes.resize(node->children.size(), 0.0f);
        for (U64 index = 0; index < level.sizes.size(); ++index) {
            if (!level.open[index]) {
                continue;
            }
            if (level.flex == index) {
                node->sizes[index] = Capped(node->sizes[index], bases[index], level.sizes[index]);
            } else if (!level.flex.has_value() || std::abs(level.sizes[index] - bases[index]) > 1e-3f) {
                node->sizes[index] = level.sizes[index];
            }
        }
        const bool shaded = from && from->minimized;
        const F32 own = !from ? 0.0f : level.vertical ? from->height : from->width;
        const F32 half = std::max(0.0f, room - kCardGap) * 0.5f;
        const F32 floor = level.row ? kSizeFloor : 0.0f;
        const F32 size = std::max(floor, own > 0.0f ? std::min(own, half) : half);
        const F32 occupied = shaded ? kStripWidth : size;
        const F32 restore = shaded ? half : size;
        for (const U64 payer : level.payers) {
            const F32 paid = level.sizes[payer] - (occupied + kCardGap) * level.sizes[payer] / room;
            level.sizes[payer] = paid;
            node->sizes[payer] = level.flex == payer ? Capped(node->sizes[payer], bases[payer], paid) : paid;
        }
        const auto at = static_cast<I64>(level.at);
        node->children.insert(node->children.begin() + at, Leaf(id));
        node->sizes.insert(node->sizes.begin() + at, restore);
        level.sizes.insert(level.sizes.begin() + at, occupied);
        level.present.insert(level.present.begin() + at, true);
        level.open.insert(level.open.begin() + at, !shaded);
        F32 cursor = StartOf(level.span, level.vertical);
        F32 start = cursor;
        bool first = true;
        for (U64 index = 0; index < level.sizes.size(); ++index) {
            if (!level.present[index]) {
                continue;
            }
            cursor += first ? 0.0f : kCardGap;
            first = false;
            start = index == level.at ? cursor : start;
            cursor += level.sizes[index];
        }
        return level.vertical ? Rect{level.span.x, start, level.span.width, occupied}
                              : Rect{start, level.span.y, occupied, level.span.height};
    }

    bool fits(const Model& shape, const Arrangement& base, const Level& level, const std::string& id,
              const Rect& rect, Rect& landed) const {
        const auto laid = arrange(shape);
        const F32 floor = kSizeFloor - kSizeEpsilon;
        if (const auto* row = laid.split({}); row && !row->vertical) {
            for (U64 index = 0; index < row->sizes.size(); ++index) {
                if (row->open[index] && row->sizes[index] < floor) {
                    return false;
                }
            }
        }
        for (const auto& leaf : laid.leaves) {
            const auto* before = base.leaf(leaf.pane);
            const bool moved = leaf.pane == id;
            const bool paying = std::find(level.paying.begin(), level.paying.end(), leaf.pane) != level.paying.end();
            if (moved ? !Near(leaf.area, rect, kPlanTolerance)
                      : !before || (!paying && !Near(leaf.area, before->area, kPlanTolerance))) {
                return false;
            }
            if ((moved || paying) && !leaf.minimized && (leaf.area.width < floor || leaf.area.height <= 0.0f)) {
                return false;
            }
        }
        const auto* moved = laid.leaf(id);
        if (!moved) {
            return false;
        }
        landed = moved->tile;
        return true;
    }

    std::optional<Plan> plan(const std::string& id, const Target& target, const Removal& removed) const {
        Plan result{.model = removed.shape};
        auto& shape = result.model;
        const auto* insert = std::get_if<ColumnInsert>(&target);
        if (!shape.tree.has_value() || (insert && insert->before.empty() && insert->after.empty())) {
            Pane item{.id = id, .width = removed.from ? removed.from->column : 0.0f};
            append(shape, item);
            const auto laid = arrange(shape);
            const auto* leaf = laid.leaf(id);
            result.rect = leaf ? leaf->tile : viewport;
            return result;
        }
        auto& root = *shape.tree;
        if (const auto* row = removed.base.split({}); row && !root.children.empty() && !root.vertical) {
            root.sizes = baked(*row, root);
        }
        auto level = insert ? insertLevel(shape, removed.base, *insert)
                            : splitLevel(shape, removed.base, std::get<TileSplit>(target));
        if (!level || level->payers.empty()) {
            return std::nullopt;
        }
        const auto rect = place(*level, shape, id, removed.from);
        if (!rect) {
            return std::nullopt;
        }
        settle(shape);
        if (!fits(shape, removed.base, *level, id, *rect, result.rect)) {
            return std::nullopt;
        }
        return result;
    }

    Rect preview(const Target& target, const Plan& planned) const {
        if (const auto* split = std::get_if<TileSplit>(&target)) {
            for (const auto& leaf : leaves) {
                if (leaf.pane == split->pane) {
                    return Slice(leaf.tile, split->side, Along(planned.rect, Vertical(split->side)));
                }
            }
        }
        return planned.rect;
    }

    void drop(const std::string& id, const Target& target) {
        const auto removed = removal(id);
        auto planned = plan(id, target, removed);
        if (!planned) {
            planned = plan(id, ColumnInsert{}, removed);
        }
        model = std::move(planned->model);
        focus(id);
        commit();
    }

    bool placePending() {
        if (pending.empty() || config.tabs) {
            return false;
        }
        const auto queued = std::exchange(pending, {});
        bool placed = false;
        for (const auto& id : queued) {
            const auto* item = pane(id);
            if (!item || (model.tree && Contains(*model.tree, id))) {
                continue;
            }
            std::optional<Plan> planned;
            if (item->placement.has_value()) {
                const auto& request = *item->placement;
                const Origin from{.minimized = model.minimized.contains(id)};
                const auto removed = removal(id, from);
                const auto* anchor = removed.base.leaf(request.anchor);
                if (anchor && !anchor->minimized) {
                    planned = plan(id, TileSplit{.pane = request.anchor, .side = request.side}, removed);
                }
            }
            if (planned) {
                model = std::move(planned->model);
            } else {
                append(model, *item);
            }
            placed = true;
        }
        if (placed) {
            settleMaximized();
            settleMinimized();
            commit();
        }
        return placed;
    }

    static bool sameSplitter(const SplitterLayout& a, const SplitterLayout& b) {
        return a.vertical == b.vertical && a.left == b.left && a.right == b.right && a.path == b.path;
    }

    DockTiles* splitNode(const SplitterLayout& splitter) {
        if (!model.tree.has_value()) {
            return nullptr;
        }
        auto* node = At(*model.tree, splitter.path);
        if (!node || node->children.empty() || splitter.left >= splitter.right ||
            splitter.right >= node->children.size()) {
            return nullptr;
        }
        return node;
    }

    F32 splitterMinimum(const SplitLayout& split, F32 left, F32 right) const {
        if (columnRow(split.path, split.vertical)) {
            return std::min(left, right) < kMinColumnWidth ? kSizeFloor : kMinColumnWidth;
        }
        return std::min(kSplitMinimum, kSplitShare * (left + right));
    }

    static F32 Capped(F32 stored, F32 base, F32 value) {
        return value < base ? value : stored;
    }

    std::vector<F32> baked(const SplitLayout& split, const DockTiles& node) const {
        auto stored = node.sizes;
        stored.resize(node.children.size(), 0.0f);
        const auto bases = resolved(node, columnRow(split.path, split.vertical));
        for (U64 index = 0; index < split.sizes.size() && index < stored.size(); ++index) {
            if (!split.open[index]) {
                continue;
            }
            if (split.flex == index) {
                stored[index] = Capped(stored[index], bases[index], split.sizes[index]);
            } else if (!split.flex.has_value() || std::abs(split.sizes[index] - bases[index]) > 1e-3f) {
                stored[index] = split.sizes[index];
            }
        }
        return stored;
    }

    bool beginResize(const SplitterLayout& splitter) {
        const auto* split = splitAt(splitter.path);
        auto* node = splitNode(splitter);
        if (!split || !node || split->sizes.size() != node->children.size()) {
            return false;
        }
        pressSplit = *split;
        pressStored = node->sizes;
        pressStored.resize(node->children.size(), 0.0f);
        return true;
    }

    bool resize(const SplitterLayout& splitter, const Extent2D<F32>& point) {
        auto* node = splitNode(splitter);
        const auto& split = pressSplit;
        if (!node || split.path != splitter.path || split.sizes.size() != node->children.size()) {
            return false;
        }
        const U64 left = splitter.left;
        const U64 right = splitter.right;
        const F32 delta = (splitter.vertical ? point.y - pressPoint.y : point.x - pressPoint.x) / pixelRatio;
        DockTiles pressed = *node;
        pressed.sizes = pressStored;
        auto sizes = baked(split, pressed);
        const auto bases = resolved(pressed, columnRow(split.path, split.vertical));
        const F32 minimum = splitterMinimum(split, split.sizes[left], split.sizes[right]);
        if (split.flex == left || split.flex == right) {
            const U64 flex = *split.flex;
            const U64 other = flex == left ? right : left;
            const F32 sign = other == left ? 1.0f : -1.0f;
            const F32 shrink = std::max(0.0f, split.sizes[other] - minimum);
            const F32 room = std::max(0.0f, split.sizes[flex] - minimum);
            const F32 grow = std::clamp(sign * delta, -shrink, room);
            sizes[other] = split.sizes[other] + grow;
            sizes[flex] = Capped(pressStored[flex], bases[flex], split.sizes[flex] - grow);
        } else {
            const F32 shift = std::clamp(delta, -std::max(0.0f, split.sizes[left] - minimum),
                                         std::max(0.0f, split.sizes[right] - minimum));
            sizes[left] = split.sizes[left] + shift;
            sizes[right] = split.sizes[right] - shift;
        }
        if (sizes == node->sizes) {
            return false;
        }
        node->sizes = std::move(sizes);
        return true;
    }

    void equalize(const SplitterLayout& splitter) {
        const auto* split = splitAt(splitter.path);
        auto* node = splitNode(splitter);
        if (!split || !node || split->sizes.size() != node->children.size()) {
            return;
        }
        const U64 left = splitter.left;
        const U64 right = splitter.right;
        const auto bases = resolved(*node, columnRow(split->path, split->vertical));
        auto sizes = baked(*split, *node);
        const F32 average = (split->sizes[left] + split->sizes[right]) * 0.5f;
        for (const U64 index : {left, right}) {
            sizes[index] = split->flex == index ? Capped(sizes[index], bases[index], average) : average;
        }
        node->sizes = std::move(sizes);
        commit();
    }

    void focus(const std::string& id) {
        if (id.empty()) {
            return;
        }
        const auto* item = pane(id);
        if (!item) {
            pendingFocus = id;
            return;
        }
        pendingFocus.clear();
        if (focusedPane == id) {
            return;
        }
        focusedPane = id;
        if (item->onFocus) {
            item->onFocus();
        }
    }

    void forgetRemovedFocus() {
        if (!known(focusedPane)) {
            focusedPane.clear();
        }
    }

    void flushPendingFocus() {
        if (!pendingFocus.empty() && known(pendingFocus)) {
            focus(std::exchange(pendingFocus, std::string()));
        }
    }

    void moveTab(const std::string& id, U64 index) {
        const auto it = std::find(order.begin(), order.end(), id);
        if (it != order.end()) {
            const U64 from = static_cast<U64>(it - order.begin());
            order.erase(it);
            if (from < index) {
                --index;
            }
        }
        index = std::min<U64>(index, order.size());
        order.insert(order.begin() + static_cast<I64>(index), id);
        active = index;
        focus(id);
    }

    void removeOwn(const std::string& id) {
        if (focusedPane == id) {
            focusedPane.clear();
        }
        if (config.tabs) {
            const auto it = std::find(order.begin(), order.end(), id);
            if (it == order.end()) {
                return;
            }
            const U64 index = static_cast<U64>(it - order.begin());
            const bool wasActive = index == active;
            order.erase(it);
            if (index < active && active > 0) {
                --active;
            }
            active = std::min<U64>(active, order.empty() ? 0 : order.size() - 1);
            if (wasActive && !order.empty()) {
                focus(order[active]);
            }
            return;
        }
        model.tree = Remove(model.tree, id);
        settle(model);
        settleMaximized();
        commit();
    }

    std::string describe(const std::string& id) const {
        return config.describe ? config.describe(id) : id;
    }

    std::optional<U64> tabAt(const Extent2D<F32>& point) const {
        for (U64 index = 0; index < tabRects.size(); ++index) {
            if (tabRects[index].contains(point.x, point.y)) {
                return index;
            }
        }
        return std::nullopt;
    }

    const LeafLayout* leafAt(const Extent2D<F32>& point) const {
        for (const auto& leaf : leaves) {
            if (leaf.tile.contains(point.x, point.y)) {
                return &leaf;
            }
        }
        return nullptr;
    }

    const LeafLayout* targetAt(const Extent2D<F32>& point, F32 slack) const {
        if (const auto* leaf = leafAt(point)) {
            return leaf;
        }
        for (const auto& leaf : leaves) {
            if (Inflate(leaf.tile, slack).contains(point.x, point.y)) {
                return &leaf;
            }
        }
        return nullptr;
    }

    std::optional<U64> splitterAt(const Extent2D<F32>& point) const {
        const F32 slack = kSplitterSlack * pixelRatio;
        for (U64 index = 0; index < splitters.size(); ++index) {
            const Rect hit = Inflate(splitters[index].rect, slack);
            if (hit.contains(point.x, point.y)) {
                return index;
            }
        }
        return std::nullopt;
    }
};


Dock::Dock() {
    impl = std::make_unique<Impl>();
    setClipsChildren(true);
}

Dock::~Dock() {
    if (impl->dragging && impl->drag && !impl->dragAlive.expired()) {
        impl->drag->reset();
    }
}

void Dock::update(Config config) {
    const bool changed = config != impl->config;
    if (changed) {
        invalidate(Dirty::Layout);
    }
    std::vector<Component*> retinted;
    for (const auto& item : config.panes) {
        const auto previous = std::find_if(impl->config.panes.begin(), impl->config.panes.end(), [&](const auto& pane) {
            return pane.id == item.id;
        });
        if (item.content && previous != impl->config.panes.end() && previous->fill != item.fill) {
            retinted.push_back(item.content);
        }
    }
    impl->config = std::move(config);
    for (auto* content : retinted) {
        Access::invalidateSubtree(*content);
    }
    std::optional<Impl::State> before;
    if (!changed) {
        before = impl->state();
    }
    impl->forgetRemovedFocus();
    if (impl->config.tabs) {
        impl->syncTabs();
    } else {
        impl->syncTiles();
    }
    impl->flushPendingFocus();
    impl->settleMaximized();
    impl->settleMinimized();
    if (before.has_value() && *before != impl->state()) {
        invalidate(Dirty::Layout);
    }
    for (const auto& item : impl->config.panes) {
        if (item.content && !hasChild(*item.content)) {
            addChild(*item.content);
            item.content->setVisible(impl->displays(item.id));
        }
    }
    const std::vector<Component*> mounted = children();
    if (impl->config.aside) {
        impl->parkedAside = impl->config.aside;
    } else if (std::find(mounted.begin(), mounted.end(), impl->parkedAside) == mounted.end()) {
        impl->parkedAside = nullptr;
    }
    std::vector<Component*> chrome;
    if (impl->config.tabs) {
        for (auto* item : {impl->config.leading, impl->config.trailing, impl->config.end, impl->parkedAside}) {
            if (item) {
                chrome.push_back(item);
            }
        }
    }
    for (auto* item : impl->mountedChrome) {
        const bool live = std::find(mounted.begin(), mounted.end(), item) != mounted.end();
        if (live && std::find(chrome.begin(), chrome.end(), item) == chrome.end()) {
            removeChild(*item);
        }
    }
    for (auto* item : chrome) {
        if (!hasChild(*item)) {
            addChild(*item);
        }
    }
    impl->mountedChrome = std::move(chrome);

    const std::vector<std::string> placed = impl->placed();
    bool captionsChanged = false;
    const auto caption = [&captionsChanged](auto& map, const std::string& id, std::string title, std::string icon) {
        auto& entry = map[id];
        if (entry.title != title || entry.icon != icon) {
            captionsChanged = true;
        }
        entry.title = std::move(title);
        entry.icon = std::move(icon);
    };
    for (const auto& item : impl->config.panes) {
        caption(impl->tabCaptions, item.id, item.title, item.icon);
        caption(impl->headerCaptions, item.id, item.title, item.icon);
        auto& actions = impl->headerCaptions[item.id].actions;
        actions.resize(item.actions.size());
        for (auto& action : actions) {
            if (!action) {
                action = std::make_unique<Text>();
            }
        }
    }
    for (const auto& id : placed) {
        if (!impl->known(id)) {
            caption(impl->headerCaptions, id, impl->describe(id), {});
            auto& placeholder = impl->placeholders[id];
            if (!placeholder) {
                placeholder = std::make_unique<Impl::Placeholder>();
            }
            placeholder->spinner.update({.diameter = kPlaceholderSpinner});
            placeholder->text.update({
                .str = "Loading",
                .font = Text::Font::Sans,
                .tone = Text::Tone::Secondary,
                .align = Text::Align::Center,
                .scale = Typography::Scale::Small,
            });
            placeholder->stack.update({
                .spacing = Spacing::Small,
                .items = {{.child = &placeholder->spinner}, {.child = &placeholder->text}},
            });
        }
    }
    const auto stale = [&](const std::string& id, bool placeholder) {
        const bool inTree = std::find(placed.begin(), placed.end(), id) != placed.end();
        return placeholder ? (impl->known(id) || !inTree) : (!impl->known(id) && !inTree);
    };
    if (captionsChanged) {
        invalidate(Dirty::Layout);
    }
    std::erase_if(impl->tabCaptions, [&](const auto& entry) {
        return !impl->known(entry.first);
    });
    std::erase_if(impl->headerCaptions, [&](const auto& entry) {
        return stale(entry.first, false);
    });
    std::erase_if(impl->placeholders, [&](const auto& entry) {
        return stale(entry.first, true);
    });
    std::vector<ColorRGBA<F32>> fills;
    if (!impl->config.tabs) {
        for (const auto& item : impl->config.panes) {
            if (item.card && item.fill.a > 0.0f && std::find(fills.begin(), fills.end(), item.fill) == fills.end()) {
                fills.push_back(item.fill);
            }
        }
    }
    impl->fillFrames.resize(fills.size());
    for (U64 index = 0; index < fills.size(); ++index) {
        if (!impl->fillFrames[index]) {
            impl->fillFrames[index] = std::make_unique<Impl::FillFrame>();
        }
        impl->fillFrames[index]->fill = fills[index];
    }
    impl->collectManaged();
    auto order = impl->childOrder();
    if (order != impl->ordered) {
        for (auto* child : children()) {
            if (child && std::find(order.begin(), order.end(), child) == order.end()) {
                child->setVisible(false);
            }
        }
        setChildren(order);
        impl->ordered = std::move(order);
    }
}

bool Dock::resizing() const {
    return impl->press == Impl::Press::Splitter;
}

std::string Dock::paneAt(const Extent2D<F32>& point) const {
    if (!visible()) {
        return {};
    }
    const auto* leaf = impl->leafAt(point);
    return leaf ? leaf->pane : std::string();
}

std::optional<Rect> Dock::tabRect(const std::string& pane) const {
    for (U64 index = 0; index < impl->tabPanes.size() && index < impl->tabRects.size(); ++index) {
        if (impl->tabPanes[index] == pane) {
            const Rect& tab = impl->tabRects[index];
            const F32 scale = impl->pixelRatio;
            return Rect{(tab.x - frame().x) / scale, (tab.y - frame().y) / scale, tab.width / scale,
                        tab.height / scale};
        }
    }
    return std::nullopt;
}

Extent2D<F32> Dock::measure(const Context&, Extent2D<F32> available) {
    return {FillLength(available.x), FillLength(available.y)};
}

void Dock::layout(const Context& ctx) {
    auto& s = *impl;
    s.bounds = frame();
    s.pixelRatio = ctx.pixelRatio;
    s.drag = ctx.dockDrag;
    s.dragAlive = s.drag ? std::weak_ptr<bool>(s.drag->alive) : std::weak_ptr<bool>();
    s.metrics.setWindow(ctx.render);
    s.settleMaximized();
    s.settleMinimized();
    s.advanceShown(ctx.time);
    if (s.shownMoving) {
        requestFrame();
    }
    const bool tabs = s.config.tabs;
    const F32 font = Px(ctx, Typography::FontSize);
    const F32 infinite = std::numeric_limits<F32>::infinity();
    const F32 line = Hairline(ctx);
    const F32 gap = Px(ctx, kCardGap);
    const F32 radius = Px(ctx, TileRounding);
    const F32 closeSize = Px(ctx, kPillClose);
    F32 tabRadius = 0.0f;
    F32 trackRadius = 0.0f;

    std::vector<Box::Instance> backgroundInstances;
    std::vector<Box::Instance> separatorInstances;
    std::vector<Box::Instance> tabInstances;
    std::vector<Box::Instance> trackInstances;
    std::vector<Box::Instance> activeInstances;
    std::vector<Box::Instance> tabRimInstances;
    std::vector<Box::Instance> dividerInstances;
    std::vector<Rect> tileRects;
    std::vector<Box::Instance> tileFillInstances;
    std::vector<Rect> frameRects;
    std::vector<Box::Instance> splitterInstances;
    std::vector<Component*> labels;
    std::vector<Component*> chrome;
    std::vector<Component*> contents;

    s.tabRects.clear();
    s.tabCloseRects.clear();
    s.tabPanes.clear();
    s.leaves.clear();
    s.splitters.clear();
    s.leadingRect = {};
    s.trailingRect = {};
    s.endRect = {};

    const auto prepareGlyph = [&](Text& glyph, const char* icon, const char* colorKey, F32 points, F32 opacity) {
        glyph.update({.str = icon, .colorKey = colorKey, .scale = points / Typography::FontSize, .opacity = opacity});
        return measureChild(glyph, ctx, {infinite, infinite});
    };
    const auto placeGlyph = [&](Text& glyph, Extent2D<F32> center, Extent2D<F32> size, const Rect& clip) {
        layoutChild(ctx, glyph, {center.x - size.x * 0.5f, center.y - size.y * 0.5f, size.x, size.y}, clip);
        labels.push_back(&glyph);
    };
    const auto glyphButton = [&](Text& glyph, const Rect& button, const char* icon, const char* colorKey,
                                 F32 points = kCloseGlyph, F32 opacity = 1.0f) {
        const auto size = prepareGlyph(glyph, icon, colorKey, points, opacity);
        placeGlyph(glyph, {button.x + button.width * 0.5f, button.y + button.height * 0.5f}, size, button);
    };

    if (tabs) {
        auto* leading = s.config.leading;
        auto* trailing = s.config.trailing;
        auto* end = s.config.end;
        Extent2D<F32> leadSize = leading ? measureChild(*leading, ctx, {infinite, infinite})
                                         : Extent2D<F32>{0.0f, 0.0f};
        const Extent2D<F32> trailSize = trailing ? measureChild(*trailing, ctx, {infinite, infinite})
                                                 : Extent2D<F32>{0.0f, 0.0f};
        const Extent2D<F32> endSize = end ? measureChild(*end, ctx, {infinite, infinite}) : Extent2D<F32>{0.0f, 0.0f};
        const bool bar = !s.order.empty() || leading || trailing || end || s.config.tabBarHeight > 0.0f;
        const F32 base = s.config.tabBarHeight > 0.0f ? std::round(Px(ctx, s.config.tabBarHeight))
                                                      : std::round(Px(ctx, kPillBarHeight));
        const F32 height = bar ? std::max({base, std::ceil(leadSize.y), std::ceil(trailSize.y)}) : 0.0f;
        s.tabBar = {s.bounds.x, s.bounds.y, s.bounds.width, height};
        const F32 margin = Px(ctx, kBarMargin);
        const F32 drop = std::round(Px(ctx, kTrackDrop));
        const F32 spacing = Px(ctx, kPillGap);
        F32 x = s.bounds.x + margin + Px(ctx, s.config.leadingInset);
        F32 limit = s.bounds.right() - margin - Px(ctx, s.config.trailingInset);
        if (leading) {
            const F32 controls = (end ? endSize.x + spacing : 0.0f) + (trailing ? trailSize.x + spacing : 0.0f);
            const F32 room = std::max(0.0f, limit - controls - spacing - x - leadSize.x);
            const F32 extra = std::clamp(Px(ctx, s.reach) - (x - s.bounds.x) - leadSize.x, 0.0f, room);
            leadSize.x += std::round(extra * Smoothstep(s.reachProgress.value_or(0.0f)));
            s.leadingRect = {x, s.tabBar.y + std::floor((height - leadSize.y) * 0.5f) + drop, leadSize.x, leadSize.y};
            layoutChild(ctx, *leading, s.leadingRect, s.leadingRect);
            chrome.push_back(leading);
            x += leadSize.x + spacing;
        }
        if (end) {
            const F32 trackHeight = std::round(std::min(Px(ctx, kTrackHeight), height - 2.0f * Px(ctx, kTrackInset)));
            const F32 centerY = s.tabBar.y + std::floor((height - trackHeight) * 0.5f) + drop + trackHeight * 0.5f;
            s.endRect = {limit - endSize.x, std::round(centerY - endSize.y * 0.5f), endSize.x, endSize.y};
            layoutChild(ctx, *end, s.endRect, s.tabBar);
            chrome.push_back(end);
            limit = s.endRect.x - spacing;
        }
        if (trailing) {
            s.trailingRect = {limit - trailSize.x, s.tabBar.y + std::floor((height - trailSize.y) * 0.5f) + drop, trailSize.x,
                              trailSize.y};
            limit = s.trailingRect.x - spacing;
        }

        if (bar) {
            separatorInstances.push_back({.rect = s.tabBar, .backgroundColor = ctx.color("background")});
        }
        const F32 inset = Px(ctx, kTrackInset);
        const F32 trackHeight = std::round(std::min(Px(ctx, kTrackHeight), height - 2.0f * inset));
        const F32 trackY = s.tabBar.y + std::floor((height - trackHeight) * 0.5f) + drop;
        const F32 tabHeight = trackHeight - 2.0f * inset;
        const F32 tabY = trackY + inset;
        const F32 padding = Px(ctx, kTabPadX);
        const F32 reserve = Px(ctx, kPillCloseGap) + closeSize + Px(ctx, kPillCloseMargin);
        tabRadius = tabHeight * 0.5f;
        trackRadius = trackHeight * 0.5f;
        const U64 count = s.config.hideTabs ? 0 : s.order.size();
        const F32 tabGap = Px(ctx, kTabGap);
        const F32 gaps = count > 1 ? tabGap * static_cast<F32>(count - 1) : 0.0f;
        const F32 room = std::max(0.0f, limit - x - 2.0f * inset - gaps);
        const F32 width = count == 0 ? 0.0f : std::floor(std::min(room / static_cast<F32>(count), Px(ctx, kTabMaxWidth)));
        if (count > 0) {
            trackInstances.push_back({
                .rect = {x, trackY, width * static_cast<F32>(count) + gaps + 2.0f * inset, trackHeight},
                .backgroundColor = ctx.color("pill_bg"),
            });
        }
        for (U64 index = 0; index < count; ++index) {
            const std::string& id = s.order[index];
            const auto* item = s.pane(id);
            const bool closable = item && item->closable;
            const Rect tab = {x + inset + (width + tabGap) * static_cast<F32>(index), tabY, width, tabHeight};
            s.tabRects.push_back(tab);
            s.tabPanes.push_back(id);
            const bool selected = index == s.active;
            const bool hovered = s.hoveredTab == index;
            if (selected) {
                activeInstances.push_back({.rect = tab, .backgroundColor = ctx.color("pill_active_bg")});
                tabRimInstances.push_back({.rect = tab,
                                           .backgroundColor = {0.0f, 0.0f, 0.0f, 0.0f}});
            } else if (hovered) {
                tabInstances.push_back({.rect = tab, .backgroundColor = ctx.color("pill_hover_bg")});
            }
            if (index + 1 < count && index != s.active && index + 1 != s.active) {
                const F32 divider = Px(ctx, kDividerHeight);
                dividerInstances.push_back({
                    .rect = {std::round(tab.right() + (tabGap - line) * 0.5f), tabY + std::floor((tabHeight - divider) * 0.5f),
                             line, divider},
                    .backgroundColor = ctx.color("tile_border"),
                });
            }
            const auto it = s.tabCaptions.find(id);
            auto* caption = it == s.tabCaptions.end() ? nullptr : &it->second;
            const auto textKey = selected ? "text_primary" : "text_secondary";
            F32 cursor = tab.x + padding;
            if (caption && !caption->icon.empty()) {
                caption->glyph->update({.str = caption->icon, .colorKey = textKey});
                const auto glyph = measureChild(*caption->glyph, ctx, {infinite, infinite});
                layoutChild(ctx, *caption->glyph,
                            {cursor, tabY + std::floor((tabHeight - glyph.y) * 0.5f), glyph.x, glyph.y}, tab);
                labels.push_back(caption->glyph.get());
                cursor += glyph.x + Px(ctx, kIconGap);
            }
            const bool unsaved = item && item->unsaved;
            const F32 markerRoom = unsaved ? Px(ctx, kUnsavedGap + kUnsavedDot) : 0.0f;
            const F32 titleRoom = std::max(0.0f, tab.right() - cursor - (closable ? reserve : padding) - markerRoom);
            if (caption) {
                auto& text = *caption->text;
                text.update({.str = Fit(s.metrics, caption->title, font, titleRoom, kTabFont),
                             .font = Text::Font::Display,
                             .colorKey = textKey});
                const Extent2D<F32> textSize = measureChild(text, ctx, {infinite, infinite});
                const F32 textWidth = std::min(textSize.x, titleRoom);
                layoutChild(ctx, text, {cursor, tabY + std::floor((tabHeight - textSize.y) * 0.5f), textWidth, textSize.y}, tab);
                labels.push_back(&text);
                const auto size = prepareGlyph(*caption->marker, ICON_LC_DOT, "cyber_blue", kUnsavedGlyph,
                                               unsaved ? 1.0f : 0.0f);
                placeGlyph(*caption->marker,
                           {cursor + textWidth + Px(ctx, kUnsavedGap + kUnsavedDot * 0.5f), tabY + tabHeight * 0.5f},
                           size, tab);
            }
            const bool showClose = closable && (selected || hovered);
            const Rect close = closable ? Rect{tab.right() - Px(ctx, kPillCloseMargin) - closeSize,
                                               tabY + std::floor((tabHeight - closeSize) * 0.5f), closeSize, closeSize}
                                        : Rect{};
            s.tabCloseRects.push_back(showClose ? close : Rect{});
            if (closable && caption) {
                glyphButton(*caption->close, close, ICON_LC_X,
                            hovered && s.hoveredTabClose ? "tile_button_hover" : textKey, kCloseGlyph,
                            !showClose ? 0.0f : hovered ? 1.0f : kIdleCloseOpacity);
            }
        }
        if (trailing && count > 0) {
            s.trailingRect.x = x + width * static_cast<F32>(count) + gaps + 2.0f * inset + spacing;
            layoutChild(ctx, *trailing, s.trailingRect, s.trailingRect);
            chrome.push_back(trailing);
        } else {
            s.trailingRect = {};
        }
        const F32 overlap = s.tabBar.height > 0.0f ? std::min(Px(ctx, kCardGap), s.tabBar.height) : 0.0f;
        const F32 contentTop = s.tabBar.height - overlap;
        s.contentArea = {s.bounds.x, s.bounds.y + contentTop, s.bounds.width,
                         std::max(0.0f, s.bounds.height - contentTop)};
        if (auto* aside = s.config.aside) {
            const F32 room = std::max(0.0f, s.contentArea.width * 0.5f - gap);
            const F32 natural = s.config.asideWidth == 0.0f
                                    ? measureChild(*aside, ctx, {room, s.contentArea.height}).x
                                    : 0.0f;
            const F32 width = std::round(ResolveLength(ctx, s.config.asideWidth, natural, room)) + gap;
            const Rect area = {s.contentArea.right() - width, s.contentArea.y, width, s.contentArea.height};
            s.contentArea.width -= width - gap;
            if (!s.config.hideTabs) {
                backgroundInstances.push_back({.rect = area, .backgroundColor = ctx.color("background")});
            }
            layoutChild(ctx, *aside, area, area);
            chrome.push_back(aside);
        }
    } else {
        s.tabBar = {};
        s.contentArea = s.bounds;
    }

    if (tabs) {
        if (s.active < s.order.size()) {
            s.leaves.push_back({.pane = s.order[s.active], .tile = s.contentArea});
        }
    } else {
        const Rect inset = Inset(s.contentArea, gap);
        const F32 left = std::round(inset.x);
        const F32 top = std::round(inset.y);
        s.viewport = {left, top, std::max(0.0f, std::round(inset.right()) - left),
                      std::max(0.0f, std::round(inset.bottom()) - top)};
        s.placePending();
        s.apply(s.arrange(s.model));
    }
    const U64 total = tabs ? (s.active < s.order.size() ? 1 : 0) : s.placed().size();
    if (!s.config.resizable) {
        s.splitters.clear();
        s.hoveredSplitter.reset();
        if (s.press == Impl::Press::Splitter) {
            s.commit();
            s.press = Impl::Press::None;
        }
    }
    s.showHeaders = !tabs && s.config.headers;
    const auto expanded = std::count_if(s.leaves.begin(), s.leaves.end(), [](const auto& leaf) {
        return !leaf.minimized;
    });

    const F32 headerHeight = std::round(Px(ctx, Dock::HeaderHeight));
    const F32 button = Px(ctx, kHeaderButton);
    const F32 titleSize = Px(ctx, kHeaderTitleSize);
    for (auto& leaf : s.leaves) {
        const auto* item = s.pane(leaf.pane);
        Component* placeholder = nullptr;
        if (!item) {
            const auto it = s.placeholders.find(leaf.pane);
            if (it == s.placeholders.end()) {
                continue;
            }
            placeholder = &it->second->stack;
        }
        const bool closable = item ? item->closable : true;
        const bool bare = !s.config.loneHeader && s.model.maximized.empty() && expanded == 1 && !leaf.minimized &&
                          item && !closable;
        Rect content = leaf.tile;
        if (!tabs && item && !item->card) {
            content = Intersect(Inflate(leaf.tile, gap), s.contentArea);
        }
        if (!tabs && (!item || item->card)) {
            tileRects.push_back(leaf.tile);
            if (item && item->fill.a > 0.0f && !leaf.minimized) {
                tileFillInstances.push_back({.rect = leaf.tile, .backgroundColor = item->fill});
            } else {
                frameRects.push_back(leaf.tile);
            }
        }
        const bool fullscreenable = item && item->maximizable && total > 1;
        const bool minimizable = item && !leaf.minimized && s.canMinimize(leaf.pane);
        const auto fullscreenButton = [&](Impl::Caption* caption, const Rect& rect) {
            leaf.maximize = rect;
            if (caption) {
                const bool fullscreen = s.model.maximized == leaf.pane;
                glyphButton(*caption->maximize, rect, fullscreen ? ICON_LC_MINIMIZE_2 : ICON_LC_MAXIMIZE_2,
                            s.hoveredHeaderMaximize == leaf.pane ? "tile_button_hover"
                            : fullscreen                         ? "cyber_blue"
                                                                 : "tile_header_text",
                            kHeaderGlyph);
            }
        };
        if (s.showHeaders && !bare && leaf.strip) {
            leaf.header = leaf.tile;
            const auto found = s.headerCaptions.find(leaf.pane);
            auto* caption = found == s.headerCaptions.end() ? nullptr : &found->second;
            const F32 left = std::round(leaf.tile.x + (leaf.tile.width - button) * 0.5f);
            const F32 margin = left - leaf.tile.x;
            if (fullscreenable) {
                fullscreenButton(caption, {left, leaf.tile.bottom() - margin - button, button, button});
            }
            if (closable) {
                leaf.close = {left, leaf.tile.y + margin, button, button};
                if (caption) {
                    glyphButton(*caption->close, leaf.close, ICON_LC_X,
                                s.hoveredHeaderClose == leaf.pane ? "tile_button_hover" : "tile_header_text",
                                kHeaderGlyph);
                }
            }
            if (caption) {
                const F32 reserve = std::max(Px(ctx, kStripPad), margin + button + Px(ctx, kHeaderButtonGap));
                const F32 room = std::max(0.0f, leaf.tile.height - 2.0f * reserve);
                const F32 spineSize = Px(ctx, kStripTitleSize);
                const std::string spine = Fit(s.metrics, caption->title, spineSize, room, kHeaderFont);
                const F32 ink = s.metrics.inkOffset(kHeaderFont, spine, spineSize) - Px(ctx, kInkNudge);
                caption->spine->update({
                    .instances = {{
                        .rect = {leaf.tile.x + ink, leaf.tile.y, leaf.tile.width, leaf.tile.height},
                        .str = spine,
                        .visible = !leaf.tile.empty(),
                        .color = ctx.color("tile_header_text"),
                        .fontSize = spineSize,
                        .alignment = {1, 1},
                        .rotation = kStripRotation,
                    }},
                    .clip = leaf.tile,
                    .fontName = kHeaderFont,
                });
                layoutChild(ctx, *caption->spine, leaf.tile);
                labels.push_back(caption->spine.get());
            }
            content = {};
        } else if (s.showHeaders && !bare) {
            leaf.header = leaf.minimized ? leaf.tile
                                         : Rect{leaf.tile.x, leaf.tile.y, leaf.tile.width,
                                                std::min(headerHeight, leaf.tile.height)};
            const F32 inset = Px(ctx, std::max(kHeaderPadX, item ? item->padding : 0.0f));
            const F32 titleY = leaf.header.y + (leaf.minimized ? std::round((leaf.header.height - titleSize) * 0.5f)
                                                               : std::round(Px(ctx, kHeaderTitleTop)));
            const F32 centerY = titleY + titleSize * 0.5f;
            const F32 buttonY = std::round(centerY - button * 0.5f);
            const auto found = s.headerCaptions.find(leaf.pane);
            auto* caption = found == s.headerCaptions.end() ? nullptr : &found->second;
            F32 right = leaf.tile.right() - inset;
            if (closable && (!item || !item->closeOnlyMinimized || leaf.minimized)) {
                const char* closeKey = s.hoveredHeaderClose == leaf.pane ? "tile_button_hover" : "tile_header_text";
                const Extent2D<F32> size = caption ? prepareGlyph(*caption->close, ICON_LC_X, closeKey, kHeaderGlyph, 1.0f)
                                                   : Extent2D<F32>{Px(ctx, kHeaderGlyph), Px(ctx, kHeaderGlyph)};
                const F32 centerX = right - size.x * 0.5f;
                leaf.close = {std::round(centerX - button * 0.5f), buttonY, button, button};
                if (caption) {
                    placeGlyph(*caption->close, {centerX, centerY}, size, leaf.close);
                }
                right = leaf.close.x - Px(ctx, kHeaderButtonGap);
            }
            if (fullscreenable) {
                fullscreenButton(caption, {right - button, buttonY, button, button});
                right = leaf.maximize.x - Px(ctx, kHeaderButtonGap);
            }
            if (minimizable) {
                leaf.minimize = {right - button, buttonY, button, button};
                right = leaf.minimize.x - Px(ctx, kHeaderButtonGap);
                if (caption) {
                    glyphButton(*caption->minimize, leaf.minimize, ICON_LC_MINUS,
                                s.hoveredHeaderMinimize == leaf.pane ? "tile_button_hover" : "tile_header_text",
                                kHeaderGlyph);
                }
            }
            if (item && !leaf.minimized) {
                for (U64 index = 0; index < item->actions.size(); ++index) {
                    const auto& action = item->actions[index];
                    const Rect rect = {right - button, buttonY, button, button};
                    right = rect.x - Px(ctx, kHeaderButtonGap);
                    leaf.actions.push_back(rect);
                    if (caption && index < caption->actions.size()) {
                        const bool hovered = s.hoveredHeaderAction ==
                                             std::optional<std::pair<std::string, U64>>(std::make_pair(leaf.pane, index));
                        glyphButton(*caption->actions[index], rect, action.icon.c_str(),
                                    hovered ? "tile_button_hover" : action.active ? "cyber_blue" : "tile_header_text",
                                    kHeaderGlyph);
                    }
                }
            }
            if (caption) {
                const F32 cursor = leaf.tile.x + inset;
                const F32 room = std::max(0.0f, right - Px(ctx, kHeaderIconGap) - cursor);
                const std::string title = Fit(s.metrics, caption->title, titleSize, room, kHeaderFont);
                const F32 ink = leaf.minimized
                                    ? std::round(s.metrics.inkOffset(kHeaderFont, title, titleSize) - Px(ctx, kInkNudge))
                                    : 0.0f;
                caption->text->update({.str = title,
                                      .font = Text::Font::DisplaySemiBold,
                                      .colorKey = "tile_header_text",
                                      .scale = kHeaderTitleSize / Typography::FontSize});
                const Extent2D<F32> textSize = measureChild(*caption->text, ctx, {infinite, infinite});
                const F32 width = std::min(textSize.x, room);
                layoutChild(ctx, *caption->text, {cursor, titleY - ink, width, textSize.y},
                            {cursor, leaf.tile.y, room, leaf.tile.height});
                labels.push_back(caption->text.get());
            }
            content = leaf.minimized ? Rect{}
                                     : Rect{leaf.tile.x, leaf.header.bottom(), leaf.tile.width,
                                            std::max(0.0f, leaf.tile.bottom() - leaf.header.bottom())};
        }
        if (tabs && !s.config.hideTabs) {
            backgroundInstances.push_back({.rect = content, .backgroundColor = ctx.color("background")});
        }
        leaf.body = content;
        leaf.content = Inset(content, Px(ctx, item ? item->padding : 0.0f));
        leaf.content.x -= std::max(0.0f, leaf.slide);
        leaf.content.width += std::abs(leaf.slide);
        if (leaf.minimized) {
            continue;
        }
        if (item && item->content) {
            contents.push_back(item->content);
        }
        if (placeholder) {
            const auto size = measureChild(*placeholder, ctx, {leaf.content.width, infinite});
            if (!leaf.body.empty() && size.y <= leaf.content.height) {
                layoutChild(ctx, *placeholder,
                            {leaf.content.x + std::floor((leaf.content.width - size.x) * 0.5f),
                             leaf.content.y + std::floor((leaf.content.height - size.y) * 0.5f), size.x, size.y},
                            leaf.body);
                labels.push_back(placeholder);
            }
        }
    }

    const F32 splitterLine = Px(ctx, kSplitterLine);
    for (U64 index = 0; index < s.splitters.size(); ++index) {
        const bool hot = s.hoveredSplitter == index ||
                         (s.press == Impl::Press::Splitter && Impl::sameSplitter(s.splitters[index], s.pressLayout));
        const Rect& rect = s.splitters[index].rect;
        const Rect mark = s.splitters[index].vertical
                              ? Rect{rect.x + radius, rect.y + std::floor((rect.height - splitterLine) * 0.5f),
                                     std::max(0.0f, rect.width - 2.0f * radius), splitterLine}
                              : Rect{rect.x + std::floor((rect.width - splitterLine) * 0.5f), rect.y + radius,
                                     splitterLine, std::max(0.0f, rect.height - 2.0f * radius)};
        splitterInstances.push_back({.rect = mark, .visible = hot, .backgroundColor = ctx.color("separator_hovered")});
    }

    std::vector<Component*> shown = std::move(labels);
    shown.insert(shown.end(), chrome.begin(), chrome.end());
    shown.insert(shown.end(), contents.begin(), contents.end());
    const std::weak_ptr<bool> token = Access::alive(*this);
    const std::vector<Component*>& mounted = children();
    std::vector<std::pair<Component*, std::weak_ptr<bool>>> managed;
    for (auto* child : s.managed) {
        if (std::find(mounted.begin(), mounted.end(), child) != mounted.end()) {
            managed.push_back({child, Access::alive(*child)});
        }
    }
    const auto present = [&](Component* child, const std::weak_ptr<bool>& alive) {
        return !token.expired() && !alive.expired() && hasChild(*child);
    };
    for (const auto& [child, alive] : managed) {
        if (!present(child, alive)) {
            continue;
        }
        const bool visible = std::find(shown.begin(), shown.end(), child) != shown.end();
        if (!visible && child->visible()) {
            cancelMouse(*child);
            if (present(child, alive)) {
                releaseFocusWithin(*child);
            }
        }
        if (present(child, alive)) {
            child->setVisible(visible);
        }
    }
    if (token.expired()) {
        return;
    }

    s.backgrounds.update({.instances = std::move(backgroundInstances), .clip = s.bounds, .capacity = kBoxCapacity});
    s.separators.update({.instances = std::move(separatorInstances), .clip = s.bounds, .capacity = kBoxCapacity});
    s.tabBoxes.update({.instances = std::move(tabInstances), .clip = s.bounds, .cornerRadius = tabRadius,
                       .capacity = kBoxCapacity});
    s.track.update({.instances = std::move(trackInstances), .clip = s.bounds, .cornerRadius = trackRadius,
                    .capacity = kBoxCapacity});
    s.activeTab.update({.instances = std::move(activeInstances), .clip = s.bounds, .cornerRadius = tabRadius,
                        .borderWidth = Px(ctx, 1.0f), .borderColor = ctx.color("tile_border"), .strokeFringe = true,
                        .capacity = kBoxCapacity});
    s.tabRim.update({.instances = std::move(tabRimInstances), .clip = s.bounds,
                     .cornerRadius = tabRadius, .borderWidth = Px(ctx, 1.0f),
                     .borderColor = RimColor(ctx.color("pill_active_bg")), .strokeFringe = true, .topLit = true,
                     .capacity = kBoxCapacity});
    s.dividers.update({.instances = std::move(dividerInstances), .clip = s.bounds, .capacity = kBoxCapacity});
    Card::Config tile = {
        .rects = std::move(tileRects),
        .clip = s.bounds,
        .cornerRadius = radius,
        .softness = Px(ctx, kShadowSoftness),
        .drop = Px(ctx, kShadowDrop),
        .borderWidth = Px(ctx, 1.0f),
        .lineWidth = Px(ctx, 1.0f),
        .layer = Card::Layer::Base,
        .capacity = kBoxCapacity,
    };
    s.tileBase.update(tile);
    for (auto& frame : s.fillFrames) {
        std::vector<Box::Instance> outlines;
        for (const auto& fill : tileFillInstances) {
            if (fill.backgroundColor == frame->fill) {
                outlines.push_back({.rect = fill.rect, .backgroundColor = {0.0f, 0.0f, 0.0f, 0.0f}});
            }
        }
        const ColorRGBA<F32>& fill = frame->fill;
        const F32 luma = 0.2126f * fill.r + 0.7152f * fill.g + 0.0722f * fill.b;
        const ColorRGBA<F32> contrast = luma < 0.5f ? ColorRGBA<F32>{1.0f, 1.0f, 1.0f, 1.0f}
                                                    : ColorRGBA<F32>{0.0f, 0.0f, 0.0f, 1.0f};
        frame->border.update({.instances = outlines, .clip = s.bounds, .cornerRadius = radius,
                              .borderWidth = Px(ctx, 1.0f), .borderColor = MixColor(fill, contrast, kFillBorder),
                              .strokeFringe = true, .capacity = kBoxCapacity});
        frame->rim.update({.instances = std::move(outlines), .clip = s.bounds, .cornerRadius = radius,
                           .borderWidth = Px(ctx, 1.0f), .borderColor = MixColor(fill, contrast, kFillRim),
                           .strokeFringe = true, .topLit = true, .capacity = kBoxCapacity});
        layoutChild(ctx, frame->border, s.bounds);
        layoutChild(ctx, frame->rim, s.bounds);
    }
    s.tileFills.update({.instances = std::move(tileFillInstances), .clip = s.bounds, .cornerRadius = radius,
                        .capacity = kBoxCapacity});
    tile.layer = Card::Layer::Frame;
    tile.rects = std::move(frameRects);
    s.tileFrame.update(std::move(tile));
    s.splitterBoxes.update({.instances = std::move(splitterInstances), .clip = s.bounds,
                            .cornerRadius = splitterLine * 0.5f, .capacity = kBoxCapacity});
    const Rect hoveredButton = s.hoveredHeaderButton();
    std::vector<Box::Instance> buttonInstances;
    F32 buttonRadius = hoveredButton.width * 0.5f;
    for (const auto& leaf : s.leaves) {
        if (s.model.maximized == leaf.pane && !leaf.maximize.empty()) {
            buttonInstances.push_back({.rect = leaf.maximize, .backgroundColor = ctx.color("tile_button_active_bg")});
            buttonRadius = leaf.maximize.width * 0.5f;
        }
    }
    const bool activeHovered = std::any_of(buttonInstances.begin(), buttonInstances.end(), [&](const auto& instance) {
        return instance.rect == hoveredButton;
    });
    if (!hoveredButton.empty() && !activeHovered) {
        buttonInstances.push_back({.rect = hoveredButton, .backgroundColor = ctx.color("chrome_button_hover")});
    }
    s.buttonHover.update({
        .instances = std::move(buttonInstances),
        .clip = s.bounds,
        .cornerRadius = buttonRadius,
        .capacity = 2,
    });
    for (auto* primitive : std::initializer_list<Component*>{&s.backgrounds, &s.separators, &s.track, &s.tabBoxes,
                                                             &s.activeTab, &s.tabRim, &s.dividers, &s.tileBase,
                                                             &s.tileFills, &s.splitterBoxes, &s.buttonHover,
                                                             &s.tileFrame}) {
        layoutChild(ctx, *primitive, s.bounds);
    }

    Context carded = ctx;
    carded.backdropColorKey = "tile_bg";
    carded.backdropTopColorKey.clear();
    for (auto& frame : s.fillFrames) {
        frame->palette = ctx.palette.get();
        const ColorRGBA<F32> base = ctx.color("tile_bg");
        const ColorRGBA<F32>& fill = frame->fill;
        frame->palette["tile_fill"] = MixColor(base, {fill.r, fill.g, fill.b, base.a}, fill.a);
    }
    for (const auto& leaf : s.leaves) {
        const auto* item = s.pane(leaf.pane);
        if (!item || !item->content || leaf.minimized) {
            continue;
        }
        if (tabs || !item->card) {
            layoutChild(ctx, *item->content, leaf.content, leaf.body);
            continue;
        }
        const auto frame = std::find_if(s.fillFrames.begin(), s.fillFrames.end(), [&](const auto& candidate) {
            return item->fill.a > 0.0f && candidate->fill == item->fill;
        });
        if (frame == s.fillFrames.end()) {
            layoutChild(carded, *item->content, leaf.content, leaf.body);
            continue;
        }
        Context tinted = carded;
        tinted.palette = std::cref((*frame)->palette);
        tinted.backdropColorKey = "tile_fill";
        layoutChild(tinted, *item->content, leaf.content, leaf.body);
    }

    const auto& adopts = s.config.adopts;
    if (!s.drag || !s.drag->active || Access::within(*this, s.drag->content) ||
        (s.drag->group != s.config.accepts &&
         std::find(adopts.begin(), adopts.end(), s.drag->group) == adopts.end()) ||
        !ctx.mouse.has_value() ||
        !s.bounds.contains(ctx.mouse->x, ctx.mouse->y)) {
        return;
    }
    const Extent2D<F32> point = *ctx.mouse;
    const std::string dragged = s.drag->pane;
    const Extent2D<F32> shift = ctx.translation;
    const auto offer = [&](Rect preview, std::function<void(const std::string&)> accept) {
        preview.x += shift.x;
        preview.y += shift.y;
        s.drag->offer({
            .preview = preview,
            .accept = [this, accept = std::move(accept)](const std::string& pane) {
                accept(pane);
                invalidate(Dirty::Layout);
            },
            .owner = this,
        });
    };
    auto* state = impl.get();

    if (tabs) {
        if (!s.tabBar.contains(point.x, point.y)) {
            return;
        }
        U64 index = s.tabRects.size();
        for (U64 tab = 0; tab < s.tabRects.size(); ++tab) {
            if (point.x < s.tabRects[tab].x + s.tabRects[tab].width * 0.5f) {
                index = tab;
                break;
            }
        }
        const Rect area = s.contentArea.empty() ? s.bounds : s.contentArea;
        offer(Inset(area, gap), [state, index](const std::string& pane) {
            state->moveTab(pane, index);
        });
        return;
    }

    const auto removed = s.removal(dragged);
    const auto target = s.aim(point, dragged, removed);
    if (const auto planned = target ? s.plan(dragged, *target, removed) : std::nullopt) {
        offer(s.preview(*target, *planned), [state, target = *target](const std::string& pane) {
            state->drop(pane, target);
        });
    }
}

bool Dock::mouseEvent(const MouseEvent& event) {
    auto& s = *impl;
    const Extent2D<F32> point = event.position;
    switch (event.type) {
        case MouseEventType::Move: {
            if (s.dragging) {
                if (s.dragPoint != point) {
                    s.dragPoint = point;
                    invalidate(Dirty::Layout);
                }
                return true;
            }
            if (s.press == Impl::Press::Splitter) {
                if (s.resize(s.pressLayout, point)) {
                    invalidate(Dirty::Layout);
                }
                return true;
            }
            if ((s.press == Impl::Press::Tab || s.press == Impl::Press::Header) && s.drag && s.known(s.pressPane)) {
                const F32 dx = point.x - s.pressPoint.x;
                const F32 dy = point.y - s.pressPoint.y;
                const F32 threshold = kDragThreshold * s.pixelRatio;
                if (dx * dx + dy * dy > threshold * threshold) {
                    const auto* item = s.pane(s.pressPane);
                    s.drag->reset();
                    s.drag->active = true;
                    s.drag->pane = s.pressPane;
                    s.drag->group = item->group;
                    s.drag->title = item->title;
                    s.drag->icon = s.config.tabs ? item->icon : std::string();
                    s.drag->source = this;
                    s.drag->content = item->content;
                    s.dragging = true;
                    s.dragPoint = point;
                    invalidate(Dirty::Layout);
                    return true;
                }
            }

            const auto tab = s.tabAt(point);
            const bool tabClose = tab.has_value() && s.tabCloseRects[*tab].contains(point.x, point.y);
            std::string headerClose;
            std::string headerMaximize;
            std::string headerMinimize;
            std::optional<std::pair<std::string, U64>> headerAction;
            for (const auto& leaf : s.leaves) {
                if (leaf.close.contains(point.x, point.y)) {
                    headerClose = leaf.pane;
                }
                if (leaf.maximize.contains(point.x, point.y)) {
                    headerMaximize = leaf.pane;
                }
                if (leaf.minimize.contains(point.x, point.y)) {
                    headerMinimize = leaf.pane;
                }
                for (U64 index = 0; index < leaf.actions.size(); ++index) {
                    if (leaf.actions[index].contains(point.x, point.y)) {
                        headerAction = std::make_pair(leaf.pane, index);
                    }
                }
            }
            const auto splitter = s.splitterAt(point);
            if (tab != s.hoveredTab || tabClose != s.hoveredTabClose || headerClose != s.hoveredHeaderClose ||
                headerMaximize != s.hoveredHeaderMaximize || headerMinimize != s.hoveredHeaderMinimize ||
                headerAction != s.hoveredHeaderAction ||
                splitter != s.hoveredSplitter) {
                s.hoveredTab = tab;
                s.hoveredTabClose = tabClose;
                s.hoveredHeaderClose = headerClose;
                s.hoveredHeaderMaximize = headerMaximize;
                s.hoveredHeaderMinimize = headerMinimize;
                s.hoveredHeaderAction = headerAction;
                s.hoveredSplitter = splitter;
                invalidate(Dirty::Layout);
            }
            if (s.press != Impl::Press::None) {
                return true;
            }
            return mouseEventChildren(event);
        }
        case MouseEventType::Click: {
            if (event.clicks < 2) {
                s.restored.clear();
            }
            if (event.button == MouseButton::Right && s.config.tabs) {
                if (const auto tab = s.tabAt(point)) {
                    if (s.config.onTabMenu && *tab < s.tabPanes.size()) {
                        s.config.onTabMenu(s.tabPanes[*tab], point);
                    }
                    return true;
                }
            }
            if (event.button == MouseButton::Left) {
                if (const auto tab = s.tabAt(point)) {
                    s.pressTab = *tab;
                    s.pressPane = *tab < s.tabPanes.size() ? s.tabPanes[*tab] : std::string();
                    s.pressPoint = point;
                    s.press = s.tabCloseRects[*tab].contains(point.x, point.y) ? Impl::Press::TabClose
                                                                                : Impl::Press::Tab;
                    return true;
                }
                if (s.config.tabs && s.tabBar.contains(point.x, point.y) &&
                    !s.trailingRect.contains(point.x, point.y) && !s.endRect.contains(point.x, point.y)) {
                    if (s.leadingRect.contains(point.x, point.y) && mouseEventChildren(event)) {
                        return true;
                    }
                    if (s.config.onBarPress) {
                        s.config.onBarPress(event.clicks);
                    }
                    return true;
                }
                for (const auto& leaf : s.leaves) {
                    if (leaf.close.contains(point.x, point.y)) {
                        s.press = Impl::Press::HeaderClose;
                        s.pressPane = leaf.pane;
                        s.pressPoint = point;
                        return true;
                    }
                    if (leaf.maximize.contains(point.x, point.y)) {
                        s.press = Impl::Press::HeaderMaximize;
                        s.pressPane = leaf.pane;
                        s.pressPoint = point;
                        return true;
                    }
                    if (leaf.minimize.contains(point.x, point.y)) {
                        s.press = Impl::Press::HeaderMinimize;
                        s.pressPane = leaf.pane;
                        s.pressPoint = point;
                        return true;
                    }
                    for (U64 index = 0; index < leaf.actions.size(); ++index) {
                        if (leaf.actions[index].contains(point.x, point.y)) {
                            s.press = Impl::Press::HeaderAction;
                            s.pressPane = leaf.pane;
                            s.pressAction = index;
                            s.pressPoint = point;
                            return true;
                        }
                    }
                    if (leaf.header.contains(point.x, point.y)) {
                        const auto* item = s.pane(leaf.pane);
                        if (event.clicks >= 2 && !leaf.minimized && s.restored != leaf.pane && item &&
                            item->maximizable && s.placed().size() > 1) {
                            s.press = Impl::Press::Handled;
                            s.model.maximized = s.model.maximized == leaf.pane ? std::string() : leaf.pane;
                            s.focus(leaf.pane);
                            s.commitView();
                            invalidate(Dirty::Layout);
                            return true;
                        }
                        s.press = Impl::Press::Header;
                        s.pressPane = leaf.pane;
                        s.pressPoint = point;
                        s.focus(leaf.pane);
                        invalidate(Dirty::Layout);
                        return true;
                    }
                }
                if (const auto splitter = s.splitterAt(point)) {
                    const auto layout = s.splitters[*splitter];
                    if (event.clicks >= 2) {
                        s.press = Impl::Press::Handled;
                        s.equalize(layout);
                        invalidate(Dirty::Layout);
                        return true;
                    }
                    if (!s.beginResize(layout)) {
                        return true;
                    }
                    s.press = Impl::Press::Splitter;
                    s.pressLayout = layout;
                    s.pressPoint = point;
                    invalidate(Dirty::Layout);
                    return true;
                }
            }
            if (const auto* leaf = s.leafAt(point)) {
                s.focus(leaf->pane);
            }
            return mouseEventChildren(event);
        }
        case MouseEventType::Release: {
            if (event.button != MouseButton::Left) {
                return mouseEventChildren(event);
            }
            const auto press = s.press;
            s.press = Impl::Press::None;
            if (s.dragging) {
                s.dragging = false;
                if (s.drag) {
                    const std::string pane = s.drag->pane;
                    const auto target = s.drag->target;
                    s.drag->reset();
                    if (target.has_value() && target->accept) {
                        if (target->owner != this) {
                            s.removeOwn(pane);
                        }
                        target->accept(pane);
                    }
                }
                invalidate(Dirty::Layout);
                return true;
            }
            if (press == Impl::Press::Tab && s.pressTab < s.tabRects.size()) {
                if (s.active != s.pressTab) {
                    s.active = s.pressTab;
                    invalidate(Dirty::Layout);
                }
                s.focus(s.pressPane);
                return true;
            }
            if (press == Impl::Press::TabClose && s.pressTab < s.tabCloseRects.size() &&
                s.tabCloseRects[s.pressTab].contains(point.x, point.y)) {
                if (const auto* item = s.pane(s.pressPane); item && item->onClose) {
                    item->onClose();
                }
                return true;
            }
            if (press == Impl::Press::HeaderMaximize) {
                for (const auto& leaf : s.leaves) {
                    if (leaf.pane == s.pressPane && leaf.maximize.contains(point.x, point.y)) {
                        s.model.maximized = s.model.maximized == leaf.pane ? std::string() : leaf.pane;
                        s.focus(leaf.pane);
                        s.commitView();
                        invalidate(Dirty::Layout);
                        break;
                    }
                }
                return true;
            }
            if (press == Impl::Press::Header) {
                for (const auto& leaf : s.leaves) {
                    if (leaf.pane == s.pressPane && leaf.minimized && leaf.tile.contains(point.x, point.y)) {
                        s.toggleMinimized(leaf.pane);
                        s.restored = leaf.pane;
                        invalidate(Dirty::Layout);
                        break;
                    }
                }
                return true;
            }
            if (press == Impl::Press::HeaderMinimize) {
                for (const auto& leaf : s.leaves) {
                    if (leaf.pane == s.pressPane && leaf.minimize.contains(point.x, point.y)) {
                        s.toggleMinimized(leaf.pane);
                        invalidate(Dirty::Layout);
                        break;
                    }
                }
                return true;
            }
            if (press == Impl::Press::HeaderAction) {
                std::function<void()> onClick;
                for (const auto& leaf : s.leaves) {
                    if (leaf.pane == s.pressPane && s.pressAction < leaf.actions.size() &&
                        leaf.actions[s.pressAction].contains(point.x, point.y)) {
                        if (const auto* item = s.pane(leaf.pane); item && s.pressAction < item->actions.size()) {
                            onClick = item->actions[s.pressAction].onClick;
                        }
                    }
                }
                if (onClick) {
                    s.focus(s.pressPane);
                    onClick();
                }
                return true;
            }
            if (press == Impl::Press::HeaderClose) {
                bool closed = false;
                for (const auto& leaf : s.leaves) {
                    if (leaf.pane == s.pressPane && leaf.close.contains(point.x, point.y)) {
                        closed = true;
                    }
                }
                if (closed) {
                    const auto onClose = s.pane(s.pressPane) ? s.pane(s.pressPane)->onClose : nullptr;
                    if (!s.config.tabs) {
                        s.removeOwn(s.pressPane);
                        invalidate(Dirty::Layout);
                    }
                    if (onClose) {
                        onClose();
                    }
                }
                return true;
            }
            if (press == Impl::Press::Splitter) {
                s.commit();
                invalidate(Dirty::Layout);
                return true;
            }
            if (press != Impl::Press::None) {
                return true;
            }
            return mouseEventChildren(event);
        }
        case MouseEventType::Leave:
        case MouseEventType::Cancel: {
            const bool changed = s.dragging || s.press != Impl::Press::None || s.hoveredTab.has_value() ||
                                 s.hoveredTabClose || !s.hoveredHeaderClose.empty() ||
                                 !s.hoveredHeaderMaximize.empty() || !s.hoveredHeaderMinimize.empty() ||
                                 s.hoveredHeaderAction.has_value() ||
                                 s.hoveredSplitter.has_value();
            if (s.dragging && s.drag) {
                s.drag->reset();
            }
            if (s.press == Impl::Press::Splitter) {
                s.commit();
            }
            s.dragging = false;
            s.press = Impl::Press::None;
            s.hoveredTab.reset();
            s.hoveredTabClose = false;
            s.hoveredHeaderClose.clear();
            s.hoveredHeaderMaximize.clear();
            s.hoveredHeaderMinimize.clear();
            s.hoveredHeaderAction.reset();
            s.hoveredSplitter.reset();
            if (changed) {
                invalidate(Dirty::Layout);
            }
            return mouseEventChildren(event);
        }
        default:
            return mouseEventChildren(event);
    }
}

std::optional<SurfaceCursor> Dock::cursorAt(const Extent2D<F32>& point) const {
    const auto& s = *impl;
    if (s.press == Impl::Press::Splitter) {
        return s.pressLayout.vertical ? SurfaceCursor::ResizeNS : SurfaceCursor::ResizeEW;
    }
    if (const auto splitter = s.splitterAt(point)) {
        return s.splitters[*splitter].vertical ? SurfaceCursor::ResizeNS : SurfaceCursor::ResizeEW;
    }
    const auto over = [&](const Rect& rect) {
        return !rect.empty() && rect.contains(point.x, point.y);
    };
    for (const auto& leaf : s.leaves) {
        if (over(leaf.close) || over(leaf.maximize) || over(leaf.minimize) || (leaf.minimized && over(leaf.tile)) ||
            std::any_of(leaf.actions.begin(), leaf.actions.end(), over)) {
            return SurfaceCursor::Hand;
        }
    }
    for (const auto& close : s.tabCloseRects) {
        if (over(close)) {
            return SurfaceCursor::Hand;
        }
    }
    return std::nullopt;
}

}  // namespace Jetstream::Sakura
