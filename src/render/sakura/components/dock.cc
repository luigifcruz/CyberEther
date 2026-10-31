#include <jetstream/render/sakura/components/dock.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/card.hh>
#include <jetstream/render/sakura/components/label.hh>
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
#include <limits>
#include <set>
#include <unordered_map>
#include <utility>

namespace Jetstream::Sakura {

namespace {

constexpr F32 kDragThreshold = 6.0f;
constexpr F32 kCardGap = Dock::ContentPadding;
constexpr F32 kColumnWidth = 420.0f;
constexpr F32 kFlexColumnWidth = 480.0f;
constexpr F32 kMinColumnWidth = 240.0f;
constexpr F32 kStoredWidthFloor = 64.0f;
constexpr F32 kSplitterSlack = 3.0f;
constexpr F32 kHeaderTitleTop = 6.0f;
constexpr F32 kHeaderPadX = 12.0f;
constexpr F32 kHeaderButton = 20.0f;
constexpr F32 kCloseGlyph = 13.0f;
constexpr F32 kHeaderGlyph = 12.0f;
constexpr F32 kStripWidth = 32.0f;
constexpr F32 kStripPad = 12.0f;
constexpr F32 kStripRotation = -90.0f;
constexpr F32 kStripTitleSize = 17.0f;
constexpr F32 kInkNudge = 0.5f;
constexpr F32 kIdleCloseOpacity = 0.35f;
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
constexpr const char* kTabFont = Typography::DisplayFont;
constexpr const char* kHeaderFont = Typography::DisplaySemiboldFont;
constexpr F32 kBarMargin = 6.0f;
constexpr F32 kSplitterLine = 2.0f;
constexpr U64 kBoxCapacity = 128;
constexpr F64 kHideDuration = 0.25;

enum class Side {
    Left,
    Right,
    Top,
    Bottom,
};

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

std::vector<F32> Weights(const DockTiles& tiles) {
    std::vector<F32> weights = tiles.weights;
    if (weights.size() != tiles.children.size()) {
        weights.assign(tiles.children.size(), 1.0f);
    }
    F32 sum = 0.0f;
    for (auto& weight : weights) {
        weight = std::max(weight, 1.0e-3f);
        sum += weight;
    }
    for (auto& weight : weights) {
        weight /= std::max(sum, 1.0e-6f);
    }
    return weights;
}

DockTiles Normalize(DockTiles tiles) {
    if (tiles.children.empty()) {
        tiles.weights.clear();
        return tiles;
    }
    const auto weights = Weights(tiles);
    std::vector<DockTiles> children;
    std::vector<F32> merged;
    for (U64 index = 0; index < tiles.children.size(); ++index) {
        auto child = Normalize(std::move(tiles.children[index]));
        if (child.children.size() >= 2 && child.vertical == tiles.vertical) {
            const auto childWeights = Weights(child);
            for (U64 inner = 0; inner < child.children.size(); ++inner) {
                children.push_back(std::move(child.children[inner]));
                merged.push_back(weights[index] * childWeights[inner]);
            }
            continue;
        }
        children.push_back(std::move(child));
        merged.push_back(weights[index]);
    }
    if (children.size() == 1) {
        return std::move(children.front());
    }
    tiles.children = std::move(children);
    tiles.weights = std::move(merged);
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
    const auto weights = Weights(tiles);
    DockTiles result;
    result.vertical = tiles.vertical;
    for (U64 index = 0; index < tiles.children.size(); ++index) {
        if (auto filtered = FilterNode(tiles.children[index], keep)) {
            result.children.push_back(std::move(*filtered));
            result.weights.push_back(weights[index]);
        }
    }
    if (result.children.empty()) {
        return std::nullopt;
    }
    if (result.children.size() == 1) {
        return std::move(result.children.front());
    }
    return result;
}

std::optional<DockTiles> Filter(const DockTiles& tiles, const std::function<bool(const std::string&)>& keep) {
    auto filtered = FilterNode(tiles, keep);
    if (!filtered.has_value()) {
        return std::nullopt;
    }
    return Normalize(std::move(*filtered));
}

std::optional<DockTiles> Remove(const DockTiles& tiles, const std::string& pane) {
    return Filter(tiles, [&](const std::string& id) {
        return id != pane;
    });
}

DockTiles Leaf(const std::string& pane) {
    return DockTiles{.pane = pane};
}

DockTiles SplitNode(DockTiles tiles, const std::string& target, const std::string& pane, Side side) {
    const bool vertical = side == Side::Top || side == Side::Bottom;
    const bool before = side == Side::Left || side == Side::Top;
    if (tiles.children.empty()) {
        if (tiles.pane != target) {
            return tiles;
        }
        DockTiles result;
        result.vertical = vertical;
        result.weights = {0.5f, 0.5f};
        result.children = before ? std::vector<DockTiles>{Leaf(pane), tiles} : std::vector<DockTiles>{tiles, Leaf(pane)};
        return result;
    }
    auto weights = Weights(tiles);
    for (U64 index = 0; index < tiles.children.size(); ++index) {
        const auto& child = tiles.children[index];
        if (child.children.empty() && child.pane == target && tiles.vertical == vertical) {
            const F32 half = weights[index] * 0.5f;
            weights[index] = half;
            const U64 at = before ? index : index + 1;
            tiles.children.insert(tiles.children.begin() + static_cast<I64>(at), Leaf(pane));
            weights.insert(weights.begin() + static_cast<I64>(at), half);
            tiles.weights = std::move(weights);
            return tiles;
        }
    }
    for (auto& child : tiles.children) {
        child = SplitNode(std::move(child), target, pane, side);
    }
    tiles.weights = std::move(weights);
    return tiles;
}

DockTiles Split(const DockTiles& tiles, const std::string& target, const std::string& pane, Side side) {
    return Normalize(SplitNode(tiles, target, pane, side));
}

DockTiles Append(const std::optional<DockTiles>& tiles, const std::string& pane) {
    if (!tiles.has_value()) {
        return Leaf(pane);
    }
    if (tiles->children.size() >= 2 && !tiles->vertical) {
        DockTiles result = *tiles;
        auto weights = Weights(result);
        const F32 count = static_cast<F32>(weights.size());
        for (auto& weight : weights) {
            weight *= count / (count + 1.0f);
        }
        weights.push_back(1.0f / (count + 1.0f));
        result.children.push_back(Leaf(pane));
        result.weights = std::move(weights);
        return result;
    }
    DockTiles result;
    result.vertical = false;
    result.weights = {0.5f, 0.5f};
    result.children = {*tiles, Leaf(pane)};
    return result;
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

struct ColumnModel {
    std::vector<std::string> panes;
    std::vector<F32> weights;
    F32 width = 0.0f;

    bool operator==(const ColumnModel&) const = default;
};

ColumnModel ColumnOf(const DockTiles& node, F32 width) {
    ColumnModel column{.width = width >= kStoredWidthFloor ? width : 0.0f};
    if (node.children.empty()) {
        column.panes = {node.pane};
        column.weights = {1.0f};
        return column;
    }
    const auto weights = Weights(node);
    for (U64 index = 0; index < node.children.size(); ++index) {
        const auto leaves = Leaves(node.children[index]);
        for (const auto& leaf : leaves) {
            column.panes.push_back(leaf);
            column.weights.push_back(weights[index] / static_cast<F32>(leaves.size()));
        }
    }
    return column;
}

void AppendColumns(const DockTiles& tiles, bool root, std::vector<ColumnModel>& columns) {
    if (tiles.children.empty() || tiles.vertical) {
        columns.push_back(ColumnOf(tiles, 0.0f));
        return;
    }
    for (U64 index = 0; index < tiles.children.size(); ++index) {
        const auto& child = tiles.children[index];
        if (!child.children.empty() && !child.vertical) {
            AppendColumns(child, false, columns);
            continue;
        }
        const F32 width = root && index < tiles.weights.size() ? tiles.weights[index] : 0.0f;
        columns.push_back(ColumnOf(child, width));
    }
}

std::vector<ColumnModel> ToColumns(const std::optional<DockTiles>& tiles) {
    std::vector<ColumnModel> columns;
    if (tiles.has_value()) {
        AppendColumns(*tiles, true, columns);
    }
    std::erase_if(columns, [](const ColumnModel& column) {
        return column.panes.empty();
    });
    return columns;
}

std::optional<DockTiles> FromColumns(const std::vector<ColumnModel>& columns) {
    DockTiles row;
    for (const auto& column : columns) {
        if (column.panes.empty()) {
            continue;
        }
        DockTiles node;
        if (column.panes.size() == 1) {
            node = Leaf(column.panes.front());
        } else {
            node.vertical = true;
            node.weights = column.weights;
            for (const auto& pane : column.panes) {
                node.children.push_back(Leaf(pane));
            }
        }
        row.children.push_back(std::move(node));
        row.weights.push_back(column.width);
    }
    if (row.children.empty()) {
        return std::nullopt;
    }
    if (row.children.size() == 1) {
        return std::move(row.children.front());
    }
    return row;
}

std::vector<F32> FitColumns(std::vector<F32> widths, F32 room, std::optional<U64> flex, F32 minimum) {
    F32 sum = 0.0f;
    for (const F32 width : widths) {
        sum += width;
    }
    if (widths.empty() || sum <= 0.0f || sum == room) {
        return widths;
    }
    if (sum < room && flex.has_value() && *flex < widths.size()) {
        widths[*flex] += room - sum;
        return widths;
    }
    const U64 count = widths.size();
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
                    free -= minimum;
                } else {
                    flexible += widths[index];
                }
            }
            if (free <= 0.0f || flexible <= 0.0f) {
                break;
            }
            for (U64 index = 0; index < count; ++index) {
                if (!pinned[index] && widths[index] * free / flexible < minimum) {
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
    std::vector<F32> fitted(count);
    F32 used = 0.0f;
    for (U64 index = 0; index + 1 < count; ++index) {
        fitted[index] = pinned[index] ? minimum : std::round(widths[index] * free / flexible);
        used += fitted[index];
    }
    fitted[count - 1] = std::max(0.0f, room - used);
    return fitted;
}

Rect Half(const Rect& rect, Side side) {
    switch (side) {
        case Side::Left:
            return {rect.x, rect.y, rect.width * 0.5f, rect.height};
        case Side::Right:
            return {rect.x + rect.width * 0.5f, rect.y, rect.width * 0.5f, rect.height};
        case Side::Top:
            return {rect.x, rect.y, rect.width, rect.height * 0.5f};
        case Side::Bottom:
            return {rect.x, rect.y + rect.height * 0.5f, rect.width, rect.height * 0.5f};
    }
    return rect;
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
        std::unique_ptr<Text> maximize = std::make_unique<Text>();
        std::unique_ptr<Text> minimize = std::make_unique<Text>();
        std::unique_ptr<Label> spine = std::make_unique<Label>();
        std::vector<std::unique_ptr<Text>> actions;
    };

    struct LeafLayout {
        std::string pane;
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
        U64 index = 0;
        bool vertical = false;
        Rect rect;
        F32 spanStart = 0.0f;
        F32 spanEnd = 0.0f;
        bool column = false;
        U64 left = 0;
        U64 right = 0;
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

    Config config;
    std::vector<std::string> order;
    U64 active = 0;
    std::optional<DockTiles> tree;
    std::optional<DockTiles> synced;
    std::string focusedPane;
    std::string pendingFocus;
    std::string maximized;
    std::set<std::string> minimized;
    std::set<std::string> introduced;
    std::vector<ColumnModel> columns;

    struct State {
        std::vector<std::string> order;
        U64 active = 0;
        std::optional<DockTiles> tree;
        std::string focusedPane;
        std::string maximized;
        std::set<std::string> minimized;
        std::vector<ColumnModel> columns;

        bool operator==(const State&) const = default;
    };

    State state() const {
        return {.order = order, .active = active, .tree = tree, .focusedPane = focusedPane,
                .maximized = maximized, .minimized = minimized, .columns = columns};
    }
    Rect viewport;
    std::vector<Rect> columnRects;
    std::vector<F32> columnWidths;
    std::vector<F32> columnBases;
    std::vector<F32> pressWidths;
    std::vector<F32> pressBases;

    Box backgrounds;
    Box separators;
    Box tabBoxes;
    Box track;
    Box activeTab;
    Box tabRim;
    Box dividers;
    Card tileBase;
    Box splitterBoxes;
    Box buttonHover;
    Card tileFrame;
    TextMetrics metrics;
    std::unordered_map<std::string, Caption> tabCaptions;
    std::unordered_map<std::string, Caption> headerCaptions;
    std::unordered_map<std::string, std::unique_ptr<Text>> placeholders;

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

    void collectManaged() {
        managed.clear();
        const auto captionParts = [&](Caption& caption) {
            managed.insert(managed.end(), {caption.text.get(), caption.glyph.get(), caption.close.get(), caption.maximize.get(),
                                            caption.minimize.get(), caption.spine.get()});
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
                managed.push_back(it->second.get());
            }
        }
        for (const auto& item : config.panes) {
            if (item.content) {
                managed.push_back(item.content);
            }
        }
        managed.insert(managed.end(), mountedChrome.begin(), mountedChrome.end());
    }

    std::vector<Component*> childOrder() {
        std::vector<Component*> order = {&backgrounds, &separators, &track, &tabBoxes, &activeTab,
                                         &tabRim, &dividers, &tileBase, &splitterBoxes, &buttonHover};
        order.insert(order.end(), managed.begin(), managed.end());
        order.push_back(&tileFrame);
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
        if (!maximized.empty()) {
            return maximized == id;
        }
        if (minimized.contains(id)) {
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

    std::optional<std::pair<U64, U64>> locate(const std::string& id) const {
        for (U64 column = 0; column < columns.size(); ++column) {
            const auto& panes = columns[column].panes;
            const auto it = std::find(panes.begin(), panes.end(), id);
            if (it != panes.end()) {
                return std::make_pair(column, static_cast<U64>(it - panes.begin()));
            }
        }
        return std::nullopt;
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

    F32 shownFactor(const ColumnModel& column) const {
        F32 factor = 0.0f;
        for (const auto& id : column.panes) {
            factor = std::max(factor, shownFactor(id));
        }
        return factor;
    }

    std::optional<U64> flexColumn() const {
        if (config.flexPane.empty()) {
            return std::nullopt;
        }
        if (const auto found = locate(config.flexPane);
            found && !collapsed(columns[found->first]) && shownFactor(columns[found->first]) >= 1.0f) {
            return found->first;
        }
        return std::nullopt;
    }

    bool collapsed(const ColumnModel& column) const {
        const auto minimizedPane = [this](const std::string& id) {
            return minimized.contains(id);
        };
        return std::any_of(column.panes.begin(), column.panes.end(), minimizedPane) &&
               std::all_of(column.panes.begin(), column.panes.end(), [&](const std::string& id) {
                   return minimizedPane(id) || hiddenPane(id);
               });
    }

    bool canMinimize(const std::string& id) const {
        if (!columnMode() || !maximized.empty()) {
            return false;
        }
        if (minimized.contains(id)) {
            return true;
        }
        U64 open = 0;
        for (const auto& column : columns) {
            for (const auto& pane : column.panes) {
                open += minimized.contains(pane) || hiddenPane(pane) ? 0 : 1;
            }
        }
        return open > 1;
    }

    void toggleMinimized(const std::string& id) {
        if (minimized.erase(id) > 0) {
            focus(id);
            return;
        }
        if (canMinimize(id)) {
            minimized.insert(id);
        }
    }

    void settleMinimized() {
        if (!columnMode() || !config.headers) {
            minimized.clear();
            return;
        }
        std::erase_if(minimized, [this](const std::string& id) {
            return !locate(id).has_value();
        });
        const bool covered = std::all_of(columns.begin(), columns.end(), [this](const ColumnModel& column) {
            return std::all_of(column.panes.begin(), column.panes.end(), [this](const std::string& id) {
                return minimized.contains(id) || hiddenPane(id);
            });
        });
        if (covered) {
            minimized.clear();
        }
    }

    void rebuild() {
        tree = FromColumns(columns);
    }

    F32 takeColumn(const std::string& id) {
        const auto found = locate(id);
        if (!found) {
            return 0.0f;
        }
        auto& column = columns[found->first];
        if (column.panes.size() == 1) {
            const F32 width = column.width;
            columns.erase(columns.begin() + static_cast<I64>(found->first));
            return width;
        }
        column.panes.erase(column.panes.begin() + static_cast<I64>(found->second));
        if (found->second < column.weights.size()) {
            column.weights.erase(column.weights.begin() + static_cast<I64>(found->second));
        }
        return 0.0f;
    }

    void syncColumns() {
        if (config.tiles != synced) {
            synced = config.tiles;
            columns = ToColumns(config.tiles);
        }
        std::erase_if(introduced, [this](const std::string& id) {
            return !known(id);
        });
        bool added = false;
        for (const auto& item : config.panes) {
            if (!locate(item.id)) {
                ColumnModel column{.panes = {item.id}, .weights = {1.0f}, .width = item.width};
                if (item.leftmost) {
                    columns.insert(columns.begin(), std::move(column));
                } else {
                    columns.push_back(std::move(column));
                }
                added = true;
            }
            if (introduced.insert(item.id).second && item.startMinimized) {
                minimized.insert(item.id);
            }
        }
        rebuild();
        if (added) {
            maximized.clear();
            commit();
        }
    }

    void dropColumn(const std::string& id, const std::string& target, Side side) {
        const auto before = locate(target);
        if (!before) {
            appendColumn(id);
            return;
        }
        U64 column = before->first;
        U64 row = before->second;
        const F32 width = takeColumn(id);
        if (target != id) {
            const auto after = locate(target);
            if (!after) {
                appendColumn(id);
                return;
            }
            column = after->first;
            row = after->second;
        }
        if (side == Side::Left || side == Side::Right) {
            const U64 at = std::min<U64>(side == Side::Left ? column : column + 1, columns.size());
            columns.insert(columns.begin() + static_cast<I64>(at),
                           ColumnModel{.panes = {id}, .weights = {1.0f}, .width = width});
        } else if (column < columns.size()) {
            auto& into = columns[column];
            into.weights.resize(into.panes.size(), 1.0f);
            const F32 half = into.weights[row] * 0.5f;
            into.weights[row] = half;
            const U64 at = side == Side::Top ? row : row + 1;
            into.panes.insert(into.panes.begin() + static_cast<I64>(at), id);
            into.weights.insert(into.weights.begin() + static_cast<I64>(at), half);
        }
        maximized.clear();
        minimized.erase(id);
        rebuild();
        focus(id);
        commit();
    }

    void appendColumn(const std::string& id) {
        const F32 width = takeColumn(id);
        columns.push_back({.panes = {id}, .weights = {1.0f}, .width = width});
        maximized.clear();
        minimized.erase(id);
        rebuild();
        focus(id);
        commit();
    }

    static bool sameSplitter(const SplitterLayout& a, const SplitterLayout& b) {
        return a.column == b.column && a.vertical == b.vertical && a.index == b.index && a.path == b.path;
    }

    void pressColumns() {
        pressWidths = columnWidths;
        pressBases = columnBases;
    }

    void bakeColumns() {
        for (U64 index = 0; index < columns.size() && index < pressWidths.size(); ++index) {
            if (!collapsed(columns[index]) && shownFactor(columns[index]) >= 1.0f) {
                columns[index].width = pressWidths[index] / pixelRatio;
            }
        }
    }

    bool resizeColumns(const SplitterLayout& splitter, const Extent2D<F32>& point) {
        const auto before = columns;
        if (splitter.column) {
            const U64 left = splitter.left;
            const U64 right = splitter.right;
            if (left >= right || right >= columns.size() || right >= pressWidths.size() || right >= pressBases.size()) {
                return false;
            }
            const F32 minimum = kMinColumnWidth * pixelRatio;
            const F32 delta = point.x - pressPoint.x;
            const auto flex = flexColumn();
            if (flex == left || flex == right) {
                const U64 other = flex == left ? right : left;
                const F32 sign = other == left ? 1.0f : -1.0f;
                const F32 shrink = std::max(0.0f, pressWidths[other] - minimum);
                const F32 room = std::max(0.0f, pressWidths[*flex] - minimum);
                const F32 grow = std::clamp(sign * delta, -shrink, room);
                columns[other].width = (pressWidths[other] + grow) / pixelRatio;
                if (pressWidths[*flex] - grow < pressBases[*flex]) {
                    columns[*flex].width = (pressWidths[*flex] - grow) / pixelRatio;
                }
            } else {
                if (!flex.has_value()) {
                    bakeColumns();
                }
                const F32 leftRoom = std::max(0.0f, pressWidths[left] - minimum);
                const F32 rightRoom = std::max(0.0f, pressWidths[right] - minimum);
                const F32 shift = std::clamp(delta, -leftRoom, rightRoom);
                columns[left].width = (pressWidths[left] + shift) / pixelRatio;
                columns[right].width = (pressWidths[right] - shift) / pixelRatio;
            }
        } else {
            if (splitter.path.empty() || splitter.path.front() >= columns.size()) {
                return false;
            }
            auto& column = columns[splitter.path.front()];
            if (splitter.left >= splitter.right || splitter.right >= column.panes.size()) {
                return false;
            }
            column.weights.resize(column.panes.size(), 1.0f);
            const F32 pair = column.weights[splitter.left] + column.weights[splitter.right];
            const F32 span = std::max(1.0f, splitter.spanEnd - splitter.spanStart);
            const F32 minimum = std::min(0.45f, (48.0f * pixelRatio) / span);
            const F32 t = std::clamp((point.y - splitter.spanStart) / span, minimum, 1.0f - minimum);
            column.weights[splitter.left] = pair * t;
            column.weights[splitter.right] = pair * (1.0f - t);
        }
        rebuild();
        return columns != before;
    }

    static void evenPair(std::vector<F32>& weights, U64 first, U64 second) {
        if (first >= weights.size() || second >= weights.size()) {
            return;
        }
        const F32 half = (weights[first] + weights[second]) * 0.5f;
        weights[first] = half;
        weights[second] = half;
    }

    void equalize(const SplitterLayout& splitter) {
        if (columnMode()) {
            if (splitter.column) {
                const U64 left = splitter.left;
                const U64 right = splitter.right;
                if (left >= right || right >= columns.size() || right >= columnWidths.size()) {
                    return;
                }
                pressColumns();
                const F32 average = (columnWidths[left] + columnWidths[right]) * 0.5f / pixelRatio;
                const auto flex = flexColumn();
                if (flex == left) {
                    columns[right].width = average;
                } else if (flex == right) {
                    columns[left].width = average;
                } else {
                    if (!flex.has_value()) {
                        bakeColumns();
                    }
                    columns[left].width = average;
                    columns[right].width = average;
                }
            } else if (!splitter.path.empty() && splitter.path.front() < columns.size()) {
                auto& column = columns[splitter.path.front()];
                column.weights.resize(column.panes.size(), 1.0f);
                evenPair(column.weights, splitter.left, splitter.right);
            }
            rebuild();
        } else if (tree.has_value()) {
            auto* node = At(*tree, splitter.path);
            if (!node || splitter.index + 1 >= node->children.size()) {
                return;
            }
            auto weights = Weights(*node);
            evenPair(weights, splitter.index, splitter.index + 1);
            node->weights = std::move(weights);
        }
        commit();
    }

    void arrangeColumns(const Rect& area, F32 gap) {
        viewport = area;
        columnRects.clear();
        columnWidths.clear();
        columnBases.clear();
        if (!maximized.empty() && locate(maximized) && !hiddenPane(maximized)) {
            leaves.push_back({.pane = maximized, .tile = area});
            return;
        }
        const U64 count = columns.size();
        if (count == 0) {
            return;
        }
        const auto flex = flexColumn();
        const F32 strip = std::round(kStripWidth * pixelRatio);
        std::vector<F32> factors(count, 1.0f);
        std::vector<F32> leads(count, 0.0f);
        std::vector<F32> widths(count, strip);
        std::vector<F32> fulls(count, strip);
        F32 earlier = 0.0f;
        F32 room = area.width;
        for (U64 index = 0; index < count; ++index) {
            factors[index] = shownFactor(columns[index]);
            if (index > 0) {
                leads[index] = std::round(gap * std::min(factors[index], earlier));
                room -= leads[index];
            }
            earlier = std::max(earlier, factors[index]);
        }
        const auto fit = [&](F32 space, const std::function<bool(U64)>& include) {
            std::vector<F32> open;
            std::optional<U64> openFlex;
            for (U64 index = 0; index < count; ++index) {
                if (!collapsed(columns[index]) && include(index)) {
                    if (flex == index) {
                        openFlex = open.size();
                    }
                    open.push_back(fulls[index]);
                }
            }
            open = FitColumns(std::move(open), std::max(0.0f, space), openFlex, kMinColumnWidth * pixelRatio);
            std::vector<F32> fitted(count, 0.0f);
            for (U64 index = 0, slot = 0; index < count; ++index) {
                if (!collapsed(columns[index]) && include(index)) {
                    fitted[index] = open[slot++];
                }
            }
            return fitted;
        };
        F32 fullRoom = room;
        for (U64 index = 0; index < count; ++index) {
            if (collapsed(columns[index])) {
                widths[index] = std::round(strip * factors[index]);
                room -= widths[index];
                fullRoom -= widths[index];
                continue;
            }
            const F32 fallback = flex == index ? kFlexColumnWidth : kColumnWidth;
            const F32 stored = columns[index].width >= kStoredWidthFloor ? columns[index].width : fallback;
            fulls[index] = std::round(std::max(stored, kMinColumnWidth) * pixelRatio);
            if (factors[index] > 0.0f && factors[index] < 1.0f) {
                fullRoom += gap - leads[index];
            }
        }
        columnBases = fulls;
        const auto natural = fit(fullRoom, [&](U64 index) {
            return factors[index] > 0.0f;
        });
        for (U64 index = 0; index < count; ++index) {
            if (!collapsed(columns[index]) && factors[index] > 0.0f && factors[index] < 1.0f) {
                fulls[index] = natural[index];
                widths[index] = std::round(fulls[index] * factors[index]);
                room -= widths[index];
            } else if (!collapsed(columns[index]) && factors[index] <= 0.0f) {
                widths[index] = 0.0f;
            }
        }
        const auto shown = fit(room, [&](U64 index) {
            return factors[index] >= 1.0f;
        });
        for (U64 index = 0; index < count; ++index) {
            if (!collapsed(columns[index]) && factors[index] >= 1.0f) {
                widths[index] = shown[index];
            }
        }
        const auto visibleBefore = [&](U64 index) -> std::optional<U64> {
            for (U64 scan = index; scan-- > 0;) {
                if (factors[scan] > 0.0f && !collapsed(columns[scan])) {
                    return scan;
                }
            }
            return std::nullopt;
        };
        F32 x = area.x;
        for (U64 index = 0; index < count; ++index) {
            x += leads[index];
            const Rect rect = {x, area.y, widths[index], area.height};
            x += widths[index];
            columnRects.push_back(rect);
            if (factors[index] <= 0.0f) {
                continue;
            }
            const auto& column = columns[index];
            const bool strips = collapsed(column);
            const F32 slide = factors[index] < 1.0f
                                  ? (flex.has_value() && index > *flex ? -1.0f : 1.0f) * (fulls[index] - widths[index])
                                  : 0.0f;
            const auto before = visibleBefore(index);
            if (before.has_value() && factors[index] >= 1.0f && factors[*before] >= 1.0f) {
                std::optional<U64> after;
                for (U64 scan = index; scan < count; ++scan) {
                    if (factors[scan] > 0.0f && !collapsed(columns[scan])) {
                        after = scan;
                        break;
                    }
                }
                if (after.has_value()) {
                    splitters.push_back({.index = index - 1, .rect = {rect.x - gap, area.y, gap, area.height},
                                         .column = true, .left = *before, .right = *after});
                }
            }
            const U64 rows = column.panes.size();
            std::vector<bool> shaded(rows, false);
            std::vector<F32> shown(rows, 1.0f);
            std::vector<F32> weights = column.weights;
            weights.resize(rows, 1.0f);
            F32 total = 0.0f;
            F32 fixed = 0.0f;
            U64 present = 0;
            bool settled = true;
            for (U64 row = 0; row < rows; ++row) {
                shown[row] = shownFactor(column.panes[row]);
                settled = settled && (shown[row] <= 0.0f || shown[row] >= 1.0f);
                if (shown[row] <= 0.0f) {
                    weights[row] = 0.0f;
                    continue;
                }
                ++present;
                shaded[row] = !strips && minimized.contains(column.panes[row]);
                if (shaded[row]) {
                    fixed += strip;
                    continue;
                }
                weights[row] = std::max(weights[row], 1.0e-3f) * shown[row];
                total += weights[row];
            }
            const F32 usable = std::max(0.0f, rect.height - gap * static_cast<F32>(present - 1) - fixed);
            std::vector<F32> heights(rows);
            F32 used = 0.0f;
            std::optional<U64> last;
            for (U64 row = 0; row < rows; ++row) {
                heights[row] = shown[row] <= 0.0f ? 0.0f
                               : shaded[row]      ? strip
                                                  : std::round(usable * weights[row] / std::max(total, 1.0e-3f));
                used += heights[row];
                if (!shaded[row] && shown[row] > 0.0f) {
                    last = row;
                }
            }
            if (last.has_value()) {
                heights[*last] = std::max(0.0f, heights[*last] + usable + fixed - used);
            }
            std::vector<F32> tops(rows);
            F32 y = rect.y;
            for (U64 row = 0; row < rows; ++row) {
                tops[row] = y;
                if (shown[row] <= 0.0f) {
                    continue;
                }
                leaves.push_back({.pane = column.panes[row], .tile = {rect.x, y, rect.width, heights[row]},
                                  .slide = slide, .minimized = minimized.contains(column.panes[row]),
                                  .strip = strips});
                y += heights[row] + gap;
            }
            for (U64 boundary = 1; settled && boundary < rows; ++boundary) {
                if (shown[boundary] <= 0.0f) {
                    continue;
                }
                std::optional<U64> upper;
                for (U64 scan = boundary; scan-- > 0;) {
                    if (!shaded[scan] && shown[scan] > 0.0f) {
                        upper = scan;
                        break;
                    }
                }
                std::optional<U64> lower;
                for (U64 scan = boundary; scan < rows; ++scan) {
                    if (!shaded[scan] && shown[scan] > 0.0f) {
                        lower = scan;
                        break;
                    }
                }
                if (!upper.has_value() || !lower.has_value()) {
                    continue;
                }
                const F32 above = tops[boundary] - gap - (tops[*upper] + heights[*upper]);
                const F32 below = tops[*lower] - tops[boundary];
                splitters.push_back({
                    .path = {static_cast<U8>(index)},
                    .index = boundary - 1,
                    .vertical = true,
                    .rect = {rect.x, tops[boundary] - gap, rect.width, gap},
                    .spanStart = tops[*upper] + above,
                    .spanEnd = tops[*lower] + heights[*lower] - below,
                    .left = *upper,
                    .right = *lower,
                });
            }
        }
        columnWidths = std::move(widths);
    }

    std::vector<std::string> placed() const {
        if (config.tabs) {
            return order;
        }
        return tree.has_value() ? Leaves(*tree) : std::vector<std::string>{};
    }

    void settleMaximized() {
        if (maximized.empty()) {
            return;
        }
        if (config.tabs || !config.headers || !tree.has_value() || !Contains(*tree, maximized) ||
            Leaves(*tree).size() < 2 || (columnMode() && hiddenPane(maximized))) {
            maximized.clear();
        }
    }

    bool hiddenPane(const std::string& id) const {
        const auto* item = pane(id);
        return item && item->hidden;
    }

    std::optional<DockTiles> visible() const {
        if (!config.tabs) {
            if (!maximized.empty() && tree.has_value() && Contains(*tree, maximized)) {
                return Leaf(maximized);
            }
            return tree;
        }
        if (active < order.size()) {
            return Leaf(order[active]);
        }
        return std::nullopt;
    }

    void commit() {
        if (config.tabs) {
            return;
        }
        synced = tree;
        if (config.onTiles) {
            config.onTiles(synced);
        }
    }

    void syncTiles() {
        if (config.tiles != synced) {
            synced = config.tiles;
            tree.reset();
            if (config.tiles.has_value()) {
                tree = Normalize(*config.tiles);
            }
        }
        bool added = false;
        for (const auto& item : config.panes) {
            if (!tree.has_value() || !Contains(*tree, item.id)) {
                tree = Append(tree, item.id);
                added = true;
            }
        }
        if (added) {
            maximized.clear();
            tree = Normalize(*tree);
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

    void splitTile(const std::string& id, const std::string& target, Side side) {
        std::optional<DockTiles> base;
        if (tree.has_value()) {
            base = Remove(*tree, id);
        }
        tree = base.has_value() ? Split(*base, target, id, side) : Leaf(id);
        maximized.clear();
        focus(id);
        commit();
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
        if (columnMode()) {
            takeColumn(id);
            rebuild();
        } else if (tree.has_value()) {
            tree = Remove(*tree, id);
        }
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

    void arrange(const DockTiles& node, const Rect& area, std::vector<U8>& path, F32 gap) {
        if (node.children.size() < 2) {
            if (node.children.size() == 1) {
                path.push_back(0);
                arrange(node.children.front(), area, path, gap);
                path.pop_back();
                return;
            }
            leaves.push_back({.pane = node.pane, .tile = area});
            return;
        }
        const auto weights = Weights(node);
        const U64 count = node.children.size();
        const F32 start = node.vertical ? area.y : area.x;
        const F32 extent = node.vertical ? area.height : area.width;
        const F32 usable = std::max(0.0f, extent - gap * static_cast<F32>(count - 1));
        F32 cursor = start;
        Rect previous;
        for (U64 index = 0; index < count; ++index) {
            const F32 end = index + 1 == count ? start + extent : std::round(cursor + usable * weights[index]);
            const F32 size = std::max(0.0f, end - cursor);
            const Rect rect = node.vertical ? Rect{area.x, cursor, area.width, size}
                                            : Rect{cursor, area.y, size, area.height};
            if (index > 0) {
                const F32 previousStart = node.vertical ? previous.y : previous.x;
                const Rect splitter = node.vertical ? Rect{area.x, cursor - gap, area.width, gap}
                                                    : Rect{cursor - gap, area.y, gap, area.height};
                splitters.push_back({
                    .path = path,
                    .index = index - 1,
                    .vertical = node.vertical,
                    .rect = splitter,
                    .spanStart = previousStart,
                    .spanEnd = end,
                });
            }
            path.push_back(static_cast<U8>(index));
            arrange(node.children[index], rect, path, gap);
            path.pop_back();
            previous = rect;
            cursor = end + gap;
        }
    }

    void resize(DockTiles& tree, const SplitterLayout& splitter, const Extent2D<F32>& point) {
        auto* node = At(tree, splitter.path);
        if (!node || splitter.index + 1 >= node->children.size()) {
            return;
        }
        auto weights = Weights(*node);
        const F32 pair = weights[splitter.index] + weights[splitter.index + 1];
        const F32 span = std::max(1.0f, splitter.spanEnd - splitter.spanStart);
        const F32 position = splitter.vertical ? point.y : point.x;
        const F32 minimum = std::min(0.45f, (48.0f * pixelRatio) / span);
        const F32 t = std::clamp((position - splitter.spanStart) / span, minimum, 1.0f - minimum);
        weights[splitter.index] = pair * t;
        weights[splitter.index + 1] = pair * (1.0f - t);
        node->weights = std::move(weights);
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
    impl->config = std::move(config);
    std::optional<Impl::State> before;
    if (!changed) {
        before = impl->state();
    }
    impl->forgetRemovedFocus();
    if (impl->config.tabs) {
        impl->syncTabs();
    } else if (impl->columnMode()) {
        impl->syncColumns();
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
                placeholder = std::make_unique<Text>();
            }
            placeholder->update({
                .str = "Loading...",
                .tone = Text::Tone::Secondary,
                .align = Text::Align::Center,
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
        F32 limit = s.bounds.right() - margin;
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
            const F32 titleRoom = std::max(0.0f, tab.right() - cursor - (closable ? reserve : padding));
            if (caption) {
                auto& text = *caption->text;
                text.update({.str = Fit(s.metrics, caption->title, font, titleRoom, kTabFont),
                             .font = Text::Font::Display,
                             .colorKey = textKey});
                const Extent2D<F32> textSize = measureChild(text, ctx, {infinite, infinite});
                const F32 textWidth = std::min(textSize.x, titleRoom);
                layoutChild(ctx, text, {cursor, tabY + std::floor((tabHeight - textSize.y) * 0.5f), textWidth, textSize.y}, tab);
                labels.push_back(&text);
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

    const auto tileTree = s.visible();
    const U64 total = tabs ? (tileTree.has_value() ? 1 : 0) : s.placed().size();
    if (s.columnMode()) {
        s.arrangeColumns(Inset(s.contentArea, gap), gap);
    } else if (tileTree) {
        std::vector<U8> path;
        s.arrange(*tileTree, tabs ? s.contentArea : Inset(s.contentArea, gap), path, gap);
    }
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
        Text* placeholder = nullptr;
        if (!item) {
            const auto it = s.placeholders.find(leaf.pane);
            if (it == s.placeholders.end()) {
                continue;
            }
            placeholder = it->second.get();
        }
        const bool closable = item ? item->closable : true;
        const bool bare = !s.config.loneHeader && s.maximized.empty() && expanded == 1 && !leaf.minimized && item &&
                          !closable;
        Rect content = leaf.tile;
        if (!tabs && item && !item->card) {
            content = Intersect(Inflate(leaf.tile, gap), s.contentArea);
        }
        if (!tabs && (!item || item->card)) {
            tileRects.push_back(leaf.tile);
        }
        const bool fullscreenable = item && item->maximizable && total > 1;
        const bool minimizable = item && !leaf.minimized && s.canMinimize(leaf.pane);
        const auto fullscreenButton = [&](Impl::Caption* caption, const Rect& rect) {
            leaf.maximize = rect;
            if (caption) {
                const bool fullscreen = s.maximized == leaf.pane;
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
            const F32 height = measureChild(*placeholder, ctx, {leaf.content.width, infinite}).y;
            layoutChild(ctx, *placeholder, {leaf.content.x, leaf.content.y + std::floor((leaf.content.height - height) * 0.5f),
                                            leaf.content.width, height});
            labels.push_back(placeholder);
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
    tile.layer = Card::Layer::Frame;
    s.tileFrame.update(std::move(tile));
    s.splitterBoxes.update({.instances = std::move(splitterInstances), .clip = s.bounds,
                            .cornerRadius = splitterLine * 0.5f, .capacity = kBoxCapacity});
    const Rect hoveredButton = s.hoveredHeaderButton();
    std::vector<Box::Instance> buttonInstances;
    F32 buttonRadius = hoveredButton.width * 0.5f;
    for (const auto& leaf : s.leaves) {
        if (s.maximized == leaf.pane && !leaf.maximize.empty()) {
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
                                                             &s.splitterBoxes, &s.buttonHover, &s.tileFrame}) {
        layoutChild(ctx, *primitive, s.bounds);
    }

    Context carded = ctx;
    carded.backdropColorKey = "tile_bg";
    carded.backdropTopColorKey.clear();
    for (const auto& leaf : s.leaves) {
        const auto* item = s.pane(leaf.pane);
        if (item && item->content && !leaf.minimized) {
            layoutChild(!tabs && item->card ? carded : ctx, *item->content, leaf.content, leaf.body);
        }
    }

    if (!s.drag || !s.drag->active || s.drag->group != s.config.accepts || !ctx.mouse.has_value() ||
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

    if (s.columnMode()) {
        if (s.columns.empty() || s.leaves.empty()) {
            offer(s.viewport, [state](const std::string& pane) {
                state->appendColumn(pane);
            });
            return;
        }
        const auto* leaf = s.targetAt(point, gap * 0.5f + 1.0f);
        if (!leaf) {
            if (!s.columnRects.empty() && point.x >= s.columnRects.back().right()) {
                const F32 x = s.columnRects.back().right() + gap;
                offer({x, s.viewport.y, std::max(0.0f, s.viewport.right() - x), s.viewport.height},
                      [state](const std::string& pane) {
                          state->appendColumn(pane);
                      });
            }
            return;
        }
        const auto side = EdgeAt(leaf->tile, point);
        const auto column = s.locate(leaf->pane);
        if (!side.has_value() || !column || column->first >= s.columnRects.size()) {
            return;
        }
        const bool across = *side == Side::Left || *side == Side::Right;
        if (leaf->pane == dragged && (!across || s.columns[column->first].panes.size() < 2)) {
            return;
        }
        const std::string target = leaf->pane;
        const Side edge = *side;
        offer(across ? Half(s.columnRects[column->first], edge) : Half(leaf->tile, edge),
              [state, target, edge](const std::string& pane) {
                  state->dropColumn(pane, target, edge);
              });
        return;
    }

    if (s.leaves.empty()) {
        offer(Inset(s.contentArea, gap), [state](const std::string& pane) {
            state->tree = Leaf(pane);
            state->focus(pane);
            state->commit();
        });
        return;
    }

    const auto* leaf = s.targetAt(point, gap * 0.5f + 1.0f);
    if (!leaf || leaf->pane == dragged) {
        return;
    }
    const auto side = EdgeAt(leaf->tile, point);
    if (!side.has_value()) {
        return;
    }
    const std::string target = leaf->pane;
    const Side edge = *side;
    offer(Half(leaf->tile, edge), [state, target, edge](const std::string& pane) {
        state->splitTile(pane, target, edge);
    });
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
            if (s.press == Impl::Press::Splitter && s.columnMode()) {
                if (s.resizeColumns(s.pressLayout, point)) {
                    invalidate(Dirty::Layout);
                }
                return true;
            }
            if (s.press == Impl::Press::Splitter) {
                if (s.tree.has_value()) {
                    const DockTiles before = *s.tree;
                    s.resize(*s.tree, s.pressLayout, point);
                    if (*s.tree != before) {
                        invalidate(Dirty::Layout);
                    }
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
                            s.maximized = s.maximized == leaf.pane ? std::string() : leaf.pane;
                            s.focus(leaf.pane);
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
                    s.press = Impl::Press::Splitter;
                    s.pressLayout = layout;
                    s.pressPoint = point;
                    if (s.columnMode()) {
                        s.pressColumns();
                    }
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
                        s.maximized = s.maximized == leaf.pane ? std::string() : leaf.pane;
                        s.focus(leaf.pane);
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
