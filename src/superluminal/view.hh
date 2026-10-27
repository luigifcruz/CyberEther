#ifndef JETSTREAM_SUPERLUMINAL_VIEW_HH
#define JETSTREAM_SUPERLUMINAL_VIEW_HH

#include "jetstream/logger.hh"
#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"
#include "jetstream/surface.hh"

#include "render/sakura/metrics.hh"
#include "render/sakura/titlebar.hh"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace Jetstream {

struct SuperluminalView {
    static constexpr const char* Group = "superluminal";
    static constexpr F32 TitleInset = 12.0f;
    static constexpr F32 TitleSize = 17.0f;
    static constexpr F32 LabelInset = 6.0f;
    static constexpr const char* GlobalPane = "global";
    static constexpr const char* GlobalTitle = "Controls";
    static constexpr const char* GlobalBox = "";
    static constexpr F32 GlobalWeight = 0.25f;
    static constexpr F32 GlobalLimit = 0.6f;
    static constexpr F32 GlobalHeight = 120.0f;
    static constexpr F32 FitSlack = 1.0f;
    static constexpr I32 SliderPrecision = 3;
    static constexpr F32 MarkdownMinimum = 48.0f;
    static constexpr U64 ParkedUpdates = 120;

    struct InterfaceCallback {
        InterfaceCallback() = default;

        template<typename Function>
            requires (!std::is_same_v<std::remove_cvref_t<Function>, InterfaceCallback> &&
                      std::is_invocable_r_v<void, std::decay_t<Function>&>)
        InterfaceCallback(Function&& function) :
            callback(std::make_shared<std::function<void()>>(std::forward<Function>(function))) {
            if (!*callback) {
                callback.reset();
            }
        }

        explicit operator bool() const {
            return static_cast<bool>(callback);
        }

        void operator()() const {
            const auto active = callback;
            if (active) {
                (*active)();
            }
        }

     private:
        std::shared_ptr<std::function<void()>> callback;
    };

    struct Item {
        enum class Kind : U8 {
            Text,
            Slider,
            Markdown,
        };

        Kind kind = Kind::Text;
        std::string str;
        F32 min = 0.0f;
        F32 max = 1.0f;
        F32 value = 0.0f;
        bool integer = false;

        bool operator==(const Item&) const = default;
    };

    struct Recorder {
        void begin(const std::string& box) {
            current = box;
            items[box].clear();
            occurrences.clear();
        }

        void end() {
            if (current.has_value()) {
                std::erase_if(pending, [this](const auto& entry) {
                    return std::get<0>(entry.first) == *current;
                });
            }
            current.reset();
        }

        Result text(const std::string& str) {
            if (!current.has_value()) {
                JST_ERROR("[SUPERLUMINAL] Text must be called from an interface, box, or global interface callback.");
                return Result::ERROR;
            }
            items[*current].push_back({.kind = Item::Kind::Text, .str = str});
            return Result::SUCCESS;
        }

        Result markdown(const std::string& str) {
            if (!current.has_value()) {
                JST_ERROR("[SUPERLUMINAL] Markdown must be called from an interface, box, or global interface callback.");
                return Result::ERROR;
            }
            items[*current].push_back({.kind = Item::Kind::Markdown, .str = str});
            return Result::SUCCESS;
        }

        Result slider(const std::string& label, F32 min, F32 max, F32& value, bool integer, bool* edited = nullptr) {
            if (edited) {
                *edited = false;
            }
            if (!current.has_value()) {
                JST_ERROR("[SUPERLUMINAL] Slider must be called from an interface, box, or global interface callback.");
                return Result::ERROR;
            }
            const U64 occurrence = occurrences[label]++;
            if (const auto it = pending.find({*current, label, occurrence}); it != pending.end()) {
                value = it->second;
                pending.erase(it);
                if (edited) {
                    *edited = true;
                }
            }
            items[*current].push_back({
                .kind = Item::Kind::Slider,
                .str = label,
                .min = min,
                .max = max,
                .value = value,
                .integer = integer,
            });
            return Result::SUCCESS;
        }

        Result slider(const std::string& label, F32 min, F32 max, U64& value) {
            F32 resolved = static_cast<F32>(value);
            bool edited = false;
            const Result result = slider(label, min, max, resolved, true, &edited);
            if (result == Result::SUCCESS && edited && std::isfinite(resolved)) {
                const F32 rounded = std::max(0.0f, std::round(resolved));
                value = rounded >= static_cast<F32>(std::numeric_limits<U64>::max())
                            ? std::numeric_limits<U64>::max()
                            : static_cast<U64>(rounded);
            }
            return result;
        }

        void slide(const std::string& box, const std::string& label, U64 occurrence, F32 value) {
            pending[{box, label, occurrence}] = value;
        }

        const std::vector<Item>& recorded(const std::string& box) {
            return items[box];
        }

     private:
        std::optional<std::string> current;
        std::unordered_map<std::string, std::vector<Item>> items;
        std::unordered_map<std::string, U64> occurrences;
        std::map<std::tuple<std::string, std::string, U64>, F32> pending;
    };

    struct Surface {
        std::shared_ptr<const Render::Texture> texture;
        std::function<void(const Sakura::SurfaceResize&)> onSize;
        std::function<void(InputEvent)> onInput;
        std::function<SurfaceCursor()> resolveCursor;
    };

    struct Tile {
        std::string name;
        Extent2D<U8> offset = {0, 0};
        Extent2D<U8> size = {1, 1};
        InterfaceCallback callback;
        std::vector<Surface> surfaces;
    };

    struct Config {
        std::string title = "Superluminal";
        Sakura::Titlebar::Chrome chrome;
        std::function<void(U8)> onBarPress;
        InterfaceCallback global;
        std::vector<Tile> tiles;
        Extent2D<U8> mosaic = {0, 0};
    };

    struct Shell : public Sakura::Component {
        struct Config {
            Sakura::Component* content = nullptr;
            Sakura::Callback<void(Extent2D<F32>)> onPress;
            Sakura::Callback<void(F32)> onLayout;

            bool operator==(const Config&) const = default;
        };

        void update(Config config) {
            if (config != this->config) {
                invalidate(Dirty::Layout);
            }
            const bool remount = config.content != this->config.content ||
                                 (config.content && !hasChild(*config.content));
            this->config = std::move(config);
            if (remount) {
                std::vector<Sakura::Component*> children;
                if (this->config.content) {
                    children.push_back(this->config.content);
                }
                mountChildren(children);
            }
        }

     protected:
        Extent2D<F32> measure(const Sakura::Context&, Extent2D<F32> available) override {
            return {Sakura::FillLength(available.x), Sakura::FillLength(available.y)};
        }

        void layout(const Sakura::Context& ctx) override {
            const Rect bounds = frame();
            if (config.content) {
                layoutChild(ctx, *config.content, bounds);
            }
            const F32 height = bounds.height / ctx.pixelRatio;
            if (config.onLayout && height != last) {
                last = height;
                config.onLayout(height);
            }
        }

        bool mouseEvent(const MouseEvent& event) override {
            if (event.type == MouseEventType::Click && config.onPress &&
                frame().contains(event.position.x, event.position.y)) {
                config.onPress(event.position);
            }
            return mouseEventChildren(event);
        }

     private:
        Config config;
        F32 last = -1.0f;
    };

    struct ScrollView : public Sakura::Component {
        struct Config {
            Sakura::Component* content = nullptr;
            F32 height = Sakura::Fill;
            Sakura::Callback<void(F32, F32)> onLayout;

            bool operator==(const Config&) const = default;
        };

        ScrollView() {
            setClipsChildren(true);
            addChild(scroller);
        }

        void update(Config config) {
            if (config != this->config) {
                invalidate(Dirty::Layout);
            }
            const bool remount = config.content != this->config.content ||
                                 (config.content && !hasChild(*config.content));
            this->config = std::move(config);
            if (remount) {
                std::vector<Sakura::Component*> children = {&scroller};
                if (this->config.content) {
                    children.push_back(this->config.content);
                }
                mountChildren(children);
            }
        }

     protected:
        Extent2D<F32> measure(const Sakura::Context& ctx, Extent2D<F32> available) override {
            const F32 width = Sakura::FillLength(available.x);
            F32 natural = 0.0f;
            if (config.height == 0.0f && config.content) {
                natural = measureChild(*config.content, ctx, {width, std::numeric_limits<F32>::infinity()}).y;
            }
            return {width, Sakura::ResolveLength(ctx, config.height, natural, available.y)};
        }

        void layout(const Sakura::Context& ctx) override {
            const Rect bounds = frame();
            const F32 infinite = std::numeric_limits<F32>::infinity();
            F32 width = bounds.width;
            F32 natural = config.content ? measureChild(*config.content, ctx, {width, infinite}).y : 0.0f;
            const bool overflow = natural > bounds.height + 0.5f;
            if (overflow && config.content) {
                width = std::max(0.0f, bounds.width - Sakura::Px(ctx, Sakura::kScrollbarThickness) -
                                           2.0f * Sakura::Px(ctx, Sakura::kScrollbarMargin));
                natural = measureChild(*config.content, ctx, {width, infinite}).y;
            }
            offset = overflow ? std::clamp(offset, 0.0f, std::max(0.0f, natural - bounds.height)) : 0.0f;
            scroller.update({
                .contentHeight = overflow ? natural : 0.0f,
                .scrollY = offset,
                .scrollbar = overflow,
                .wheel = overflow,
                .wheelStep = Sakura::Px(ctx, Sakura::kWheelStep),
                .thickness = Sakura::Px(ctx, Sakura::kScrollbarThickness),
                .margin = Sakura::Px(ctx, Sakura::kScrollbarMargin),
                .trackColorKey = "scrollbar_bg",
                .thumbColorKey = "scrollbar_grab",
                .onScrollY = [this](F32 next) {
                    if (next != offset) {
                        offset = next;
                        invalidate(Dirty::Layout);
                    }
                },
                .onOverscroll = [this] {
                    requestFrame();
                },
            });
            layoutChild(ctx, scroller, bounds);
            if (config.content) {
                const F32 overscroll = scroller.overscroll(ctx).y;
                layoutChild(ctx, *config.content,
                            {bounds.x, bounds.y - offset - overscroll, width, overflow ? natural : bounds.height},
                            bounds);
            }
            const Extent2D<F32> reported = {std::ceil(natural) / ctx.pixelRatio, bounds.height / ctx.pixelRatio};
            if (config.onLayout && reported != last) {
                last = reported;
                config.onLayout(reported.x, reported.y);
            }
        }

     private:
        Config config;
        Sakura::Scroller scroller;
        F32 offset = 0.0f;
        Extent2D<F32> last = {-1.0f, -1.0f};
    };

    struct Keys : public Sakura::Component {
        struct Config {
            Sakura::Component* content = nullptr;
            Sakura::Callback<bool()> active;
            Sakura::Callback<void(KeyEvent)> onKey;

            bool operator==(const Config&) const = default;
        };

        void update(Config config) {
            if (config != this->config) {
                invalidate(Dirty::Layout);
            }
            const bool remount = config.content != this->config.content ||
                                 (config.content && !hasChild(*config.content));
            this->config = std::move(config);
            if (remount) {
                std::vector<Sakura::Component*> children;
                if (this->config.content) {
                    children.push_back(this->config.content);
                }
                mountChildren(children);
            }
        }

     protected:
        Extent2D<F32> measure(const Sakura::Context&, Extent2D<F32> available) override {
            return {Sakura::FillLength(available.x), Sakura::FillLength(available.y)};
        }

        void layout(const Sakura::Context& ctx) override {
            if (config.content) {
                layoutChild(ctx, *config.content, frame());
            }
        }

        bool keyEvent(const KeyEvent& event) override {
            return event.key == KeyCode::Space;
        }

        bool shortcutEvent(const KeyEvent& event) override {
            if (event.key != KeyCode::Space || textEditorFocused() || !config.active || !config.active() ||
                !config.onKey) {
                return false;
            }
            config.onKey(event);
            return true;
        }

     private:
        Config config;
    };

    struct Mosaic : public Sakura::Component {
        struct Item {
            Sakura::Component* child = nullptr;
            Extent2D<U8> offset = {0, 0};
            Extent2D<U8> size = {1, 1};

            bool operator==(const Item&) const = default;
        };

        struct Config {
            std::vector<Item> items;
            Extent2D<U8> grid = {1, 1};
            F32 top = Sakura::Dock::ContentPadding;

            bool operator==(const Config&) const = default;
        };

        struct Cell : public Sakura::Component {
            void hold(Sakura::Component* next) {
                if (next == content && (!next || hasChild(*next))) {
                    return;
                }
                content = next;
                invalidate(Dirty::Layout);
                std::vector<Sakura::Component*> children;
                if (next) {
                    children.push_back(next);
                }
                mountChildren(children);
            }

            void place(const Rect& next) {
                tile = next;
            }

            const Rect& area() const {
                return tile;
            }

         protected:
            Extent2D<F32> measure(const Sakura::Context&, Extent2D<F32> available) override {
                return {Sakura::FillLength(available.x), Sakura::FillLength(available.y)};
            }

            void layout(const Sakura::Context& ctx) override {
                if (content) {
                    layoutChild(ctx, *content, frame());
                }
            }

            bool occludes(const Extent2D<F32>& point) const override {
                return tile.contains(point.x, point.y);
            }

         private:
            Sakura::Component* content = nullptr;
            Rect tile;
        };

        void update(Config config) {
            if (config == this->config) {
                this->config = std::move(config);
                return;
            }
            invalidate(Dirty::Layout);
            while (cells.size() < config.items.size()) {
                cells.push_back(std::make_unique<Cell>());
            }
            std::vector<Sakura::Component*> children;
            for (U64 index = 0; index < config.items.size(); ++index) {
                cells[index]->hold(config.items[index].child);
                children.push_back(cells[index].get());
            }
            this->config = std::move(config);
            captured = nullptr;
            buttons.fill(false);
            mountChildren(children);
        }

     protected:
        Extent2D<F32> measure(const Sakura::Context&, Extent2D<F32> available) override {
            return {Sakura::FillLength(available.x), Sakura::FillLength(available.y)};
        }

        void layout(const Sakura::Context& ctx) override {
            const Rect bounds = frame();
            const F32 gap = Sakura::Px(ctx, Sakura::Dock::ContentPadding);
            const F32 top = Sakura::Px(ctx, config.top);
            const Rect area = {bounds.x + gap, bounds.y + top, std::max(0.0f, bounds.width - 2.0f * gap),
                               std::max(0.0f, bounds.height - top - gap)};
            const U32 columns = std::max<U32>(1, config.grid.x);
            const U32 rows = std::max<U32>(1, config.grid.y);
            const F32 unitX = (area.width + gap) / static_cast<F32>(columns);
            const F32 unitY = (area.height + gap) / static_cast<F32>(rows);
            for (U64 index = 0; index < config.items.size(); ++index) {
                const auto& item = config.items[index];
                const F32 x0 = std::round(area.x + unitX * static_cast<F32>(item.offset.x));
                const F32 y0 = std::round(area.y + unitY * static_cast<F32>(item.offset.y));
                const F32 x1 = std::round(area.x + unitX * static_cast<F32>(item.offset.x + std::max<U8>(1, item.size.x)) - gap);
                const F32 y1 = std::round(area.y + unitY * static_cast<F32>(item.offset.y + std::max<U8>(1, item.size.y)) - gap);
                const Rect tile = {x0, y0, std::max(0.0f, x1 - x0), std::max(0.0f, y1 - y0)};
                cells[index]->place(tile);
                layoutChild(ctx, *cells[index], {tile.x - gap, tile.y - gap, tile.width + 2.0f * gap, tile.height + 2.0f * gap});
            }
        }

        bool mouseEvent(const MouseEvent& event) override {
            switch (event.type) {
                case MouseEventType::Click: {
                    if (!captured) {
                        captured = topmost(event.position);
                    }
                    if (!captured) {
                        return false;
                    }
                    held(event.button) = true;
                    deliverChild(*captured, event);
                    return true;
                }
                case MouseEventType::Scroll: {
                    Cell* target = topmost(event.position);
                    if (!target) {
                        return false;
                    }
                    deliverChild(*target, event);
                    return true;
                }
                case MouseEventType::Move: {
                    Cell* target = captured ? captured : topmost(event.position);
                    MouseEvent away = event;
                    away.position = {-1.0e9f, -1.0e9f};
                    bool handled = false;
                    for (U64 index = config.items.size(); index-- > 0;) {
                        Cell* cell = cells[index].get();
                        if (cell == target) {
                            handled = deliverChild(*cell, event) || handled;
                        } else {
                            deliverChild(*cell, away);
                        }
                    }
                    return handled;
                }
                case MouseEventType::Release: {
                    const bool handled = mouseEventChildren(event);
                    held(event.button) = false;
                    if (std::none_of(buttons.begin(), buttons.end(), [](bool down) { return down; })) {
                        captured = nullptr;
                    }
                    return handled;
                }
                default:
                    captured = nullptr;
                    buttons.fill(false);
                    return mouseEventChildren(event);
            }
        }

     private:
        bool& held(MouseButton button) {
            return buttons[std::min<U64>(static_cast<U64>(button), buttons.size() - 1)];
        }

        Cell* topmost(const Extent2D<F32>& point) const {
            for (U64 index = config.items.size(); index-- > 0;) {
                if (cells[index]->area().contains(point.x, point.y)) {
                    return cells[index].get();
                }
            }
            return nullptr;
        }

        Config config;
        std::vector<std::unique_ptr<Cell>> cells;
        Cell* captured = nullptr;
        std::array<bool, 3> buttons{};
    };

    struct Box {
        struct Config {
            std::string id;
            std::vector<Item> items;
            F32 height = Sakura::Fill;
            std::function<void(const std::string&, U64, F32)> onSlide;
            std::function<void(F32, F32)> onLayout;
        };

        void update(Config config) {
            this->config = std::move(config);
            const bool fill = this->config.height == Sakura::Fill && !this->config.onLayout;
            U64 textCount = 0;
            std::unordered_map<std::string, U64> occurrences;
            ++generation;
            U64 markdownCount = 0;
            U64 gridCount = 0;
            std::vector<Sakura::NodeContentStack::Item> rows;
            std::vector<Sakura::NodeFieldGrid::Item> cells;
            bool labelled = false;
            const auto push = [&](Sakura::Component& child, std::optional<Sakura::NodeContentStack::Flex> flex = std::nullopt) {
                rows.push_back({
                    .child = &child,
                    .flex = flex,
                    .gap = labelled ? std::optional<F32>(0.0f) : std::nullopt,
                });
                labelled = false;
            };
            const auto flush = [&] {
                if (cells.empty()) {
                    return;
                }
                if (grids.size() <= gridCount) {
                    grids.push_back(std::make_unique<Sakura::NodeFieldGrid>());
                }
                auto& grid = *grids[gridCount];
                grid.update({
                    .id = this->config.id + ":grid:" + std::to_string(gridCount),
                    .items = std::move(cells),
                });
                ++gridCount;
                cells.clear();
                push(grid);
            };
            for (U64 index = 0; index < this->config.items.size(); ++index) {
                const auto& item = this->config.items[index];
                if (item.kind == Item::Kind::Markdown) {
                    flush();
                    if (markdowns.size() <= markdownCount) {
                        markdowns.push_back(std::make_unique<Sakura::Markdown>());
                    }
                    const bool last = fill && index + 1 == this->config.items.size();
                    auto& markdown = *markdowns[markdownCount];
                    markdown.update({
                        .id = this->config.id + ":markdown:" + std::to_string(markdownCount),
                        .value = item.str,
                        .scrollbar = last,
                    });
                    ++markdownCount;
                    std::optional<Sakura::NodeContentStack::Flex> flex;
                    if (last) {
                        flex = Sakura::NodeContentStack::Flex{.minimum = MarkdownMinimum};
                    }
                    push(markdown, flex);
                    continue;
                }
                if (item.kind == Item::Kind::Text) {
                    flush();
                    if (texts.size() <= textCount) {
                        texts.push_back(std::make_unique<Label>());
                    }
                    auto& label = *texts[textCount++];
                    label.text.update({
                        .str = item.str,
                        .font = Sakura::Text::Font::DisplaySemiBold,
                        .wrapped = true,
                        .maxCharacters = std::bit_ceil(std::max<U64>(256, item.str.size())),
                    });
                    label.frame.update({
                        .content = &label.text,
                        .padding = LabelInset,
                        .border = false,
                        .colorKey = "transparent",
                    });
                    push(label.frame);
                    labelled = true;
                    continue;
                }
                const U64 occurrence = occurrences[item.str]++;
                const std::string identity = item.str + "#" + std::to_string(occurrence);
                auto& entry = sliders[identity];
                if (!entry) {
                    entry = std::make_unique<Slider>();
                }
                auto& slider = *entry;
                slider.seen = generation;
                const std::string id = this->config.id + ":" + identity;
                slider.input.update({
                    .id = id + ":input",
                    .min = item.min,
                    .max = item.max,
                    .value = item.value,
                    .integer = item.integer,
                    .precision = SliderPrecision,
                    .onChange = [this, label = item.str, occurrence](F32 value) {
                        if (this->config.onSlide) {
                            this->config.onSlide(label, occurrence, value);
                        }
                    },
                });
                slider.field.update({
                    .id = id,
                    .label = item.str,
                    .content = &slider.input,
                });
                cells.push_back({.child = &slider.field});
            }
            flush();
            std::erase_if(sliders, [&](const auto& entry) {
                return entry.second->seen + ParkedUpdates < generation;
            });
            stack.update({
                .id = this->config.id + ":stack",
                .items = std::move(rows),
            });
            panel.update({
                .id = this->config.id + ":panel",
                .content = &stack,
                .height = 0.0f,
            });
            Sakura::Callback<void(F32, F32)> report;
            if (this->config.onLayout) {
                report = [this](F32 natural, F32 height) {
                    if (this->config.onLayout) {
                        this->config.onLayout(natural, height);
                    }
                };
            }
            viewport.update({
                .content = &panel,
                .height = this->config.height,
                .onLayout = std::move(report),
            });
        }

        Sakura::Component& component() {
            return viewport;
        }

     private:
        struct Slider {
            Sakura::NodeField field;
            Sakura::NodeRangeInput input;
            U64 seen = 0;
        };

        struct Label {
            Sakura::Text text;
            Sakura::Div frame;
        };

        Config config;
        std::vector<std::unique_ptr<Label>> texts;
        std::unordered_map<std::string, std::unique_ptr<Slider>> sliders;
        U64 generation = 0;
        std::vector<std::unique_ptr<Sakura::Markdown>> markdowns;
        std::vector<std::unique_ptr<Sakura::NodeFieldGrid>> grids;
        Sakura::NodeContentStack stack;
        Sakura::NodeFieldPanel panel;
        ScrollView viewport;
    };

    struct Plot {
        void update(std::vector<Surface> next, std::function<bool()> active) {
            surfaces = std::move(next);
            while (images.size() < surfaces.size()) {
                images.push_back(std::make_unique<Sakura::SurfaceImage>());
            }
            std::vector<Sakura::Stack::Item> items;
            for (U64 index = 0; index < surfaces.size(); ++index) {
                const auto& surface = surfaces[index];
                images[index]->update({
                    .texture = surface.texture,
                    .onSize = surface.onSize,
                    .onInput = surface.onInput,
                    .resolveCursor = surface.resolveCursor,
                });
                items.push_back({.child = images[index].get(), .grow = 1.0f});
            }
            layout.update({
                .spacing = Sakura::Dock::ContentPadding,
                .items = std::move(items),
            });
            keys.update({
                .content = &layout,
                .active = std::move(active),
                .onKey = [this](KeyEvent event) {
                    for (const auto& surface : surfaces) {
                        if (surface.onInput) {
                            surface.onInput(event);
                        }
                    }
                },
            });
        }

        Sakura::Component& component() {
            return keys;
        }

     private:
        std::vector<Surface> surfaces;
        std::vector<std::unique_ptr<Sakura::SurfaceImage>> images;
        Sakura::Stack layout;
        Keys keys;
    };

    void update(Config config) {
        std::vector<Tile> tiles = std::move(config.tiles);
        std::stable_sort(tiles.begin(), tiles.end(), [](const Tile& a, const Tile& b) {
            return std::tie(a.offset.y, a.offset.x, a.name) < std::tie(b.offset.y, b.offset.x, b.name);
        });

        const bool hasGlobal = static_cast<bool>(config.global);
        std::vector<std::string> names;
        for (const auto& tile : tiles) {
            names.push_back(tile.name);
        }
        Extent2D<U8> dims = config.mosaic;
        for (const auto& tile : tiles) {
            dims.x = std::max<U8>(dims.x, static_cast<U8>(tile.offset.x + std::max<U8>(1, tile.size.x)));
            dims.y = std::max<U8>(dims.y, static_cast<U8>(tile.offset.y + std::max<U8>(1, tile.size.y)));
        }
        if (names != arranged || dims != grid || hasGlobal != withGlobal) {
            arranged = names;
            grid = dims;
            withGlobal = hasGlobal;
            mosaic = Arrange(tiles, dims);
            free = !tiles.empty() && !mosaic.has_value();
            fitted = !hasGlobal;
            globalHeight = GlobalHeight;
            layout = free ? (hasGlobal ? std::optional<Sakura::DockTiles>(Sakura::DockTiles{.pane = GlobalPane})
                                       : std::nullopt)
                          : Stacked(GlobalWeight);
        }

        std::unordered_set<std::string> boxNames;
        std::unordered_set<std::string> plotNames;
        std::unordered_set<std::string> freeNames;
        std::vector<Sakura::Dock::Pane> panes;
        std::vector<Mosaic::Item> placed;
        const auto group = [this](const std::string& name) {
            return free ? std::string(Group) + ":" + name : std::string(Group);
        };
        if (hasGlobal) {
            record(GlobalBox, config.global);
            global.update({
                .id = "superluminal:global",
                .items = recorder.recorded(GlobalBox),
                .height = Sakura::Fill,
                .onSlide = [this](const std::string& label, U64 occurrence, F32 value) {
                    recorder.slide(GlobalBox, label, occurrence, value);
                },
                .onLayout = [this](F32 natural, F32 height) {
                    settle(natural, height);
                },
            });
            panes.push_back({
                .id = GlobalPane,
                .title = GlobalTitle,
                .group = group(GlobalPane),
                .icon = ICON_LC_SLIDERS_HORIZONTAL,
                .content = &global.component(),
                .closable = false,
                .padding = Sakura::Dock::ContentPadding,
            });
        }
        stacking.clear();
        for (U64 index = 0; index < tiles.size(); ++index) {
            const auto& tile = tiles[index];
            const std::string id = PaneId(index);
            Sakura::Component* content = nullptr;
            const bool interface = static_cast<bool>(tile.callback);
            if (interface) {
                record(tile.name, tile.callback);
                auto& box = boxes[tile.name];
                if (!box) {
                    box = std::make_unique<Box>();
                }
                box->update({
                    .id = "superluminal:box:" + tile.name,
                    .items = recorder.recorded(tile.name),
                    .onSlide = [this, name = tile.name](const std::string& label, U64 occurrence, F32 value) {
                        recorder.slide(name, label, occurrence, value);
                    },
                });
                boxNames.insert(tile.name);
                content = &box->component();
            } else {
                auto& plot = plots[tile.name];
                if (!plot) {
                    plot = std::make_unique<Plot>();
                }
                plot->update(tile.surfaces, [this, id]() {
                    return selected == id;
                });
                plotNames.insert(tile.name);
                content = &plot->component();
            }
            Sakura::Dock::Pane pane = {
                .id = id,
                .title = tile.name,
                .group = group(id),
                .icon = interface ? ICON_LC_SLIDERS_HORIZONTAL : ICON_LC_CHART_AREA,
                .content = content,
                .closable = false,
                .padding = Sakura::Dock::ContentPadding,
            };
            if (!free) {
                panes.push_back(std::move(pane));
                continue;
            }
            auto& frame = frames[tile.name];
            if (!frame) {
                frame = std::make_unique<Sakura::Dock>();
            }
            frame->update({
                .tabs = false,
                .accepts = group(id),
                .panes = {std::move(pane)},
                .tiles = Sakura::DockTiles{.pane = id},
                .loneHeader = true,
            });
            freeNames.insert(tile.name);
            placed.push_back({.child = frame.get(), .offset = tile.offset, .size = tile.size});
            stacking.push_back(frame.get());
        }
        std::erase_if(boxes, [&](const auto& entry) {
            return !boxNames.contains(entry.first);
        });
        std::erase_if(plots, [&](const auto& entry) {
            return !plotNames.contains(entry.first);
        });
        std::erase_if(frames, [&](const auto& entry) {
            return !freeNames.contains(entry.first);
        });

        dock.update({
            .tabs = false,
            .accepts = group(GlobalPane),
            .panes = std::move(panes),
            .tiles = layout,
            .onTiles = [this](std::optional<Sakura::DockTiles> tiles) {
                const F32 before = Fraction(layout);
                layout = std::move(tiles);
                if (std::abs(Fraction(layout) - before) > 1e-3f) {
                    fitted = true;
                }
            },
            .loneHeader = free,
        });
        mosaicFrames.update({
            .items = std::move(placed),
            .grid = grid,
            .top = hasGlobal ? 0.0f : Sakura::Dock::ContentPadding,
        });

        const F32 barHeight = config.chrome.height > 0.0f ? config.chrome.height : Sakura::Titlebar::MinimumHeight;
        const F32 barCenter = config.chrome.controlsCenter > 0.0f ? config.chrome.controlsCenter : barHeight * 0.5f;
        barTitle.update({
            .str = config.title,
            .font = Sakura::Text::Font::DisplaySemiBold,
            .scale = TitleSize / Sakura::Typography::FontSize,
        });
        barRow.update({
            .items = {{
                .child = &barTitle,
                .anchor = Sakura::Anchor::TopLeft,
                .offset = {config.chrome.leadingInset + TitleInset, barCenter - TitleSize * 0.5f},
            }},
        });
        Sakura::Callback<void()> press;
        Sakura::Callback<void()> zoom;
        if (config.onBarPress) {
            press = [onBarPress = config.onBarPress] {
                onBarPress(1);
            };
            zoom = [onBarPress = config.onBarPress] {
                onBarPress(2);
            };
        }
        bar.update({
            .content = &barRow,
            .border = false,
            .colorKey = "background",
            .hoveredColorKey = "background",
            .selectedColorKey = "background",
            .onClick = std::move(press),
            .onDoubleClick = std::move(zoom),
        });
        std::vector<Sakura::Stack::Item> rows = {
            {.child = &bar, .size = std::max(0.0f, barHeight - Sakura::Dock::ContentPadding)},
        };
        if (!free) {
            rows.push_back({.child = &dock, .grow = 1.0f});
        } else {
            if (hasGlobal) {
                const F32 limit = windowHeight > 0.0f ? GlobalLimit * windowHeight : globalHeight;
                rows.push_back({.child = &dock, .size = std::min(globalHeight, limit)});
            }
            rows.push_back({.child = &mosaicFrames, .grow = 1.0f});
        }
        shell.update({
            .spacing = 0.0f,
            .items = std::move(rows),
        });

        window.update({
            .content = &shell,
            .onPress = [this](Extent2D<F32> point) {
                select(point);
            },
            .onLayout = [this](F32 height) {
                windowHeight = height;
            },
        });

        background.update({.particles = false});
        content.update({.content = &window});
        toasts.update();
    }

    std::vector<Sakura::Layer*> layers() {
        return {&background, &content, &toasts};
    }

    Recorder& interface() {
        return recorder;
    }

    const std::optional<Sakura::DockTiles>& tiles() const {
        return layout;
    }

    bool freeform() const {
        return free;
    }

    static std::string PaneId(U64 index) {
        return "tile:" + std::to_string(index);
    }

    static std::optional<Sakura::DockTiles> Arrange(const std::vector<Tile>& tiles, Extent2D<U8> dims) {
        std::vector<Span> spans;
        const U32 width = dims.x;
        const U32 height = dims.y;
        std::vector<U8> cells(static_cast<U64>(width) * height, 0);
        for (U64 index = 0; index < tiles.size(); ++index) {
            const auto& tile = tiles[index];
            const Span span = {
                .pane = PaneId(index),
                .x0 = tile.offset.x,
                .y0 = tile.offset.y,
                .x1 = static_cast<U32>(tile.offset.x) + std::max<U32>(1, tile.size.x),
                .y1 = static_cast<U32>(tile.offset.y) + std::max<U32>(1, tile.size.y),
            };
            if (span.x1 > width || span.y1 > height) {
                return std::nullopt;
            }
            for (U32 y = span.y0; y < span.y1; ++y) {
                for (U32 x = span.x0; x < span.x1; ++x) {
                    if (cells[static_cast<U64>(y) * width + x]++ != 0) {
                        return std::nullopt;
                    }
                }
            }
            spans.push_back(span);
        }
        if (std::any_of(cells.begin(), cells.end(), [](U8 cell) { return cell == 0; })) {
            return std::nullopt;
        }
        bool failed = false;
        auto arranged = Cut(spans, 0, 0, width, height, failed);
        if (failed) {
            return std::nullopt;
        }
        return arranged;
    }

 private:
    struct Span {
        std::string pane;
        U32 x0 = 0;
        U32 y0 = 0;
        U32 x1 = 1;
        U32 y1 = 1;
    };

    static std::optional<Sakura::DockTiles> Cut(const std::vector<Span>& spans, U32 x0, U32 y0, U32 x1, U32 y1,
                                                bool& failed) {
        if (spans.empty() || failed) {
            return std::nullopt;
        }
        if (spans.size() == 1) {
            return Sakura::DockTiles{.pane = spans.front().pane};
        }
        for (const bool vertical : {true, false}) {
            const U32 start = vertical ? y0 : x0;
            const U32 end = vertical ? y1 : x1;
            std::vector<U32> cuts = {start};
            for (U32 line = start + 1; line < end; ++line) {
                const bool clear = std::all_of(spans.begin(), spans.end(), [&](const Span& span) {
                    const U32 low = vertical ? span.y0 : span.x0;
                    const U32 high = vertical ? span.y1 : span.x1;
                    return high <= line || low >= line;
                });
                if (clear) {
                    cuts.push_back(line);
                }
            }
            cuts.push_back(end);
            if (cuts.size() <= 2) {
                continue;
            }
            Sakura::DockTiles node = {.vertical = vertical};
            for (U64 index = 0; index + 1 < cuts.size(); ++index) {
                const U32 low = cuts[index];
                const U32 high = cuts[index + 1];
                std::vector<Span> band;
                for (const auto& span : spans) {
                    const U32 begin = vertical ? span.y0 : span.x0;
                    if (begin >= low && begin < high) {
                        band.push_back(span);
                    }
                }
                const auto child = vertical ? Cut(band, x0, low, x1, high, failed) : Cut(band, low, y0, high, y1, failed);
                if (child.has_value()) {
                    node.children.push_back(*child);
                    node.weights.push_back(static_cast<F32>(high - low));
                }
            }
            if (node.children.size() == 1) {
                return node.children.front();
            }
            return node;
        }
        failed = true;
        return std::nullopt;
    }

    static F32 Fraction(const std::optional<Sakura::DockTiles>& tiles) {
        if (!tiles.has_value() || !tiles->vertical || tiles->children.size() < 2 ||
            tiles->children.front() != Sakura::DockTiles{.pane = GlobalPane}) {
            return -1.0f;
        }
        F32 sum = 0.0f;
        for (U64 index = 0; index < tiles->children.size(); ++index) {
            sum += index < tiles->weights.size() ? tiles->weights[index] : 1.0f;
        }
        const F32 first = tiles->weights.empty() ? 1.0f : tiles->weights.front();
        return sum > 0.0f ? first / sum : -1.0f;
    }

    std::optional<Sakura::DockTiles> Stacked(F32 weight) const {
        if (!withGlobal) {
            return mosaic;
        }
        if (!mosaic.has_value()) {
            return Sakura::DockTiles{.pane = GlobalPane};
        }
        return Sakura::DockTiles{
            .vertical = true,
            .weights = {weight, 1.0f - weight},
            .children = {Sakura::DockTiles{.pane = GlobalPane}, *mosaic},
        };
    }

    void settle(F32 natural, F32 height) {
        const F32 chrome = Sakura::Dock::HeaderHeight + 2.0f * Sakura::Dock::ContentPadding;
        const F32 target = natural + FitSlack + chrome;
        if (free) {
            const F32 next = target + 2.0f * Sakura::Dock::ContentPadding;
            if (std::abs(next - globalHeight) > 0.5f) {
                globalHeight = next;
            }
            return;
        }
        const F32 current = Fraction(layout);
        if (fitted || height <= 0.0f || current <= 0.0f || dock.resizing()) {
            return;
        }
        const F32 tile = height + chrome;
        if (std::abs(target - tile) <= 1.0f) {
            return;
        }
        const F32 usable = tile / current;
        const F32 weight = std::clamp(target / std::max(usable, 1.0f), 0.05f, GlobalLimit);
        if (std::abs(weight - current) < 1e-4f || current >= 1.0f) {
            return;
        }
        auto next = *layout;
        next.weights.resize(next.children.size(), 1.0f);
        F32 sum = 0.0f;
        for (const F32 value : next.weights) {
            sum += value;
        }
        const F32 scale = (1.0f - weight) / (1.0f - current);
        for (U64 index = 1; index < next.weights.size(); ++index) {
            next.weights[index] *= scale;
        }
        next.weights.front() = weight * sum;
        layout = std::move(next);
    }

    void select(Extent2D<F32> point) {
        std::string pane = dock.paneAt(point);
        for (auto it = stacking.rbegin(); it != stacking.rend() && pane.empty(); ++it) {
            pane = (*it)->paneAt(point);
        }
        if (!pane.empty()) {
            selected = pane;
        }
    }

    void record(const std::string& box, const InterfaceCallback& callback) {
        recorder.begin(box);
        callback();
        recorder.end();
    }

    Recorder recorder;
    std::vector<std::string> arranged;
    std::optional<Sakura::DockTiles> mosaic;
    std::optional<Sakura::DockTiles> layout;
    bool fitted = true;
    bool free = false;
    F32 globalHeight = GlobalHeight;
    F32 windowHeight = 0.0f;
    Extent2D<U8> grid = {0, 0};
    bool withGlobal = false;
    std::vector<Sakura::Dock*> stacking;
    std::string selected;
    std::unordered_map<std::string, std::unique_ptr<Box>> boxes;
    std::unordered_map<std::string, std::unique_ptr<Plot>> plots;
    std::unordered_map<std::string, std::unique_ptr<Sakura::Dock>> frames;
    Box global;
    Sakura::Dock dock;
    Mosaic mosaicFrames;
    Sakura::Text barTitle;
    Sakura::Overlay barRow;
    Sakura::Div bar;
    Sakura::Stack shell;
    Shell window;
    Sakura::BackgroundLayer background;
    Sakura::Layer content;
    Sakura::ToastLayer toasts;
};

}  // namespace Jetstream

#endif  // JETSTREAM_SUPERLUMINAL_VIEW_HH
