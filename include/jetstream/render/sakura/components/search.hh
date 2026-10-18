#ifndef JETSTREAM_RENDER_SAKURA_SEARCH_HH
#define JETSTREAM_RENDER_SAKURA_SEARCH_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct PillField : public Component {
    struct Config {
        std::string icon;
        std::string iconColorKey = "text_secondary";
        std::string placeholder;
        std::string query;
        std::string trailing;
        bool editing = false;
        Extent2D<F32> size = {0.0f, 34.0f};
        Callback<void()> onOpen;
        Callback<void(const std::string&)> onChange;
        Callback<void(const std::string&)> onSubmit;
        Callback<void()> onCancel;

        bool operator==(const Config&) const = default;
    };

    PillField();
    ~PillField();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;
    bool mouseEvent(const MouseEvent& event) override;
    std::optional<SurfaceCursor> cursorAt(const Extent2D<F32>& point) const override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

struct PillStepper : public Component {
    struct Config {
        std::string label;
        bool canDecrement = true;
        bool canIncrement = true;
        Extent2D<F32> size = {0.0f, 34.0f};
        Callback<void()> onDecrement;
        Callback<void()> onIncrement;
        Callback<void()> onReset;

        bool operator==(const Config&) const = default;
    };

    PillStepper();
    ~PillStepper();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;
    bool mouseEvent(const MouseEvent& event) override;
    std::optional<SurfaceCursor> cursorAt(const Extent2D<F32>& point) const override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

struct PillButton : public Component {
    struct Config {
        std::string icon;
        std::string label;
        std::string colorKey = "text_primary";
        std::string backgroundColorKey = "pill_bg";
        std::string hoverColorKey = "pill_hover_bg";
        bool raised = false;
        F32 height = 28.0f;
        Callback<void()> onClick;

        bool operator==(const Config&) const = default;
    };

    PillButton();
    ~PillButton();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;
    bool mouseEvent(const MouseEvent& event) override;
    std::optional<SurfaceCursor> cursorAt(const Extent2D<F32>& point) const override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

struct SearchCatalog : public Component {
    struct Item {
        std::string id;
        std::string icon;
        std::string title;
        std::string summary;
        std::string group;
        std::vector<std::string> badges;
        std::string documentation;

        bool operator==(const Item&) const = default;
    };

    enum class DocsSide {
        Left,
        Right,
    };

    struct Config {
        std::vector<Item> items;
        std::vector<std::string> groups;
        std::string query;
        std::string activeGroup;
        U64 selected = 0;
        Component* field = nullptr;
        DocsSide docsSide = DocsSide::Right;
        std::string emptyLabel = "No matches.";
        F32 listWidth = 300.0f;
        F32 docsWidth = 440.0f;
        F32 maxHeight = 460.0f;
        Callback<void(U64)> onSelect;
        Callback<void(const std::string&, std::optional<U64>)> onActivate;
        Callback<void(const std::string&)> onGroup;

        bool operator==(const Config&) const = default;
    };

    SearchCatalog();
    ~SearchCatalog();

    void update(Config config);
    const std::vector<std::string>& results() const;

    static std::vector<std::string> Groups(const std::vector<Item>& items, const std::vector<std::string>& groups);
    static std::vector<U64> Rank(const std::vector<Item>& items,
                                 const std::vector<std::string>& groups,
                                 const std::string& query,
                                 const std::string& activeGroup = "");

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;

    void mountParts();
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_SEARCH_HH
