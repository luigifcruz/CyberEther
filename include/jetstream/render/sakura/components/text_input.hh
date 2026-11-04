#ifndef JETSTREAM_RENDER_SAKURA_TEXT_INPUT_HH
#define JETSTREAM_RENDER_SAKURA_TEXT_INPUT_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/typography.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace Jetstream::Sakura {

struct TextInput : public Component {
    enum class Submit {
        OnEdit,
        OnEnter,
        OnCommit,
    };

    struct Config {
        std::string value;
        std::string hint;
        Submit submit = Submit::OnEnter;
        bool focus = false;
        bool selectAllOnFocus = false;
        bool keepFocusOnSubmit = false;
        bool multiline = false;
        bool password = false;
        bool disabled = false;
        F32 height = 0.0f;
        Callback<void(const std::string&)> onChange;
        Callback<void(const std::string&)> onSubmit;
        std::string backgroundColorKey = "card";
        Callback<void()> onCancel;
        Extent2D<F32> padding = {12.0f, 5.0f};
        F32 fontSize = Typography::FontSize;
        std::string fontName = Typography::MonoFont;
        std::string borderColorKey;
        F32 rounding = 8.0f;

        bool operator==(const Config&) const = default;
    };

    TextInput();
    ~TextInput();

    void update(Config config);
    bool focused() const;
    void focus(bool selectAll = false);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;
    bool mouseEvent(const MouseEvent& event) override;
    std::optional<SurfaceCursor> cursorOverChildren(const Extent2D<F32>& point) const override;
    std::optional<SurfaceCursor> cursorAt(const Extent2D<F32>& point) const override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_TEXT_INPUT_HH
