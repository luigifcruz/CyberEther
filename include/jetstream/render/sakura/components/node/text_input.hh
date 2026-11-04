#ifndef JETSTREAM_RENDER_SAKURA_NODE_TEXT_INPUT_HH
#define JETSTREAM_RENDER_SAKURA_NODE_TEXT_INPUT_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct NodeTextInput : public Component {
    enum class Submit : U8 {
        OnEnter,
        OnCommit,
    };

    struct Config {
        std::string id;
        std::string value;
        std::string hint;
        std::string unit;
        Submit submit = Submit::OnEnter;
        bool selectAllOnFocus = false;
        Callback<bool(U32)> characterFilter;
        bool background = true;
        std::string textColorKey = "text_primary";
        F32 height = 0.0f;
        Callback<void(const std::string&)> onChange;
        Callback<void()> onDeactivate;

        bool operator==(const Config&) const = default;
    };

    NodeTextInput();
    ~NodeTextInput();

    void update(Config config);
    bool focused() const;
    void focus(bool selectAll = false);
    void focusWhenPlaced(bool selectAll = false);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;
    bool mouseEvent(const MouseEvent& event) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_NODE_TEXT_INPUT_HH
