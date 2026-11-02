#ifndef JETSTREAM_RENDER_SAKURA_PILL_INPUT_HH
#define JETSTREAM_RENDER_SAKURA_PILL_INPUT_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/components/text_input.hh>
#include <jetstream/types.hh>

#include <memory>
#include <optional>
#include <string>

namespace Jetstream::Sakura {

struct PillInput : public Component {
    struct Config {
        std::string icon;
        std::string value;
        std::string hint;
        TextInput::Submit submit = TextInput::Submit::OnCommit;
        bool password = false;
        bool disabled = false;
        F32 width = Fill;
        Callback<void(const std::string&)> onChange;

        bool operator==(const Config&) const = default;
    };

    PillInput();
    ~PillInput();

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

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_PILL_INPUT_HH
