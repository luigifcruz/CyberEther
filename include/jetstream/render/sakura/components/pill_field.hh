#ifndef JETSTREAM_RENDER_SAKURA_PILL_FIELD_HH
#define JETSTREAM_RENDER_SAKURA_PILL_FIELD_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>
#include <optional>
#include <string>

namespace Jetstream::Sakura {

struct PillField : public Component {
    struct Config {
        std::string icon;
        std::string iconColorKey = "text_secondary";
        std::string placeholder;
        std::string query;
        std::string trailing;
        bool editing = false;
        F32 scale = 1.0f;
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

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_PILL_FIELD_HH
