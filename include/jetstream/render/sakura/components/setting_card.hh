#ifndef JETSTREAM_RENDER_SAKURA_SETTING_CARD_HH
#define JETSTREAM_RENDER_SAKURA_SETTING_CARD_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct SettingCard : public Component {
    struct Row {
        std::string label;
        std::string description;
        Component* control = nullptr;
        Callback<void()> onClick;
        bool wide = false;

        bool operator==(const Row&) const = default;
    };

    struct Config {
        std::vector<Row> rows;
        std::string empty;

        bool operator==(const Config&) const = default;
    };

    SettingCard();
    ~SettingCard();

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

#endif  // JETSTREAM_RENDER_SAKURA_SETTING_CARD_HH
