#ifndef JETSTREAM_RENDER_SAKURA_STATUS_BAR_HH
#define JETSTREAM_RENDER_SAKURA_STATUS_BAR_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/typography.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <string>

namespace Jetstream::Sakura {

enum class StatusTone : U8 {
    Info,
    Success,
    Warning,
    Error,
};

struct StatusBar : public Component {
    struct Config {
        std::string id;
        std::string status;
        StatusTone tone = StatusTone::Info;
        std::string toggleText;
        bool toggleExpanded = false;
        F32 fontSize = Typography::FontSize;
        F32 cornerRadius = 0.0f;
        Callback<void()> onToggle;

        bool operator==(const Config&) const = default;
    };

    StatusBar();
    ~StatusBar();

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

#endif  // JETSTREAM_RENDER_SAKURA_STATUS_BAR_HH
