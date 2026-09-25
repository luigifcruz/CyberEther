#ifndef JETSTREAM_RENDER_SAKURA_RETAINED_STATUS_BAR_HH
#define JETSTREAM_RENDER_SAKURA_RETAINED_STATUS_BAR_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/typography.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <string>

namespace Jetstream::Sakura::Retained {

enum class StatusTone : U8 {
    Info,
    Success,
    Warning,
    Error,
};

struct StatusBar : public Component {
    static constexpr F32 Height = 24.0f;

    struct Config {
        std::string id;
        std::string status;
        StatusTone tone = StatusTone::Info;
        std::string toggleText;
        bool toggleExpanded = false;
        F32 fontSize = Typography::FontSize;
        F32 pixelRatio = 1.0f;
        std::function<void()> onToggle;
    };

    static F32 HeightPixels(F32 pixelRatio);

    StatusBar();
    ~StatusBar();

    bool update(Config config);

 protected:
    void layout(const Context& ctx) override;
    bool event(const MouseEvent& event) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura::Retained

#endif  // JETSTREAM_RENDER_SAKURA_RETAINED_STATUS_BAR_HH
