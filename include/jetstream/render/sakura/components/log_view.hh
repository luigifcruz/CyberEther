#ifndef JETSTREAM_RENDER_SAKURA_LOG_VIEW_HH
#define JETSTREAM_RENDER_SAKURA_LOG_VIEW_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/typography.hh>
#include <jetstream/types.hh>

#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct LogView : public Component {
    struct Config {
        std::string value;
        F32 height = 300.0f;
        F32 fontSize = Typography::FontSize;
        F32 padding = 6.0f;
        bool stickToBottom = false;
        std::string backgroundColorKey = "editor_console_background";
        std::string textColorKey = "editor_text";

        bool operator==(const Config&) const = default;
    };

    LogView();
    ~LogView();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_LOG_VIEW_HH
