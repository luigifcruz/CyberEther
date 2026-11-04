#ifndef JETSTREAM_RENDER_SAKURA_CONSOLE_HH
#define JETSTREAM_RENDER_SAKURA_CONSOLE_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/typography.hh>
#include <jetstream/render/sakura/components/status_bar.hh>
#include <jetstream/types.hh>

#include <memory>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct Console : public Component {
    using StatusTone = Sakura::StatusTone;

    struct Config {
        std::string id;
        std::vector<std::string> output;
        std::string status;
        std::string emptyText = "No output.";
        StatusTone statusTone = StatusTone::Info;
        F32 height = 160.0f;
        F32 fontSize = Typography::FontSize;
        std::string backgroundColorKey = "editor_console_background";

        bool operator==(const Config&) const = default;
    };

    Console();
    ~Console();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_CONSOLE_HH
