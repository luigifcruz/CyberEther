#ifndef JETSTREAM_RENDER_SAKURA_RETAINED_CONSOLE_HH
#define JETSTREAM_RENDER_SAKURA_RETAINED_CONSOLE_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/components/retained/status_bar.hh>
#include <jetstream/types.hh>

#include <memory>
#include <string>
#include <vector>

namespace Jetstream::Sakura::Retained {

struct Console {
    using StatusTone = Retained::StatusTone;

    struct Config {
        std::string id;
        std::vector<std::string> output;
        std::string status;
        std::string emptyText = "No output.";
        StatusTone statusTone = StatusTone::Info;
        Extent2D<F32> size = {0.0f, 160.0f};
        F32 fontSize = 15.0f;
        std::string backgroundColorKey = "editor_console_background";
    };

    Console();
    ~Console();

    Console(Console&&) noexcept;
    Console& operator=(Console&&) noexcept;

    Console(const Console&) = delete;
    Console& operator=(const Console&) = delete;

    bool update(Config config);
    void render(const Sakura::Context& ctx);

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura::Retained

#endif  // JETSTREAM_RENDER_SAKURA_RETAINED_CONSOLE_HH
