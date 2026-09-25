#ifndef JETSTREAM_RENDER_SAKURA_NODE_CONSOLE_HH
#define JETSTREAM_RENDER_SAKURA_NODE_CONSOLE_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/typography.hh>
#include <jetstream/render/sakura/components/retained/console.hh>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct NodeConsole {
    using StatusTone = Retained::Console::StatusTone;

    struct Config {
        std::string id;
        std::vector<std::string> output;
        std::string status;
        std::string emptyText = "No output.";
        StatusTone statusTone = StatusTone::Info;
        std::optional<F32> height;
        F32 fontSize = Typography::FontSize;
    };

    NodeConsole();
    ~NodeConsole();

    NodeConsole(NodeConsole&&) noexcept;
    NodeConsole& operator=(NodeConsole&&) noexcept;

    NodeConsole(const NodeConsole&) = delete;
    NodeConsole& operator=(const NodeConsole&) = delete;

    bool update(Config config);
    void render(const Context& ctx) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_NODE_CONSOLE_HH
