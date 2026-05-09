#pragma once

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/context.hh>
#include <jetstream/types.hh>

#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct DebugWindow : public Component {
    struct Config {
        std::string id;
        bool visible = false;
        std::string title = "Latency Debug";
        Extent2D<F32> size = {200.0f, 120.0f};
        F32 verticalRatio = 0.25f;
    };

    DebugWindow();
    ~DebugWindow();

    DebugWindow(DebugWindow&&) noexcept;
    DebugWindow& operator=(DebugWindow&&) noexcept;

    DebugWindow(const DebugWindow&) = delete;
    DebugWindow& operator=(const DebugWindow&) = delete;

    bool update(Config config);
    void render(const Context& ctx);

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura
