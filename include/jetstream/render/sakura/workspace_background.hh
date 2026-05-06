#pragma once

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/context.hh>
#include <jetstream/types.hh>

#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct WorkspaceBackground : public Component {
    struct Config {
        std::string id;
        bool particles = true;
        F32 topOffset = 0.0f;
        U64 particleCount = 60;
        std::string backgroundColorKey = "background";
        std::string particleColorKey = "cyber_blue";
    };

    WorkspaceBackground();
    ~WorkspaceBackground();

    WorkspaceBackground(WorkspaceBackground&&) noexcept;
    WorkspaceBackground& operator=(WorkspaceBackground&&) noexcept;

    WorkspaceBackground(const WorkspaceBackground&) = delete;
    WorkspaceBackground& operator=(const WorkspaceBackground&) = delete;

    bool update(Config config);
    void render(const Context& ctx);

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura
