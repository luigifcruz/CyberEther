#ifndef JETSTREAM_RENDER_SAKURA_WORKSPACE_BACKGROUND_HH
#define JETSTREAM_RENDER_SAKURA_WORKSPACE_BACKGROUND_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct WorkspaceBackground {
    struct Config {
        std::string id;
        bool particles = true;
        F32 topOffset = 0.0f;
        U64 particleCount = 60;
        std::string backgroundColorKey = "background";
        std::string particleColorKey = "workspace_particle";
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

#endif  // JETSTREAM_RENDER_SAKURA_WORKSPACE_BACKGROUND_HH
