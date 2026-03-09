#ifndef JETSTREAM_INSTANCE_HH
#define JETSTREAM_INSTANCE_HH

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

#include "jetstream/types.hh"
#include "jetstream/compositor.hh"
#include "jetstream/flowgraph.hh"
#include "jetstream/viewport/types.hh"

namespace Jetstream {

namespace Render { class Window; }
namespace Viewport { class Generic; }

class JETSTREAM_API Instance : public std::enable_shared_from_this<Instance> {
 public:
    struct Impl;
    struct Remote;

    struct Config {
        std::optional<DeviceType> device{};
        std::optional<CompositorType> compositor{};
        bool headless{false};
        Extent2D<U64> size{1920, 1080};
        U64 framerate{60};
    };

    Instance();
    ~Instance();

    Result create(const Config& config);
    Result destroy();

    Result start();

    bool computing() const;
    Result compute();

    bool presenting() const;
    Result present(const std::function<Result()>& callback = {});

    bool polling() const;
    Result poll(const bool wait = true);

    Result stop();

    const std::shared_ptr<Remote>& remote();

    Result flowgraphCreate(const std::string name,
                           const Flowgraph::Config& config,
                           std::shared_ptr<Flowgraph>& flowgraph);
    Result flowgraphDestroy(const std::string name);
    Result flowgraphList(std::unordered_map<std::string, std::shared_ptr<Flowgraph>>& flowgraphs);

    Result compositorGet(std::shared_ptr<Compositor>& compositor);
    Result viewportGet(std::shared_ptr<Viewport::Generic>& viewport);
    Result renderGet(std::shared_ptr<Render::Window>& render);

 private:
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream

#endif  // JETSTREAM_INSTANCE_HH
