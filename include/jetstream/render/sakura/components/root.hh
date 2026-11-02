#ifndef JETSTREAM_RENDER_SAKURA_ROOT_HH
#define JETSTREAM_RENDER_SAKURA_ROOT_HH

#include <jetstream/render/sakura/components/layer.hh>
#include <jetstream/types.hh>

#include <memory>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct Root {
    struct Config {
        std::string id;
        F32 topOffset = 0.0f;
        std::vector<Layer*> layers;

        bool operator==(const Config&) const = default;
    };

    Root();
    ~Root();

    Root(const Root&) = delete;
    Root& operator=(const Root&) = delete;

    void update(Config config);
    void render(const Sakura::Context& ctx);

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_ROOT_HH
