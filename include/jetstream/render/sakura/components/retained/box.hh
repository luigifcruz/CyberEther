#ifndef JETSTREAM_RENDER_SAKURA_RETAINED_BOX_HH
#define JETSTREAM_RENDER_SAKURA_RETAINED_BOX_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Jetstream::Sakura::Retained {

struct Box : public Component {
    struct Instance {
        Rect rect;
        bool visible = true;
        ColorRGBA<F32> backgroundColor = {1.0f, 1.0f, 1.0f, 1.0f};

        bool operator==(const Instance&) const = default;
    };

    struct Config {
        std::string id;
        std::vector<Instance> instances;
        std::optional<Rect> clip;
        F32 cornerRadius = 0.0f;
        F32 borderWidth = 0.0f;
        ColorRGBA<F32> borderColor = {0.0f, 0.0f, 0.0f, 1.0f};
        U64 capacity = 0;
    };

    Box();
    ~Box();

    bool update(Config config);

 protected:
    Result build(Context& ctx) override;
    Result paint() override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura::Retained

#endif  // JETSTREAM_RENDER_SAKURA_RETAINED_BOX_HH
