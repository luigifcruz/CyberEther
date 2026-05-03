#pragma once

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/context.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct SliderFloat : public Component {
    struct Config {
        std::string id;
        F32 value = 0.0f;
        F32 min = 0.0f;
        F32 max = 1.0f;
        std::string format = "%.3f";
        std::function<void(F32)> onChange;
    };

    SliderFloat();
    ~SliderFloat();

    SliderFloat(SliderFloat&&) noexcept;
    SliderFloat& operator=(SliderFloat&&) noexcept;

    SliderFloat(const SliderFloat&) = delete;
    SliderFloat& operator=(const SliderFloat&) = delete;

    bool update(Config config);
    void render(const Context& ctx) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura
