#pragma once

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/context.hh>
#include <jetstream/types.hh>

#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct ProgressBar : public Component {
    struct Config {
        std::string id;
        F32 value = 0.0f;
        std::string overlay;
        Extent2D<F32> size = {-1.0f, 20.0f};
        std::string colorKey = "action_btn";
    };

    ProgressBar();
    ~ProgressBar();

    ProgressBar(ProgressBar&&) noexcept;
    ProgressBar& operator=(ProgressBar&&) noexcept;

    ProgressBar(const ProgressBar&) = delete;
    ProgressBar& operator=(const ProgressBar&) = delete;

    bool update(Config config);
    void render(const Context& ctx) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura
