#ifndef JETSTREAM_RENDER_SAKURA_COLLAPSE_CHEVRON_HH
#define JETSTREAM_RENDER_SAKURA_COLLAPSE_CHEVRON_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct CollapseChevron {
    struct Config {
        std::string id;
        F32 stripHeight = 8.0f;
    };

    CollapseChevron();
    ~CollapseChevron();

    CollapseChevron(CollapseChevron&&) noexcept;
    CollapseChevron& operator=(CollapseChevron&&) noexcept;

    CollapseChevron(const CollapseChevron&) = delete;
    CollapseChevron& operator=(const CollapseChevron&) = delete;

    bool update(Config config);

    bool render(const Context& ctx, bool expanded) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_COLLAPSE_CHEVRON_HH
