#ifndef JETSTREAM_RENDER_SAKURA_SPLIT_VIEW_HH
#define JETSTREAM_RENDER_SAKURA_SPLIT_VIEW_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct SplitView {
    using Child = std::function<void(const Context&)>;
    using Children = std::vector<Child>;

    struct Config {
        std::string id;
        F32 leftWidth = 220.0f;
        F32 height = 0.0f;
        bool fillHeight = false;
        F32 reservedHeight = 0.0f;
    };

    SplitView();
    ~SplitView();

    SplitView(SplitView&&) noexcept;
    SplitView& operator=(SplitView&&) noexcept;

    SplitView(const SplitView&) = delete;
    SplitView& operator=(const SplitView&) = delete;

    bool update(Config config);
    void render(const Context& ctx, Children children) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_SPLIT_VIEW_HH
