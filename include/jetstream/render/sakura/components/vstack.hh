#ifndef JETSTREAM_RENDER_SAKURA_VSTACK_HH
#define JETSTREAM_RENDER_SAKURA_VSTACK_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct VStack {
    using Child = std::function<void(const Context&)>;
    using Children = std::vector<Child>;

    struct Config {
        std::string id;
        F32 spacing = 0.0f;
    };

    VStack();
    ~VStack();

    VStack(VStack&&) noexcept;
    VStack& operator=(VStack&&) noexcept;

    VStack(const VStack&) = delete;
    VStack& operator=(const VStack&) = delete;

    bool update(Config config);
    void render(const Context& ctx, Children children) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_VSTACK_HH
