#ifndef JETSTREAM_RENDER_SAKURA_VSTACK_HH
#define JETSTREAM_RENDER_SAKURA_VSTACK_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct VStack {
    using Child = std::function<void(const Context&)>;
    using Children = std::vector<Child>;

    struct Flex {
        F32 minimum = 0.0f;
        F32 grow = 1.0f;
        std::optional<F32> basis;
    };

    struct Item {
        std::string id;
        std::optional<Flex> flex;
    };

    struct Config {
        std::string id;
        F32 spacing = 0.0f;
        std::optional<F32> height;
        std::vector<Item> items;
    };

    struct Layout {
        bool measured = false;
        F32 fixedHeight = 0.0f;
        std::optional<F32> minimumHeight;
        std::vector<std::optional<F32>> itemHeights;

        std::optional<F32> itemHeight(U64 index) const;
    };

    VStack();
    ~VStack();

    VStack(VStack&&) noexcept;
    VStack& operator=(VStack&&) noexcept;

    VStack(const VStack&) = delete;
    VStack& operator=(const VStack&) = delete;

    bool update(Config config);
    const Layout* layout() const;
    void render(const Context& ctx, Children children) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_VSTACK_HH
