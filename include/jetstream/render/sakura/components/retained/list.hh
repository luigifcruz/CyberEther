#ifndef JETSTREAM_RENDER_SAKURA_RETAINED_LIST_HH
#define JETSTREAM_RENDER_SAKURA_RETAINED_LIST_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace Jetstream::Sakura::Retained {

struct List : public Component {
    struct Config {
        std::string id;
        F32 rowHeight = 24.0f;
        U64 rowCount = 0;
        F32 scrollY = 0.0f;
        U64 slotCapacity = 32;
        std::function<void(F32 scrollY)> onScrollY;
        std::function<void(U64 slot,
                           std::optional<U64> row,
                           const Rect& rowRect,
                           const std::optional<Rect>& clip)> onBindRow;
    };

    List();
    ~List();

    bool update(Config config);

 protected:
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura::Retained

#endif  // JETSTREAM_RENDER_SAKURA_RETAINED_LIST_HH
