#ifndef JETSTREAM_RENDER_SAKURA_DROPDOWN_HH
#define JETSTREAM_RENDER_SAKURA_DROPDOWN_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct Dropdown {
    struct Config {
        std::string id;
        std::vector<std::string> options;
        std::optional<U64> selected;
        F32 rowGap = 0.0f;
        F32 padding = 12.0f;
        Extent2D<F32> rowPadding = {12.0f, 5.0f};
        U64 visibleRows = 8;
        Callback<void(U64)> onSelect;

        bool operator==(const Config&) const = default;
    };

    Dropdown();
    ~Dropdown();

    Dropdown(const Dropdown&) = delete;
    Dropdown& operator=(const Dropdown&) = delete;

    void open();
    void close();
    bool isOpen() const;

    void update(Config config);
    void request(const Sakura::Context& ctx, const Rect& anchor);

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_DROPDOWN_HH
