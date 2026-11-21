#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_PAGINATION_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_PAGINATION_HH

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct Pagination : public Sakura::Stack {
    static constexpr F32 ButtonSize = 36.0f;
    static constexpr F32 Rounding = Sakura::Rounding::Control;
    static constexpr F32 Spacing = Sakura::Spacing::Small;
    static constexpr F32 NumberScale = Sakura::Typography::Scale::Lead;
    static constexpr U64 Window = 7;

    struct Config {
        U64 page = 0;
        U64 pages = 0;
        Sakura::Callback<void(U64)> onSelect;
    };

    static std::vector<std::optional<U64>> Slots(U64 page, U64 pages);
    void update(Config config);

 private:
    void style(Sakura::Button& button, std::string label, bool selected, bool disabled, U64 target);

    Config config;
    Sakura::Button previous;
    std::vector<std::unique_ptr<Sakura::Button>> numbers;
    Sakura::Button next;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_PAGINATION_HH
