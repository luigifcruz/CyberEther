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

    static std::vector<std::optional<U64>> Slots(U64 page, U64 pages) {
        std::vector<std::optional<U64>> slots;
        if (pages <= Window) {
            for (U64 index = 0; index < pages; ++index) {
                slots.push_back(index);
            }
            return slots;
        }
        const U64 first = std::clamp<U64>(page, 2, pages - 3) - 1;
        const U64 last = first + 2;
        slots.push_back(0);
        if (first > 1) {
            slots.push_back(std::nullopt);
        }
        for (U64 index = first; index <= last; ++index) {
            slots.push_back(index);
        }
        if (last < pages - 2) {
            slots.push_back(std::nullopt);
        }
        slots.push_back(pages - 1);
        return slots;
    }

    void update(Config config) {
        this->config = std::move(config);
        const U64 pages = this->config.pages;
        const U64 page = std::min(this->config.page, pages > 0 ? pages - 1 : 0);
        const auto slots = Slots(page, pages);
        while (numbers.size() < slots.size()) {
            numbers.push_back(std::make_unique<Sakura::Button>());
        }

        std::vector<Sakura::Stack::Item> items;
        style(previous, ICON_LC_CHEVRON_LEFT, false, page == 0, page > 0 ? page - 1 : 0);
        items.push_back({.child = &previous});
        for (U64 index = 0; index < slots.size(); ++index) {
            const auto slot = slots[index];
            if (slot.has_value()) {
                style(*numbers[index], jst::fmt::format("{}", *slot + 1), *slot == page, false, *slot);
            } else {
                style(*numbers[index], "...", false, true, page);
            }
            items.push_back({.child = numbers[index].get()});
        }
        style(next, ICON_LC_CHEVRON_RIGHT, false, page + 1 >= pages, page + 1);
        items.push_back({.child = &next});

        Stack::update({
            .axis = Sakura::Stack::Axis::Horizontal,
            .spacing = Spacing,
            .items = std::move(items),
        });
    }

 private:
    void style(Sakura::Button& button, std::string label, bool selected, bool disabled, U64 target) {
        button.update({
            .str = std::move(label),
            .size = {ButtonSize, ButtonSize},
            .disabled = disabled,
            .chrome = true,
            .colorKey = selected ? "primary_btn" : "card_surface",
            .hoveredColorKey = selected ? "primary_btn_hovered" : "card_surface_hovered",
            .activeColorKey = selected ? "primary_btn_hovered" : "card_surface_hovered",
            .borderColorKey = selected ? "" : "setting_card_border",
            .textColorKey = selected ? "primary_btn_text" : "text_secondary",
            .hoveredTextColorKey = selected ? "primary_btn_text" : "text_primary",
            .textScale = NumberScale,
            .rounding = Rounding,
            .padding = {0.0f, 0.0f},
            .onClick = [this, target]() {
                if (config.onSelect && target != config.page) {
                    config.onSelect(target);
                }
            },
            .font = Sakura::Text::Font::SansBold,
        });
    }

    Config config;
    Sakura::Button previous;
    std::vector<std::unique_ptr<Sakura::Button>> numbers;
    Sakura::Button next;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_PAGINATION_HH
