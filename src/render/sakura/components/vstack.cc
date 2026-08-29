#include <jetstream/render/sakura/components/vstack.hh>

#include "../helpers.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace Jetstream::Sakura {

struct VStack::Impl {
    struct Measurement {
        U64 revision = 0;
        F32 height = 0.0f;
        std::vector<F32> itemHeights;
    };

    Config config;
    U64 revision = 0;
    std::optional<Measurement> measurement;
    std::optional<F32> lastMeasuredFixedHeight;
    std::optional<Layout> layout;
};

namespace {

F32 Normalize(const F32 value) {
    return std::isfinite(value) ? std::max(0.0f, value) : 0.0f;
}

bool SameFlex(const std::optional<VStack::Flex>& lhs,
              const std::optional<VStack::Flex>& rhs) {
    if (lhs.has_value() != rhs.has_value()) {
        return false;
    }
    if (!lhs.has_value()) {
        return true;
    }
    // Basis changes only redistribute known flexible space; fixed children
    // do not need to be measured again.
    return lhs->minimum == rhs->minimum && lhs->grow == rhs->grow;
}

bool SameLayoutContract(const VStack::Config& lhs, const VStack::Config& rhs) {
    if (lhs.id != rhs.id || lhs.spacing != rhs.spacing || lhs.items.size() != rhs.items.size()) {
        return false;
    }
    for (U64 i = 0; i < lhs.items.size(); ++i) {
        if (lhs.items[i].id != rhs.items[i].id ||
            !SameFlex(lhs.items[i].flex, rhs.items[i].flex)) {
            return false;
        }
    }
    return true;
}

F32 FloorToF32(const F64 value) {
    F32 result = static_cast<F32>(std::min(
        value,
        static_cast<F64>(std::numeric_limits<F32>::max())));
    if (static_cast<F64>(result) > value) {
        result = std::nextafter(result, 0.0f);
    }
    return std::max(0.0f, result);
}

F32 MeasuredHeight(const Context& ctx, const F32 start) {
    const F32 height = std::max(0.0f,
        ImGui::GetCursorScreenPos().y - start - ImGui::GetStyle().ItemSpacing.y);
    return Unscale(ctx, height);
}

}  // namespace

VStack::VStack() {
    this->impl = std::make_unique<Impl>();
}

VStack::~VStack() = default;
VStack::VStack(VStack&&) noexcept = default;
VStack& VStack::operator=(VStack&&) noexcept = default;

bool VStack::update(Config config) {
    config.spacing = Normalize(config.spacing);
    if (config.height.has_value()) {
        config.height = Normalize(*config.height);
    }
    for (auto& item : config.items) {
        if (item.flex.has_value()) {
            item.flex->minimum = Normalize(item.flex->minimum);
            item.flex->grow = Normalize(item.flex->grow);
            if (item.flex->basis.has_value()) {
                item.flex->basis = std::max(item.flex->minimum,
                                            Normalize(*item.flex->basis));
            }
        }
    }

    if (!SameLayoutContract(this->impl->config, config)) {
        ++this->impl->revision;
        this->impl->measurement.reset();
        this->impl->lastMeasuredFixedHeight.reset();
    }
    this->impl->config = std::move(config);
    this->impl->layout.reset();

    const auto& currentConfig = this->impl->config;
    if (currentConfig.items.empty()) {
        return true;
    }

    Layout layout{
        .itemHeights = std::vector<std::optional<F32>>(currentConfig.items.size()),
    };
    const auto& measurement = this->impl->measurement;
    const bool hasMeasurement = measurement.has_value() &&
                                measurement->revision == this->impl->revision &&
                                measurement->itemHeights.size() == currentConfig.items.size();
    if (hasMeasurement) {
        layout.measured = true;
    }
    F32 measuredFlexibleHeight = 0.0f;
    F64 minimumFlexibleHeight = 0.0;
    F64 basisFlexibleHeight = 0.0;
    F64 totalGrow = 0.0;
    U64 flexibleItemCount = 0;
    for (U64 i = 0; i < currentConfig.items.size(); ++i) {
        const auto& flex = currentConfig.items[i].flex;
        if (!flex.has_value()) {
            continue;
        }
        if (hasMeasurement) {
            measuredFlexibleHeight += measurement->itemHeights[i];
        }
        minimumFlexibleHeight += flex->minimum;
        basisFlexibleHeight += flex->basis.value_or(flex->minimum);
        totalGrow += flex->grow;
        ++flexibleItemCount;
    }
    if (hasMeasurement) {
        layout.fixedHeight = std::max(0.0f, measurement->height - measuredFlexibleHeight);
        this->impl->lastMeasuredFixedHeight = layout.fixedHeight;
    } else if (this->impl->lastMeasuredFixedHeight.has_value()) {
        layout.fixedHeight = *this->impl->lastMeasuredFixedHeight;
    }

    if (!currentConfig.height.has_value() || flexibleItemCount == 0) {
        this->impl->layout = std::move(layout);
        return true;
    }

    const F32 availableHeight = std::max(0.0f, *currentConfig.height - layout.fixedHeight);
    if (hasMeasurement) {
        layout.minimumHeight = layout.fixedHeight + static_cast<F32>(minimumFlexibleHeight);
    }

    const F64 available = availableHeight;
    const F64 growHeight = std::max(0.0, available - basisFlexibleHeight);
    const F64 growUnit = totalGrow > 0.0 ? growHeight / totalGrow : 0.0;
    const F64 minimumScale = minimumFlexibleHeight > 0.0
        ? std::min(1.0, available / minimumFlexibleHeight)
        : 1.0;
    const F64 basisScale = basisFlexibleHeight > minimumFlexibleHeight
        ? std::clamp((available - minimumFlexibleHeight) /
                         (basisFlexibleHeight - minimumFlexibleHeight),
                     0.0,
                     1.0)
        : 1.0;
    U64 lastFlexibleItem = 0;
    for (U64 i = 0; i < currentConfig.items.size(); ++i) {
        if (currentConfig.items[i].flex.has_value()) {
            lastFlexibleItem = i;
        }
    }
    const bool fillAvailable = available <= basisFlexibleHeight ||
                               totalGrow > 0.0;
    F64 remainingHeight = availableHeight;
    for (U64 i = 0; i < currentConfig.items.size(); ++i) {
        const auto& flex = currentConfig.items[i].flex;
        if (flex.has_value()) {
            const F64 minimum = flex->minimum;
            const F64 basis = flex->basis.value_or(flex->minimum);
            F64 height = minimum * minimumScale;
            if (available >= minimumFlexibleHeight) {
                height = minimum + (basis - minimum) * basisScale;
            }
            if (available >= basisFlexibleHeight) {
                height = basis + growUnit * static_cast<F64>(flex->grow);
            }
            const F64 boundedHeight = fillAvailable && i == lastFlexibleItem
                ? remainingHeight
                : std::min(remainingHeight, height);
            const F32 allocatedHeight = FloorToF32(boundedHeight);
            layout.itemHeights[i] = allocatedHeight;
            remainingHeight = std::max(0.0,
                remainingHeight - static_cast<F64>(allocatedHeight));
        }
    }
    this->impl->layout = std::move(layout);
    return true;
}

const VStack::Layout* VStack::layout() const {
    return this->impl->layout.has_value() ? &*this->impl->layout : nullptr;
}

std::optional<F32> VStack::Layout::itemHeight(const U64 index) const {
    return index < itemHeights.size() ? itemHeights[index] : std::nullopt;
}

void VStack::render(const Context& ctx, Children children) const {
    const auto& config = this->impl->config;

    ImGui::PushID(config.id.c_str());

    const bool managed = !config.items.empty() && config.items.size() == children.size();
    const F32 stackStart = managed ? ImGui::GetCursorScreenPos().y : 0.0f;
    std::vector<F32> measuredItemHeights(managed ? children.size() : 0);
    for (U64 i = 0; i < children.size(); ++i) {
        const F32 itemStart = managed ? ImGui::GetCursorScreenPos().y : 0.0f;
        children[i](ctx);
        if (managed) {
            measuredItemHeights[i] = MeasuredHeight(ctx, itemStart);
        }

        if (config.spacing > 0.0f && i + 1 < children.size()) {
            ImGui::Dummy(Private::ToImVec2({0.0f, Scale(ctx, config.spacing)}));
        }
    }

    if (managed) {
        this->impl->measurement = Impl::Measurement{
            .revision = this->impl->revision,
            .height = MeasuredHeight(ctx, stackStart),
            .itemHeights = std::move(measuredItemHeights),
        };
    } else if (!config.items.empty()) {
        this->impl->measurement.reset();
        this->impl->layout.reset();
    }

    ImGui::PopID();
}

}  // namespace Jetstream::Sakura
