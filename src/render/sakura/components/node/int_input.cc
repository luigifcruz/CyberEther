#include <jetstream/render/sakura/components/node/int_input.hh>

#include <jetstream/render/sakura/components/node/text_input.hh>

#include "../../metrics.hh"
#include "parse.hh"

#include <cerrno>
#include <cstdlib>
#include <limits>
#include <utility>

namespace Jetstream::Sakura {

namespace {

std::optional<I64> ParseInt(const std::string& text) {
    const std::string trimmed = Trim(text);
    if (trimmed.empty()) {
        return std::nullopt;
    }
    char* end = nullptr;
    errno = 0;
    const long long value = std::strtoll(trimmed.c_str(), &end, 10);
    if (errno == ERANGE) {
        return std::nullopt;
    }
    if (end == trimmed.c_str()) {
        const auto asFloat = ParseFloat(trimmed);
        constexpr F64 lowest = static_cast<F64>(std::numeric_limits<I64>::min());
        constexpr F64 highest = static_cast<F64>(std::numeric_limits<I64>::max());
        if (!asFloat.has_value() || *asFloat < lowest || *asFloat >= highest) {
            return std::nullopt;
        }
        return static_cast<I64>(*asFloat);
    }
    return static_cast<I64>(value);
}

}  // namespace

struct NodeIntInput::Impl {
    Config config;
    NodeTextInput input;
};

NodeIntInput::NodeIntInput() {
    this->impl = std::make_unique<Impl>();
    addChild(impl->input);
}

NodeIntInput::~NodeIntInput() = default;

void NodeIntInput::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
    impl->input.update({
        .id = impl->config.id + ":value",
        .value = jst::fmt::format("{}", impl->config.value),
        .unit = impl->config.unit,
        .submit = NodeTextInput::Submit::OnEnter,
        .selectAllOnFocus = true,
        .characterFilter = DecimalCharacter,
        .onChange = [this](const std::string& text) {
            const auto parsed = ParseInt(text);
            if (parsed.has_value() && *parsed != impl->config.value && impl->config.onChange) {
                impl->config.onChange(*parsed);
            }
        },
    });
}

Extent2D<F32> NodeIntInput::measure(const Context& ctx, Extent2D<F32> available) {
    return {FillLength(available.x), FitLength(Px(ctx, NodeFrameHeight(ctx)), available.y)};
}

void NodeIntInput::layout(const Context& ctx) {
    layoutChild(ctx, impl->input, frame());
}

}  // namespace Jetstream::Sakura
