#include <jetstream/render/sakura/components/node/uint_input.hh>

#include <jetstream/render/sakura/components/node/text_input.hh>

#include "common.hh"

#include <cerrno>
#include <cstdlib>
#include <utility>

namespace Jetstream::Sakura {

using namespace NodeStyle;

namespace {

std::optional<U64> ParseUInt(const std::string& text) {
    const std::string trimmed = Trim(text);
    if (trimmed.empty() || trimmed.front() == '-') {
        return std::nullopt;
    }
    char* end = nullptr;
    errno = 0;
    const unsigned long long value = std::strtoull(trimmed.c_str(), &end, 10);
    if (end == trimmed.c_str() || errno == ERANGE) {
        return std::nullopt;
    }
    return static_cast<U64>(value);
}

}  // namespace

struct NodeUIntInput::Impl {
    Config config;
    NodeTextInput input;
};

NodeUIntInput::NodeUIntInput() {
    this->impl = std::make_unique<Impl>();
    addChild(impl->input);
}

NodeUIntInput::~NodeUIntInput() = default;

void NodeUIntInput::update(Config config) {
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
            const auto parsed = ParseUInt(text);
            if (parsed.has_value() && *parsed != impl->config.value && impl->config.onChange) {
                impl->config.onChange(*parsed);
            }
        },
    });
}

Extent2D<F32> NodeUIntInput::measure(const Context& ctx, Extent2D<F32> available) {
    return {FillLength(available.x), FitLength(Px(ctx, FrameHeight(ctx)), available.y)};
}

void NodeUIntInput::layout(const Context& ctx) {
    layoutChild(ctx, impl->input, frame());
}

}  // namespace Jetstream::Sakura
