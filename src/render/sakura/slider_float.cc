#include <jetstream/render/sakura/slider_float.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct SliderFloat::Impl {
    Config config;
};

SliderFloat::SliderFloat() {
    this->impl = std::make_unique<Impl>();
}

SliderFloat::~SliderFloat() = default;
SliderFloat::SliderFloat(SliderFloat&&) noexcept = default;
SliderFloat& SliderFloat::operator=(SliderFloat&&) noexcept = default;

bool SliderFloat::update(Config config) {
    this->impl->config = std::move(config);
    return true;
}

void SliderFloat::render(const Context& ctx) const {
    (void)ctx;
    const auto& config = this->impl->config;

    ImGui::PushID(config.id.c_str());
    ImGui::SetNextItemWidth(-FLT_MIN);

    F32 value = config.value;
    const bool changed = ImGui::SliderFloat("##slider",
                                            &value,
                                            config.min,
                                            config.max,
                                            config.format.c_str());
    if (changed && config.onChange) {
        config.onChange(value);
    }
    ImGui::PopID();
}

}  // namespace Jetstream::Sakura
