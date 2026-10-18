#include "runtime.hh"

namespace Jetstream::Sakura {

struct Runtime::Impl {
    Config config;
    InputFrame input;
};

Runtime::Runtime() {
    this->impl = std::make_unique<Impl>();
}

Runtime::~Runtime() = default;
Runtime::Runtime(Runtime&&) noexcept = default;
Runtime& Runtime::operator=(Runtime&&) noexcept = default;

void Runtime::update(Config config) {
    this->impl->config = std::move(config);
}

Context Runtime::context() {
    const auto& config = this->impl->config;
    Private::SampleInput(this->impl->input);
    return Context{
        .palette = config.palette ? *config.palette : EmptyPalette(),
        .render = config.render,
        .input = &this->impl->input,
    };
}

}  // namespace Jetstream::Sakura
