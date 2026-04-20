#include "jetstream/tensor_link.hh"

namespace Jetstream {

bool TensorLink::resolved() const {
    return producer.has_value() && !producer->module.empty() && !producer->port.empty() && !tensor.empty();
}

void TensorLink::requested(const std::string& block, const std::string& port) {
    external = BlockEndpoint{block, port};
    producer = std::nullopt;
    tensor = Tensor{};
}

void TensorLink::produced(const std::string& module, const std::string& port, const Tensor& value) {
    producer = ModuleEndpoint{module, port};
    external = std::nullopt;
    tensor = value;
}

void TensorLink::exposedAs(const std::string& block, const std::string& port) {
    external = BlockEndpoint{block, port};
}

}  // namespace Jetstream
