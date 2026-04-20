#ifndef JETSTREAM_TENSOR_LINK_HH
#define JETSTREAM_TENSOR_LINK_HH

#include <optional>
#include <string>
#include <unordered_map>

#include "jetstream/memory/tensor.hh"

namespace Jetstream {

struct BlockEndpoint {
    std::string block;
    std::string port;
};

struct ModuleEndpoint {
    std::string module;
    std::string port;
};

struct TensorLink {
    std::optional<ModuleEndpoint> producer;
    std::optional<BlockEndpoint> external;
    Tensor tensor;

    void requested(const std::string& block, const std::string& port);
    void produced(const std::string& module, const std::string& port, const Tensor& tensor);
    void exposedAs(const std::string& block, const std::string& port);

    bool resolved() const;
};

using TensorMap = std::unordered_map<std::string, TensorLink>;

}  // namespace Jetstream

#endif  // JETSTREAM_TENSOR_LINK_HH
