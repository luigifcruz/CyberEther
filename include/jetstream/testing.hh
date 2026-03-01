#ifndef JETSTREAM_TESTING_HH
#define JETSTREAM_TESTING_HH

#include <memory>
#include <string>

#include "jetstream/types.hh"
#include "jetstream/memory/tensor.hh"
#include "jetstream/parser.hh"
#include "jetstream/runtime.hh"
#include "jetstream/provider.hh"
#include "jetstream/module.hh"

namespace Jetstream {

class JETSTREAM_API TestContext {
 public:
    TestContext(const std::string& moduleType,
                DeviceType device,
                RuntimeType runtime,
                const ProviderType& provider);
    ~TestContext();

    TestContext(const TestContext&) = delete;
    TestContext& operator=(const TestContext&) = delete;
    TestContext(TestContext&&) noexcept;
    TestContext& operator=(TestContext&&) noexcept;

    template<typename T>
    TypedTensor<T> createTensor(const Shape& shape) {
        return TypedTensor<T>(DeviceType::CPU, shape);
    }

    void setInput(const std::string& name, Tensor& tensor);
    void setConfig(const Module::Config& config);

    Result run();
    Tensor& output(const std::string& name);

    DeviceType device() const;
    RuntimeType runtime() const;
    const ProviderType& provider() const;

 private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

}  // namespace Jetstream

#endif  // JETSTREAM_TESTING_HH
