#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct SoapyImplNativeCpu : public SoapyImpl,
                            public Runtime::Context,
                            public Scheduler::Context {
 public:
    Result create() final;

    Result computeSubmit() override;
    Result hasPendingCompute() override;
};

Result SoapyImplNativeCpu::create() {
    JST_CHECK(SoapyImpl::create());

    return Result::SUCCESS;
}

Result SoapyImplNativeCpu::hasPendingCompute() {
    if (circularBuffer.getOccupancy() < buffer.size()) {
        return circularBuffer.waitBufferOccupancy(buffer.size());
    }

    return Result::SUCCESS;
}

Result SoapyImplNativeCpu::computeSubmit() {
    if (errored) {
        return Result::ERROR;
    }

    if (circularBuffer.getOccupancy() < buffer.size()) {
        return Result::YIELD;
    }

    circularBuffer.get(reinterpret_cast<CF32*>(buffer.data()), buffer.size());

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(SoapyImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
