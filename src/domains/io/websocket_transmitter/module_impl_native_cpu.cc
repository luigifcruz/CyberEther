#include <string_view>

#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct WebsocketTransmitterImplNativeCpu : public WebsocketTransmitterImpl,
                                           public NativeCpuRuntimeContext,
                                           public Scheduler::Context {
 public:
    Result computeSubmit() override;
};

Result WebsocketTransmitterImplNativeCpu::computeSubmit() {
    const auto& input = inputs().at("buffer").tensor;
    const std::string_view data(static_cast<const char*>(input.data()), input.sizeBytes());

    updateThroughput(send(data) * data.size());

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(WebsocketTransmitterImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
