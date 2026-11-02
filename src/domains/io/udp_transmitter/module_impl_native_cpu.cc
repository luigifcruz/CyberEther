// TODO: Explore batching UDP transmissions to reduce overhead and improve throughput.

#include <algorithm>

#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct UdpTransmitterImplNativeCpu : public UdpTransmitterImpl,
                                     public NativeCpuRuntimeContext,
                                     public Scheduler::Context {
 public:
    Result computeSubmit() override;
};

Result UdpTransmitterImplNativeCpu::computeSubmit() {
    const auto& input = inputs().at("buffer").tensor;
    const auto* data = static_cast<const U8*>(input.data());
    const U64 totalBytes = input.sizeBytes();

    U64 sentBytes = 0;
    U64 dropped = 0;
    for (U64 offset = 0; offset < totalBytes; offset += payloadBytes) {
        const U64 size = std::min(payloadBytes, totalBytes - offset);
        const Result result = socket.send(data + offset, size);
        if (result == Result::SUCCESS) {
            sentBytes += size;
        } else if (result == Result::TIMEOUT) {
            dropped += (totalBytes - offset + payloadBytes - 1) / payloadBytes;
            break;
        } else {
            dropped += 1;
        }
    }

    if (dropped > 0) {
        if (!warnedDrop) {
            JST_WARN("[MODULE_UDP_TRANSMITTER] Failed to send packets to '{}' port {}.",
                     address, port);
            warnedDrop = true;
        }
        droppedPackets.publish(droppedPackets.get() + dropped);
    }

    updateThroughput(sentBytes);

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(UdpTransmitterImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
