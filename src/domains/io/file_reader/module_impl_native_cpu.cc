#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct FileReaderImplNativeCpu : public FileReaderImpl,
                                 public NativeCpuRuntimeContext,
                                 public Scheduler::Context {
 public:
    Result create() final;

    Result computeSubmit() override;
};

Result FileReaderImplNativeCpu::create() {
    JST_CHECK(FileReaderImpl::create());

    return Result::SUCCESS;
}

// TODO: Make the file reading asynchronous.

Result FileReaderImplNativeCpu::computeSubmit() {
    if (!dataFile.is_open() || !playing) {
        return Result::SUCCESS;
    }

    const U64 bytesToRead = buffer.sizeBytes();
    const U64 totalSize = fileSize.get();
    U64 currentOffset = currentPosition.get();
    U64 remainingBytes = totalSize - currentOffset;

    if (remainingBytes == 0) {
        if (loop) {
            dataFile.seekg(0, std::ios::beg);
            currentOffset = 0;
            currentPosition.publish(currentOffset);
            remainingBytes = totalSize;
        } else {
            return Result::SUCCESS;
        }
    }

    const U64 actualBytesToRead = std::min(bytesToRead, remainingBytes);

    dataFile.read(reinterpret_cast<char*>(buffer.data()), actualBytesToRead);
    const U64 actualBytesRead = static_cast<U64>(dataFile.gcount());
    currentOffset += actualBytesRead;
    currentPosition.publish(currentOffset);

    if (actualBytesRead > 0) {
        updateBandwidth(actualBytesRead);
    }

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(FileReaderImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
