#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct FileWriterImplNativeCpu : public FileWriterImpl,
                                 public Runtime::Context,
                                 public Scheduler::Context {
 public:
    Result create() final;

    Result computeSubmit() override;
};

Result FileWriterImplNativeCpu::create() {
    JST_CHECK(FileWriterImpl::create());

    const auto& input = inputs().at("buffer").tensor;

    if (input.dtype() != DataType::CF32 &&
        input.dtype() != DataType::F32 &&
        input.dtype() != DataType::CI8 &&
        input.dtype() != DataType::I16 &&
        input.dtype() != DataType::CI16 &&
        input.dtype() != DataType::I8 &&
        input.dtype() != DataType::CU8 &&
        input.dtype() != DataType::U8 &&
        input.dtype() != DataType::CU16 &&
        input.dtype() != DataType::U16) {
        JST_ERROR("[MODULE_FILE_WRITER_NATIVE_CPU] Unsupported input data type.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result FileWriterImplNativeCpu::computeSubmit() {
    if (!dataFile.is_open() || !recording) {
        return Result::SUCCESS;
    }

    const auto& input = inputs().at("buffer").tensor;
    const U64 bytesToWrite = input.sizeBytes();

    dataFile.write(reinterpret_cast<const char*>(input.data()), bytesToWrite);
    if (!dataFile.good()) {
        JST_ERROR("[MODULE_FILE_WRITER_NATIVE_CPU] Failed writing to '{}'.", filePath.string());
        return Result::ERROR;
    }

    dataFile.flush();
    if (!dataFile.good()) {
        JST_ERROR("[MODULE_FILE_WRITER_NATIVE_CPU] Failed flushing '{}'.", filePath.string());
        return Result::ERROR;
    }

    bytesWritten += bytesToWrite;

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(FileWriterImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
