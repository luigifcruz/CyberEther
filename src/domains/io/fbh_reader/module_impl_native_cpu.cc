#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct FbhReaderImplNativeCpu : public FbhReaderImpl,
                                public NativeCpuRuntimeContext,
                                public Scheduler::Context {
 public:
    Result create() final;
    Result computeSubmit() override;
};

Result FbhReaderImplNativeCpu::create() {
    JST_CHECK(FbhReaderImpl::create());
    return Result::SUCCESS;
}

Result FbhReaderImplNativeCpu::computeSubmit() {
    if (fileId < 0 || !playing) {
        return Result::SUCCESS;
    }

    const U64 numChannels = static_cast<U64>(dimNifs) * static_cast<U64>(dimNchans);
    const U64 totalRows   = static_cast<U64>(dimNtimes);

    U64 rowsRemaining = totalRows - currentRow;
    if (rowsRemaining == 0) {
        if (loop) {
            currentRow    = 0;
            rowsRemaining = totalRows;
        } else {
            return Result::SUCCESS;
        }
    }

    const U64 rowsToRead = std::min(static_cast<U64>(batchSize), rowsRemaining);

    // Select hyperslab: [currentRow, 0, 0] → [rowsToRead, dimNifs, dimNchans]
    const hsize_t start[3]  = {static_cast<hsize_t>(currentRow), 0, 0};
    const hsize_t count[3]  = {static_cast<hsize_t>(rowsToRead),
                                static_cast<hsize_t>(dimNifs),
                                static_cast<hsize_t>(dimNchans)};

    if (H5Sselect_hyperslab(dataspaceId, H5S_SELECT_SET,
                             start, nullptr, count, nullptr) < 0) {
        JST_ERROR("[MODULE_FBH_READER] H5Sselect_hyperslab failed.");
        return Result::ERROR;
    }

    // Memory dataspace: flat [rowsToRead * numChannels]
    const hsize_t memDims[1] = {static_cast<hsize_t>(rowsToRead * numChannels)};
    hid_t memspace = H5Screate_simple(1, memDims, nullptr);
    if (memspace < 0) {
        JST_ERROR("[MODULE_FBH_READER] H5Screate_simple failed.");
        return Result::ERROR;
    }

    herr_t status = H5Dread(datasetId, H5T_NATIVE_FLOAT, memspace,
                             dataspaceId, H5P_DEFAULT,
                             buffer.data<F32>());
    H5Sclose(memspace);

    if (status < 0) {
        JST_ERROR("[MODULE_FBH_READER] H5Dread failed at row {}.", currentRow);
        return Result::ERROR;
    }

    currentRow += rowsToRead;
    snapshotCurrentRow.publish(currentRow);

    const U64 bytesRead = rowsToRead * numChannels * sizeof(F32);
    updateBandwidth(bytesRead);

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(FbhReaderImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
