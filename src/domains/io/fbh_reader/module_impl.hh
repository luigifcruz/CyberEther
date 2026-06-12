#ifndef JETSTREAM_DOMAINS_IO_FBH_READER_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_IO_FBH_READER_MODULE_IMPL_HH

#include <chrono>
#include <hdf5.h>

#include <jetstream/domains/io/fbh_reader/module.hh>
#include <jetstream/detail/module_impl.hh>
#include <jetstream/tools/snapshot.hh>

namespace Jetstream::Modules {

struct FbhReaderImpl : public Module::Impl, public DynamicConfig<FbhReader> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;
    Result destroy() override;
    Result reconfigure() override;

    U64 getCurrentRow() const;
    U64 getTotalRows() const;
    F32 getCurrentBandwidth() const;

 protected:
    void updateBandwidth(const U64 deltaBytes);

    Tensor buffer;

    hid_t fileId    = H5I_INVALID_HID;
    hid_t datasetId = H5I_INVALID_HID;
    hid_t dataspaceId = H5I_INVALID_HID;

    // dataset dims: [ntimes, nifs, nchans]
    hsize_t dimNtimes = 0;
    hsize_t dimNifs   = 0;
    hsize_t dimNchans = 0;

    U64 currentRow = 0;

    U64 bytesSinceLastMeasurement = 0;
    std::chrono::steady_clock::time_point lastMeasurementTime{};

    Tools::Snapshot<U64> snapshotTotalRows{0};
    Tools::Snapshot<U64> snapshotCurrentRow{0};
    Tools::Snapshot<F32> snapshotBandwidth{0.0f};
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_IO_FBH_READER_MODULE_IMPL_HH
