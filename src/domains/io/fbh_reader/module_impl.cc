#include "module_impl.hh"

namespace Jetstream::Modules {

Result FbhReaderImpl::validate() {
    const auto& config = *candidate();

    if (config.batchSize == 0) {
        JST_ERROR("[MODULE_FBH_READER] Batch size cannot be zero.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result FbhReaderImpl::define() {
    JST_CHECK(defineInterfaceOutput("signal"));
    return Result::SUCCESS;
}

Result FbhReaderImpl::create() {
    snapshotTotalRows.publish(0);
    snapshotCurrentRow.publish(0);
    snapshotBandwidth.publish(0.0f);
    bytesSinceLastMeasurement = 0;
    lastMeasurementTime = std::chrono::steady_clock::now();

    if (filepath.empty()) {
        JST_WARN("[MODULE_FBH_READER] File path is empty.");
        return Result::INCOMPLETE;
    }

    fileId = H5Fopen(filepath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (fileId < 0) {
        JST_WARN("[MODULE_FBH_READER] Cannot open HDF5 file '{}'.", filepath);
        return Result::INCOMPLETE;
    }

    datasetId = H5Dopen2(fileId, "/data", H5P_DEFAULT);
    if (datasetId < 0) {
        H5Fclose(fileId);
        fileId = H5I_INVALID_HID;
        JST_WARN("[MODULE_FBH_READER] Dataset '/data' not found in '{}'.", filepath);
        return Result::INCOMPLETE;
    }

    dataspaceId = H5Dget_space(datasetId);
    if (dataspaceId < 0) {
        H5Dclose(datasetId);
        H5Fclose(fileId);
        datasetId = H5I_INVALID_HID;
        fileId    = H5I_INVALID_HID;
        JST_ERROR("[MODULE_FBH_READER] Failed to get dataspace for '/data'.");
        return Result::ERROR;
    }

    const int ndims = H5Sget_simple_extent_ndims(dataspaceId);
    if (ndims != 3) {
        H5Sclose(dataspaceId);
        H5Dclose(datasetId);
        H5Fclose(fileId);
        dataspaceId = H5I_INVALID_HID;
        datasetId   = H5I_INVALID_HID;
        fileId      = H5I_INVALID_HID;
        JST_ERROR("[MODULE_FBH_READER] Expected 3-D dataset [ntimes, nifs, nchans], got {}.", ndims);
        return Result::ERROR;
    }

    hsize_t dims[3];
    H5Sget_simple_extent_dims(dataspaceId, dims, nullptr);
    dimNtimes = dims[0];
    dimNifs   = dims[1];
    dimNchans = dims[2];

    currentRow = 0;
    snapshotTotalRows.publish(static_cast<U64>(dimNtimes));
    snapshotCurrentRow.publish(0);

    const U64 numChannels = static_cast<U64>(dimNifs) * static_cast<U64>(dimNchans);
    JST_CHECK(buffer.create(device(), DataType::F32, {batchSize, numChannels}));
    outputs()["signal"].produced(name(), "signal", buffer);

    JST_INFO("[MODULE_FBH_READER] Opened '{}': ntimes={} nifs={} nchans={}.",
             filepath, dimNtimes, dimNifs, dimNchans);

    return Result::SUCCESS;
}

Result FbhReaderImpl::destroy() {
    if (dataspaceId >= 0) { H5Sclose(dataspaceId); dataspaceId = H5I_INVALID_HID; }
    if (datasetId >= 0)   { H5Dclose(datasetId);   datasetId   = H5I_INVALID_HID; }
    if (fileId >= 0)      { H5Fclose(fileId);       fileId      = H5I_INVALID_HID; }

    snapshotBandwidth.publish(0.0f);
    bytesSinceLastMeasurement = 0;

    return Result::SUCCESS;
}

Result FbhReaderImpl::reconfigure() {
    const auto& config = *candidate();

    if (config.filepath  != filepath  ||
        config.batchSize != batchSize) {
        return Result::RECREATE;
    }

    loop    = config.loop;
    playing = config.playing;
    return Result::SUCCESS;
}

U64 FbhReaderImpl::getCurrentRow() const {
    return snapshotCurrentRow.get();
}

U64 FbhReaderImpl::getTotalRows() const {
    return snapshotTotalRows.get();
}

F32 FbhReaderImpl::getCurrentBandwidth() const {
    return snapshotBandwidth.get();
}

void FbhReaderImpl::updateBandwidth(const U64 deltaBytes) {
    constexpr double kPeriodSeconds = 0.10;
    constexpr double kEmaAlpha      = 0.3;

    bytesSinceLastMeasurement += deltaBytes;

    const auto now = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(now - lastMeasurementTime).count();
    if (elapsed < kPeriodSeconds) {
        return;
    }

    const double instant = static_cast<double>(bytesSinceLastMeasurement) /
                           static_cast<double>(JST_MB) / elapsed;
    const double smoothed = kEmaAlpha * instant +
                            (1.0 - kEmaAlpha) * snapshotBandwidth.get();
    snapshotBandwidth.publish(static_cast<F32>(smoothed));

    bytesSinceLastMeasurement = 0;
    lastMeasurementTime = now;
}

}  // namespace Jetstream::Modules
