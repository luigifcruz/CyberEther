#include <jetstream/domains/io/fbh_reader/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/io/fbh_reader/module.hh>
#include "module_impl.hh"

namespace Jetstream::Blocks {

struct FbhReaderImpl : public Block::Impl, public DynamicConfig<Blocks::FbhReader> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::FbhReader> moduleConfig = std::make_shared<Modules::FbhReader>();
    Modules::FbhReaderImpl* moduleImpl = nullptr;
};

Result FbhReaderImpl::configure() {
    moduleConfig->filepath  = filepath;
    moduleConfig->batchSize = batchSize;
    moduleConfig->loop      = loop;
    moduleConfig->playing   = playing;
    return Result::SUCCESS;
}

Result FbhReaderImpl::define() {
    JST_CHECK(defineInterfaceOutput("signal",
                                    "Output",
                                    "F32 tensor [batchSize, nifs × nchans] containing filterbank samples."));

    JST_CHECK(defineInterfaceConfig("filepath",
                                    "File Path",
                                    "Path to the FBH5 (HDF5) beamformer filterbank file.",
                                    "filepicker:fbh5,h5,hdf5"));

    JST_CHECK(defineInterfaceConfig("batchSize",
                                    "Batch Size",
                                    "Number of time rows to read per processing cycle.",
                                    "int:samples"));

    JST_CHECK(defineInterfaceConfig("loop",
                                    "Loop",
                                    "Restart from the beginning when reaching the end of the file.",
                                    "bool"));

    JST_CHECK(defineInterfaceConfig("playing",
                                    "Playing",
                                    "Pause or resume reading.",
                                    "bool"));

    JST_CHECK(defineInterfaceMetric("progress",
                                    "Position",
                                    "Current position in the file.",
                                    "progressbar",
        [this]() -> std::any {
            if (!moduleImpl) {
                return std::pair<std::string, F32>{"0.0%", 0.0f};
            }
            const U64 total = moduleImpl->getTotalRows();
            if (total == 0) {
                return std::pair<std::string, F32>{"0.0%", 0.0f};
            }
            const F32 progress = static_cast<F32>(moduleImpl->getCurrentRow()) /
                                 static_cast<F32>(total);
            return std::pair<std::string, F32>{jst::fmt::format("{:.1f}%", progress * 100.0f), progress};
        }));

    JST_CHECK(defineInterfaceMetric("currentBandwidth",
                                    "Bandwidth",
                                    "Smoothed HDF5 read throughput.",
                                    "label",
        [this]() -> std::any {
            if (!moduleImpl) {
                return std::string("N/A");
            }
            return jst::fmt::format("{:.1f} MB/s", moduleImpl->getCurrentBandwidth());
        }));

    JST_CHECK(defineInterfaceMetric("fileInfo",
                                    "File Info",
                                    "Shape of the opened dataset.",
                                    "label",
        [this]() -> std::any {
            if (!moduleImpl) {
                return std::string("—");
            }
            const U64 total = moduleImpl->getTotalRows();
            if (total == 0) {
                return std::string("—");
            }
            return jst::fmt::format("{} rows", total);
        }));

    return Result::SUCCESS;
}

Result FbhReaderImpl::create() {
    JST_CHECK(moduleCreate("fbh_reader", moduleConfig, {}));
    JST_CHECK(moduleExposeOutput("signal", {"fbh_reader", "signal"}));

    moduleImpl = moduleHandle("fbh_reader")->getImpl<Modules::FbhReaderImpl>();

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(FbhReaderImpl);

}  // namespace Jetstream::Blocks
