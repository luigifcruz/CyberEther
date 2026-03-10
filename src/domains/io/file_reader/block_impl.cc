#include <jetstream/domains/io/file_reader/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/io/file_reader/module.hh>
#include "module_impl.hh"

namespace Jetstream::Blocks {

struct FileReaderImpl : public Block::Impl, public DynamicConfig<Blocks::FileReader> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::FileReader> moduleConfig = std::make_shared<Modules::FileReader>();
    Modules::FileReaderImpl* moduleImpl = nullptr;
};

Result FileReaderImpl::configure() {
    moduleConfig->filepath = filepath;
    moduleConfig->fileFormat = fileFormat;
    moduleConfig->dataType = dataType;
    moduleConfig->batchSize = batchSize;
    moduleConfig->loop = loop;
    moduleConfig->playing = playing;

    return Result::SUCCESS;
}

Result FileReaderImpl::define() {
    JST_CHECK(defineInterfaceOutput("signal",
                                    "Output",
                                    "The output buffer containing samples read from the file."));

    JST_CHECK(defineInterfaceConfig("filepath",
                                    "File Path",
                                    "Path to the raw binary file to read.",
                                    "filepicker:bin,raw,iq,wav,dat"));

    JST_CHECK(defineInterfaceConfig("fileFormat",
                                    "File Format",
                                    "The format of the input file.",
                                    "dropdown:raw(Raw)"));

    JST_CHECK(defineInterfaceConfig("dataType",
                                    "Data Type",
                                    "The data type of samples in the file.",
                                    "dropdown:CF32(CF32),F32(F32),CI8(CI8),I8(I8),CU8(CU8),U8(U8),CI16(CI16),I16(I16),CU16(CU16),U16(U16)"));

    JST_CHECK(defineInterfaceConfig("batchSize",
                                    "Batch Size",
                                    "Number of samples to read per processing cycle.",
                                    "int:samples"));

    JST_CHECK(defineInterfaceConfig("loop",
                                    "Loop",
                                    "Whether to loop back to the start when reaching the end of the file.",
                                    "bool"));

    JST_CHECK(defineInterfaceConfig("playing",
                                    "Playing",
                                    "Start or stop reading from the file.",
                                    "bool"));

    JST_CHECK(defineInterfaceMetric("progress",
                                    "Position",
                                    "Current file position.",
                                    "progressbar",
        [this]() -> std::any {
            if (!moduleImpl) {
                return F32(0.0f);
            }
            const U64& size = moduleImpl->getFileSize();
            if (size == 0) {
                return F32(0.0f);
            }
            return static_cast<F32>(moduleImpl->getCurrentPosition()) /
                   static_cast<F32>(size);
        }));

    return Result::SUCCESS;
}

Result FileReaderImpl::create() {
    JST_CHECK(moduleCreate("file_reader", moduleConfig, {}));
    JST_CHECK(moduleExposeOutput("signal", {"file_reader", "signal"}));

    moduleImpl = moduleHandle("file_reader")->getImpl<Modules::FileReaderImpl>();

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(FileReaderImpl);

}  // namespace Jetstream::Blocks
