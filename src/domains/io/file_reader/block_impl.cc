#include <jetstream/domains/io/file_reader/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/io/file_reader/module.hh>
#include "module_impl.hh"

namespace Jetstream::Blocks {

namespace {

std::string FormatBytes(const U64 bytes) {
    if (bytes < 1024) {
        return std::to_string(bytes) + " B";
    }
    if (bytes < 1024 * 1024) {
        return std::to_string(bytes / 1024) + " KiB";
    }
    if (bytes < 1024 * 1024 * 1024) {
        return std::to_string(bytes / (1024 * 1024)) + " MiB";
    }
    return std::to_string(bytes / (1024 * 1024 * 1024)) + " GiB";
}

}  // namespace

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
    moduleConfig->realTime = realTime;

    return Result::SUCCESS;
}

Result FileReaderImpl::define() {
    const auto& config = *candidate();
    JST_CHECK(defineInterfaceOutput("signal",
                                    "Output",
                                    "The output buffer containing samples read from the file."));

    JST_CHECK(defineInterfaceConfig("filepath",
                                    "File Path",
                                    "Path to the file or stream to read, such as /dev/stdin. "
                                    "SigMF takes the .sigmf-meta or .sigmf-data file.",
                                    {{"type", "filepicker"}, {"extensions", Parser::MakeSequence({"bin", "raw", "iq", "wav", "dat", "sigmf-meta", "sigmf-data"})}}));

    Parser::Sequence formats;
    for (const auto& option : Modules::FileReaderFormat::Options()) {
        formats.emplace_back(Parser::Map{{"label", option.label}, {"value", option.value}});
    }
    JST_CHECK(defineInterfaceConfig("fileFormat",
                                    "File Format",
                                    "The format of the input file.",
                                    {{"type", "dropdown"}, {"options", std::move(formats)}}));

    const auto format = Modules::FileReaderFormat::Create(config.fileFormat);
    const bool describesSamples = format && format->describesSamples();

    if (!describesSamples) {
        Parser::Sequence dataTypes;
        for (const std::string type : {"CF64", "F64", "CF32", "F32", "CI8", "I8", "CU8", "U8", "CI16", "I16", "CU16", "U16"}) {
            dataTypes.emplace_back(Parser::Map{{"label", type}, {"value", type}});
        }
        JST_CHECK(defineInterfaceConfig("dataType",
                                        "Data Type",
                                        "The data type of samples in the file.",
                                        {{"type", "dropdown"}, {"options", std::move(dataTypes)}}));
    }

    JST_CHECK(defineInterfaceConfig("batchSize",
                                    "Samples",
                                    "Samples read per cycle and the FFT size downstream.",
                                    {{"type", "uint"}, {"unit", "samples"}}));

    JST_CHECK(defineInterfaceConfig("loop",
                                    "Loop",
                                    "Whether to loop back to the start when reaching the end of the file.",
                                    {{"type", "bool"}}));

    JST_CHECK(defineInterfaceConfig("playing",
                                    "Playing",
                                    "Start or stop reading from the file.",
                                    {{"type", "bool"}}));

    if (describesSamples) {
        JST_CHECK(defineInterfaceConfig("realTime",
                                        "Real Time",
                                        "Read at the recorded sample rate instead of as fast as the chain runs.",
                                        {{"type", "bool"}}));
    }

    JST_CHECK(defineInterfaceMetric("progress",
                                    "Position",
                                    "Current file position, or the data read so far from a stream.",
                                    {{"type", "progressbar"}},
        [this]() -> std::any {
            if (!moduleImpl) {
                return std::pair<std::string, F32>{"0.0%", 0.0f};
            }
            if (moduleImpl->isStream()) {
                return std::pair<std::string, F32>{FormatBytes(moduleImpl->getCurrentPosition()), 0.0f};
            }
            const U64 size = moduleImpl->getFileSize();
            if (size == 0) {
                return std::pair<std::string, F32>{"0.0%", 0.0f};
            }
            const F32 progress = static_cast<F32>(moduleImpl->getCurrentPosition()) /
                                 static_cast<F32>(size);
            return std::pair<std::string, F32>{jst::fmt::format("{:.1f}%", progress * 100.0f), progress};
        }));

    JST_CHECK(defineInterfaceMetric("throughput",
                                    "Throughput",
                                    "Smoothed recent read rate.",
                                    {{"type", "label"}},
        [this]() -> std::any {
            if (!moduleImpl) {
                return std::string("N/A");
            }
            return jst::fmt::format("{:.1f} MB/s", moduleImpl->getThroughput());
        }));

    return Result::SUCCESS;
}

Result FileReaderImpl::create() {
    JST_CHECK(moduleCreate("file_reader", moduleConfig, {}));
    JST_CHECK(moduleExposeOutput("signal", {"file_reader", "signal"}));

    moduleImpl = moduleHandle("file_reader")->getImpl<Modules::FileReaderImpl>();

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(FileReaderImpl, {"file_reader"});

}  // namespace Jetstream::Blocks
