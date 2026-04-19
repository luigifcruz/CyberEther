#include <jetstream/domains/io/file_writer/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/io/file_writer/module.hh>
#include "module_impl.hh"

namespace Jetstream::Blocks {

namespace {

std::string FormatBytes(const U64 bytes) {
    if (bytes < 1024) {
        return std::to_string(bytes) + " B";
    }
    if (bytes < 1024 * 1024) {
        return std::to_string(bytes / 1024) + " KB";
    }
    if (bytes < 1024 * 1024 * 1024) {
        return std::to_string(bytes / (1024 * 1024)) + " MB";
    }
    return std::to_string(bytes / (1024 * 1024 * 1024)) + " GB";
}

}  // namespace

struct FileWriterImpl : public Block::Impl,
                        public DynamicConfig<Blocks::FileWriter> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::FileWriter> moduleConfig =
        std::make_shared<Modules::FileWriter>();
    Modules::FileWriterImpl* moduleImpl = nullptr;
};

Result FileWriterImpl::configure() {
    moduleConfig->filepath = filepath;
    moduleConfig->fileFormat = fileFormat;
    moduleConfig->overwrite = overwrite;
    moduleConfig->recording = recording;

    return Result::SUCCESS;
}

Result FileWriterImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer",
                                    "Input",
                                    "The input buffer containing samples "
                                    "to write to the file."));

    JST_CHECK(defineInterfaceConfig("filepath",
                                    "File Path",
                                    "Path to the output file.",
                                    "filesave:bin,raw,iq,wav,dat"));

    JST_CHECK(defineInterfaceConfig("fileFormat",
                                    "File Format",
                                    "The format of the output file.",
                                    "dropdown:raw(Raw)"));

    JST_CHECK(defineInterfaceConfig("overwrite",
                                    "Overwrite",
                                    "Whether to overwrite the file if it "
                                    "already exists.",
                                    "bool"));

    JST_CHECK(defineInterfaceConfig("recording",
                                    "Recording",
                                    "Start or stop recording to the file.",
                                    "bool"));

    JST_CHECK(defineInterfaceMetric("fileSize",
                                    "File Size",
                                    "Current size of the output file on disk.",
                                    "label",
        [this]() -> std::any {
            if (!moduleImpl) {
                return std::string("0 B");
            }
            return FormatBytes(moduleImpl->getBytesWritten());
        }));

    JST_CHECK(defineInterfaceMetric("currentBandwidth",
                                    "Bandwidth",
                                    "Smoothed recent file write rate.",
                                    "label",
        [this]() -> std::any {
            if (!moduleImpl) {
                return std::string("N/A");
            }
            return jst::fmt::format("{:.1f} MB/s", moduleImpl->getCurrentBandwidth());
        }));

    return Result::SUCCESS;
}

Result FileWriterImpl::create() {
    JST_CHECK(moduleCreate("file_writer", moduleConfig, {
        {"buffer", inputs().at("buffer")}
    }));

    moduleImpl = moduleHandle("file_writer")->getImpl<Modules::FileWriterImpl>();

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(FileWriterImpl);

}  // namespace Jetstream::Blocks
