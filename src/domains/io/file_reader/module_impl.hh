#ifndef JETSTREAM_DOMAINS_IO_FILE_READER_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_IO_FILE_READER_MODULE_IMPL_HH

#include <chrono>
#include <filesystem>
#include <fstream>

#include <jetstream/domains/io/file_reader/module.hh>
#include <jetstream/detail/module_impl.hh>
#include <jetstream/tools/snapshot.hh>

namespace Jetstream::Modules {

struct FileReaderImpl : public Module::Impl, public DynamicConfig<FileReader> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;
    Result destroy() override;
    Result reconfigure() override;

    U64 getCurrentPosition() const;
    U64 getFileSize() const;
    F32 getCurrentBandwidth() const;

 protected:
    void updateBandwidth(const U64 deltaBytes);

    Tensor buffer;

    std::ifstream dataFile;
    std::filesystem::path filePath;
    U64 bytesSinceLastMeasurement = 0;
    std::chrono::steady_clock::time_point lastMeasurementTime{};
    Tools::Snapshot<U64> fileSize{0};
    Tools::Snapshot<U64> currentPosition{0};
    Tools::Snapshot<F32> currentBandwidth{0.0f};
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_IO_FILE_READER_MODULE_IMPL_HH
