#ifndef JETSTREAM_DOMAINS_IO_FILE_WRITER_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_IO_FILE_WRITER_MODULE_IMPL_HH

#include <chrono>
#include <filesystem>
#include <fstream>

#include <jetstream/domains/io/file_writer/module.hh>
#include <jetstream/detail/module_impl.hh>
#include <jetstream/tools/snapshot.hh>

namespace Jetstream::Modules {

struct FileWriterImpl : public Module::Impl, public DynamicConfig<FileWriter> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;
    Result destroy() override;

    U64 getBytesWritten() const;
    F32 getCurrentBandwidth() const;

 protected:
    void updateBandwidth(const U64 deltaBytes);

    std::ofstream dataFile;
    std::filesystem::path filePath;
    U64 bytesSinceLastMeasurement = 0;
    std::chrono::steady_clock::time_point lastMeasurementTime{};
    Tools::Snapshot<U64> bytesWritten{0};
    Tools::Snapshot<F32> currentBandwidth{0.0f};
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_IO_FILE_WRITER_MODULE_IMPL_HH
