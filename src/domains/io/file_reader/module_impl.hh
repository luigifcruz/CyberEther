#ifndef JETSTREAM_DOMAINS_IO_FILE_READER_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_IO_FILE_READER_MODULE_IMPL_HH

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <optional>
#include <vector>

#include <jetstream/domains/io/file_reader/module.hh>
#include <jetstream/detail/module_impl.hh>
#include <jetstream/platform.hh>
#include <jetstream/tools/snapshot.hh>

#include "formats/format.hh"

namespace Jetstream::Modules {

struct FileReaderImpl : public Module::Impl, public DynamicConfig<FileReader> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;
    Result destroy() override;
    Result reconfigure() override;

    bool isStream() const;
    U64 getCurrentPosition() const;
    U64 getFileSize() const;
    F32 getThroughput() const;

 protected:
    Result publishCapture(U64 sampleIndex);
    Result createStream();
    Result readStream();
    void streamLoop();
    void updateThroughput(const U64 deltaBytes);

    Tensor buffer;
    DataType validatedDataType = DataType::None;
    U64 validatedOutputSizeBytes = 0;

    FileReaderFormat::Recording recording;
    std::optional<F64> publishedFrequency;
    std::chrono::steady_clock::time_point paceStart{};
    U64 pacedSamples = 0;

    std::atomic<bool> livePlaying{true};
    std::atomic<bool> liveRealTime{false};
    std::atomic<bool> streaming{false};
    std::atomic<bool> receiving{false};
    std::atomic<bool> starved{false};
    std::atomic<bool> ended{false};
    std::atomic<bool> errored{false};
    U64 streamFilled = 0;
    U64 streamTotal = 0;
    std::mutex streamMutex;
    std::condition_variable streamCondition;
    Platform::InputStream stream;
    Platform::WorkerThread streamThread;

    std::ifstream dataFile;
    std::filesystem::path filePath;
    U64 bytesSinceLastMeasurement = 0;
    std::chrono::steady_clock::time_point lastMeasurementTime{};
    Tools::Snapshot<U64> fileSize{0};
    Tools::Snapshot<U64> currentPosition{0};
    Tools::Snapshot<F32> throughputMBs{0.0f};
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_IO_FILE_READER_MODULE_IMPL_HH
