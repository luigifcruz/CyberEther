#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_SUPPORT_CRASH_REPORT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_SUPPORT_CRASH_REPORT_HH

#include "jetstream/types.hh"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace Jetstream {

class CrashReport {
 public:
    struct Snapshot {
        enum class Status {
            Idle,
            Submitting,
            Success,
            Error,
        };

        Status status = Status::Idle;
        std::string message;
    };

    static constexpr U64 MaxSize = 256 * 1024;
    static constexpr const char* Host = "https://cyberether.org";
    static constexpr const char* Path = "/api/v1/crash";

    CrashReport();
    ~CrashReport();

    CrashReport(const CrashReport&) = delete;
    CrashReport& operator=(const CrashReport&) = delete;

    static std::vector<std::filesystem::path> Pending();
    static std::string Read(const std::filesystem::path& report);
    static void Discard(const std::vector<std::filesystem::path>& reports);

    void submit(const std::vector<std::filesystem::path>& reports);
    Snapshot snapshot() const;
    void clear();
    void shutdown();

 private:
    struct Impl;
    std::shared_ptr<Impl> pimpl;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_SUPPORT_CRASH_REPORT_HH
