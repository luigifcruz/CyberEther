#include "crash_report.hh"

#include "jetstream/logger.hh"
#include "jetstream/platform.hh"

#include <fstream>
#include <iterator>
#include <mutex>
#include <system_error>
#include <thread>

#include <nlohmann/json.hpp>

namespace Jetstream {

struct CrashReport::Impl {
    mutable std::mutex mutex;
    Snapshot state;
    bool active = true;
    U64 generation = 0;

    void finish(const U64 expected, const Snapshot::Status status, const std::string& message) {
        std::lock_guard<std::mutex> lock(mutex);
        if (expected != generation || !active) {
            return;
        }
        state.status = status;
        state.message = message;
    }
};

CrashReport::CrashReport() : pimpl(std::make_shared<Impl>()) {}

CrashReport::~CrashReport() {
    shutdown();
}

std::vector<std::filesystem::path> CrashReport::Pending() {
    std::vector<std::filesystem::path> reports;
#if defined(JETSTREAM_PLATFORM_CRASH_AVAILABLE)
    if (Platform::PendingCrashReports(reports) != Result::SUCCESS) {
        reports.clear();
    }
#endif
    return reports;
}

std::string CrashReport::Read(const std::filesystem::path& report) {
    std::error_code error;
    const auto size = std::filesystem::file_size(report, error);
    if (error || size == 0 || size > MaxSize) {
        return {};
    }

    std::ifstream file(report, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

void CrashReport::Discard(const std::vector<std::filesystem::path>& reports) {
    for (const auto& report : reports) {
        std::error_code error;
        std::filesystem::remove(report, error);
    }
}

CrashReport::Outcome CrashReport::Judge(const bool failed, const I32 status) {
    if (failed) {
        return Outcome::Retry;
    }
    if (status >= 200 && status < 300) {
        return Outcome::Accepted;
    }
    if (status >= 400 && status < 500 && status != 408 && status != 429) {
        return Outcome::Rejected;
    }
    return Outcome::Retry;
}

void CrashReport::submit(const std::vector<std::filesystem::path>& reports) {
    U64 generation = 0;
    {
        std::lock_guard<std::mutex> lock(pimpl->mutex);
        generation = ++pimpl->generation;
        pimpl->state = {.status = Snapshot::Status::Submitting};
    }

    std::thread([impl = pimpl, reports, generation]() {
        U64 rejected = 0;
        for (const auto& report : reports) {
            const std::string content = Read(report);
            if (content.empty()) {
                Discard({report});
                continue;
            }

            const auto response = Platform::Http::Client::Send({
                .method = "POST",
                .host = Host,
                .path = Path,
                .headers = {{"Content-Type", "application/json"}},
                .body = nlohmann::json{{"report", content}}.dump(
                    -1, ' ', false, nlohmann::json::error_handler_t::replace),
                .timeoutSeconds = 15,
            });
            switch (Judge(static_cast<bool>(response.error), response.status)) {
                case Outcome::Accepted:
                    Discard({report});
                    break;
                case Outcome::Rejected:
                    Discard({report});
                    ++rejected;
                    break;
                case Outcome::Retry:
                    impl->finish(generation, Snapshot::Status::Error,
                                 response.error
                                     ? "Unable to reach the crash report server. CyberEther will ask again next launch."
                                     : jst::fmt::format("The crash report server answered with HTTP {}. "
                                                        "CyberEther will ask again next launch.", response.status));
                    return;
            }
        }

        if (rejected > 0) {
            impl->finish(generation, Snapshot::Status::Error,
                         "The crash report server did not accept the report. It was discarded.");
            return;
        }
        impl->finish(generation, Snapshot::Status::Success, {});
    }).detach();
}

CrashReport::Snapshot CrashReport::snapshot() const {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    return pimpl->state;
}

void CrashReport::clear() {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    ++pimpl->generation;
    pimpl->state = {};
}

void CrashReport::shutdown() {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    pimpl->active = false;
    ++pimpl->generation;
}

}  // namespace Jetstream
