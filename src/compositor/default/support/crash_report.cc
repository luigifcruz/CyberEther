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
#if defined(JST_OS_MAC)
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

void CrashReport::submit(const std::vector<std::filesystem::path>& reports) {
    U64 generation = 0;
    {
        std::lock_guard<std::mutex> lock(pimpl->mutex);
        generation = ++pimpl->generation;
        pimpl->state = {.status = Snapshot::Status::Submitting};
    }

    std::thread([impl = pimpl, reports, generation]() {
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
            if (response.error) {
                impl->finish(generation, Snapshot::Status::Error,
                             "Unable to reach the crash report server. CyberEther will ask again next launch.");
                return;
            }
            if (response.status < 200 || response.status >= 300) {
                impl->finish(generation, Snapshot::Status::Error,
                             jst::fmt::format("The crash report server answered with HTTP {}. "
                                              "CyberEther will ask again next launch.", response.status));
                return;
            }

            Discard({report});
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
