#include <catch2/catch_test_macros.hpp>

#include "compositor/default/actions/crash_report.hh"

#include <atomic>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace Jetstream;

namespace {

struct Reports {
    std::filesystem::path directory;
    std::vector<std::filesystem::path> paths;

    explicit Reports(const U64 count) {
        static std::atomic<U64> sequence = 0;
        directory = std::filesystem::temp_directory_path() /
                    ("cyberether-crash-report-" + std::to_string(sequence.fetch_add(1)));
        std::filesystem::create_directories(directory);
        for (U64 i = 0; i < count; ++i) {
            paths.push_back(directory / (std::to_string(i) + ".crash"));
            std::ofstream(paths.back()) << "CyberEther Crash Report 1\nsignal " << i << "\n";
        }
    }

    ~Reports() {
        std::error_code error;
        std::filesystem::remove_all(directory, error);
    }

    U64 remaining() const {
        U64 count = 0;
        for (const auto& path : paths) {
            count += std::filesystem::exists(path) ? 1 : 0;
        }
        return count;
    }
};

struct Rig {
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    std::vector<std::vector<std::filesystem::path>> submitted;
    CrashReportActions actions{state, callbacks};

    Rig() {
        callbacks.submitCrashReports = [this](const std::vector<std::filesystem::path>& reports) {
            submitted.push_back(reports);
        };
    }
};

}  // namespace

TEST_CASE("Crash report stays hidden outside official packages and when no report is pending",
          "[core][compositor][crash_report]") {
    CHECK_FALSE(CrashReportActions::Detect(false).has_value());
    CHECK_FALSE(CrashReportActions::Detect(true).has_value());
}

TEST_CASE("Crash report reads a saved report and rejects missing or oversized ones",
          "[core][compositor][crash_report]") {
    const Reports reports(1);
    CHECK(CrashReport::Read(reports.paths.front()) == "CyberEther Crash Report 1\nsignal 0\n");
    CHECK(CrashReport::Read(reports.directory / "missing.crash").empty());

    const auto oversized = reports.directory / "oversized.crash";
    std::ofstream(oversized) << std::string(CrashReport::MaxSize + 1, 'x');
    CHECK(CrashReport::Read(oversized).empty());
}

TEST_CASE("Crash report uploads retry on transport or server trouble and drop permanent rejections",
          "[core][compositor][crash_report]") {
    using Outcome = CrashReport::Outcome;
    CHECK(CrashReport::Judge(true, 0) == Outcome::Retry);
    CHECK(CrashReport::Judge(true, 200) == Outcome::Retry);
    CHECK(CrashReport::Judge(false, 200) == Outcome::Accepted);
    CHECK(CrashReport::Judge(false, 202) == Outcome::Accepted);
    CHECK(CrashReport::Judge(false, 400) == Outcome::Rejected);
    CHECK(CrashReport::Judge(false, 404) == Outcome::Rejected);
    CHECK(CrashReport::Judge(false, 413) == Outcome::Rejected);
    CHECK(CrashReport::Judge(false, 408) == Outcome::Retry);
    CHECK(CrashReport::Judge(false, 429) == Outcome::Retry);
    CHECK(CrashReport::Judge(false, 500) == Outcome::Retry);
    CHECK(CrashReport::Judge(false, 301) == Outcome::Retry);
}

TEST_CASE("Opening a crash report shows the modal with the pending reports", "[core][compositor][crash_report]") {
    const Reports reports(2);
    Rig rig;
    rig.state.modal.flowgraph = "a";

    REQUIRE(rig.actions.handle(MailOpenCrashReport{.reports = reports.paths, .preview = "preview"}) ==
            Result::SUCCESS);
    CHECK(rig.state.modal.content == std::optional<ModalContent>(ModalContent::CrashReport));
    CHECK_FALSE(rig.state.modal.flowgraph.has_value());
    CHECK(rig.state.crashReport.reports == reports.paths);
    CHECK(rig.state.crashReport.preview == "preview");
    CHECK_FALSE(rig.state.crashReport.submitting);
}

TEST_CASE("Submitting a crash report hands the reports over once and closes the modal",
          "[core][compositor][crash_report]") {
    const Reports reports(2);
    Rig rig;

    REQUIRE(rig.actions.handle(MailSubmitCrashReport{}) == Result::SUCCESS);
    CHECK(rig.submitted.empty());

    REQUIRE(rig.actions.handle(MailOpenCrashReport{.reports = reports.paths}) == Result::SUCCESS);
    REQUIRE(rig.actions.handle(MailSubmitCrashReport{}) == Result::SUCCESS);
    REQUIRE(rig.actions.handle(MailSubmitCrashReport{}) == Result::SUCCESS);
    REQUIRE(rig.submitted.size() == 1);
    CHECK(rig.submitted.front() == reports.paths);
    CHECK(rig.state.crashReport.submitting);
    CHECK_FALSE(rig.state.modal.content.has_value());
    CHECK(reports.remaining() == 2);
}

TEST_CASE("Dismissing a crash report deletes the pending reports and closes the modal",
          "[core][compositor][crash_report]") {
    const Reports reports(2);
    Rig rig;

    REQUIRE(rig.actions.handle(MailOpenCrashReport{.reports = reports.paths}) == Result::SUCCESS);
    REQUIRE(rig.actions.handle(MailDismissCrashReport{}) == Result::SUCCESS);
    CHECK_FALSE(rig.state.modal.content.has_value());
    CHECK(rig.state.crashReport.reports.empty());
    CHECK(reports.remaining() == 0);
    CHECK(rig.submitted.empty());
}

TEST_CASE("Dismissing a crash report keeps reports that are being sent", "[core][compositor][crash_report]") {
    const Reports reports(1);
    Rig rig;

    REQUIRE(rig.actions.handle(MailOpenCrashReport{.reports = reports.paths}) == Result::SUCCESS);
    REQUIRE(rig.actions.handle(MailSubmitCrashReport{}) == Result::SUCCESS);
    REQUIRE(rig.actions.handle(MailDismissCrashReport{}) == Result::SUCCESS);
    CHECK(rig.state.crashReport.submitting);
    CHECK(reports.remaining() == 1);
}

TEST_CASE("Dismissing a crash report leaves a modal that replaced it open", "[core][compositor][crash_report]") {
    const Reports reports(1);
    Rig rig;

    REQUIRE(rig.actions.handle(MailOpenCrashReport{.reports = reports.paths}) == Result::SUCCESS);
    rig.state.modal.content = ModalContent::Feedback;
    REQUIRE(rig.actions.handle(MailDismissCrashReport{}) == Result::SUCCESS);
    CHECK(rig.state.modal.content == std::optional<ModalContent>(ModalContent::Feedback));
    CHECK(reports.remaining() == 0);
}
