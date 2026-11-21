#include <catch2/catch_test_macros.hpp>

#include "compositor/default/actions/file_picker.hh"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <optional>
#include <string>

using namespace Jetstream;

namespace {

struct RemotePicker {
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    FilePickerActions actions{state, callbacks};
    std::filesystem::path folder;

    RemotePicker() {
        state.remote.started = true;
        state.remote.clientCount = 1;
        const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
        folder = std::filesystem::current_path() / ("cyberether-file-picker-" + std::to_string(nonce));
        std::filesystem::create_directories(folder);
        std::ofstream(folder / "chosen.yml") << "title: Chosen\n";
    }

    ~RemotePicker() {
        std::error_code error;
        std::filesystem::remove_all(folder, error);
    }

    void select(Sakura::Callback<void(std::string)> callback) {
        REQUIRE(actions.request({
                    .initialPath = Platform::PathToUtf8(folder),
                    .extensions = {"yml"},
                    .callback = std::move(callback),
                }) == Result::INCOMPLETE);
        const auto& entries = state.filePicker.entries;
        const auto entry = std::find_if(entries.begin(), entries.end(), [](const auto& item) {
            return item.name == "chosen.yml";
        });
        REQUIRE(entry != entries.end());
        actions.handle(MailFilePickerSelect{.generation = state.filePicker.generation, .path = entry->path});
    }
};

}  // namespace

TEST_CASE("Releasing a callback keeps an empty callback empty", "[core][compositor][file-picker]") {
    Sakura::Callback<void(Result)> empty;
    CHECK_FALSE(static_cast<bool>(std::move(empty).release()));

    bool called = false;
    Sakura::Callback<void(Result)> set = [&](Result) {
        called = true;
    };
    auto released = std::move(set).release();
    REQUIRE(static_cast<bool>(released));
    released(Result::SUCCESS);
    CHECK(called);
}

TEST_CASE("Remote file pickers confirm without a completion callback", "[core][compositor][file-picker]") {
    RemotePicker picker;
    picker.select({});
    CHECK_NOTHROW(picker.actions.handle(MailFilePickerConfirm{.generation = picker.state.filePicker.generation}));
    CHECK_FALSE(picker.state.filePicker.active);
}

TEST_CASE("Remote file pickers hand the confirmed path to the completion callback",
          "[core][compositor][file-picker]") {
    RemotePicker picker;
    std::optional<std::string> chosen;
    picker.select([&](std::string path) {
        chosen = std::move(path);
    });
    picker.actions.handle(MailFilePickerConfirm{.generation = picker.state.filePicker.generation});
    REQUIRE(chosen.has_value());
    CHECK(std::filesystem::equivalent(Platform::PathFromUtf8(*chosen), picker.folder / "chosen.yml"));
    CHECK_FALSE(picker.state.filePicker.active);
}
