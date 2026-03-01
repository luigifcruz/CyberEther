#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "jetstream/domains/io/file_reader/module.hh"
#include "jetstream/registry.hh"
#include "jetstream/testing.hh"

using namespace Jetstream;

namespace {

std::filesystem::path TestFilePath(const std::string& suffix) {
    auto path = std::filesystem::temp_directory_path() /
                ("jst_test_file_reader_" + suffix + ".raw");
    return path;
}

void Cleanup(const std::filesystem::path& path) {
    if (std::filesystem::exists(path)) {
        std::filesystem::remove(path);
    }
}

}  // namespace

TEST_CASE("FileReader module reads the configured first F32 batch",
          "[modules][file_reader][f32]") {
    auto implementations = Registry::ListAvailableModules("file_reader");
    REQUIRE(!implementations.empty());

    const auto path = TestFilePath("f32_order");
    Cleanup(path);

    {
        std::ofstream file(path, std::ios::binary);
        const std::vector<F32> data = {1.5f, -2.0f, 3.0f, 4.25f};
        file.write(reinterpret_cast<const char*>(data.data()),
                   static_cast<std::streamsize>(data.size() * sizeof(F32)));
    }

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            TestContext ctx("file_reader", impl.device, impl.runtime,
                            impl.provider);

            Modules::FileReader config;
            config.filepath = path.string();
            config.dataType = "F32";
            config.batchSize = 2;
            config.loop = false;
            config.playing = true;
            ctx.setConfig(config);

            REQUIRE(ctx.run() == Result::SUCCESS);
            auto& out = ctx.output("signal");
            REQUIRE(out.dtype() == DataType::F32);
            REQUIRE(out.rank() == 1);
            REQUIRE(out.shape(0) == 2);
            REQUIRE(out.at<F32>(0) == 1.5f);
            REQUIRE(out.at<F32>(1) == -2.0f);

            REQUIRE(ctx.run() == Result::SUCCESS);
            REQUIRE(out.at<F32>(0) == 1.5f);
            REQUIRE(out.at<F32>(1) == -2.0f);
        }
    }

    Cleanup(path);
}

TEST_CASE("FileReader module validation rejects invalid config",
          "[modules][file_reader][validation]") {
    auto implementations = Registry::ListAvailableModules("file_reader");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        SECTION("invalid file format") {
            TestContext ctx("file_reader", impl.device, impl.runtime,
                            impl.provider);
            Modules::FileReader config;
            config.fileFormat = "wav";
            ctx.setConfig(config);
            REQUIRE(ctx.run() == Result::ERROR);
        }

        SECTION("invalid data type") {
            TestContext ctx("file_reader", impl.device, impl.runtime,
                            impl.provider);
            Modules::FileReader config;
            config.dataType = "I32";
            ctx.setConfig(config);
            REQUIRE(ctx.run() == Result::ERROR);
        }

        SECTION("zero batch size") {
            TestContext ctx("file_reader", impl.device, impl.runtime,
                            impl.provider);
            Modules::FileReader config;
            config.batchSize = 0;
            ctx.setConfig(config);
            REQUIRE(ctx.run() == Result::ERROR);
        }
    }
}

TEST_CASE("FileReader module reports incomplete when file is missing",
          "[modules][file_reader][errors]") {
    auto implementations = Registry::ListAvailableModules("file_reader");
    REQUIRE(!implementations.empty());

    const auto missing = TestFilePath("missing");
    Cleanup(missing);

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            TestContext ctx("file_reader", impl.device, impl.runtime,
                            impl.provider);
            Modules::FileReader config;
            config.filepath = missing.string();
            config.dataType = "CF32";
            config.batchSize = 4;
            ctx.setConfig(config);

            REQUIRE(ctx.run() == Result::INCOMPLETE);
        }
    }
}

TEST_CASE("FileReader module loop wraps at end-of-file",
          "[modules][file_reader][state]") {
    auto implementations = Registry::ListAvailableModules("file_reader");
    REQUIRE(!implementations.empty());

    const auto path = TestFilePath("loop");
    Cleanup(path);

    {
        std::ofstream file(path, std::ios::binary);
        const std::vector<F32> data = {7.0f, 9.0f};
        file.write(reinterpret_cast<const char*>(data.data()),
                   static_cast<std::streamsize>(data.size() * sizeof(F32)));
    }

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            TestContext ctx("file_reader", impl.device, impl.runtime,
                            impl.provider);
            Modules::FileReader config;
            config.filepath = path.string();
            config.dataType = "F32";
            config.batchSize = 2;
            config.loop = true;
            config.playing = true;
            ctx.setConfig(config);

            REQUIRE(ctx.run() == Result::SUCCESS);
            auto& out = ctx.output("signal");
            REQUIRE(out.at<F32>(0) == 7.0f);
            REQUIRE(out.at<F32>(1) == 9.0f);

            REQUIRE(ctx.run() == Result::SUCCESS);
            REQUIRE(out.at<F32>(0) == 7.0f);
            REQUIRE(out.at<F32>(1) == 9.0f);
        }
    }

    Cleanup(path);
}

TEST_CASE("FileReader module playing=false is a no-op success",
          "[modules][file_reader][state]") {
    auto implementations = Registry::ListAvailableModules("file_reader");
    REQUIRE(!implementations.empty());

    const auto path = TestFilePath("paused");
    Cleanup(path);

    {
        std::ofstream file(path, std::ios::binary);
        const std::vector<F32> data = {1.0f, 2.0f, 3.0f, 4.0f};
        file.write(reinterpret_cast<const char*>(data.data()),
                   static_cast<std::streamsize>(data.size() * sizeof(F32)));
    }

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            TestContext ctx("file_reader", impl.device, impl.runtime,
                            impl.provider);
            Modules::FileReader config;
            config.filepath = path.string();
            config.dataType = "F32";
            config.batchSize = 2;
            config.loop = false;
            config.playing = false;
            ctx.setConfig(config);

            REQUIRE(ctx.run() == Result::SUCCESS);
            auto& out = ctx.output("signal");
            REQUIRE(out.dtype() == DataType::F32);
            REQUIRE(out.shape(0) == 2);
        }
    }

    Cleanup(path);
}
