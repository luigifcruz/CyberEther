#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <fstream>

#include "jetstream/testing.hh"
#include "jetstream/registry.hh"
#include "jetstream/domains/io/file_writer/module.hh"

using namespace Jetstream;

namespace {

std::filesystem::path getTestFilePath(const std::string& suffix) {
    auto path = std::filesystem::temp_directory_path() / ("jst_test_file_writer_" + suffix + ".raw");
    return path;
}

void cleanupTestFile(const std::filesystem::path& path) {
    if (std::filesystem::exists(path)) {
        std::filesystem::remove(path);
    }
}

void cleanupTestDirectory(const std::filesystem::path& path) {
    if (std::filesystem::exists(path)) {
        std::filesystem::remove_all(path);
    }
}

}  // namespace

TEST_CASE("FileWriter Module - F32", "[modules][file_writer][F32]") {
    auto implementations = Registry::ListAvailableModules("file_writer");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            auto testPath = getTestFilePath("f32");
            cleanupTestFile(testPath);

            TestContext ctx("file_writer", impl.device,
                           impl.runtime, impl.provider);

            Modules::FileWriter config;
            config.filepath = testPath.string();

            config.overwrite = true;
            config.recording = true;
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({4});
            input.at(0) = 1.0f;
            input.at(1) = 2.0f;
            input.at(2) = 3.0f;
            input.at(3) = 4.0f;

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);
            REQUIRE(std::filesystem::exists(testPath));
            REQUIRE(std::filesystem::file_size(testPath) ==
                    4 * sizeof(F32));

            std::ifstream verify(testPath, std::ios::binary);
            F32 values[4];
            verify.read(reinterpret_cast<char*>(values), sizeof(values));
            REQUIRE(values[0] == 1.0f);
            REQUIRE(values[1] == 2.0f);
            REQUIRE(values[2] == 3.0f);
            REQUIRE(values[3] == 4.0f);

            cleanupTestFile(testPath);
        }
    }
}

TEST_CASE("FileWriter Module - CF32", "[modules][file_writer][CF32]") {
    auto implementations = Registry::ListAvailableModules("file_writer");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            auto testPath = getTestFilePath("cf32");
            cleanupTestFile(testPath);

            TestContext ctx("file_writer", impl.device,
                           impl.runtime, impl.provider);

            Modules::FileWriter config;
            config.filepath = testPath.string();

            config.overwrite = true;
            config.recording = true;
            ctx.setConfig(config);

            auto input = ctx.createTensor<CF32>({2});
            input.at(0) = {1.0f, 2.0f};
            input.at(1) = {3.0f, 4.0f};

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);
            REQUIRE(std::filesystem::exists(testPath));
            REQUIRE(std::filesystem::file_size(testPath) ==
                    2 * sizeof(CF32));

            std::ifstream verify(testPath, std::ios::binary);
            CF32 values[2];
            verify.read(reinterpret_cast<char*>(values), sizeof(values));
            REQUIRE(values[0].real() == 1.0f);
            REQUIRE(values[0].imag() == 2.0f);
            REQUIRE(values[1].real() == 3.0f);
            REQUIRE(values[1].imag() == 4.0f);

            cleanupTestFile(testPath);
        }
    }
}

TEST_CASE("FileWriter Module - No overwrite protection",
          "[modules][file_writer]") {
    auto implementations = Registry::ListAvailableModules("file_writer");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            auto testPath = getTestFilePath("nooverwrite");

            // Create existing file.
            {
                std::ofstream f(testPath, std::ios::binary);
                f << "existing";
            }

            TestContext ctx("file_writer", impl.device,
                           impl.runtime, impl.provider);

            Modules::FileWriter config;
            config.filepath = testPath.string();

            config.overwrite = false;
            config.recording = true;
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({4});
            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::ERROR);

            cleanupTestFile(testPath);
        }
    }
}

TEST_CASE("FileWriter Module - Missing parent directory",
          "[modules][file_writer]") {
    auto implementations = Registry::ListAvailableModules("file_writer");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            const auto parentDir = std::filesystem::temp_directory_path() /
                                   "jst_test_file_writer_missing_parent";
            const auto testPath = parentDir / "missing_parent.raw";

            cleanupTestDirectory(parentDir);

            TestContext ctx("file_writer", impl.device,
                            impl.runtime, impl.provider);

            Modules::FileWriter config;
            config.filepath = testPath.string();
            config.overwrite = true;
            config.recording = true;
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({4});
            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::INCOMPLETE);
            REQUIRE(!std::filesystem::exists(testPath));

            cleanupTestDirectory(parentDir);
        }
    }
}

TEST_CASE("FileWriter Module - Recording disabled",
          "[modules][file_writer]") {
    auto implementations = Registry::ListAvailableModules("file_writer");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            auto testPath = getTestFilePath("disabled");
            cleanupTestFile(testPath);

            TestContext ctx("file_writer", impl.device,
                            impl.runtime, impl.provider);

            Modules::FileWriter config;
            config.filepath = testPath.string();
            config.overwrite = true;
            config.recording = false;
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({4});
            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);
            REQUIRE(!std::filesystem::exists(testPath));
        }
    }
}

TEST_CASE("FileWriter Module - Invalid file format",
          "[modules][file_writer][validation]") {
    auto implementations = Registry::ListAvailableModules("file_writer");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            TestContext ctx("file_writer", impl.device,
                            impl.runtime, impl.provider);

            Modules::FileWriter config;
            config.fileFormat = "wav";
            config.recording = false;
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({4});
            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::ERROR);
        }
    }
}

TEST_CASE("FileWriter Module - Unsupported input dtype",
          "[modules][file_writer][validation]") {
    auto implementations = Registry::ListAvailableModules("file_writer");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            const auto testPath = getTestFilePath("bad_dtype");
            cleanupTestFile(testPath);

            TestContext ctx("file_writer", impl.device,
                            impl.runtime, impl.provider);

            Modules::FileWriter config;
            config.filepath = testPath.string();
            config.overwrite = true;
            config.recording = true;
            ctx.setConfig(config);

            auto input = ctx.createTensor<I32>({4});
            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::ERROR);

            cleanupTestFile(testPath);
        }
    }
}

TEST_CASE("FileWriter Module - Multiple runs overwrite with latest buffer",
          "[modules][file_writer][state]") {
    auto implementations = Registry::ListAvailableModules("file_writer");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            const auto testPath = getTestFilePath("multi_run");
            cleanupTestFile(testPath);

            TestContext ctx("file_writer", impl.device,
                            impl.runtime, impl.provider);

            Modules::FileWriter config;
            config.filepath = testPath.string();
            config.overwrite = true;
            config.recording = true;
            ctx.setConfig(config);

            auto input = ctx.createTensor<U8>({3});
            input.at(0) = 10;
            input.at(1) = 20;
            input.at(2) = 30;
            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);
            REQUIRE(std::filesystem::file_size(testPath) == 3 * sizeof(U8));

            input.at(0) = 40;
            input.at(1) = 50;
            input.at(2) = 60;

            REQUIRE(ctx.run() == Result::SUCCESS);
            REQUIRE(std::filesystem::file_size(testPath) == 3 * sizeof(U8));

            std::ifstream verify(testPath, std::ios::binary);
            U8 values[3] = {};
            verify.read(reinterpret_cast<char*>(values), sizeof(values));
            REQUIRE(values[0] == 40);
            REQUIRE(values[1] == 50);
            REQUIRE(values[2] == 60);

            cleanupTestFile(testPath);
        }
    }
}
