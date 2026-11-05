#include <catch2/catch_test_macros.hpp>
#include <filesystem>

#include "jetstream/platform.hh"
#include <fstream>

#include "jetstream/testing.hh"
#include "jetstream/registry.hh"
#include "jetstream/domains/io/file_writer/module.hh"

#include <nlohmann/json.hpp>

#include "formats/format.hh"

using namespace Jetstream;

namespace {

std::filesystem::path getTestFilePath(const std::string& suffix) {
    auto path = std::filesystem::temp_directory_path() /
                Platform::PathFromUtf8("jst_test_file_writer_" + suffix + ".raw");
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

void requireFileWriterDtypeValidationError(const Registry::ModuleRegistration& impl,
                                           const std::filesystem::path& path) {
    Tensor input;
    REQUIRE(input.create(impl.device, DataType::I32, {4}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("test", "buffer");
    inputs["buffer"].tensor = input;

    Modules::FileWriter config;
    config.filepath = Platform::PathToUtf8(path);
    config.overwrite = true;
    config.recording = true;

    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("file_writer", impl.device, impl.runtime,
                                  impl.provider, module) == Result::SUCCESS);
    REQUIRE(module->create("test", config, inputs) == Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
}

template<typename T>
void expectFileWriterSuccess(const std::string& suffix,
                             const std::vector<T>& expectedData) {
    auto implementations = Registry::ListAvailableModules("file_writer");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            const auto testPath = getTestFilePath(suffix);
            cleanupTestFile(testPath);

            TestContext ctx("file_writer", impl.device,
                            impl.runtime, impl.provider);

            Modules::FileWriter config;
            config.filepath = Platform::PathToUtf8(testPath);
            config.overwrite = true;
            config.recording = true;
            ctx.setConfig(config);

            auto input = ctx.createTensor<T>({expectedData.size()});
            for (U64 i = 0; i < expectedData.size(); ++i) {
                input.at(i) = expectedData[i];
            }

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);
            REQUIRE(std::filesystem::exists(testPath));
            REQUIRE(std::filesystem::file_size(testPath) ==
                    expectedData.size() * sizeof(T));

            std::ifstream verify(testPath, std::ios::binary);
            std::vector<T> values(expectedData.size());
            verify.read(reinterpret_cast<char*>(values.data()),
                        static_cast<std::streamsize>(values.size() * sizeof(T)));
            REQUIRE(values == expectedData);
            verify.close();

            cleanupTestFile(testPath);
        }
    }
}

}  // namespace

TEST_CASE("FileWriter Module - F32", "[modules][file_writer][F32]") {
    expectFileWriterSuccess<F32>("f32", {1.0f, 2.0f, 3.0f, 4.0f});
}

TEST_CASE("FileWriter Module - CF32", "[modules][file_writer][CF32]") {
    expectFileWriterSuccess<CF32>("cf32", {{1.0f, 2.0f}, {3.0f, 4.0f}});
}

TEST_CASE("FileWriter Module - CI8", "[modules][file_writer][CI8]") {
    expectFileWriterSuccess<CI8>("ci8", {{1, -2}, {3, 4}, {-5, 6}});
}

TEST_CASE("FileWriter Module - I8", "[modules][file_writer][I8]") {
    expectFileWriterSuccess<I8>("i8", {1, -2, 3, 4});
}

TEST_CASE("FileWriter Module - CU8", "[modules][file_writer][CU8]") {
    expectFileWriterSuccess<CU8>("cu8", {{1, 2}, {3, 4}, {5, 6}});
}

TEST_CASE("FileWriter Module - U8", "[modules][file_writer][U8]") {
    expectFileWriterSuccess<U8>("u8", {1, 2, 3, 4});
}

TEST_CASE("FileWriter Module - CI16", "[modules][file_writer][CI16]") {
    expectFileWriterSuccess<CI16>("ci16", {{1024, -2048}, {4096, 8192}});
}

TEST_CASE("FileWriter Module - I16", "[modules][file_writer][I16]") {
    expectFileWriterSuccess<I16>("i16", {1024, -2048, 4096, 8192});
}

TEST_CASE("FileWriter Module - CU16", "[modules][file_writer][CU16]") {
    expectFileWriterSuccess<CU16>("cu16", {{1024, 2048}, {4096, 8192}});
}

TEST_CASE("FileWriter Module - U16", "[modules][file_writer][U16]") {
    expectFileWriterSuccess<U16>("u16", {1024, 2048, 4096, 8192});
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
            config.filepath = Platform::PathToUtf8(testPath);

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
            config.filepath = Platform::PathToUtf8(testPath);
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
            config.filepath = Platform::PathToUtf8(testPath);
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
            config.fileFormat = "unknown";
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

            requireFileWriterDtypeValidationError(impl, testPath);
            REQUIRE(!std::filesystem::exists(testPath));

            const std::string existingData = "existing file contents";
            {
                std::ofstream file(testPath, std::ios::binary);
                file.write(existingData.data(),
                           static_cast<std::streamsize>(existingData.size()));
            }
            REQUIRE(std::filesystem::file_size(testPath) == existingData.size());

            requireFileWriterDtypeValidationError(impl, testPath);
            REQUIRE(std::filesystem::file_size(testPath) == existingData.size());

            {
                std::ifstream verify(testPath, std::ios::binary);
                std::string actualData(existingData.size(), '\0');
                verify.read(actualData.data(),
                            static_cast<std::streamsize>(actualData.size()));
                REQUIRE(actualData == existingData);
            }

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
            config.filepath = Platform::PathToUtf8(testPath);
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
            verify.close();

            cleanupTestFile(testPath);
        }
    }
}

TEST_CASE("FileWriter SigMF - records samples lost between frames",
          "[modules][file_writer][sigmf][losses]") {
    const auto base = std::filesystem::temp_directory_path() /
                      Platform::PathFromUtf8("jst_test_file_writer_losses");
    auto format = Modules::FileWriterFormat::Create("sigmf");
    REQUIRE(format);
    const auto dataPath = format->resolve(base);
    const auto metaPath = format->companions().front();

    constexpr U64 kFrame = 8;
    Tensor frame;
    REQUIRE(frame.create(DeviceType::CPU, DataType::CI16, {kFrame}) == Result::SUCCESS);
    REQUIRE(frame.setAttribute("sampleRate", 61.44e6f) == Result::SUCCESS);
    REQUIRE(frame.setAttribute("frequency", 915.0e6f) == Result::SUCCESS);
    const auto tag = [&](const U64 position, const U64 radioLost, const U64 overflows) {
        REQUIRE(frame.setAttribute("streamPosition", position) == Result::SUCCESS);
        REQUIRE(frame.setAttribute("radioLostSamples", radioLost) == Result::SUCCESS);
        REQUIRE(frame.setAttribute("deviceOverflows", overflows) == Result::SUCCESS);
    };

    std::ofstream file(dataPath, std::ios::binary);
    tag(100, 7, 2);
    REQUIRE(format->start(file, frame) == Result::SUCCESS);
    // Contiguous, then 24 samples missing, then a restart from zero (retune).
    U64 written = 0;
    for (const auto& [position, radioLost, overflows] :
         std::vector<std::tuple<U64, U64, U64>>{
             {100, 7, 2}, {108, 7, 2}, {140, 19, 3}, {0, 19, 3}, {8, 19, 3}}) {
        tag(position, radioLost, overflows);
        REQUIRE(format->track(frame, written) == Result::SUCCESS);
        written += kFrame;
    }
    REQUIRE(format->finish(file, written * 4) == Result::SUCCESS);
    file.close();

    std::ifstream metaFile(metaPath);
    const auto meta = nlohmann::json::parse(metaFile);
    REQUIRE(meta["global"]["core:datatype"] == "ci16_le");
    REQUIRE(meta["global"]["cyberether:lost_samples"] == 24);
    REQUIRE(meta["global"]["cyberether:radio_lost_samples"] == 12);
    REQUIRE(meta["global"]["cyberether:overflows"] == 1);
    REQUIRE(meta["annotations"].size() == 1);
    REQUIRE(meta["annotations"][0]["core:sample_start"] == 16);

    cleanupTestFile(dataPath);
    cleanupTestFile(metaPath);
}

TEST_CASE("FileWriter SigMF - no loss fields without a radio source",
          "[modules][file_writer][sigmf][losses]") {
    const auto base = std::filesystem::temp_directory_path() /
                      Platform::PathFromUtf8("jst_test_file_writer_plain");
    auto format = Modules::FileWriterFormat::Create("sigmf");
    const auto dataPath = format->resolve(base);
    const auto metaPath = format->companions().front();
    Tensor frame;
    REQUIRE(frame.create(DeviceType::CPU, DataType::CF32, {4}) == Result::SUCCESS);
    std::ofstream file(dataPath, std::ios::binary);
    REQUIRE(format->start(file, frame) == Result::SUCCESS);
    REQUIRE(format->track(frame, 0) == Result::SUCCESS);
    REQUIRE(format->finish(file, 32) == Result::SUCCESS);
    file.close();
    std::ifstream metaFile(metaPath);
    const auto meta = nlohmann::json::parse(metaFile);
    REQUIRE_FALSE(meta["global"].contains("cyberether:lost_samples"));
    cleanupTestFile(dataPath);
    cleanupTestFile(metaPath);
}
