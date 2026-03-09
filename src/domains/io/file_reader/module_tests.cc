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

template<typename T>
void WriteRawFile(const std::filesystem::path& path, const std::vector<T>& data) {
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(data.data()),
               static_cast<std::streamsize>(data.size() * sizeof(T)));
}

template<typename T>
void ExpectFirstBatch(const std::string& suffix,
                      const std::string& dataType,
                      const DataType expectedDtype,
                      const std::vector<T>& data,
                      const std::vector<T>& expectedBatch) {
    auto implementations = Registry::ListAvailableModules("file_reader");
    REQUIRE(!implementations.empty());

    const auto path = TestFilePath(suffix);
    Cleanup(path);
    WriteRawFile(path, data);

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            TestContext ctx("file_reader", impl.device, impl.runtime,
                            impl.provider);

            Modules::FileReader config;
            config.filepath = path.string();
            config.dataType = dataType;
            config.batchSize = expectedBatch.size();
            config.loop = false;
            config.playing = true;
            ctx.setConfig(config);

            REQUIRE(ctx.run() == Result::SUCCESS);
            auto& out = ctx.output("signal");
            REQUIRE(out.dtype() == expectedDtype);
            REQUIRE(out.rank() == 1);
            REQUIRE(out.shape(0) == expectedBatch.size());

            for (U64 i = 0; i < expectedBatch.size(); ++i) {
                REQUIRE(out.at<T>(i) == expectedBatch[i]);
            }
        }
    }

    Cleanup(path);
}

}  // namespace

TEST_CASE("FileReader module reads the configured first F32 batch",
          "[modules][file_reader][f32]") {
    ExpectFirstBatch<F32>("f32_order",
                          "F32",
                          DataType::F32,
                          {1.5f, -2.0f, 3.0f, 4.25f},
                          {1.5f, -2.0f});
}

TEST_CASE("FileReader module reads the configured first CI8 batch",
          "[modules][file_reader][ci8]") {
    ExpectFirstBatch<CI8>("ci8_order",
                          "CI8",
                          DataType::CI8,
                          {CI8(1, -2), CI8(3, 4), CI8(-5, 6)},
                          {CI8(1, -2), CI8(3, 4)});
}

TEST_CASE("FileReader module reads the configured first I8 batch",
          "[modules][file_reader][i8]") {
    ExpectFirstBatch<I8>("i8_order",
                         "I8",
                         DataType::I8,
                         {I8(1), I8(-2), I8(3), I8(4)},
                         {I8(1), I8(-2)});
}

TEST_CASE("FileReader module reads the configured first CU8 batch",
          "[modules][file_reader][cu8]") {
    ExpectFirstBatch<CU8>("cu8_order",
                          "CU8",
                          DataType::CU8,
                          {CU8(1, 2), CU8(3, 4), CU8(5, 6)},
                          {CU8(1, 2), CU8(3, 4)});
}

TEST_CASE("FileReader module reads the configured first U8 batch",
          "[modules][file_reader][u8]") {
    ExpectFirstBatch<U8>("u8_order",
                         "U8",
                         DataType::U8,
                         {U8(1), U8(2), U8(3), U8(4)},
                         {U8(1), U8(2)});
}

TEST_CASE("FileReader module reads the configured first CI16 batch",
          "[modules][file_reader][ci16]") {
    ExpectFirstBatch<CI16>("ci16_order",
                           "CI16",
                           DataType::CI16,
                           {CI16(1024, -2048), CI16(4096, 8192), CI16(-1, 1)},
                           {CI16(1024, -2048), CI16(4096, 8192)});
}

TEST_CASE("FileReader module reads the configured first I16 batch",
          "[modules][file_reader][i16]") {
    ExpectFirstBatch<I16>("i16_order",
                          "I16",
                          DataType::I16,
                          {I16(1024), I16(-2048), I16(4096), I16(8192)},
                          {I16(1024), I16(-2048)});
}

TEST_CASE("FileReader module reads the configured first CU16 batch",
          "[modules][file_reader][cu16]") {
    ExpectFirstBatch<CU16>("cu16_order",
                           "CU16",
                           DataType::CU16,
                           {CU16(1024, 2048), CU16(4096, 8192), CU16(1, 2)},
                           {CU16(1024, 2048), CU16(4096, 8192)});
}

TEST_CASE("FileReader module reads the configured first U16 batch",
          "[modules][file_reader][u16]") {
    ExpectFirstBatch<U16>("u16_order",
                          "U16",
                          DataType::U16,
                          {U16(1024), U16(2048), U16(4096), U16(8192)},
                          {U16(1024), U16(2048)});
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

        SECTION("new data types pass validation") {
            for (const auto& dataType : {"CI8", "I8", "CU8", "U8", "CI16", "I16", "CU16", "U16"}) {
                TestContext ctx("file_reader", impl.device, impl.runtime,
                                impl.provider);
                Modules::FileReader config;
                config.dataType = dataType;
                ctx.setConfig(config);
                REQUIRE(ctx.run() != Result::ERROR);
            }
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

    WriteRawFile(path, std::vector<F32>{7.0f, 9.0f});

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

    WriteRawFile(path, std::vector<F32>{1.0f, 2.0f, 3.0f, 4.0f});

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
