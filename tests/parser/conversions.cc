#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "common.hh"

using namespace Jetstream;
using namespace TestParser;

namespace {

void RequireF32VectorEq(const std::vector<F32>& actual, const std::vector<F32>& expected) {
    REQUIRE(actual.size() == expected.size());

    for (U64 i = 0; i < actual.size(); ++i) {
        REQUIRE(actual[i] == Catch::Approx(expected[i]));
    }
}

void RequireF64VectorEq(const std::vector<F64>& actual, const std::vector<F64>& expected) {
    REQUIRE(actual.size() == expected.size());

    for (U64 i = 0; i < actual.size(); ++i) {
        REQUIRE(actual[i] == Catch::Approx(expected[i]));
    }
}

}  // namespace

TEST_CASE("Parser::SplitString preserves all segments", "[parser][conversions]") {
    SECTION("splits around repeated delimiters") {
        const auto values = Parser::SplitString("alpha, beta, gamma", ", ");
        const std::vector<std::string> expected = {"alpha", "beta", "gamma"};
        REQUIRE(values == expected);
    }

    SECTION("returns the full string when delimiter is absent") {
        const auto values = Parser::SplitString("alpha", ", ");
        const std::vector<std::string> expected = {"alpha"};
        REQUIRE(values == expected);
    }

    SECTION("keeps leading and trailing empty segments") {
        const auto values = Parser::SplitString(",alpha,", ",");
        const std::vector<std::string> expected = {"", "alpha", ""};
        REQUIRE(values == expected);
    }
}

TEST_CASE("Parser string conversions round-trip scalar values", "[parser][conversions]") {
    SECTION("std::string") {
        const std::string input = "parser";
        std::string encoded;
        REQUIRE(Parser::TypedToString(std::any(input), encoded) == Result::SUCCESS);
        REQUIRE(encoded == input);

        std::string decoded;
        REQUIRE(Parser::StringToTyped<std::string>(encoded, decoded) == Result::SUCCESS);
        REQUIRE(decoded == input);
    }

    SECTION("I32") {
        constexpr I32 input = -42;
        std::string encoded;
        REQUIRE(Parser::TypedToString(std::any(input), encoded) == Result::SUCCESS);
        REQUIRE(encoded == "-42");

        I32 decoded = 0;
        REQUIRE(Parser::StringToTyped<I32>(encoded, decoded) == Result::SUCCESS);
        REQUIRE(decoded == input);
    }

    SECTION("U64") {
        constexpr U64 input = 42;
        std::string encoded;
        REQUIRE(Parser::TypedToString(std::any(input), encoded) == Result::SUCCESS);
        REQUIRE(encoded == "42");

        U64 decoded = 0;
        REQUIRE(Parser::StringToTyped<U64>(encoded, decoded) == Result::SUCCESS);
        REQUIRE(decoded == input);
    }

    SECTION("F32") {
        constexpr F32 input = 1.25f;
        std::string encoded;
        REQUIRE(Parser::TypedToString(std::any(input), encoded) == Result::SUCCESS);

        F32 decoded = 0.0f;
        REQUIRE(Parser::StringToTyped<F32>(encoded, decoded) == Result::SUCCESS);
        REQUIRE(decoded == Catch::Approx(input));
    }

    SECTION("F64") {
        constexpr F64 input = 2.5;
        std::string encoded;
        REQUIRE(Parser::TypedToString(std::any(input), encoded) == Result::SUCCESS);

        F64 decoded = 0.0;
        REQUIRE(Parser::StringToTyped<F64>(encoded, decoded) == Result::SUCCESS);
        REQUIRE(decoded == Catch::Approx(input));
    }

    SECTION("bool") {
        std::string encoded;
        REQUIRE(Parser::TypedToString(std::any(true), encoded) == Result::SUCCESS);
        REQUIRE(encoded == "true");

        bool decoded = false;
        REQUIRE(Parser::StringToTyped<bool>(encoded, decoded) == Result::SUCCESS);
        REQUIRE(decoded);
    }
}

TEST_CASE("Parser string conversions round-trip enums", "[parser][conversions]") {
    SECTION("DeviceType") {
        std::string encoded;
        REQUIRE(Parser::TypedToString(std::any(DeviceType::CPU), encoded) == Result::SUCCESS);
        REQUIRE(encoded == "cpu");

        DeviceType decoded = DeviceType::None;
        REQUIRE(Parser::StringToTyped<DeviceType>(encoded, decoded) == Result::SUCCESS);
        REQUIRE(decoded == DeviceType::CPU);
    }

    SECTION("RuntimeType") {
        std::string encoded;
        REQUIRE(Parser::TypedToString(std::any(RuntimeType::NATIVE), encoded) == Result::SUCCESS);
        REQUIRE(encoded == "native");

        RuntimeType decoded = RuntimeType::NONE;
        REQUIRE(Parser::StringToTyped<RuntimeType>(encoded, decoded) == Result::SUCCESS);
        REQUIRE(decoded == RuntimeType::NATIVE);
    }

    SECTION("SchedulerType") {
        std::string encoded;
        REQUIRE(Parser::TypedToString(std::any(SchedulerType::SYNCHRONOUS), encoded) == Result::SUCCESS);
        REQUIRE(encoded == "synchronous");

        SchedulerType decoded = SchedulerType::NONE;
        REQUIRE(Parser::StringToTyped<SchedulerType>(encoded, decoded) == Result::SUCCESS);
        REQUIRE(decoded == SchedulerType::SYNCHRONOUS);
    }
}

TEST_CASE("Parser string conversions round-trip aggregates", "[parser][conversions]") {
    SECTION("std::vector<U64>") {
        const std::vector<U64> input = {1, 2, 3};
        std::string encoded;
        REQUIRE(Parser::TypedToString(std::any(input), encoded) == Result::SUCCESS);
        REQUIRE(encoded == "[1, 2, 3]");

        std::vector<U64> decoded;
        REQUIRE(Parser::StringToTyped<std::vector<U64>>(encoded, decoded) == Result::SUCCESS);
        REQUIRE(decoded == input);
    }

    SECTION("std::vector<F32>") {
        const std::vector<F32> input = {1.25f, 2.5f};
        std::string encoded;
        REQUIRE(Parser::TypedToString(std::any(input), encoded) == Result::SUCCESS);

        std::vector<F32> decoded;
        REQUIRE(Parser::StringToTyped<std::vector<F32>>(encoded, decoded) == Result::SUCCESS);
        RequireF32VectorEq(decoded, input);
    }

    SECTION("std::vector<F64>") {
        const std::vector<F64> input = {1.25, 2.5};
        std::string encoded;
        REQUIRE(Parser::TypedToString(std::any(input), encoded) == Result::SUCCESS);

        std::vector<F64> decoded;
        REQUIRE(Parser::StringToTyped<std::vector<F64>>(encoded, decoded) == Result::SUCCESS);
        RequireF64VectorEq(decoded, input);
    }

    SECTION("Range<F32>") {
        const Range<F32> input{1.5f, 9.5f};
        std::string encoded;
        REQUIRE(Parser::TypedToString(std::any(input), encoded) == Result::SUCCESS);
        REQUIRE(encoded == "[1.5, 9.5]");

        Range<F32> decoded{0.0f, 0.0f};
        REQUIRE(Parser::StringToTyped<Range<F32>>(encoded, decoded) == Result::SUCCESS);
        REQUIRE(decoded.min == Catch::Approx(input.min));
        REQUIRE(decoded.max == Catch::Approx(input.max));
    }

    SECTION("Extent2D<U64>") {
        const Extent2D<U64> input{8, 16};
        std::string encoded;
        REQUIRE(Parser::TypedToString(std::any(input), encoded) == Result::SUCCESS);
        REQUIRE(encoded == "[8, 16]");

        Extent2D<U64> decoded{0, 0};
        REQUIRE(Parser::StringToTyped<Extent2D<U64>>(encoded, decoded) == Result::SUCCESS);
        REQUIRE(decoded == input);
    }

    SECTION("Extent2D<F32>") {
        const Extent2D<F32> input{8.5f, 16.25f};
        std::string encoded;
        REQUIRE(Parser::TypedToString(std::any(input), encoded) == Result::SUCCESS);

        Extent2D<F32> decoded{0.0f, 0.0f};
        REQUIRE(Parser::StringToTyped<Extent2D<F32>>(encoded, decoded) == Result::SUCCESS);
        REQUIRE(decoded.x == Catch::Approx(input.x));
        REQUIRE(decoded.y == Catch::Approx(input.y));
    }

    SECTION("CF32 with positive imaginary value") {
        const CF32 input{3.0f, 4.0f};
        std::string encoded;
        REQUIRE(Parser::TypedToString(std::any(input), encoded) == Result::SUCCESS);
        REQUIRE(encoded == "3+4");

        CF32 decoded{};
        REQUIRE(Parser::StringToTyped<CF32>(encoded, decoded) == Result::SUCCESS);
        REQUIRE(decoded.real() == Catch::Approx(input.real()));
        REQUIRE(decoded.imag() == Catch::Approx(input.imag()));
    }

    SECTION("CF32 with negative imaginary value") {
        const CF32 input{3.0f, -4.0f};
        std::string encoded;
        REQUIRE(Parser::TypedToString(std::any(input), encoded) == Result::SUCCESS);
        REQUIRE(encoded == "3-4");

        CF32 decoded{};
        REQUIRE(Parser::StringToTyped<CF32>(encoded, decoded) == Result::SUCCESS);
        REQUIRE(decoded.real() == Catch::Approx(input.real()));
        REQUIRE(decoded.imag() == Catch::Approx(input.imag()));
    }
}

TEST_CASE("Parser::StringToTyped<bool> accepts common truthy values", "[parser][conversions]") {
    bool value = false;

    REQUIRE(Parser::StringToTyped<bool>("TRUE", value) == Result::SUCCESS);
    REQUIRE(value);

    REQUIRE(Parser::StringToTyped<bool>("1", value) == Result::SUCCESS);
    REQUIRE(value);

    REQUIRE(Parser::StringToTyped<bool>("false", value) == Result::SUCCESS);
    REQUIRE(!value);

    REQUIRE(Parser::StringToTyped<bool>("0", value) == Result::SUCCESS);
    REQUIRE(!value);
}

TEST_CASE("Parser::TypedToString rejects unsupported values", "[parser][conversions]") {
    std::string encoded = "unchanged";
    REQUIRE(Parser::TypedToString(std::any(UnsupportedValue{}), encoded) == Result::ERROR);
}
