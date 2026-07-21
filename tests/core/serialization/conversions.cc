#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>
#include <ostream>
#include <streambuf>

#include "fixture.hh"

using namespace Jetstream;
using namespace TestSerialization;

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

template<typename T>
void RequireIntegerRoundTrip(const T input) {
    std::string encoded;
    REQUIRE(Parser::TypedToString(std::any(input), encoded) == Result::SUCCESS);

    T decoded = {};
    REQUIRE(Parser::StringToTyped<T>(encoded, decoded) == Result::SUCCESS);
    REQUIRE(decoded == input);
}

struct ResultThrowingLogBuffer : std::streambuf {
    bool attempted = false;

    int_type overflow(int_type) override {
        attempted = true;
        throw Result::FATAL;
    }
};

struct LogSinkRestore {
    explicit LogSinkRestore(std::ostream& sink) : sink(&sink) {}
    ~LogSinkRestore() { JST_LOG_SET_SINK(sink); }

    std::ostream* sink;
};

}  // namespace

TEST_CASE("Parser::SplitString preserves all segments", "[core][serialization][conversions]") {
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

    SECTION("preserves an empty input segment") {
        REQUIRE(Parser::SplitString("", ",") == std::vector<std::string>{""});
    }

    SECTION("handles a delimiter spanning multiple characters") {
        const std::vector<std::string> expected = {"", "alpha", ""};
        REQUIRE(Parser::SplitString("::alpha::", "::") == expected);
    }
}

TEST_CASE("Parser integer conversions round-trip type boundaries", "[core][serialization][conversions]") {
    RequireIntegerRoundTrip<I8>(std::numeric_limits<I8>::min());
    RequireIntegerRoundTrip<I8>(std::numeric_limits<I8>::max());
    RequireIntegerRoundTrip<I16>(std::numeric_limits<I16>::min());
    RequireIntegerRoundTrip<I16>(std::numeric_limits<I16>::max());
    RequireIntegerRoundTrip<I32>(std::numeric_limits<I32>::min());
    RequireIntegerRoundTrip<I32>(std::numeric_limits<I32>::max());
    RequireIntegerRoundTrip<I64>(std::numeric_limits<I64>::min());
    RequireIntegerRoundTrip<I64>(std::numeric_limits<I64>::max());
    RequireIntegerRoundTrip<U8>(std::numeric_limits<U8>::min());
    RequireIntegerRoundTrip<U8>(std::numeric_limits<U8>::max());
    RequireIntegerRoundTrip<U16>(std::numeric_limits<U16>::min());
    RequireIntegerRoundTrip<U16>(std::numeric_limits<U16>::max());
    RequireIntegerRoundTrip<U32>(std::numeric_limits<U32>::min());
    RequireIntegerRoundTrip<U32>(std::numeric_limits<U32>::max());
    RequireIntegerRoundTrip<U64>(std::numeric_limits<U64>::min());
    RequireIntegerRoundTrip<U64>(std::numeric_limits<U64>::max());
}

TEST_CASE("Parser string conversions round-trip scalar values", "[core][serialization][conversions]") {
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

        I32 decoded;
        REQUIRE(Parser::StringToTyped<I32>(encoded, decoded) == Result::SUCCESS);
        REQUIRE(decoded == input);
    }

    SECTION("I64") {
        constexpr I64 input = -1;
        std::string encoded;
        REQUIRE(Parser::TypedToString(std::any(input), encoded) == Result::SUCCESS);
        REQUIRE(encoded == "-1");

        I64 decoded = 0;
        REQUIRE(Parser::StringToTyped<I64>(encoded, decoded) == Result::SUCCESS);
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

TEST_CASE("Parser string conversions reject invalid I64 values",
          "[core][serialization][conversions][errors]") {
    I64 decoded = 0;
    REQUIRE(Parser::StringToTyped<I64>("9223372036854775807", decoded) == Result::SUCCESS);
    REQUIRE(decoded == std::numeric_limits<I64>::max());
    REQUIRE(Parser::StringToTyped<I64>("-9223372036854775808", decoded) == Result::SUCCESS);
    REQUIRE(decoded == std::numeric_limits<I64>::min());

    decoded = 7;

    REQUIRE(Parser::StringToTyped<I64>("not-a-number", decoded) == Result::ERROR);
    REQUIRE(decoded == 7);

    REQUIRE(Parser::StringToTyped<I64>("9223372036854775808", decoded) == Result::ERROR);
    REQUIRE(decoded == 7);

    REQUIRE(Parser::StringToTyped<I64>("-9223372036854775809", decoded) == Result::ERROR);
    REQUIRE(decoded == 7);
}

TEST_CASE("Parser string conversions round-trip enums", "[core][serialization][conversions]") {
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

TEST_CASE("Parser string conversions round-trip aggregates", "[core][serialization][conversions]") {
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

    SECTION("std::vector<CF32>") {
        const std::vector<CF32> input = {{1.5f, 2.5f}, {3.0f, -4.0f}};
        std::string encoded;
        REQUIRE(Parser::TypedToString(std::any(input), encoded) == Result::SUCCESS);
        REQUIRE(encoded == "[1.5+2.5, 3-4]");

        std::vector<CF32> decoded;
        REQUIRE(Parser::StringToTyped<std::vector<CF32>>(encoded, decoded) == Result::SUCCESS);
        REQUIRE(decoded.size() == input.size());
        for (U64 i = 0; i < decoded.size(); ++i) {
            REQUIRE(decoded[i].real() == Catch::Approx(input[i].real()));
            REQUIRE(decoded[i].imag() == Catch::Approx(input[i].imag()));
        }
    }

    SECTION("std::vector<CF64>") {
        const std::vector<CF64> input = {{1.25, -2.75}, {0.5, 0.0}};
        std::string encoded;
        REQUIRE(Parser::TypedToString(std::any(input), encoded) == Result::SUCCESS);
        REQUIRE(encoded == "[1.25-2.75, 0.5+0]");

        std::vector<CF64> decoded;
        REQUIRE(Parser::StringToTyped<std::vector<CF64>>(encoded, decoded) == Result::SUCCESS);
        REQUIRE(decoded.size() == input.size());
        for (U64 i = 0; i < decoded.size(); ++i) {
            REQUIRE(decoded[i].real() == Catch::Approx(input[i].real()));
            REQUIRE(decoded[i].imag() == Catch::Approx(input[i].imag()));
        }
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

    SECTION("CF64 with positive imaginary value") {
        const CF64 input{3.25, 4.75};
        std::string encoded;
        REQUIRE(Parser::TypedToString(std::any(input), encoded) == Result::SUCCESS);
        REQUIRE(encoded == "3.25+4.75");

        CF64 decoded{};
        REQUIRE(Parser::StringToTyped<CF64>(encoded, decoded) == Result::SUCCESS);
        REQUIRE(decoded.real() == Catch::Approx(input.real()));
        REQUIRE(decoded.imag() == Catch::Approx(input.imag()));
    }

    SECTION("CF64 with negative imaginary value") {
        const CF64 input{3.25, -4.75};
        std::string encoded;
        REQUIRE(Parser::TypedToString(std::any(input), encoded) == Result::SUCCESS);
        REQUIRE(encoded == "3.25-4.75");

        CF64 decoded{};
        REQUIRE(Parser::StringToTyped<CF64>(encoded, decoded) == Result::SUCCESS);
        REQUIRE(decoded.real() == Catch::Approx(input.real()));
        REQUIRE(decoded.imag() == Catch::Approx(input.imag()));
    }

    SECTION("non-finite complex components") {
        const CF64 infiniteInput{std::numeric_limits<F64>::infinity(), 2.0};
        std::string encoded;
        REQUIRE(Parser::TypedToString(std::any(infiniteInput), encoded) == Result::SUCCESS);

        CF64 decoded{};
        REQUIRE(Parser::StringToTyped<CF64>(encoded, decoded) == Result::SUCCESS);
        REQUIRE(std::isinf(decoded.real()));
        REQUIRE(decoded.imag() == 2.0);

        const CF64 nanInput{1.0, std::numeric_limits<F64>::quiet_NaN()};
        REQUIRE(Parser::TypedToString(std::any(nanInput), encoded) == Result::SUCCESS);
        REQUIRE(Parser::StringToTyped<CF64>(encoded, decoded) == Result::SUCCESS);
        REQUIRE(decoded.real() == 1.0);
        REQUIRE(std::isnan(decoded.imag()));
    }
}

TEST_CASE("Parser::TypedToString serializes nested parser values", "[core][serialization][conversions]") {
    Parser::Sequence sequence;
    sequence.push_back(std::string("alpha"));
    sequence.push_back(U64{7});

    Parser::Map map;
    map["z"] = std::string("omega");
    map["a"] = sequence;

    std::string encoded;
    REQUIRE(Parser::TypedToString(std::any(map), encoded) == Result::SUCCESS);
    REQUIRE(encoded == "{a: [alpha, 7], z: omega}");

    REQUIRE(Parser::TypedToString(std::any(sequence), encoded) == Result::SUCCESS);
    REQUIRE(encoded == "[alpha, 7]");
}

TEST_CASE("Parser::TypedToString rejects unsupported nested parser values", "[core][serialization][conversions]") {
    Parser::Map map;
    map["bad"] = UnsupportedValue{};

    std::string encoded;
    REQUIRE(Parser::TypedToString(std::any(map), encoded) == Result::ERROR);

    Parser::Sequence sequence;
    sequence.push_back(UnsupportedValue{});
    REQUIRE(Parser::TypedToString(std::any(sequence), encoded) == Result::ERROR);
}

TEST_CASE("Parser::StringToTyped<bool> accepts common truthy values", "[core][serialization][conversions]") {
    bool value = false;

    REQUIRE(Parser::StringToTyped<bool>("TRUE", value) == Result::SUCCESS);
    REQUIRE(value);

    REQUIRE(Parser::StringToTyped<bool>("1", value) == Result::SUCCESS);
    REQUIRE(value);

    REQUIRE(Parser::StringToTyped<bool>("false", value) == Result::SUCCESS);
    REQUIRE(!value);

    REQUIRE(Parser::StringToTyped<bool>("FALSE", value) == Result::SUCCESS);
    REQUIRE(!value);

    REQUIRE(Parser::StringToTyped<bool>("0", value) == Result::SUCCESS);
    REQUIRE(!value);
}

TEST_CASE("Parser::StringToTyped rejects malformed scalar input", "[core][serialization][conversions][errors]") {
    SECTION("invalid integers return an error") {
        I32 value = 17;
        Result result = Result::SUCCESS;

        REQUIRE_NOTHROW(result = Parser::StringToTyped<I32>("not-a-number", value));
        REQUIRE(result == Result::ERROR);
        REQUIRE(value == 17);
    }

    SECTION("numeric suffixes are not silently ignored") {
        I32 value = 17;

        REQUIRE(Parser::StringToTyped<I32>("42junk", value) == Result::ERROR);
        REQUIRE(value == 17);
    }

    SECTION("floating-point suffixes are not silently ignored") {
        F32 value = 1.5f;

        REQUIRE(Parser::StringToTyped<F32>("2.5junk", value) == Result::ERROR);
        REQUIRE(value == 1.5f);
    }

    SECTION("out-of-range integers do not throw") {
        I32 value = 17;
        Result result = Result::SUCCESS;

        REQUIRE_NOTHROW(result = Parser::StringToTyped<I32>("999999999999999999999", value));
        REQUIRE(result == Result::ERROR);
        REQUIRE(value == 17);
    }

    SECTION("unknown boolean spellings return an error") {
        bool value = true;

        REQUIRE(Parser::StringToTyped<bool>("not-a-bool", value) == Result::ERROR);
        REQUIRE(value);
    }

    SECTION("malformed complex values return an error") {
        CF32 value{1.0f, 2.0f};

        REQUIRE(Parser::StringToTyped<CF32>("not-complex", value) == Result::ERROR);
        REQUIRE(value == CF32{1.0f, 2.0f});
    }

    SECTION("complex suffixes are not silently ignored") {
        CF32 value{1.0f, 2.0f};

        REQUIRE(Parser::StringToTyped<CF32>("3+4junk", value) == Result::ERROR);
        REQUIRE(value == CF32{1.0f, 2.0f});
    }

    SECTION("complex components reject doubled signs") {
        for (const std::string encoded : {"3++4", "3+-4", "3--4", "3-+4"}) {
            CAPTURE(encoded);
            CF32 value{1.0f, 2.0f};

            REQUIRE(Parser::StringToTyped<CF32>(encoded, value) == Result::ERROR);
            REQUIRE(value == CF32{1.0f, 2.0f});
        }
    }

    SECTION("logger exceptions remain contained") {
        ResultThrowingLogBuffer buffer;
        std::ostream sink(&buffer);
        sink.exceptions(std::ios::badbit | std::ios::failbit);
        LogSinkRestore restore(JST_LOG_SINK());
        JST_LOG_SET_SINK(&sink);
        bool value = true;
        Result result = Result::SUCCESS;

        REQUIRE_NOTHROW(result = Parser::StringToTyped<bool>("not-a-bool", value));
        REQUIRE(buffer.attempted);
        REQUIRE(result == Result::ERROR);
        REQUIRE(value);
    }
}

TEST_CASE("Parser::StringToTyped validates narrow and unsigned ranges", "[core][serialization][conversions][errors]") {
    I8 signedValue = 7;
    REQUIRE(Parser::StringToTyped<I8>("128", signedValue) == Result::ERROR);
    REQUIRE(signedValue == 7);
    REQUIRE(Parser::StringToTyped<I8>("-129", signedValue) == Result::ERROR);
    REQUIRE(signedValue == 7);

    U8 unsignedValue = 7;
    REQUIRE(Parser::StringToTyped<U8>("256", unsignedValue) == Result::ERROR);
    REQUIRE(unsignedValue == 7);
    REQUIRE(Parser::StringToTyped<U8>("-1", unsignedValue) == Result::ERROR);
    REQUIRE(unsignedValue == 7);
}

TEST_CASE("Parser vector conversion is atomic on element errors", "[core][serialization][conversions][errors]") {
    SECTION("unsigned integers") {
        std::vector<U64> value = {8, 13};

        REQUIRE(Parser::StringToTyped<std::vector<U64>>("[1, -2]", value) == Result::ERROR);
        REQUIRE(value == std::vector<U64>({8, 13}));
    }

    SECTION("floating point") {
        std::vector<F32> value = {8.0f, 13.0f};

        REQUIRE(Parser::StringToTyped<std::vector<F32>>("[1.0, bad]", value) == Result::ERROR);
        REQUIRE(value == std::vector<F32>({8.0f, 13.0f}));
    }

    SECTION("complex") {
        std::vector<CF64> value = {{8.0, 13.0}};

        REQUIRE(Parser::StringToTyped<std::vector<CF64>>("[1+2, bad]", value) == Result::ERROR);
        REQUIRE(value == std::vector<CF64>({{8.0, 13.0}}));
    }
}

TEST_CASE("Parser aggregate conversion is atomic on element errors", "[core][serialization][conversions][errors]") {
    SECTION("range") {
        Range<F32> value{8.0f, 13.0f};

        REQUIRE(Parser::StringToTyped<Range<F32>>("[1.0, 2.0junk]", value) == Result::ERROR);
        REQUIRE(value.min == 8.0f);
        REQUIRE(value.max == 13.0f);
    }

    SECTION("unsigned extent") {
        Extent2D<U64> value{8, 13};

        REQUIRE(Parser::StringToTyped<Extent2D<U64>>("[1, -2]", value) == Result::ERROR);
        REQUIRE(value == Extent2D<U64>{8, 13});
    }

    SECTION("floating-point extent") {
        Extent2D<F32> value{8.0f, 13.0f};

        REQUIRE(Parser::StringToTyped<Extent2D<F32>>("[1.0, bad]", value) == Result::ERROR);
        REQUIRE(value == Extent2D<F32>{8.0f, 13.0f});
    }
}

TEST_CASE("Parser::TypedToString rejects unsupported values", "[core][serialization][conversions]") {
    std::string encoded = "unchanged";
    REQUIRE(Parser::TypedToString(std::any(UnsupportedValue{}), encoded) == Result::ERROR);
    REQUIRE(encoded == "unchanged");

    REQUIRE(Parser::TypedToString(std::any{}, encoded) == Result::ERROR);
    REQUIRE(encoded == "unchanged");
}
