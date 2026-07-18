#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <any>
#include <array>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "jetstream/memory/tensor.hh"
#include "jetstream/tools/automatic_iterator.hh"

namespace {

using namespace Jetstream;

struct Pixel {
    U16 red;
    U16 green;
};

void FillLinear(Tensor& tensor, F32 start = 0.0f) {
    for (U64 i = 0; i < tensor.size(); ++i) {
        tensor.data<F32>()[i] = start + static_cast<F32>(i);
    }
}

TEST_CASE("Tensor default and CPU creation state is coherent",
          "[core][memory][tensor][creation]") {
    SECTION("default state") {
        Tensor tensor;

        REQUIRE(tensor.empty());
        REQUIRE_FALSE(tensor.validShape());
        REQUIRE(tensor.device() == DeviceType::None);
        REQUIRE(tensor.nativeDevice() == DeviceType::None);
        REQUIRE(tensor.dtype() == DataType::None);
        REQUIRE(tensor.size() == 0);
        REQUIRE(tensor.sizeBytes() == 0);
        REQUIRE(tensor.elementSize() == 0);
        REQUIRE(tensor.rank() == 0);
        REQUIRE(tensor.ndims() == 0);
        REQUIRE(tensor.shape().empty());
        REQUIRE(tensor.stride().empty());
        REQUIRE(tensor.shapeMinusOne().empty());
        REQUIRE(tensor.backstride().empty());
        REQUIRE(tensor.offset() == 0);
        REQUIRE(tensor.offsetBytes() == 0);
        REQUIRE(tensor.contiguous());
        REQUIRE(tensor.data() == nullptr);
        REQUIRE(tensor.id() == 0);
        REQUIRE_THROWS_AS(tensor.buffer(), std::runtime_error);
    }

    SECTION("three-dimensional CPU tensor") {
        Tensor tensor(DeviceType::CPU, DataType::F32, {2, 3, 4});

        REQUIRE(tensor.validShape());
        REQUIRE_FALSE(tensor.empty());
        REQUIRE(tensor.device() == DeviceType::CPU);
        REQUIRE(tensor.nativeDevice() == DeviceType::CPU);
        REQUIRE(tensor.dtype() == DataType::F32);
        REQUIRE(tensor.shape() == Shape{2, 3, 4});
        REQUIRE(tensor.stride() == Shape{12, 4, 1});
        REQUIRE(tensor.shapeMinusOne() == Shape{1, 2, 3});
        REQUIRE(tensor.backstride() == Shape{12, 8, 3});
        REQUIRE(tensor.size() == 24);
        REQUIRE(tensor.sizeBytes() == 24 * sizeof(F32));
        REQUIRE(tensor.elementSize() == sizeof(F32));
        REQUIRE(tensor.rank() == 3);
        REQUIRE(tensor.ndims() == 3);
        REQUIRE(tensor.contiguous());
        REQUIRE(tensor.offset() == 0);
        REQUIRE(tensor.offsetBytes() == 0);
        REQUIRE(tensor.buffer().valid());
        REQUIRE(tensor.buffer().device() == DeviceType::CPU);
        REQUIRE(tensor.data() != nullptr);
        REQUIRE(std::all_of(tensor.data<F32>(), tensor.data<F32>() + tensor.size(),
                            [](F32 value) { return value == 0.0f; }));
    }

    SECTION("zero extent") {
        Tensor tensor(DeviceType::CPU, DataType::F32, {2, 0, 3});

        REQUIRE(tensor.validShape());
        REQUIRE(tensor.empty());
        REQUIRE(tensor.shape() == Shape{2, 0, 3});
        REQUIRE(tensor.stride() == Shape{0, 3, 1});
        REQUIRE(tensor.size() == 0);
        REQUIRE(tensor.sizeBytes() == 0);
        REQUIRE(tensor.buffer().valid());
        REQUIRE(tensor.data() == nullptr);
    }

    SECTION("zero-byte recreation clears borrowed storage") {
        Tensor tensor(DeviceType::CPU, DataType::F32, {2});
        void* const storage = tensor.data();
        REQUIRE(storage != nullptr);

        REQUIRE(tensor.create(storage, DeviceType::CPU, DataType::F32, {0}) ==
                Result::SUCCESS);
        REQUIRE(tensor.validShape());
        REQUIRE(tensor.empty());
        REQUIRE(tensor.buffer().valid());
        REQUIRE(tensor.buffer().isBorrowed());
        REQUIRE(tensor.data() == nullptr);
    }

    SECTION("invalid type and shape") {
        Tensor tensor;

        REQUIRE(tensor.create(DeviceType::CPU, DataType::None, {1}) == Result::ERROR);
        REQUIRE(tensor.create(DeviceType::CPU, DataType::F32, {}) == Result::ERROR);
        REQUIRE_FALSE(tensor.validShape());
        REQUIRE(tensor.device() == DeviceType::None);
        REQUIRE(tensor.dtype() == DataType::None);
    }
}

TEST_CASE("Tensor allocates every public data type", "[core][memory][tensor][dtype]") {
    constexpr std::array<DataType, 20> dataTypes = {
        DataType::F32, DataType::F64,
        DataType::I8, DataType::I16, DataType::I32, DataType::I64,
        DataType::U8, DataType::U16, DataType::U32, DataType::U64,
        DataType::CF32, DataType::CF64,
        DataType::CI8, DataType::CI16, DataType::CI32, DataType::CI64,
        DataType::CU8, DataType::CU16, DataType::CU32, DataType::CU64,
    };

    for (const auto dataType : dataTypes) {
        Tensor tensor(DeviceType::CPU, dataType, {2});

        REQUIRE(tensor.validShape());
        REQUIRE(tensor.dtype() == dataType);
        REQUIRE(tensor.elementSize() == DataTypeSize(dataType));
        REQUIRE(tensor.size() == 2);
        REQUIRE(tensor.sizeBytes() == 2 * DataTypeSize(dataType));
    }
}

TEST_CASE("Tensor borrows CPU storage and provides typed access",
          "[core][memory][tensor][access]") {
    SECTION("borrowed storage") {
        std::array<F32, 6> storage = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        Tensor tensor(storage.data(), DeviceType::CPU, DataType::F32, {2, 3});

        REQUIRE(tensor.validShape());
        REQUIRE(tensor.buffer().isBorrowed());
        REQUIRE(tensor.data() == storage.data());
        REQUIRE(tensor.at<F32>(1, 2) == 5.0f);
        REQUIRE(tensor.shapeToOffset({1, 2}) == 5);

        tensor.at<F32>(0, 1) = 12.5f;
        REQUIRE(storage[1] == 12.5f);

        const Tensor& constTensor = tensor;
        REQUIRE(constTensor.data<F32>() == storage.data());
        REQUIRE(constTensor.at<F32>(0, 1) == 12.5f);
    }

    SECTION("typed numeric tensor") {
        TypedTensor<I32> tensor(DeviceType::CPU, {2, 3});

        REQUIRE(tensor.dtype() == DataType::I32);
        REQUIRE(tensor.shape() == Shape{2, 3});
        REQUIRE(tensor.size() == 6);
        tensor.at(1, 2) = -17;
        REQUIRE(tensor.at(1, 2) == -17);
    }

    SECTION("typed object tensor") {
        TypedTensor<Pixel> pixels(DeviceType::CPU, {3});

        REQUIRE(pixels.dtype() == DataType::U8);
        REQUIRE(pixels.shape() == Shape{3 * sizeof(Pixel)});
        REQUIRE(pixels.size() == 3);
        REQUIRE(pixels.sizeBytes() == 3 * sizeof(Pixel));

        pixels.at(2) = Pixel{11, 29};
        REQUIRE(pixels.at(2).red == 11);
        REQUIRE(pixels.at(2).green == 29);
    }
}

TEST_CASE("Custom TypedTensor supports multidimensional and empty layouts",
          "[core][memory][tensor][typed]") {
    SECTION("multidimensional indexing uses logical object coordinates") {
        std::array<Pixel, 6> storage{};
        TypedTensor<Pixel> pixels(storage.data(), DeviceType::CPU, DataType::U8,
                                  {2, 3 * sizeof(Pixel)});
        storage[5] = Pixel{11, 29};

        REQUIRE(pixels.shape() == Shape{2, 3 * sizeof(Pixel)});
        REQUIRE(pixels.size() == 6);
        REQUIRE(pixels.sizeBytes() == 6 * sizeof(Pixel));
        REQUIRE(&pixels.at(1, 2) == pixels.data() + 5);
        REQUIRE(pixels.at(1, 2).red == 11);
        REQUIRE(pixels.at(1, 2).green == 29);

        const auto& constPixels = pixels;
        REQUIRE(&constPixels.at(1, 2) == constPixels.data() + 5);
        REQUIRE_THROWS_AS(pixels.at(2, 0), std::out_of_range);
        REQUIRE_THROWS_AS(pixels.at(0, 3), std::out_of_range);
        REQUIRE_THROWS_AS(pixels.at(0), std::out_of_range);
        REQUIRE_THROWS_AS(pixels.at(std::numeric_limits<U64>::max(), 0),
                          std::out_of_range);
    }

    SECTION("offset views apply their storage offset once") {
        std::array<Pixel, 6> storage{};
        TypedTensor<Pixel> pixels(storage.data(), DeviceType::CPU, DataType::U8,
                                  {2, 3 * sizeof(Pixel)});
        storage[5] = Pixel{37, 41};

        REQUIRE(pixels.slice({Token(U64{1}), Token()}) == Result::SUCCESS);
        REQUIRE(pixels.shape() == Shape{3 * sizeof(Pixel)});
        REQUIRE(&pixels.at(2) == storage.data() + 5);
        REQUIRE(pixels.at(2).red == 37);
        REQUIRE(pixels.at(2).green == 41);
    }

    SECTION("layouts with a moved packed axis are rejected") {
        TypedTensor<Pixel> pixels(DeviceType::CPU, {4, 2});

        REQUIRE(pixels.permute({1, 0}) == Result::SUCCESS);
        REQUIRE_THROWS_AS(pixels.at(2, 0), std::logic_error);
    }

    SECTION("byte-sliced object offsets are rejected") {
        TypedTensor<Pixel> pixels(DeviceType::CPU, {2});

        REQUIRE(pixels.slice({Token(U64{2}, U64{6})}) == Result::SUCCESS);
        REQUIRE_THROWS_AS(pixels.at(0), std::logic_error);
    }

    SECTION("malformed backing storage is rejected") {
        std::array<Pixel, 2> storage{};
        TypedTensor<Pixel> pixels(storage.data(), DeviceType::CPU, DataType::U8,
                                  {2 * sizeof(Pixel)});

        Buffer undersized;
        REQUIRE(undersized.create(DeviceType::CPU, sizeof(Pixel)) == Result::SUCCESS);
        pixels.buffer() = undersized;
        REQUIRE_THROWS_AS(pixels.at(1), std::out_of_range);

        REQUIRE(pixels.buffer().destroy() == Result::SUCCESS);
        REQUIRE_THROWS_AS(pixels.at(0), std::logic_error);
    }

    SECTION("incompatible runtime storage is rejected") {
        std::array<F32, 1> storage{};
        TypedTensor<Pixel> pixels(storage.data(), DeviceType::CPU, DataType::F32, {1});

        REQUIRE_THROWS_AS(pixels.at(0), std::logic_error);
    }

    SECTION("misaligned borrowed storage is rejected") {
        alignas(Pixel) std::array<std::uint8_t, sizeof(Pixel) + 1> storage{};
        TypedTensor<Pixel> pixels(storage.data() + 1, DeviceType::CPU, DataType::U8,
                                  {sizeof(Pixel)});

        REQUIRE_THROWS_AS(pixels.at(0), std::runtime_error);
    }

    SECTION("zero extent") {
        TypedTensor<Pixel> pixels(DeviceType::CPU, {2, 0, 3});

        REQUIRE(pixels.validShape());
        REQUIRE(pixels.empty());
        REQUIRE(pixels.shape() == Shape{2, 0, 3 * sizeof(Pixel)});
        REQUIRE(pixels.size() == 0);
        REQUIRE(pixels.sizeBytes() == 0);
        REQUIRE(pixels.data() == nullptr);
        REQUIRE_THROWS_AS(pixels.at(0, 0, 0), std::out_of_range);
    }

    SECTION("shape conversion overflow preserves existing storage") {
        TypedTensor<Pixel> pixels(DeviceType::CPU, {2});
        const Shape shape = pixels.shape();
        const Index id = pixels.id();
        void* const data = pixels.data();
        const U64 overflowingExtent = std::numeric_limits<U64>::max() / sizeof(Pixel) + 1;

        REQUIRE(pixels.create(DeviceType::CPU, {overflowingExtent}) == Result::ERROR);
        REQUIRE(pixels.shape() == shape);
        REQUIRE(pixels.id() == id);
        REQUIRE(pixels.data() == data);
    }
}

TEST_CASE("Failed Tensor creation preserves the complete existing state",
          "[core][memory][tensor][creation][errors]") {
    std::array<F32, 12> storage{};
    Tensor tensor(storage.data(), DeviceType::CPU, DataType::F32, {3, 4});
    REQUIRE(tensor.slice({Token(U64{1}, U64{3}), Token(U64{1}, U64{4}, U64{2})}) ==
            Result::SUCCESS);

    const DataType dtype = tensor.dtype();
    const Shape shape = tensor.shape();
    const Shape stride = tensor.stride();
    const U64 size = tensor.size();
    const U64 sizeBytes = tensor.sizeBytes();
    const U64 offset = tensor.offset();
    const U64 offsetBytes = tensor.offsetBytes();
    const bool contiguous = tensor.contiguous();
    const Index id = tensor.id();
    void* const data = tensor.data();

    REQUIRE(tensor.create(static_cast<void*>(nullptr), DeviceType::CPU, DataType::I32, {7}) ==
            Result::ERROR);
    REQUIRE(tensor.dtype() == dtype);
    REQUIRE(tensor.shape() == shape);
    REQUIRE(tensor.stride() == stride);
    REQUIRE(tensor.size() == size);
    REQUIRE(tensor.sizeBytes() == sizeBytes);
    REQUIRE(tensor.offset() == offset);
    REQUIRE(tensor.offsetBytes() == offsetBytes);
    REQUIRE(tensor.contiguous() == contiguous);
    REQUIRE(tensor.id() == id);
    REQUIRE(tensor.data() == data);

    REQUIRE(tensor.create(tensor.data(), DeviceType::CPU, DataType::F32, {2, 2}) ==
            Result::ERROR);
    REQUIRE(tensor.dtype() == dtype);
    REQUIRE(tensor.shape() == shape);
    REQUIRE(tensor.stride() == stride);
    REQUIRE(tensor.size() == size);
    REQUIRE(tensor.sizeBytes() == sizeBytes);
    REQUIRE(tensor.offset() == offset);
    REQUIRE(tensor.offsetBytes() == offsetBytes);
    REQUIRE(tensor.contiguous() == contiguous);
    REQUIRE(tensor.id() == id);
    REQUIRE(tensor.data() == data);
}

TEST_CASE("Tensor rejects size arithmetic overflow before allocation",
          "[core][memory][tensor][creation][overflow][errors]") {
    SECTION("zero extent short-circuits element count") {
        Tensor tensor;

        REQUIRE(tensor.create(DeviceType::CPU, DataType::F32,
                              {std::numeric_limits<U64>::max(), 2, 0}) ==
                Result::SUCCESS);
        REQUIRE(tensor.validShape());
        REQUIRE(tensor.empty());
        REQUIRE(tensor.stride() == Shape{0, 0, 1});
        REQUIRE(tensor.sizeBytes() == 0);
    }

    SECTION("element count overflow") {
        Tensor tensor;
        const U64 halfRangePlusOne = std::numeric_limits<U64>::max() / 2 + 1;

        REQUIRE(tensor.create(DeviceType::CPU, DataType::F32, {halfRangePlusOne, 2}) ==
                Result::ERROR);
        REQUIRE_FALSE(tensor.validShape());
    }

    SECTION("byte count overflow") {
        Tensor tensor;
        const U64 elements = std::numeric_limits<U64>::max() / sizeof(F64) + 1;

        REQUIRE(tensor.create(DeviceType::CPU, DataType::F64, {elements}) == Result::ERROR);
        REQUIRE_FALSE(tensor.validShape());
    }

    SECTION("overflow preserves existing state") {
        Tensor tensor(DeviceType::CPU, DataType::F32, {2, 3});
        const Shape shape = tensor.shape();
        const Index id = tensor.id();
        void* const data = tensor.data();
        const U64 halfRangePlusOne = std::numeric_limits<U64>::max() / 2 + 1;

        REQUIRE(tensor.create(DeviceType::CPU, DataType::F32, {halfRangePlusOne, 2}) ==
                Result::ERROR);
        REQUIRE(tensor.shape() == shape);
        REQUIRE(tensor.id() == id);
        REQUIRE(tensor.data() == data);
    }
}

TEST_CASE("Tensor value copies and clones have distinct semantics",
          "[core][memory][tensor][ownership]") {
    SECTION("value copy shares layout and storage") {
        Tensor source(DeviceType::CPU, DataType::F32, {2, 3});
        FillLinear(source);

        Tensor alias = source;
        REQUIRE(alias.id() == source.id());
        REQUIRE(alias.data() == source.data());

        alias.at<F32>(1, 2) = 42.0f;
        REQUIRE(source.at<F32>(1, 2) == 42.0f);
        REQUIRE(alias.reshape({3, 2}) == Result::SUCCESS);
        REQUIRE(source.shape() == Shape{3, 2});

        const Index id = source.id();
        void* const data = source.data();
        REQUIRE(source.create(DeviceType::CPU, alias) == Result::ERROR);
        REQUIRE(source.id() == id);
        REQUIRE(source.data() == data);
        REQUIRE(source.shape() == Shape{3, 2});
    }

    SECTION("clone shares storage but owns its layout") {
        Tensor source(DeviceType::CPU, DataType::F32, {2, 3});
        FillLinear(source);

        Tensor clone = source.clone();
        REQUIRE(clone.id() != source.id());
        REQUIRE(clone.data() == source.data());
        REQUIRE(clone.reshape({3, 2}) == Result::SUCCESS);
        REQUIRE(clone.shape() == Shape{3, 2});
        REQUIRE(source.shape() == Shape{2, 3});

        clone.at<F32>(2, 1) = 77.0f;
        REQUIRE(source.at<F32>(1, 2) == 77.0f);
    }

    SECTION("move retains initialized state") {
        Tensor source(DeviceType::CPU, DataType::F32, {2});
        source.at<F32>(1) = 8.0f;
        const Index id = source.id();

        Tensor moved = std::move(source);
        REQUIRE(moved.id() == id);
        REQUIRE(moved.shape() == Shape{2});
        REQUIRE(moved.at<F32>(1) == 8.0f);
        REQUIRE(source.empty());
        REQUIRE_FALSE(source.validShape());
        REQUIRE(source.data() == nullptr);
    }
}

TEST_CASE("Tensor copyFrom copies compatible CPU storage",
          "[core][memory][tensor][copy]") {
    Tensor source(DeviceType::CPU, DataType::F32, {2, 3});
    Tensor destination(DeviceType::CPU, DataType::F32, {2, 3});
    FillLinear(source, 10.0f);

    REQUIRE(destination.copyFrom(source) == Result::SUCCESS);
    REQUIRE(destination.data() != source.data());
    for (U64 i = 0; i < source.size(); ++i) {
        REQUIRE(destination.data<F32>()[i] == source.data<F32>()[i]);
    }

    source.data<F32>()[0] = 99.0f;
    REQUIRE(destination.data<F32>()[0] == 10.0f);

    Tensor wrongSize(DeviceType::CPU, DataType::F32, {5});
    std::fill(wrongSize.data<F32>(), wrongSize.data<F32>() + wrongSize.size(), -1.0f);
    REQUIRE(wrongSize.copyFrom(source) == Result::ERROR);
    REQUIRE(std::all_of(wrongSize.data<F32>(), wrongSize.data<F32>() + wrongSize.size(),
                        [](F32 value) { return value == -1.0f; }));
}

TEST_CASE("Tensor copyFrom safely copies overlapping borrowed CPU storage", "[core][memory][tensor][copy]") {
    std::array<F32, 6> storage = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Tensor source(storage.data(), DeviceType::CPU, DataType::F32, {4});
    Tensor destination(storage.data() + 1, DeviceType::CPU, DataType::F32, {4});

    REQUIRE(destination.copyFrom(source) == Result::SUCCESS);
    REQUIRE(storage == std::array<F32, 6>{1.0f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f});
}

TEST_CASE("Tensor copyFrom rejects a different logical shape",
          "[core][memory][tensor][copy][errors]") {
    Tensor source(DeviceType::CPU, DataType::F32, {2, 3});
    Tensor destination(DeviceType::CPU, DataType::F32, {6});
    FillLinear(source, 1.0f);
    std::fill(destination.data<F32>(), destination.data<F32>() + destination.size(), -1.0f);

    const Shape shape = destination.shape();
    const Shape stride = destination.stride();
    const U64 offset = destination.offset();

    REQUIRE(destination.copyFrom(source) == Result::ERROR);
    REQUIRE(destination.shape() == shape);
    REQUIRE(destination.stride() == stride);
    REQUIRE(destination.offset() == offset);
    REQUIRE(std::all_of(destination.data<F32>(),
                        destination.data<F32>() + destination.size(),
                        [](F32 value) { return value == -1.0f; }));
}

TEST_CASE("Tensor copyFrom rejects a different dtype atomically",
          "[core][memory][tensor][copy][errors]") {
    Tensor source(DeviceType::CPU, DataType::I32, {4});
    Tensor destination(DeviceType::CPU, DataType::F32, {4});
    std::fill(source.data<I32>(), source.data<I32>() + source.size(), I32{42});
    std::fill(destination.data<F32>(), destination.data<F32>() + destination.size(), -1.0f);

    const DataType dtype = destination.dtype();
    const Shape shape = destination.shape();
    const bool contiguous = destination.contiguous();
    const U64 offset = destination.offset();

    REQUIRE(destination.copyFrom(source) == Result::ERROR);
    REQUIRE(destination.dtype() == dtype);
    REQUIRE(destination.shape() == shape);
    REQUIRE(destination.contiguous() == contiguous);
    REQUIRE(destination.offset() == offset);
    REQUIRE(std::all_of(destination.data<F32>(),
                        destination.data<F32>() + destination.size(),
                        [](F32 value) { return value == -1.0f; }));
}

TEST_CASE("Tensor copyFrom rejects non-contiguous sources",
          "[core][memory][tensor][copy][errors]") {
    Tensor source(DeviceType::CPU, DataType::F32, {2, 3});
    Tensor destination(DeviceType::CPU, DataType::F32, {3, 2});
    FillLinear(source, 1.0f);
    std::fill(destination.data<F32>(), destination.data<F32>() + destination.size(), -1.0f);
    REQUIRE(source.permute({1, 0}) == Result::SUCCESS);
    REQUIRE_FALSE(source.contiguous());

    REQUIRE(destination.copyFrom(source) == Result::ERROR);
    REQUIRE(std::all_of(destination.data<F32>(),
                        destination.data<F32>() + destination.size(),
                        [](F32 value) { return value == -1.0f; }));
}

TEST_CASE("Tensor copyFrom rejects offset views atomically",
          "[core][memory][tensor][copy][errors]") {
    Tensor sourceStorage(DeviceType::CPU, DataType::F32, {2, 4});
    Tensor destinationStorage(DeviceType::CPU, DataType::F32, {2, 4});
    FillLinear(sourceStorage, 1.0f);
    std::fill(destinationStorage.data<F32>(),
              destinationStorage.data<F32>() + destinationStorage.size(), -1.0f);

    Tensor source = sourceStorage.clone();
    Tensor destination = destinationStorage.clone();
    REQUIRE(source.slice({Token(U64{1})}) == Result::SUCCESS);
    REQUIRE(destination.slice({Token(U64{0})}) == Result::SUCCESS);
    REQUIRE(source.offset() == 4);
    REQUIRE(destination.offset() == 0);
    REQUIRE(source.contiguous());
    REQUIRE(destination.contiguous());

    REQUIRE(destination.copyFrom(source) == Result::ERROR);
    REQUIRE(destination.offset() == 0);
    REQUIRE(destination.shape() == Shape{4});
    REQUIRE(std::all_of(destinationStorage.data<F32>(),
                        destinationStorage.data<F32>() + destinationStorage.size(),
                        [](F32 value) { return value == -1.0f; }));
}

TEST_CASE("Tensor copyFrom rejects incompatible destination storage atomically", "[core][memory][tensor][copy][errors]") {
    SECTION("non-contiguous destination") {
        Tensor source(DeviceType::CPU, DataType::F32, {3, 2});
        Tensor destination(DeviceType::CPU, DataType::F32, {2, 3});
        FillLinear(source, 1.0f);
        std::fill(destination.data<F32>(), destination.data<F32>() + destination.size(), -1.0f);
        REQUIRE(destination.permute({1, 0}) == Result::SUCCESS);
        REQUIRE_FALSE(destination.contiguous());

        REQUIRE(destination.copyFrom(source) == Result::ERROR);
        REQUIRE(std::all_of(destination.data<F32>(), destination.data<F32>() + destination.size(), [](F32 value) { return value == -1.0f; }));
    }

    SECTION("destination offset") {
        Tensor source(DeviceType::CPU, DataType::F32, {4});
        Tensor destinationStorage(DeviceType::CPU, DataType::F32, {2, 4});
        FillLinear(source, 1.0f);
        std::fill(destinationStorage.data<F32>(), destinationStorage.data<F32>() + destinationStorage.size(), -1.0f);
        Tensor destination = destinationStorage.clone();
        REQUIRE(destination.slice({Token(U64{1})}) == Result::SUCCESS);
        REQUIRE(destination.offset() == 4);

        REQUIRE(destination.copyFrom(source) == Result::ERROR);
        REQUIRE(std::all_of(destinationStorage.data<F32>(), destinationStorage.data<F32>() + destinationStorage.size(), [](F32 value) { return value == -1.0f; }));
    }

    SECTION("partial backing storage") {
        Tensor sourceStorage(DeviceType::CPU, DataType::F32, {2, 4});
        Tensor destinationStorage(DeviceType::CPU, DataType::F32, {2, 4});
        FillLinear(sourceStorage, 1.0f);
        std::fill(destinationStorage.data<F32>(), destinationStorage.data<F32>() + destinationStorage.size(), -1.0f);
        Tensor source = sourceStorage.clone();
        Tensor destination = destinationStorage.clone();
        REQUIRE(source.slice({Token(U64{0})}) == Result::SUCCESS);
        REQUIRE(destination.slice({Token(U64{0})}) == Result::SUCCESS);
        REQUIRE(source.offset() == 0);
        REQUIRE(destination.offset() == 0);

        REQUIRE(destination.copyFrom(source) == Result::ERROR);
        REQUIRE(std::all_of(destinationStorage.data<F32>(), destinationStorage.data<F32>() + destinationStorage.size(), [](F32 value) { return value == -1.0f; }));
    }

    SECTION("invalid source storage") {
        Tensor source(DeviceType::CPU, DataType::F32, {4});
        Tensor destination(DeviceType::CPU, DataType::F32, {4});
        std::fill(destination.data<F32>(), destination.data<F32>() + destination.size(), -1.0f);
        REQUIRE(source.buffer().destroy() == Result::SUCCESS);

        REQUIRE(destination.copyFrom(source) == Result::ERROR);
        REQUIRE(std::all_of(destination.data<F32>(), destination.data<F32>() + destination.size(), [](F32 value) { return value == -1.0f; }));
    }

    SECTION("invalid destination storage") {
        Tensor source(DeviceType::CPU, DataType::F32, {4});
        Tensor destination(DeviceType::CPU, DataType::F32, {4});
        FillLinear(source, 1.0f);
        REQUIRE(destination.buffer().destroy() == Result::SUCCESS);

        REQUIRE(destination.copyFrom(source) == Result::ERROR);
        REQUIRE_FALSE(destination.buffer().valid());
    }
}

TEST_CASE("Tensor dimension operations update cached layout",
          "[core][memory][tensor][layout]") {
    SECTION("expand and squeeze") {
        Tensor tensor(DeviceType::CPU, DataType::F32, {2, 3});

        REQUIRE(tensor.expandDims(1) == Result::SUCCESS);
        REQUIRE(tensor.shape() == Shape{2, 1, 3});
        REQUIRE(tensor.stride() == Shape{3, 3, 1});
        REQUIRE(tensor.shapeMinusOne() == Shape{1, 0, 2});
        REQUIRE(tensor.backstride() == Shape{3, 0, 2});
        REQUIRE(tensor.size() == 6);
        REQUIRE(tensor.squeezeDims(1) == Result::SUCCESS);
        REQUIRE(tensor.shape() == Shape{2, 3});
        REQUIRE(tensor.stride() == Shape{3, 1});
    }

    SECTION("invalid squeeze is atomic") {
        Tensor tensor(DeviceType::CPU, DataType::F32, {2, 3});
        const Shape shape = tensor.shape();
        const Shape stride = tensor.stride();

        REQUIRE(tensor.squeezeDims(0) == Result::ERROR);
        REQUIRE(tensor.squeezeDims(2) == Result::ERROR);
        REQUIRE(tensor.shape() == shape);
        REQUIRE(tensor.stride() == stride);
    }

    SECTION("reshape preserves data") {
        Tensor tensor(DeviceType::CPU, DataType::F32, {2, 3});
        FillLinear(tensor, 4.0f);
        void* data = tensor.data();

        REQUIRE(tensor.reshape({3, 2}) == Result::SUCCESS);
        REQUIRE(tensor.shape() == Shape{3, 2});
        REQUIRE(tensor.stride() == Shape{2, 1});
        REQUIRE(tensor.data() == data);
        REQUIRE(tensor.at<F32>(2, 1) == 9.0f);

        REQUIRE(tensor.reshape({7}) == Result::ERROR);
        REQUIRE(tensor.shape() == Shape{3, 2});
    }

    SECTION("broadcast uses zero strides") {
        Tensor tensor(DeviceType::CPU, DataType::F32, {1, 3});
        tensor.at<F32>(0, 0) = 3.0f;
        tensor.at<F32>(0, 1) = 5.0f;
        tensor.at<F32>(0, 2) = 7.0f;

        REQUIRE(tensor.broadcastTo({2, 3}) == Result::SUCCESS);
        REQUIRE(tensor.shape() == Shape{2, 3});
        REQUIRE(tensor.stride() == Shape{0, 1});
        REQUIRE(tensor.backstride() == Shape{0, 2});
        REQUIRE_FALSE(tensor.contiguous());
        REQUIRE(tensor.at<F32>(1, 0) == 3.0f);
        REQUIRE(tensor.at<F32>(1, 2) == 7.0f);
    }
}

TEST_CASE("Tensor broadcast rejects shrinking dimensions",
          "[core][memory][tensor][broadcast][errors]") {
    Tensor tensor(DeviceType::CPU, DataType::F32, {2, 3});

    // Current failure: broadcast accepts a target extent of one and keeps the larger extent.
    REQUIRE(tensor.broadcastTo({1, 3}) == Result::ERROR);
}

TEST_CASE("Failed Tensor broadcast preserves its original layout",
          "[core][memory][tensor][broadcast][errors]") {
    Tensor tensor(DeviceType::CPU, DataType::F32, {2, 3});
    const Shape shape = tensor.shape();
    const Shape stride = tensor.stride();

    REQUIRE(tensor.broadcastTo({2, 2, 4}) == Result::ERROR);
    // Current failure: rank expansion mutates the layout before compatibility is validated.
    REQUIRE(tensor.shape() == shape);
    REQUIRE(tensor.stride() == stride);
}

TEST_CASE("Tensor slicing tracks offsets, strides, and values",
          "[core][memory][tensor][slice]") {
    Tensor storage(DeviceType::CPU, DataType::F32, {4, 5});
    FillLinear(storage);
    Tensor view = storage.clone();

    REQUIRE(view.slice({Token(U64{1}, U64{4}), Token(U64{0}, U64{5}, U64{2})}) ==
            Result::SUCCESS);
    REQUIRE(view.shape() == Shape{3, 3});
    REQUIRE(view.stride() == Shape{5, 2});
    REQUIRE(view.shapeMinusOne() == Shape{2, 2});
    REQUIRE(view.backstride() == Shape{10, 4});
    REQUIRE(view.offset() == 5);
    REQUIRE(view.offsetBytes() == 5 * sizeof(F32));
    REQUIRE(view.data<F32>() == storage.data<F32>() + 5);
    REQUIRE_FALSE(view.contiguous());
    REQUIRE(view.at<F32>(0, 0) == 5.0f);
    REQUIRE(view.at<F32>(1, 1) == 12.0f);
    REQUIRE(view.at<F32>(2, 2) == 19.0f);
}

TEST_CASE("Tensor slicing supports rank reduction and ellipsis",
          "[core][memory][tensor][slice]") {
    SECTION("single index") {
        Tensor tensor(DeviceType::CPU, DataType::F32, {3, 4});
        FillLinear(tensor);

        REQUIRE(tensor.slice({Token(U64{1})}) == Result::SUCCESS);
        REQUIRE(tensor.shape() == Shape{4});
        REQUIRE(tensor.stride() == Shape{1});
        REQUIRE(tensor.offset() == 4);
        REQUIRE(tensor.contiguous());
        REQUIRE(tensor.at<F32>(0) == 4.0f);
        REQUIRE(tensor.at<F32>(3) == 7.0f);
    }

    SECTION("middle ellipsis") {
        Tensor tensor(DeviceType::CPU, DataType::F32, {2, 3, 4});
        FillLinear(tensor);

        REQUIRE(tensor.slice({Token("..."), Token(U64{1})}) == Result::SUCCESS);
        REQUIRE(tensor.shape() == Shape{2, 3});
        REQUIRE(tensor.stride() == Shape{12, 4});
        REQUIRE(tensor.offset() == 1);
        REQUIRE(tensor.at<F32>(0, 0) == 1.0f);
        REQUIRE(tensor.at<F32>(1, 2) == 21.0f);
    }

    SECTION("invalid slices preserve layout") {
        Tensor tensor(DeviceType::CPU, DataType::F32, {4, 4});
        const Shape shape = tensor.shape();
        const Shape stride = tensor.stride();

        REQUIRE(tensor.slice({Token(U64{0}, U64{4}, U64{0})}) == Result::ERROR);
        REQUIRE(tensor.slice({Token(U64{5})}) == Result::ERROR);
        REQUIRE(tensor.slice({Token("..."), Token("...")}) == Result::ERROR);
        REQUIRE(tensor.shape() == shape);
        REQUIRE(tensor.stride() == stride);
        REQUIRE(tensor.offset() == 0);
    }
}

TEST_CASE("Tensor trailing ellipsis preserves all remaining dimensions",
          "[core][memory][tensor][slice][errors]") {
    Tensor tensor(DeviceType::CPU, DataType::F32, {2, 3, 4});

    REQUIRE(tensor.slice({Token(U64{1}), Token("...")}) == Result::SUCCESS);
    // Current failure: a trailing ellipsis drops the final unconsumed dimension.
    REQUIRE(tensor.shape() == Shape{3, 4});
    REQUIRE(tensor.stride() == Shape{4, 1});
}

TEST_CASE("Tensor slicing validates token count and bounds atomically",
          "[core][memory][tensor][slice][errors]") {
    Tensor tensor(DeviceType::CPU, DataType::F32, {2, 3});
    const Shape shape = tensor.shape();
    const Shape stride = tensor.stride();

    REQUIRE(tensor.slice({Token(U64{0}), Token(U64{0}), Token(U64{0})}) ==
            Result::ERROR);
    REQUIRE(tensor.slice({Token(U64{0}, U64{3}), Token()}) == Result::ERROR);
    REQUIRE(tensor.slice({Token(), Token(U64{0}, U64{4})}) == Result::ERROR);
    REQUIRE(tensor.shape() == shape);
    REQUIRE(tensor.stride() == stride);
    REQUIRE(tensor.offset() == 0);
}

TEST_CASE("Tensor slicing produces a scalar when every axis is indexed",
          "[core][memory][tensor][slice][errors]") {
    Tensor tensor(DeviceType::CPU, DataType::F32, {2, 3});
    FillLinear(tensor, 1.0f);

    REQUIRE(tensor.slice({Token(U64{1}), Token(U64{2})}) == Result::SUCCESS);
    REQUIRE(tensor.rank() == 0);
    REQUIRE(tensor.validShape());
    REQUIRE(tensor.shape().empty());
    REQUIRE(tensor.stride().empty());
    REQUIRE(tensor.offset() == 5);
    REQUIRE(tensor.offsetBytes() == 5 * sizeof(F32));
    REQUIRE(tensor.size() == 1);
    REQUIRE(tensor.sizeBytes() == sizeof(F32));
    REQUIRE_FALSE(tensor.empty());
    REQUIRE(tensor.data<F32>() != nullptr);
    REQUIRE(*tensor.data<F32>() == 6.0f);
}

TEST_CASE("Tensor slicing preserves zero extents",
          "[core][memory][tensor][slice][errors]") {
    SECTION("full slice of an empty axis") {
        Tensor tensor(DeviceType::CPU, DataType::F32, {2, 0, 3});

        // Current failure: a full slice rejects a zero-sized dimension as out of bounds.
        REQUIRE(tensor.slice({Token(), Token(), Token()}) == Result::SUCCESS);
        REQUIRE(tensor.shape() == Shape{2, 0, 3});
        REQUIRE(tensor.size() == 0);
        REQUIRE(tensor.offset() == 0);
    }

    SECTION("empty range at the upper bound") {
        Tensor tensor(DeviceType::CPU, DataType::F32, {4});

        // Current failure: Python-style empty boundary ranges are rejected.
        REQUIRE(tensor.slice({Token(U64{4}, U64{4})}) == Result::SUCCESS);
        REQUIRE(tensor.shape() == Shape{0});
        REQUIRE(tensor.size() == 0);
        REQUIRE(tensor.offset() == 4);
    }
}

TEST_CASE("Repeated Tensor slices accumulate offsets exactly once",
          "[core][memory][tensor][slice]") {
    Tensor storage(DeviceType::CPU, DataType::F32, {4, 5});
    FillLinear(storage);
    Tensor view = storage.clone();

    REQUIRE(view.slice({Token(U64{1}, U64{4}), Token(U64{1}, U64{5})}) ==
            Result::SUCCESS);
    REQUIRE(view.offset() == 6);
    REQUIRE(view.slice({Token(U64{1}, U64{3}), Token(U64{1}, U64{4}, U64{2})}) ==
            Result::SUCCESS);

    REQUIRE(view.shape() == Shape{2, 2});
    REQUIRE(view.stride() == Shape{5, 2});
    REQUIRE(view.offset() == 12);
    REQUIRE(view.offsetBytes() == 12 * sizeof(F32));
    REQUIRE(view.data<F32>() == storage.data<F32>() + 12);
    REQUIRE(view.at<F32>(0, 0) == 12.0f);
    REQUIRE(view.at<F32>(0, 1) == 14.0f);
    REQUIRE(view.at<F32>(1, 0) == 17.0f);
    REQUIRE(view.at<F32>(1, 1) == 19.0f);
}

TEST_CASE("Tensor permutation validates axes and preserves indexing",
          "[core][memory][tensor][permute]") {
    Tensor tensor(DeviceType::CPU, DataType::F32, {2, 3});
    FillLinear(tensor);

    REQUIRE(tensor.permute({1, 0}) == Result::SUCCESS);
    REQUIRE(tensor.shape() == Shape{3, 2});
    REQUIRE(tensor.stride() == Shape{1, 3});
    REQUIRE_FALSE(tensor.contiguous());
    REQUIRE(tensor.at<F32>(0, 0) == 0.0f);
    REQUIRE(tensor.at<F32>(0, 1) == 3.0f);
    REQUIRE(tensor.at<F32>(2, 1) == 5.0f);

    const Shape shape = tensor.shape();
    const Shape stride = tensor.stride();
    REQUIRE(tensor.permute({0}) == Result::ERROR);
    REQUIRE(tensor.permute({0, 0}) == Result::ERROR);
    REQUIRE(tensor.permute({0, 2}) == Result::ERROR);
    REQUIRE(tensor.shape() == shape);
    REQUIRE(tensor.stride() == stride);
}

TEST_CASE("Tensor singleton permutations retain logical contiguity",
          "[core][memory][tensor][permute][errors]") {
    Tensor tensor(DeviceType::CPU, DataType::F32, {1, 3});

    REQUIRE(tensor.permute({1, 0}) == Result::SUCCESS);
    REQUIRE(tensor.shape() == Shape{3, 1});
    REQUIRE(tensor.contiguous());
}

TEST_CASE("Tensor attributes inherit through clones and propagation",
          "[core][memory][tensor][attributes]") {
    Tensor source(DeviceType::CPU, DataType::F32, {2});
    U64 generation = 1;

    REQUIRE(source.setAttribute("name", std::string("source")) == Result::SUCCESS);
    REQUIRE(source.setAttribute("sampleRate", F32{48'000.0f}) == Result::SUCCESS);
    REQUIRE(source.setDerivedAttribute("generation", [&]() { return std::any(generation); }) ==
            Result::SUCCESS);
    REQUIRE(source.hasAttribute("name"));
    REQUIRE_FALSE(source.hasAttribute("missing"));
    REQUIRE(std::any_cast<std::string>(source.attribute("name")) == "source");
    REQUIRE(std::any_cast<F32>(source.attribute("sampleRate")) == 48'000.0f);
    REQUIRE(std::any_cast<U64>(source.attribute("generation")) == 1);
    REQUIRE(source.attributeKeys() ==
            std::vector<std::string>{"generation", "name", "sampleRate"});
    REQUIRE_FALSE(source.attribute("missing").has_value());

    generation = 2;
    Tensor clone = source.clone();
    REQUIRE(std::any_cast<U64>(clone.attribute("generation")) == 2);
    REQUIRE(std::any_cast<std::string>(clone.attribute("name")) == "source");
    REQUIRE(clone.setAttribute("name", std::string("clone")) == Result::SUCCESS);
    REQUIRE(std::any_cast<std::string>(clone.attribute("name")) == "clone");
    REQUIRE(std::any_cast<std::string>(source.attribute("name")) == "source");

    Tensor destination(DeviceType::CPU, DataType::F32, {2});
    REQUIRE(destination.propagateAttributes(source) == Result::SUCCESS);
    REQUIRE(std::any_cast<F32>(destination.attribute("sampleRate")) == 48'000.0f);
}

TEST_CASE("Tensor swaps compatible storage without replacing metadata",
          "[core][memory][tensor][swap]") {
    Tensor left(DeviceType::CPU, DataType::F32, {2, 2});
    Tensor right(DeviceType::CPU, DataType::F32, {2, 2});
    std::fill(left.data<F32>(), left.data<F32>() + left.size(), 1.0f);
    std::fill(right.data<F32>(), right.data<F32>() + right.size(), 2.0f);
    REQUIRE(left.setAttribute("side", std::string("left")) == Result::SUCCESS);

    Tensor leftClone = left.clone();
    const Index leftId = left.id();
    const Index rightId = right.id();

    REQUIRE(left.swapBuffers(right) == Result::SUCCESS);
    REQUIRE(left.id() == leftId);
    REQUIRE(right.id() == rightId);
    REQUIRE(std::any_cast<std::string>(left.attribute("side")) == "left");
    REQUIRE(std::all_of(left.data<F32>(), left.data<F32>() + left.size(),
                        [](F32 value) { return value == 2.0f; }));
    REQUIRE(std::all_of(leftClone.data<F32>(), leftClone.data<F32>() + leftClone.size(),
                        [](F32 value) { return value == 2.0f; }));
    REQUIRE(std::all_of(right.data<F32>(), right.data<F32>() + right.size(),
                        [](F32 value) { return value == 1.0f; }));

    Tensor wrongShape(DeviceType::CPU, DataType::F32, {4});
    Tensor wrongType(DeviceType::CPU, DataType::I32, {2, 2});
    REQUIRE(left.swapBuffers(wrongShape) == Result::ERROR);
    REQUIRE(left.swapBuffers(wrongType) == Result::ERROR);
}

TEST_CASE("AutomaticIterator honors one-dimensional sliced strides",
          "[core][memory][iterator][slice]") {
    Tensor inputStorage(DeviceType::CPU, DataType::F32, {8});
    Tensor outputStorage(DeviceType::CPU, DataType::F32, {8});
    FillLinear(inputStorage, 1.0f);
    std::fill(outputStorage.data<F32>(), outputStorage.data<F32>() + outputStorage.size(), -1.0f);

    Tensor input = inputStorage.clone();
    Tensor output = outputStorage.clone();
    REQUIRE(input.slice({Token(U64{0}, U64{8}, U64{2})}) == Result::SUCCESS);
    REQUIRE(output.slice({Token(U64{0}, U64{8}, U64{2})}) == Result::SUCCESS);

    REQUIRE(AutomaticIterator<F32, F32>(
        [](const F32 value, F32& result) {
            result = value * 2.0f;
        },
        input, output) == Result::SUCCESS);

    for (U64 i = 0; i < outputStorage.size(); ++i) {
        if (i % 2 == 0) {
            REQUIRE(outputStorage.data<F32>()[i] == inputStorage.data<F32>()[i] * 2.0f);
        } else {
            REQUIRE(outputStorage.data<F32>()[i] == -1.0f);
        }
    }
}

TEST_CASE("AutomaticIterator mixes contiguous and transposed tensors",
          "[core][memory][iterator][permute]") {
    Tensor inputStorage(DeviceType::CPU, DataType::F32, {3, 4});
    Tensor input = inputStorage.clone();
    Tensor scale(DeviceType::CPU, DataType::F32, {4, 3});
    Tensor output(DeviceType::CPU, DataType::F32, {4, 3});
    FillLinear(inputStorage, 1.0f);
    FillLinear(scale, 10.0f);

    REQUIRE(input.permute({1, 0}) == Result::SUCCESS);
    REQUIRE_FALSE(input.contiguous());
    REQUIRE(scale.contiguous());
    REQUIRE(output.contiguous());

    REQUIRE(AutomaticIterator<F32, F32, F32>(
        [](const F32 value, const F32 multiplier, F32& result) {
            result = value * multiplier;
        },
        input, scale, output) == Result::SUCCESS);

    for (U64 row = 0; row < output.shape(0); ++row) {
        for (U64 column = 0; column < output.shape(1); ++column) {
            REQUIRE(output.at<F32>(row, column) ==
                    input.at<F32>(row, column) * scale.at<F32>(row, column));
        }
    }
}

TEST_CASE("AutomaticIterator handles offset and broadcast rank-four views",
          "[core][memory][iterator][broadcast]") {
    Tensor inputStorage(DeviceType::CPU, DataType::F32, {2, 2, 2, 2, 2});
    Tensor outputStorage(DeviceType::CPU, DataType::F32, {2, 2, 2, 2, 2});
    Tensor scaleStorage(DeviceType::CPU, DataType::F32, {2, 1, 2, 1, 2});
    FillLinear(inputStorage, 1.0f);
    FillLinear(scaleStorage, 1.0f);
    std::fill(outputStorage.data<F32>(), outputStorage.data<F32>() + outputStorage.size(), -1.0f);

    Tensor input = inputStorage.clone();
    Tensor output = outputStorage.clone();
    Tensor scale = scaleStorage.clone();
    REQUIRE(input.slice({Token(U64{1}), Token(), Token(), Token(), Token()}) == Result::SUCCESS);
    REQUIRE(output.slice({Token(U64{1}), Token(), Token(), Token(), Token()}) == Result::SUCCESS);
    REQUIRE(scale.slice({Token(U64{1}), Token(), Token(), Token(), Token()}) == Result::SUCCESS);
    REQUIRE(input.permute({1, 0, 2, 3}) == Result::SUCCESS);
    REQUIRE(output.permute({1, 0, 2, 3}) == Result::SUCCESS);
    REQUIRE(scale.broadcastTo(input.shape()) == Result::SUCCESS);
    REQUIRE(input.offset() != 0);
    REQUIRE(output.offset() != 0);
    REQUIRE_FALSE(input.contiguous());
    REQUIRE_FALSE(output.contiguous());
    REQUIRE_FALSE(scale.contiguous());
    REQUIRE(scale.stride(0) == 0);
    REQUIRE(scale.stride(2) == 0);

    REQUIRE(AutomaticIterator<F32, F32, F32>(
        [](const F32 value, const F32 multiplier, F32& result) {
            result = value * multiplier;
        },
        input, scale, output) == Result::SUCCESS);

    for (U64 i = 0; i < 2; ++i) {
        for (U64 j = 0; j < 2; ++j) {
            for (U64 k = 0; k < 2; ++k) {
                for (U64 l = 0; l < 2; ++l) {
                    REQUIRE(output.at<F32>(i, j, k, l) ==
                            input.at<F32>(i, j, k, l) * scale.at<F32>(i, j, k, l));
                }
            }
        }
    }
}

TEST_CASE("AutomaticIterator has no fixed generic rank limit",
          "[core][memory][iterator][rank]") {
    Shape shape(17, 1);
    shape.front() = 2;
    shape.back() = 2;
    Tensor input(DeviceType::CPU, DataType::F32, shape);
    Tensor output(DeviceType::CPU, DataType::F32, shape);
    FillLinear(input, 1.0f);

    Shape axes(17);
    std::iota(axes.begin(), axes.end(), U64{0});
    std::swap(axes.front(), axes.back());
    REQUIRE(input.permute(axes) == Result::SUCCESS);
    REQUIRE(output.permute(axes) == Result::SUCCESS);
    REQUIRE(input.rank() == 17);
    REQUIRE_FALSE(input.contiguous());
    REQUIRE_FALSE(output.contiguous());

    REQUIRE(AutomaticIterator<F32, F32>(
        [](const F32 value, F32& result) {
            result = value * 3.0f;
        },
        input, output) == Result::SUCCESS);

    for (U64 i = 0; i < output.size(); ++i) {
        REQUIRE(output.data<F32>()[i] == input.data<F32>()[i] * 3.0f);
    }
}

TEST_CASE("AutomaticIterator accepts empty CPU tensors",
          "[core][memory][iterator][empty]") {
    Tensor input(DeviceType::CPU, DataType::F32, {0});
    Tensor output(DeviceType::CPU, DataType::F32, {0});
    U64 invocations = 0;

    REQUIRE(AutomaticIterator<F32, F32>(
        [&](const F32, F32&) {
            ++invocations;
        },
        input, output) == Result::SUCCESS);
    REQUIRE(invocations == 0);
}

TEST_CASE("AutomaticIterator accepts cv-qualified element types", "[core][memory][iterator][types]") {
    Tensor input(DeviceType::CPU, DataType::F32, {3});
    Tensor output(DeviceType::CPU, DataType::F32, {3});
    FillLinear(input, 1.0f);

    REQUIRE(AutomaticIterator<const F32&, F32&>([](const F32& value, F32& destination) { destination = value * 2.0f; }, input, output) == Result::SUCCESS);
    REQUIRE(output.data<F32>()[0] == 2.0f);
    REQUIRE(output.data<F32>()[1] == 4.0f);
    REQUIRE(output.data<F32>()[2] == 6.0f);
}

TEST_CASE("AutomaticIterator rejects incompatible tensor metadata before iteration",
          "[core][memory][iterator][validation][errors]") {
    SECTION("rank mismatch") {
        std::array<F32, 4> inputStorage = {1.0f, 2.0f, 3.0f, 4.0f};
        std::array<F32, 4> outputStorage = {-1.0f, -1.0f, -1.0f, -1.0f};
        Tensor input(inputStorage.data(), DeviceType::CPU, DataType::F32, {2, 2});
        Tensor output(outputStorage.data(), DeviceType::CPU, DataType::F32, {4});
        U64 invocations = 0;

        const Result result = AutomaticIterator<F32, F32>(
            [&](const F32 value, F32& destination) {
                ++invocations;
                destination = value;
            },
            input, output);

        REQUIRE(result == Result::ERROR);
        REQUIRE(invocations == 0);
        REQUIRE(outputStorage == std::array<F32, 4>{-1.0f, -1.0f, -1.0f, -1.0f});
    }

    SECTION("shape mismatch") {
        std::array<F32, 4> inputStorage = {1.0f, 2.0f, 3.0f, 4.0f};
        std::array<F32, 4> outputStorage = {-1.0f, -1.0f, -1.0f, -1.0f};
        Tensor input(inputStorage.data(), DeviceType::CPU, DataType::F32, {2, 2});
        Tensor output(outputStorage.data(), DeviceType::CPU, DataType::F32, {1, 4});
        U64 invocations = 0;

        const Result result = AutomaticIterator<F32, F32>(
            [&](const F32 value, F32& destination) {
                ++invocations;
                destination = value;
            },
            input, output);

        REQUIRE(result == Result::ERROR);
        REQUIRE(invocations == 0);
        REQUIRE(outputStorage == std::array<F32, 4>{-1.0f, -1.0f, -1.0f, -1.0f});
    }

    SECTION("size mismatch") {
        // Both arrays have max(size) elements so today's missing check remains in bounds.
        std::array<F32, 3> inputStorage = {1.0f, 2.0f, 91.0f};
        std::array<F32, 3> outputStorage = {-1.0f, -1.0f, -1.0f};
        Tensor input(inputStorage.data(), DeviceType::CPU, DataType::F32, {2});
        Tensor output(outputStorage.data(), DeviceType::CPU, DataType::F32, {3});
        U64 invocations = 0;

        const Result result = AutomaticIterator<F32, F32>(
            [&](const F32 value, F32& destination) {
                ++invocations;
                destination = value;
            },
            input, output);

        REQUIRE(result == Result::ERROR);
        REQUIRE(invocations == 0);
        REQUIRE(outputStorage == std::array<F32, 3>{-1.0f, -1.0f, -1.0f});
    }

    SECTION("dtype mismatch") {
        // Runtime metadata is intentionally wrong; the backing objects remain F32 for safe access.
        std::array<F32, 3> inputStorage = {1.0f, 2.0f, 3.0f};
        std::array<F32, 3> outputStorage = {-1.0f, -1.0f, -1.0f};
        Tensor input(inputStorage.data(), DeviceType::CPU, DataType::I32, {3});
        Tensor output(outputStorage.data(), DeviceType::CPU, DataType::F32, {3});
        U64 invocations = 0;

        const Result result = AutomaticIterator<F32, F32>(
            [&](const F32 value, F32& destination) {
                ++invocations;
                destination = value;
            },
            input, output);

        REQUIRE(result == Result::ERROR);
        REQUIRE(invocations == 0);
        REQUIRE(outputStorage == std::array<F32, 3>{-1.0f, -1.0f, -1.0f});
    }

    SECTION("uninitialized tensor") {
        Tensor input;
        Tensor output(DeviceType::CPU, DataType::F32, {0});
        U64 invocations = 0;

        REQUIRE(AutomaticIterator<F32, F32>(
                    [&](const F32, F32&) {
                        ++invocations;
                    },
                    input, output) == Result::ERROR);
        REQUIRE(invocations == 0);
    }

    SECTION("invalid CPU storage") {
        Tensor input(DeviceType::CPU, DataType::F32, {0});
        Tensor output(DeviceType::CPU, DataType::F32, {0});
        U64 invocations = 0;
        REQUIRE(input.buffer().destroy() == Result::SUCCESS);
        REQUIRE_FALSE(input.buffer().valid());

        const Result result = AutomaticIterator<F32, F32>(
            [&](const F32, F32&) {
                ++invocations;
            },
            input, output);

        REQUIRE(result == Result::ERROR);
        REQUIRE(invocations == 0);
    }
}

}  // namespace
