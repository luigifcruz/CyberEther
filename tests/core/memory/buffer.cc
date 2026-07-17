#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <utility>

#include "jetstream/memory/buffer.hh"

namespace {

using namespace Jetstream;

constexpr U64 kBufferBytes = 64;

U8* Bytes(Buffer& buffer) {
    return static_cast<U8*>(buffer.data());
}

const U8* Bytes(const Buffer& buffer) {
    return static_cast<const U8*>(buffer.data());
}

TEST_CASE("Buffer has a stable empty lifecycle", "[core][memory][buffer][lifecycle]") {
    Buffer buffer;

    REQUIRE_FALSE(buffer.valid());
    REQUIRE_FALSE(buffer.isBorrowed());
    REQUIRE(buffer.sizeBytes() == 0);
    REQUIRE(buffer.device() == DeviceType::None);
    REQUIRE(buffer.nativeDevice() == DeviceType::None);
    REQUIRE(buffer.location() == Location::None);
    REQUIRE(buffer.data() == nullptr);
    REQUIRE(buffer.backend() == nullptr);

    REQUIRE(buffer.destroy() == Result::SUCCESS);
    REQUIRE(buffer.create(DeviceType::CPU, 0) == Result::SUCCESS);
    REQUIRE(buffer.valid());
    REQUIRE_FALSE(buffer.isBorrowed());
    REQUIRE(buffer.sizeBytes() == 0);
    REQUIRE(buffer.device() == DeviceType::CPU);
    REQUIRE(buffer.nativeDevice() == DeviceType::CPU);
    REQUIRE(buffer.location() == Location::Host);
    REQUIRE(buffer.data() == nullptr);
    REQUIRE(buffer.backend() != nullptr);

    REQUIRE(buffer.destroy() == Result::SUCCESS);
    REQUIRE(buffer.destroy() == Result::SUCCESS);
    REQUIRE_FALSE(buffer.valid());
    REQUIRE(buffer.device() == DeviceType::None);
    REQUIRE(buffer.nativeDevice() == DeviceType::None);
    REQUIRE(buffer.location() == Location::None);
}

TEST_CASE("Buffer owns zero-initialized CPU memory", "[core][memory][buffer][cpu]") {
    Buffer buffer;
    REQUIRE(buffer.create(DeviceType::CPU, kBufferBytes) == Result::SUCCESS);

    REQUIRE(buffer.valid());
    REQUIRE_FALSE(buffer.isBorrowed());
    REQUIRE(buffer.sizeBytes() == kBufferBytes);
    REQUIRE(buffer.device() == DeviceType::CPU);
    REQUIRE(buffer.nativeDevice() == DeviceType::CPU);
    REQUIRE(buffer.location() == Location::Host);
    REQUIRE(buffer.data() != nullptr);
    REQUIRE(std::all_of(Bytes(buffer), Bytes(buffer) + kBufferBytes,
                        [](U8 value) { return value == 0; }));

    for (U64 i = 0; i < kBufferBytes; ++i) {
        Bytes(buffer)[i] = static_cast<U8>(i);
    }

    const Buffer& constBuffer = buffer;
    REQUIRE(constBuffer.data() == buffer.data());
    for (U64 i = 0; i < kBufferBytes; ++i) {
        REQUIRE(Bytes(constBuffer)[i] == static_cast<U8>(i));
    }
}

TEST_CASE("Buffer can borrow external CPU memory", "[core][memory][buffer][borrow]") {
    std::array<U8, 8> storage = {1, 2, 3, 4, 5, 6, 7, 8};
    Buffer buffer;

    REQUIRE(buffer.create(DeviceType::CPU, storage.data(), storage.size()) == Result::SUCCESS);
    REQUIRE(buffer.valid());
    REQUIRE(buffer.isBorrowed());
    REQUIRE(buffer.data() == storage.data());
    REQUIRE(buffer.sizeBytes() == storage.size());
    REQUIRE(buffer.location() == Location::Host);

    storage[2] = 42;
    REQUIRE(Bytes(buffer)[2] == 42);
    Bytes(buffer)[5] = 99;
    REQUIRE(storage[5] == 99);

    REQUIRE(buffer.destroy() == Result::SUCCESS);
    REQUIRE(storage == std::array<U8, 8>{1, 2, 42, 4, 5, 99, 7, 8});

    Buffer emptyBorrow;
    REQUIRE(emptyBorrow.create(DeviceType::CPU, nullptr, 0) == Result::SUCCESS);
    REQUIRE(emptyBorrow.valid());
    REQUIRE(emptyBorrow.isBorrowed());
    REQUIRE(emptyBorrow.data() == nullptr);
}

TEST_CASE("Buffer copyFrom creates independent storage", "[core][memory][buffer][copy]") {
    Buffer source;
    REQUIRE(source.create(DeviceType::CPU, kBufferBytes) == Result::SUCCESS);
    for (U64 i = 0; i < kBufferBytes; ++i) {
        Bytes(source)[i] = static_cast<U8>(i * 3);
    }

    Buffer destination;
    REQUIRE(destination.copyFrom(source) == Result::SUCCESS);
    REQUIRE(destination.valid());
    REQUIRE_FALSE(destination.isBorrowed());
    REQUIRE(destination.data() != source.data());
    REQUIRE(destination.sizeBytes() == source.sizeBytes());
    REQUIRE(std::equal(Bytes(source), Bytes(source) + kBufferBytes, Bytes(destination)));

    Bytes(source)[0] = 255;
    REQUIRE(Bytes(destination)[0] == 0);

    Buffer existing;
    REQUIRE(existing.create(DeviceType::CPU, kBufferBytes) == Result::SUCCESS);
    std::fill(Bytes(existing), Bytes(existing) + kBufferBytes, U8{7});
    REQUIRE(existing.copyFrom(source) == Result::SUCCESS);
    REQUIRE(std::equal(Bytes(source), Bytes(source) + kBufferBytes, Bytes(existing)));

    Buffer wrongSize;
    REQUIRE(wrongSize.create(DeviceType::CPU, kBufferBytes / 2) == Result::SUCCESS);
    std::fill(Bytes(wrongSize), Bytes(wrongSize) + wrongSize.sizeBytes(), U8{11});
    REQUIRE(wrongSize.copyFrom(source) == Result::ERROR);
    REQUIRE(std::all_of(Bytes(wrongSize), Bytes(wrongSize) + wrongSize.sizeBytes(),
                        [](U8 value) { return value == 11; }));
}

TEST_CASE("Buffer value copies share ownership", "[core][memory][buffer][ownership]") {
    SECTION("copy") {
        Buffer original;
        REQUIRE(original.create(DeviceType::CPU, 4) == Result::SUCCESS);
        Bytes(original)[0] = 17;

        Buffer alias = original;
        REQUIRE(alias.data() == original.data());
        REQUIRE(Bytes(alias)[0] == 17);

        Bytes(alias)[0] = 23;
        REQUIRE(Bytes(original)[0] == 23);

        REQUIRE(alias.destroy() == Result::SUCCESS);
        REQUIRE_FALSE(alias.valid());
        REQUIRE_FALSE(original.valid());
    }

    SECTION("move") {
        Buffer source;
        REQUIRE(source.create(DeviceType::CPU, 4) == Result::SUCCESS);
        Bytes(source)[0] = 31;

        Buffer moved = std::move(source);
        REQUIRE(moved.valid());
        REQUIRE(Bytes(moved)[0] == 31);
        REQUIRE_FALSE(source.valid());
        REQUIRE(source.data() == nullptr);
    }
}

TEST_CASE("Buffer assignment is a zero-copy ownership operation",
          "[core][memory][buffer][ownership][assignment]") {
    SECTION("copy assignment retains displaced aliases") {
        Buffer source;
        REQUIRE(source.create(DeviceType::CPU, 8) == Result::SUCCESS);
        Bytes(source)[0] = 17;

        Buffer destination;
        REQUIRE(destination.create(DeviceType::CPU, 4) == Result::SUCCESS);
        Bytes(destination)[0] = 29;
        Buffer displaced = destination;
        void* const displacedData = displaced.data();

        destination = source;
        REQUIRE(destination.valid());
        REQUIRE(destination.data() == source.data());
        REQUIRE(destination.backend() == source.backend());
        REQUIRE(Bytes(destination)[0] == 17);
        REQUIRE(displaced.valid());
        REQUIRE(displaced.data() == displacedData);
        REQUIRE(Bytes(displaced)[0] == 29);

        Bytes(destination)[0] = 41;
        REQUIRE(Bytes(source)[0] == 41);
    }

    SECTION("move assignment transfers ownership") {
        Buffer source;
        REQUIRE(source.create(DeviceType::CPU, 4) == Result::SUCCESS);
        Bytes(source)[0] = 53;
        void* const sourceData = source.data();

        Buffer destination;
        REQUIRE(destination.create(DeviceType::CPU, 2) == Result::SUCCESS);
        destination = std::move(source);

        REQUIRE(destination.valid());
        REQUIRE(destination.data() == sourceData);
        REQUIRE(Bytes(destination)[0] == 53);
        REQUIRE_FALSE(source.valid());
        REQUIRE(source.data() == nullptr);
    }

    SECTION("copy and move self-assignment") {
        Buffer buffer;
        REQUIRE(buffer.create(DeviceType::CPU, 4) == Result::SUCCESS);
        Bytes(buffer)[0] = 67;
        void* const data = buffer.data();
        void* const backend = buffer.backend();
        Buffer& self = buffer;

        buffer = self;
        REQUIRE(buffer.valid());
        REQUIRE(buffer.data() == data);
        REQUIRE(buffer.backend() == backend);
        REQUIRE(Bytes(buffer)[0] == 67);

        buffer = std::move(self);
        REQUIRE(buffer.valid());
        REQUIRE(buffer.data() == data);
        REQUIRE(buffer.backend() == backend);
        REQUIRE(Bytes(buffer)[0] == 67);
    }
}

TEST_CASE("Buffer aliases outlive the handle that created storage",
          "[core][memory][buffer][ownership][lifetime]") {
    Buffer survivor;
    void* originalData = nullptr;

    {
        Buffer owner;
        REQUIRE(owner.create(DeviceType::CPU, 4) == Result::SUCCESS);
        Bytes(owner)[0] = 79;
        originalData = owner.data();
        survivor = owner;
    }

    REQUIRE(survivor.valid());
    REQUIRE(survivor.data() == originalData);
    REQUIRE(Bytes(survivor)[0] == 79);
}

TEST_CASE("Buffer copies zero-byte CPU storage without materializing data",
          "[core][memory][buffer][copy]") {
    Buffer source;
    REQUIRE(source.create(DeviceType::CPU, 0) == Result::SUCCESS);

    Buffer destination;
    REQUIRE(destination.copyFrom(source) == Result::SUCCESS);
    REQUIRE(destination.valid());
    REQUIRE(destination.device() == DeviceType::CPU);
    REQUIRE(destination.sizeBytes() == 0);
    REQUIRE(destination.data() == nullptr);
    REQUIRE(destination.backend() != source.backend());

    void* const backend = destination.backend();
    REQUIRE(destination.copyFrom(source) == Result::SUCCESS);
    REQUIRE(destination.backend() == backend);
    REQUIRE(destination.data() == nullptr);
}

TEST_CASE("Buffer rejects invalid creation and copy requests",
          "[core][memory][buffer][errors]") {
    SECTION("unsupported device leaves the object reusable") {
        Buffer buffer;
        REQUIRE(buffer.create(DeviceType::None, 4) == Result::ERROR);
        REQUIRE_FALSE(buffer.valid());
        REQUIRE(buffer.create(DeviceType::CPU, 4) == Result::SUCCESS);
    }

    SECTION("double creation preserves the original allocation") {
        Buffer buffer;
        REQUIRE(buffer.create(DeviceType::CPU, 4) == Result::SUCCESS);
        void* original = buffer.data();

        REQUIRE(buffer.create(DeviceType::CPU, 8) == Result::ERROR);
        REQUIRE(buffer.data() == original);
        REQUIRE(buffer.sizeBytes() == 4);
    }

    SECTION("invalid sources are rejected") {
        Buffer source;
        Buffer destination;

        REQUIRE(destination.copyFrom(source) == Result::ERROR);
        REQUIRE(destination.create(DeviceType::CPU, source) == Result::ERROR);
        REQUIRE_FALSE(destination.valid());
    }

    SECTION("same-device mirrors are rejected") {
        Buffer source;
        Buffer destination;
        REQUIRE(source.create(DeviceType::CPU, 4) == Result::SUCCESS);

        REQUIRE(destination.create(DeviceType::CPU, source) == Result::ERROR);
        REQUIRE_FALSE(destination.valid());
    }

    SECTION("failed copy leaves existing storage reusable") {
        Buffer tooLarge;
        Buffer compatible;
        Buffer destination;
        REQUIRE(tooLarge.create(DeviceType::CPU, 8) == Result::SUCCESS);
        REQUIRE(compatible.create(DeviceType::CPU, 4) == Result::SUCCESS);
        REQUIRE(destination.create(DeviceType::CPU, 4) == Result::SUCCESS);
        std::fill(Bytes(compatible), Bytes(compatible) + compatible.sizeBytes(), U8{37});
        std::fill(Bytes(destination), Bytes(destination) + destination.sizeBytes(), U8{11});

        REQUIRE(destination.copyFrom(tooLarge) == Result::ERROR);
        REQUIRE(std::all_of(Bytes(destination), Bytes(destination) + destination.sizeBytes(),
                            [](U8 value) { return value == 11; }));
        REQUIRE(destination.copyFrom(compatible) == Result::SUCCESS);
        REQUIRE(std::all_of(Bytes(destination), Bytes(destination) + destination.sizeBytes(),
                            [](U8 value) { return value == 37; }));
    }
}

TEST_CASE("Failed Buffer creation does not initialize the object",
          "[core][memory][buffer][errors]") {
    Buffer buffer;

    REQUIRE(buffer.create(DeviceType::CPU, nullptr, 4) == Result::ERROR);
    REQUIRE(buffer.data() == nullptr);
    REQUIRE(buffer.sizeBytes() == 0);
    // Current failure: Buffer retains the CPU backend after its backend create call fails.
    REQUIRE_FALSE(buffer.valid());
}

TEST_CASE("Buffer is reusable after a failed CPU creation",
          "[core][memory][buffer][errors]") {
    Buffer buffer;

    REQUIRE(buffer.create(DeviceType::CPU, nullptr, 4) == Result::ERROR);
    // Current failure: the failed backend remains installed and rejects the next create call.
    REQUIRE(buffer.create(DeviceType::CPU, 4) == Result::SUCCESS);
    REQUIRE(buffer.valid());
    REQUIRE(buffer.sizeBytes() == 4);
}

}  // namespace
