#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include <sstream>
#include <vector>
#include <array>

#include "jetstream/memory2/buffer.hh"
#include "jetstream/memory2/types.hh"
#include "jetstream/logger.hh"

#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
#include "jetstream/backend/devices/metal/base.hh"
#endif

using namespace Jetstream;
using namespace mem2;

namespace {

constexpr U64 TEST_SIZE = 1024;

// Helper function to get CPU-accessible pointer from Metal buffer
#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
void* GetMetalBufferContents(const Buffer& buf) {
    if (buf.device() != Device::Metal) {
        return nullptr;
    }
    auto* metal_buffer = static_cast<MTL::Buffer*>(const_cast<void*>(buf.data()));
    return metal_buffer ? metal_buffer->contents() : nullptr;
}
#endif

// Helper function to write test pattern to buffer
void WriteTestPattern(const Buffer& buf, uint8_t pattern) {
    if (buf.device() == Device::CPU) {
        void* ptr = const_cast<void*>(buf.data());
        if (ptr) {
            std::memset(ptr, pattern, buf.size_bytes());
        }
    }
#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
    else if (buf.device() == Device::Metal) {
        void* contents = GetMetalBufferContents(buf);
        if (contents) {
            std::memset(contents, pattern, buf.size_bytes());
        }
    }
#endif
}

// Helper function to verify test pattern in buffer
bool VerifyTestPattern(const Buffer& buf, uint8_t pattern) {
    if (buf.size_bytes() == 0) {
        return true;
    }

    const uint8_t* ptr = nullptr;
    if (buf.device() == Device::CPU) {
        ptr = static_cast<const uint8_t*>(buf.data());
    }
#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
    else if (buf.device() == Device::Metal) {
        ptr = static_cast<const uint8_t*>(GetMetalBufferContents(buf));
    }
#endif

    if (!ptr) {
        return false;
    }

    for (U64 i = 0; i < buf.size_bytes(); ++i) {
        if (ptr[i] != pattern) {
            return false;
        }
    }
    return true;
}

}  // anonymous namespace

TEST_CASE("Buffer Creation and Destruction", "[buffer][creation]") {
    SECTION("Default Constructor State") {
        Buffer buf;
        REQUIRE_FALSE(buf.valid());
        REQUIRE(buf.size_bytes() == 0);
        REQUIRE(buf.device() == Device::None);
        REQUIRE(buf.native_device() == Device::None);
        REQUIRE(buf.location() == Location::None);
        REQUIRE_FALSE(buf.is_borrowed());
        REQUIRE(buf.data() == nullptr);
    }

    SECTION("CPU Creation - Empty") {
        Buffer buf;
        REQUIRE(buf.create(Device::CPU, 0) == Result::SUCCESS);
        REQUIRE(buf.valid());
        REQUIRE(buf.size_bytes() == 0);
        REQUIRE(buf.device() == Device::CPU);
        REQUIRE(buf.native_device() == Device::CPU);
        REQUIRE(buf.location() == Location::Host);
        REQUIRE_FALSE(buf.is_borrowed());
        REQUIRE(buf.data() == nullptr);
    }

    SECTION("CPU Creation - Non-Empty") {
        Buffer buf;
        REQUIRE(buf.create(Device::CPU, TEST_SIZE) == Result::SUCCESS);
        REQUIRE(buf.valid());
        REQUIRE(buf.size_bytes() == TEST_SIZE);
        REQUIRE(buf.device() == Device::CPU);
        REQUIRE(buf.native_device() == Device::CPU);
        REQUIRE(buf.location() == Location::Host);
        REQUIRE_FALSE(buf.is_borrowed());
        void* ptr = buf.data();
        REQUIRE(ptr != nullptr);

        // Write and read back to verify
        WriteTestPattern(buf, 0xAA);
        REQUIRE(VerifyTestPattern(buf, 0xAA));

        // Destroy and verify state
        REQUIRE(buf.destroy() == Result::SUCCESS);
        REQUIRE_FALSE(buf.valid());
        REQUIRE(buf.size_bytes() == 0);
        REQUIRE(buf.device() == Device::None);
        REQUIRE(buf.native_device() == Device::None);
        REQUIRE(buf.data() == nullptr);
    }

#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
    SECTION("Metal Creation - Empty") {
        Buffer buf;
        REQUIRE(buf.create(Device::Metal, 0) == Result::SUCCESS);
        REQUIRE(buf.valid());
        REQUIRE(buf.size_bytes() == 0);
        REQUIRE(buf.device() == Device::Metal);
        REQUIRE(buf.native_device() == Device::Metal);
        REQUIRE(buf.location() == Location::Unified);
        REQUIRE_FALSE(buf.is_borrowed());
        REQUIRE(buf.data() == nullptr);
    }

    SECTION("Metal Creation - Non-Empty") {
        Buffer buf;
        REQUIRE(buf.create(Device::Metal, TEST_SIZE) == Result::SUCCESS);
        REQUIRE(buf.valid());
        REQUIRE(buf.size_bytes() == TEST_SIZE);
        REQUIRE(buf.device() == Device::Metal);
        REQUIRE(buf.native_device() == Device::Metal);
        REQUIRE(buf.location() == Location::Unified);
        REQUIRE_FALSE(buf.is_borrowed());

        void* metal_buf_ptr = buf.data();
        REQUIRE(metal_buf_ptr != nullptr);

        // Verify we can access unified memory contents
        void* contents = GetMetalBufferContents(buf);
        REQUIRE(contents != nullptr);

        // Metal unified memory should be writable
        WriteTestPattern(buf, 0xBB);
        REQUIRE(VerifyTestPattern(buf, 0xBB));
    }
#endif

    SECTION("Double Creation") {
        Buffer buf;
        REQUIRE(buf.create(Device::CPU, TEST_SIZE) == Result::SUCCESS);
        REQUIRE(buf.create(Device::CPU, TEST_SIZE) == Result::ERROR);  // Already initialized
        // Original buffer should still be valid
        REQUIRE(buf.valid());
        REQUIRE(buf.size_bytes() == TEST_SIZE);
    }

    SECTION("Multiple Destruction") {
        Buffer buf;
        REQUIRE(buf.create(Device::CPU, TEST_SIZE) == Result::SUCCESS);
        REQUIRE(buf.destroy() == Result::SUCCESS);
        REQUIRE(buf.destroy() == Result::SUCCESS);  // Should be safe
        REQUIRE_FALSE(buf.valid());
    }
}

TEST_CASE("Buffer Copy Operations", "[buffer][copy]") {
    SECTION("CPU to CPU - Fresh Copy") {
        Buffer src;
        REQUIRE(src.create(Device::CPU, TEST_SIZE) == Result::SUCCESS);
        WriteTestPattern(src, 0xCC);

        Buffer dst;
        REQUIRE(dst.copy_from(src) == Result::SUCCESS);
        REQUIRE(dst.valid());
        REQUIRE(dst.size_bytes() == TEST_SIZE);
        REQUIRE(dst.device() == Device::CPU);
        REQUIRE(dst.native_device() == Device::CPU);
        REQUIRE_FALSE(dst.is_borrowed());

        REQUIRE(VerifyTestPattern(dst, 0xCC));
    }

    SECTION("CPU to CPU - To Existing Buffer") {
        Buffer src;
        REQUIRE(src.create(Device::CPU, TEST_SIZE) == Result::SUCCESS);
        WriteTestPattern(src, 0xDD);

        Buffer dst;
        REQUIRE(dst.create(Device::CPU, TEST_SIZE) == Result::SUCCESS);
        WriteTestPattern(dst, 0xEE);  // Fill with different pattern first

        REQUIRE(dst.copy_from(src) == Result::SUCCESS);
        REQUIRE(VerifyTestPattern(dst, 0xDD));
    }

    SECTION("Mismatched Sizes") {
        Buffer src;
        REQUIRE(src.create(Device::CPU, TEST_SIZE) == Result::SUCCESS);

        Buffer dst;
        REQUIRE(dst.create(Device::CPU, TEST_SIZE * 2) == Result::SUCCESS);
        REQUIRE(dst.copy_from(src) == Result::ERROR);  // Size mismatch
    }

    SECTION("Mismatched Devices - Existing Buffer") {
        Buffer src;
        REQUIRE(src.create(Device::CPU, TEST_SIZE) == Result::SUCCESS);

#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
        Buffer dst;
        REQUIRE(dst.create(Device::Metal, TEST_SIZE) == Result::SUCCESS);
        REQUIRE(dst.copy_from(src) == Result::ERROR);  // Device mismatch
#endif
    }

    SECTION("Empty Buffer Copy") {
        Buffer src;
        REQUIRE(src.create(Device::CPU, 0) == Result::SUCCESS);

        Buffer dst;
        REQUIRE(dst.copy_from(src) == Result::SUCCESS);
        REQUIRE(dst.size_bytes() == 0);
        REQUIRE(dst.device() == Device::CPU);
        REQUIRE(dst.valid());
    }

    SECTION("Copy from Invalid Buffer") {
        Buffer src;  // Not initialized
        Buffer dst;
        REQUIRE(dst.copy_from(src) == Result::ERROR);
        REQUIRE_FALSE(dst.valid());
    }

#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
    SECTION("Metal to CPU Copy") {
        Buffer src;
        REQUIRE(src.create(Device::Metal, TEST_SIZE) == Result::SUCCESS);
        WriteTestPattern(src, 0xEE);

        Buffer tmp;
        REQUIRE(tmp.create(Device::CPU, src) == Result::SUCCESS);

        Buffer dst;
        REQUIRE(dst.create(Device::CPU, TEST_SIZE) == Result::SUCCESS);
        REQUIRE(dst.copy_from(tmp) == Result::SUCCESS);
        REQUIRE(dst.device() == Device::CPU);
        REQUIRE(dst.size_bytes() == TEST_SIZE);
        REQUIRE_FALSE(dst.is_borrowed());

        REQUIRE(VerifyTestPattern(dst, 0xEE));
    }

    SECTION("CPU to Metal Copy") {
        Buffer src;
        REQUIRE(src.create(Device::CPU, TEST_SIZE) == Result::SUCCESS);
        WriteTestPattern(src, 0xFF);

        Buffer tmp;
        REQUIRE(tmp.create(Device::Metal, src) == Result::SUCCESS);

        Buffer dst;
        REQUIRE(dst.create(Device::Metal, TEST_SIZE) == Result::SUCCESS);
        REQUIRE(dst.copy_from(tmp) == Result::SUCCESS);
        REQUIRE(dst.device() == Device::Metal);
        REQUIRE(dst.size_bytes() == TEST_SIZE);
        REQUIRE_FALSE(dst.is_borrowed());

        REQUIRE(VerifyTestPattern(dst, 0xFF));
    }

    SECTION("Metal to Metal Copy") {
        Buffer src;
        REQUIRE(src.create(Device::Metal, TEST_SIZE) == Result::SUCCESS);
        WriteTestPattern(src, 0x11);

        Buffer dst;
        REQUIRE(dst.copy_from(src) == Result::SUCCESS);
        REQUIRE(dst.device() == Device::Metal);
        REQUIRE(dst.size_bytes() == TEST_SIZE);
        REQUIRE_FALSE(dst.is_borrowed());

        REQUIRE(VerifyTestPattern(dst, 0x11));
    }
#endif
}

TEST_CASE("Buffer Mirroring", "[buffer][mirror]") {
    SECTION("CPU Mirror - Self (Invalid)") {
        Buffer src;
        REQUIRE(src.create(Device::CPU, TEST_SIZE) == Result::SUCCESS);

        Buffer dst;
        REQUIRE(dst.create(Device::CPU, src) == Result::ERROR);  // Same device
        REQUIRE_FALSE(dst.valid());
    }

    SECTION("Mirror from Invalid Buffer") {
        Buffer src;  // Not initialized
        Buffer dst;
        REQUIRE(dst.create(Device::CPU, src) == Result::ERROR);
        REQUIRE_FALSE(dst.valid());
    }

    SECTION("Mirror Empty Buffer") {
        Buffer src;
        REQUIRE(src.create(Device::CPU, 0) == Result::SUCCESS);

#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
        Buffer dst;
        REQUIRE(dst.create(Device::Metal, src) == Result::SUCCESS);
        REQUIRE(dst.valid());
        REQUIRE(dst.size_bytes() == 0);
        REQUIRE(dst.device() == Device::Metal);
        REQUIRE(dst.native_device() == Device::CPU);  // Should inherit native device
#endif
    }

#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
    SECTION("Metal to CPU Mirror - Unified") {
        Buffer src;
        REQUIRE(src.create(Device::Metal, TEST_SIZE) == Result::SUCCESS);
        WriteTestPattern(src, 0x22);

        Buffer dst;
        REQUIRE(dst.create(Device::CPU, src) == Result::SUCCESS);
        REQUIRE(dst.valid());
        REQUIRE(dst.device() == Device::CPU);
        REQUIRE(dst.native_device() == Device::Metal);  // Should inherit native device
        REQUIRE(dst.size_bytes() == TEST_SIZE);
        REQUIRE(dst.is_borrowed());  // Should be borrowed from unified
        REQUIRE(dst.location() == Location::Unified);

        // Verify zero-copy mirroring - should share same memory
        void* dst_ptr = dst.data();
        void* src_contents = GetMetalBufferContents(src);
        REQUIRE(dst_ptr == src_contents);  // Same pointer for zero-copy

        // Verify data is accessible
        REQUIRE(VerifyTestPattern(dst, 0x22));

        // Verify that changes to one are visible in the other
        WriteTestPattern(dst, 0x33);
        REQUIRE(VerifyTestPattern(src, 0x33));
    }

    SECTION("CPU to Metal Mirror") {
        Buffer src;
        REQUIRE(src.create(Device::CPU, TEST_SIZE) == Result::SUCCESS);
        WriteTestPattern(src, 0x44);

        Buffer dst;
        REQUIRE(dst.create(Device::Metal, src) == Result::SUCCESS);
        REQUIRE(dst.valid());
        REQUIRE(dst.device() == Device::Metal);
        REQUIRE(dst.native_device() == Device::CPU);
        REQUIRE(dst.size_bytes() == TEST_SIZE);
        REQUIRE(dst.is_borrowed());
        REQUIRE(dst.location() == Location::Unified);

        // Verify the data is accessible through Metal buffer
        REQUIRE(VerifyTestPattern(dst, 0x44));

        // Verify shared memory - changes should be visible both ways
        WriteTestPattern(src, 0x55);
        REQUIRE(VerifyTestPattern(dst, 0x55));
    }
#endif

    SECTION("Double Mirror Creation") {
        Buffer src;
        REQUIRE(src.create(Device::CPU, TEST_SIZE) == Result::SUCCESS);

#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
        Buffer dst;
        REQUIRE(dst.create(Device::Metal, src) == Result::SUCCESS);
        REQUIRE(dst.create(Device::Metal, src) == Result::ERROR);  // Already initialized
        REQUIRE(dst.valid());  // Should still be valid
#endif
    }
}

TEST_CASE("Buffer Edge Cases", "[buffer][edge]") {
    SECTION("Large Allocation") {
        U64 large_size = 1024 * 1024;  // 1MB
        Buffer buf;
        REQUIRE(buf.create(Device::CPU, large_size) == Result::SUCCESS);
        REQUIRE(buf.size_bytes() == large_size);
        void* ptr = buf.data();
        REQUIRE(ptr != nullptr);

        WriteTestPattern(buf, 0x33);
        REQUIRE(VerifyTestPattern(buf, 0x33));

        REQUIRE(buf.destroy() == Result::SUCCESS);
    }

    SECTION("Very Large Size") {
        // Test with a size that might cause allocation failure
        U64 huge_size = static_cast<U64>(1024) * 1024 * 1024 * 16;  // 16GB
        Buffer buf;
        Result result = buf.create(Device::CPU, huge_size);
        // Don't require specific result as it depends on system memory
        // Just verify that if it fails, buffer state is correct
        if (result != Result::SUCCESS) {
            REQUIRE_FALSE(buf.valid());
            REQUIRE(buf.size_bytes() == 0);
            REQUIRE(buf.data() == nullptr);
        }
    }

    SECTION("Borrowed Buffer Destruction Order") {
#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
        Buffer src;
        REQUIRE(src.create(Device::Metal, TEST_SIZE) == Result::SUCCESS);
        WriteTestPattern(src, 0x66);

        Buffer dst;
        REQUIRE(dst.create(Device::CPU, src) == Result::SUCCESS);
        REQUIRE(dst.is_borrowed());

        // Destroy borrowed buffer first
        REQUIRE(dst.destroy() == Result::SUCCESS);
        REQUIRE_FALSE(dst.is_borrowed());
        REQUIRE_FALSE(dst.valid());

        // Source should still be valid and accessible
        REQUIRE(src.valid());
        REQUIRE(VerifyTestPattern(src, 0x66));

        // Destroy source
        REQUIRE(src.destroy() == Result::SUCCESS);
        REQUIRE_FALSE(src.valid());
#endif
    }

    SECTION("Copy Constructor and Assignment") {
        Buffer src;
        REQUIRE(src.create(Device::CPU, TEST_SIZE) == Result::SUCCESS);
        WriteTestPattern(src, 0x77);

        // Copy constructor
        Buffer copy1(src);
        REQUIRE(copy1.valid());
        REQUIRE(copy1.size_bytes() == TEST_SIZE);
        REQUIRE(VerifyTestPattern(copy1, 0x77));

        // Assignment operator
        Buffer copy2;
        copy2 = src;
        REQUIRE(copy2.valid());
        REQUIRE(copy2.size_bytes() == TEST_SIZE);
        REQUIRE(VerifyTestPattern(copy2, 0x77));

        // Move constructor
        Buffer moved(std::move(copy1));
        REQUIRE(moved.valid());
        REQUIRE(moved.size_bytes() == TEST_SIZE);
        REQUIRE(VerifyTestPattern(moved, 0x77));

        // Move assignment
        Buffer moved2;
        moved2 = std::move(copy2);
        REQUIRE(moved2.valid());
        REQUIRE(moved2.size_bytes() == TEST_SIZE);
        REQUIRE(VerifyTestPattern(moved2, 0x77));
    }

    SECTION("Buffer State After Operations") {
        Buffer buf;

        // Initial state
        REQUIRE_FALSE(buf.valid());
        REQUIRE(buf.size_bytes() == 0);
        REQUIRE(buf.device() == Device::None);
        REQUIRE(buf.native_device() == Device::None);
        REQUIRE(buf.location() == Location::None);
        REQUIRE_FALSE(buf.is_borrowed());
        REQUIRE(buf.data() == nullptr);

        // After creation
        REQUIRE(buf.create(Device::CPU, TEST_SIZE) == Result::SUCCESS);
        REQUIRE(buf.valid());
        REQUIRE(buf.size_bytes() == TEST_SIZE);
        REQUIRE(buf.device() == Device::CPU);
        REQUIRE(buf.native_device() == Device::CPU);
        REQUIRE(buf.location() == Location::Host);
        REQUIRE_FALSE(buf.is_borrowed());
        REQUIRE(buf.data() != nullptr);

        // After destruction
        REQUIRE(buf.destroy() == Result::SUCCESS);
        REQUIRE_FALSE(buf.valid());
        REQUIRE(buf.size_bytes() == 0);
        REQUIRE(buf.device() == Device::None);
        REQUIRE(buf.native_device() == Device::None);
        REQUIRE(buf.location() == Location::None);
        REQUIRE_FALSE(buf.is_borrowed());
        REQUIRE(buf.data() == nullptr);
    }
}

TEST_CASE("Buffer Memory Access Patterns", "[buffer][access]") {
    SECTION("CPU Buffer - Sequential Access") {
        Buffer buf;
        REQUIRE(buf.create(Device::CPU, TEST_SIZE) == Result::SUCCESS);

        auto* data = static_cast<uint8_t*>(buf.data());
        REQUIRE(data != nullptr);

        // Write sequential pattern
        for (U64 i = 0; i < TEST_SIZE; ++i) {
            data[i] = static_cast<uint8_t>(i & 0xFF);
        }

        // Verify sequential pattern
        for (U64 i = 0; i < TEST_SIZE; ++i) {
            REQUIRE(data[i] == static_cast<uint8_t>(i & 0xFF));
        }
    }

#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
    SECTION("Metal Buffer - Unified Memory Access") {
        Buffer buf;
        REQUIRE(buf.create(Device::Metal, TEST_SIZE) == Result::SUCCESS);

        void* contents = GetMetalBufferContents(buf);
        REQUIRE(contents != nullptr);

        auto* data = static_cast<uint8_t*>(contents);

        // Write sequential pattern
        for (U64 i = 0; i < TEST_SIZE; ++i) {
            data[i] = static_cast<uint8_t>((i * 2) & 0xFF);
        }

        // Verify sequential pattern
        for (U64 i = 0; i < TEST_SIZE; ++i) {
            REQUIRE(data[i] == static_cast<uint8_t>((i * 2) & 0xFF));
        }
    }
#endif
}

int main(int argc, char* argv[]) {
    JST_LOG_SET_DEBUG_LEVEL(4);

    return Catch::Session().run(argc, argv);
}
