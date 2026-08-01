#include <array>
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <jetstream/tools/circular_buffer.hh>

using namespace Jetstream;
using OverflowPolicy = Tools::CircularBufferOverflowPolicy;

TEST_CASE("Circular buffer supports wrapped peeking and selective discard",
          "[tools][circular_buffer]") {
    Tools::CircularBuffer<F32> buffer(4, OverflowPolicy::Reject);

    const std::array<F32, 3> first = {1.0f, 2.0f, 3.0f};
    REQUIRE(buffer.push(first.data(), first.size()) == Result::SUCCESS);

    std::array<F32, 2> peeked = {};
    REQUIRE(buffer.peek(1, peeked.data(), peeked.size()) == Result::SUCCESS);
    const std::array<F32, 2> expectedPeek = {2.0f, 3.0f};
    REQUIRE(peeked == expectedPeek);
    REQUIRE(buffer.size() == 3);

    REQUIRE(buffer.discard(2) == Result::SUCCESS);
    const std::array<F32, 3> second = {4.0f, 5.0f, 6.0f};
    REQUIRE(buffer.push(second.data(), second.size()) == Result::SUCCESS);
    REQUIRE(buffer.full());

    std::array<F32, 4> output = {};
    REQUIRE(buffer.pop(output.data(), output.size()) == Result::SUCCESS);
    const std::array<F32, 4> expected = {3.0f, 4.0f, 5.0f, 6.0f};
    REQUIRE(output == expected);
    REQUIRE(buffer.empty());
}

TEST_CASE("Circular buffer applies explicit overflow policies",
          "[tools][circular_buffer][overflow]") {
    SECTION("overwrite retains the newest elements") {
        Tools::CircularBuffer<F32> buffer(4, OverflowPolicy::OverwriteOldest);
        const std::array<F32, 3> first = {1.0f, 2.0f, 3.0f};
        const std::array<F32, 3> second = {4.0f, 5.0f, 6.0f};
        REQUIRE(buffer.push(first.data(), first.size()) == Result::SUCCESS);
        REQUIRE(buffer.push(second.data(), second.size()) == Result::SUCCESS);
        REQUIRE(buffer.overflows() == 1);

        std::array<F32, 4> output = {};
        REQUIRE(buffer.pop(output.data(), output.size()) == Result::SUCCESS);
        const std::array<F32, 4> expected = {3.0f, 4.0f, 5.0f, 6.0f};
        REQUIRE(output == expected);
    }

    SECTION("oversized writes retain one full latest window") {
        Tools::CircularBuffer<F32> buffer(4, OverflowPolicy::OverwriteOldest);
        const std::array<F32, 6> input = {1.0f, 2.0f, 3.0f,
                                          4.0f, 5.0f, 6.0f};
        REQUIRE(buffer.push(input.data(), input.size()) == Result::SUCCESS);
        REQUIRE(buffer.overflows() == 1);

        std::array<F32, 4> output = {};
        REQUIRE(buffer.pop(output.data(), output.size()) == Result::SUCCESS);
        const std::array<F32, 4> expected = {3.0f, 4.0f, 5.0f, 6.0f};
        REQUIRE(output == expected);
    }

    SECTION("reject leaves existing data untouched") {
        Tools::CircularBuffer<F32> buffer(2, OverflowPolicy::Reject);
        const std::array<F32, 2> first = {1.0f, 2.0f};
        const F32 extra = 3.0f;
        REQUIRE(buffer.push(first.data(), first.size()) == Result::SUCCESS);
        REQUIRE(buffer.push(&extra, 1) == Result::INCOMPLETE);
        REQUIRE(buffer.overflows() == 1);

        std::array<F32, 2> output = {};
        REQUIRE(buffer.pop(output.data(), output.size()) == Result::SUCCESS);
        REQUIRE(output == first);
    }
}

TEST_CASE("Circular buffer accepts strided input in one operation",
          "[tools][circular_buffer][strided]") {
    Tools::CircularBuffer<F32> buffer(3);
    const std::array<F32, 6> interleaved = {
        1.0f, 10.0f, 2.0f, 20.0f, 3.0f, 30.0f};

    REQUIRE(buffer.pushStrided(interleaved.data(), 3, 2) == Result::SUCCESS);

    std::array<F32, 3> output = {};
    REQUIRE(buffer.pop(output.data(), output.size()) == Result::SUCCESS);
    const std::array<F32, 3> expected = {1.0f, 2.0f, 3.0f};
    REQUIRE(output == expected);

    REQUIRE(buffer.push(expected.data(), expected.size()) == Result::SUCCESS);
    std::array<F32, 6> interleavedOutput = {};
    REQUIRE(buffer.popStrided(interleavedOutput.data(), expected.size(), 2) ==
            Result::SUCCESS);
    const std::array<F32, 6> expectedInterleavedOutput = {
        1.0f, 0.0f, 2.0f, 0.0f, 3.0f, 0.0f};
    REQUIRE(interleavedOutput == expectedInterleavedOutput);
}

TEST_CASE("Circular buffer pops strided data across physical wraparound",
          "[tools][circular_buffer][strided][wrap]") {
    Tools::CircularBuffer<F32> buffer(4, OverflowPolicy::Reject);
    const std::array<F32, 3> first = {1.0f, 2.0f, 3.0f};
    REQUIRE(buffer.push(first.data(), first.size()) == Result::SUCCESS);

    std::array<F32, 2> discarded = {};
    REQUIRE(buffer.pop(discarded.data(), discarded.size()) == Result::SUCCESS);
    const std::array<F32, 3> second = {4.0f, 5.0f, 6.0f};
    REQUIRE(buffer.push(second.data(), second.size()) == Result::SUCCESS);

    std::array<F32, 8> output = {};
    REQUIRE(buffer.popStrided(output.data(), 4, 2) == Result::SUCCESS);
    const std::array<F32, 8> expected = {
        3.0f, 0.0f, 4.0f, 0.0f, 5.0f, 0.0f, 6.0f, 0.0f};
    REQUIRE(output == expected);
}

TEST_CASE("Circular buffer retains the newest strided values on overflow",
          "[tools][circular_buffer][strided][overflow]") {
    Tools::CircularBuffer<F32> buffer(3, OverflowPolicy::OverwriteOldest);
    const std::array<F32, 10> interleaved = {
        1.0f, 10.0f, 2.0f, 20.0f, 3.0f,
        30.0f, 4.0f, 40.0f, 5.0f, 50.0f};

    REQUIRE(buffer.pushStrided(interleaved.data(), 5, 2) == Result::SUCCESS);
    REQUIRE(buffer.overflows() == 1);

    std::array<F32, 3> output = {};
    REQUIRE(buffer.pop(output.data(), output.size()) == Result::SUCCESS);
    const std::array<F32, 3> expected = {3.0f, 4.0f, 5.0f};
    REQUIRE(output == expected);
}

TEST_CASE("Circular buffer clear and resize reset state but preserve policy",
          "[tools][circular_buffer][lifecycle]") {
    Tools::CircularBuffer<F32> buffer(2, OverflowPolicy::Reject);
    const std::array<F32, 2> input = {1.0f, 2.0f};
    const F32 extra = 3.0f;

    REQUIRE(buffer.push(input.data(), input.size()) == Result::SUCCESS);
    REQUIRE(buffer.push(&extra, 1) == Result::INCOMPLETE);
    REQUIRE(buffer.overflows() == 1);
    REQUIRE(buffer.clear() == Result::SUCCESS);
    REQUIRE(buffer.capacity() == 2);
    REQUIRE(buffer.empty());
    REQUIRE(buffer.overflows() == 0);
    REQUIRE(buffer.throughput() == 0.0);

    REQUIRE(buffer.push(input.data(), input.size()) == Result::SUCCESS);
    REQUIRE(buffer.push(&extra, 1) == Result::INCOMPLETE);
    REQUIRE(buffer.resize(3) == Result::SUCCESS);
    REQUIRE(buffer.capacity() == 3);
    REQUIRE(buffer.empty());
    REQUIRE(buffer.overflows() == 0);
    REQUIRE(buffer.throughput() == 0.0);

    const std::array<F32, 3> resizedInput = {4.0f, 5.0f, 6.0f};
    REQUIRE(buffer.push(resizedInput.data(), resizedInput.size()) ==
            Result::SUCCESS);
    REQUIRE(buffer.push(&extra, 1) == Result::INCOMPLETE);
}

TEST_CASE("Circular buffer handles zero-sized operations",
          "[tools][circular_buffer][boundary]") {
    Tools::CircularBuffer<F32> buffer;

    REQUIRE(buffer.capacity() == 0);
    REQUIRE(buffer.empty());
    REQUIRE_FALSE(buffer.full());
    REQUIRE(buffer.push(nullptr, 0) == Result::SUCCESS);
    REQUIRE(buffer.pop(nullptr, 0) == Result::SUCCESS);
    REQUIRE(buffer.peek(0, nullptr, 0) == Result::SUCCESS);
    REQUIRE(buffer.discard(0) == Result::SUCCESS);
    REQUIRE(buffer.waitForSize(0, std::chrono::milliseconds(0)) ==
            Result::SUCCESS);

    const F32 value = 1.0f;
    REQUIRE(buffer.push(&value, 1) == Result::ERROR);
    REQUIRE(buffer.overflows() == 1);
    REQUIRE(buffer.clear() == Result::SUCCESS);
    REQUIRE(buffer.overflows() == 0);
}

TEST_CASE("Circular buffer wait unblocks when producer pushes data",
          "[tools][circular_buffer][blocking]") {
    Tools::CircularBuffer<F32> buffer(2);
    Result producerResult = Result::ERROR;
    std::jthread producer([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        const F32 value = 7.0f;
        producerResult = buffer.push(&value, 1);
    });

    F32 output = 0.0f;
    REQUIRE(buffer.waitForSize(1) == Result::SUCCESS);
    REQUIRE(buffer.pop(&output, 1) == Result::SUCCESS);
    producer.join();
    REQUIRE(producerResult == Result::SUCCESS);
    REQUIRE(output == 7.0f);
}

TEST_CASE("Circular buffer wait observes capacity changes",
          "[tools][circular_buffer][blocking][resize]") {
    Tools::CircularBuffer<F32> buffer(2);
    Result resizeResult = Result::ERROR;
    std::jthread resizer([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        resizeResult = buffer.resize(1);
    });

    REQUIRE(buffer.waitForSize(2) == Result::ERROR);
    resizer.join();
    REQUIRE(resizeResult == Result::SUCCESS);
}

TEST_CASE("Circular buffer wait reports timeout without data",
          "[tools][circular_buffer][blocking][timeout]") {
    Tools::CircularBuffer<F32> buffer(1);
    REQUIRE(buffer.waitForSize(1, std::chrono::milliseconds(1)) ==
            Result::TIMEOUT);
}

TEST_CASE("Circular buffer remains reusable after move",
          "[tools][circular_buffer][move]") {
    Tools::CircularBuffer<F32> source(2);
    const F32 first = 1.0f;
    REQUIRE(source.push(&first, 1) == Result::SUCCESS);

    Tools::CircularBuffer<F32> destination(std::move(source));
    REQUIRE(source.empty());
    REQUIRE(source.resize(1) == Result::SUCCESS);
    const F32 second = 2.0f;
    REQUIRE(source.push(&second, 1) == Result::SUCCESS);

    Tools::CircularBuffer<F32> assigned(1);
    assigned = std::move(destination);
    REQUIRE(destination.empty());

    F32 output = 0.0f;
    REQUIRE(assigned.pop(&output, 1) == Result::SUCCESS);
    REQUIRE(output == first);
    REQUIRE(source.pop(&output, 1) == Result::SUCCESS);
    REQUIRE(output == second);
}

TEST_CASE("Circular buffer preserves ordering under concurrent reuse",
          "[tools][circular_buffer][concurrency]") {
    constexpr U64 transferCount = 4096;
    Tools::CircularBuffer<F32> buffer(31, OverflowPolicy::Reject);
    std::atomic<bool> producerFailed{false};

    std::jthread producer([&] {
        for (U64 i = 0; i < transferCount; ++i) {
            const F32 value = static_cast<F32>(i);
            while (true) {
                const Result result = buffer.push(&value, 1);
                if (result == Result::SUCCESS) {
                    break;
                }
                if (result != Result::INCOMPLETE) {
                    producerFailed = true;
                    return;
                }
                std::this_thread::yield();
            }
        }
    });

    std::vector<F32> output(transferCount);
    for (U64 i = 0; i < transferCount; ++i) {
        while (true) {
            const Result result = buffer.pop(&output[i], 1);
            if (result == Result::SUCCESS) {
                break;
            }
            REQUIRE(result == Result::INCOMPLETE);
            REQUIRE(buffer.waitForSize(1, std::chrono::seconds(1)) ==
                    Result::SUCCESS);
        }
    }
    producer.join();

    REQUIRE_FALSE(producerFailed);
    REQUIRE(buffer.empty());
    for (U64 i = 0; i < transferCount; ++i) {
        REQUIRE(output[i] == static_cast<F32>(i));
    }
}

TEST_CASE("Circular buffer validates incomplete operations",
          "[tools][circular_buffer][validation]") {
    Tools::CircularBuffer<F32> buffer(2);
    F32 value = 0.0f;

    REQUIRE(buffer.pop(&value, 1) == Result::INCOMPLETE);
    REQUIRE(buffer.peek(0, &value, 1) == Result::INCOMPLETE);
    REQUIRE(buffer.discard(1) == Result::INCOMPLETE);
    REQUIRE(buffer.push(nullptr, 1) == Result::ERROR);
    REQUIRE(buffer.peek(0, nullptr, 1) == Result::ERROR);
    REQUIRE(buffer.peek(1, &value, 0) == Result::INCOMPLETE);
    REQUIRE(buffer.waitForSize(3, std::chrono::milliseconds(1)) == Result::ERROR);

    const std::array<F32, 3> oversized = {1.0f, 2.0f, 3.0f};
    Tools::CircularBuffer<F32> rejectingBuffer(2, OverflowPolicy::Reject);
    REQUIRE(rejectingBuffer.push(oversized.data(), oversized.size()) ==
            Result::ERROR);
    REQUIRE(rejectingBuffer.empty());
    REQUIRE(rejectingBuffer.overflows() == 1);
}
