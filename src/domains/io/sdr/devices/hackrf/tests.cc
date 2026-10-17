#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <vector>

#include "domains/io/sdr/devices/hackrf/tools/firmware.hh"
#include "domains/io/sdr/devices/hackrf/tools/sweep_decoder.hh"
#include "domains/io/sdr/devices/hackrf/tools/transfer_queue.hh"

using namespace Jetstream;

TEST_CASE("HackRF firmware sweeps follow the interleaved firmware sequence",
          "[devices][sdr][hackrf][layout]") {
    const auto firmware = [&](const U64 low, const U64 high) {
        std::vector<U64> sequence;
        U64 frequency = low * 1000000;
        bool odd = true;
        const U64 step = 20000000;
        while (true) {
            sequence.push_back(frequency);
            if (!odd && frequency + step >= high * 1000000) {
                break;
            }
            frequency += odd ? step / 4 : 3 * step / 4;
            odd = !odd;
        }
        return sequence;
    };
    using Range = Radio::HackRfFirmware::Range;
    REQUIRE(Radio::HackRfFirmware::Headers(Range{88, 108}) == firmware(88, 108));
    REQUIRE(Radio::HackRfFirmware::Headers(Range{88, 128}) ==
            std::vector<U64>{88000000, 93000000, 108000000, 113000000});
    REQUIRE(Radio::HackRfFirmware::Headers(Range{2400, 2500}) == firmware(2400, 2500));

    const auto range = Radio::HackRfFirmware::Snap(88.3e6, 2707.9e6);
    REQUIRE(range);
    REQUIRE(range->start == 88);
    REQUIRE(range->end == 2708);
    REQUIRE_FALSE(Radio::HackRfFirmware::Snap(5990.0e6, 5991.0e6));
}

TEST_CASE("HackRF firmware captures fit the sweep block payload",
          "[devices][sdr][hackrf][layout]") {
    const auto capability = Radio::HackRfFirmware::Capability();
    REQUIRE(capability.maxCaptureLength * 2 + Radio::HackRfSweepDecoder::HeaderBytes <=
            Radio::HackRfFirmware::BlockBytes);
    REQUIRE((capability.maxCaptureLength - capability.captureBase) % capability.captureStep == 0);

    Radio::Device::Layout layout;
    REQUIRE(capability.layout(88.3e6, 107.9e6, 2004, layout));
    REQUIRE(layout.startFrequency == 88.0e6);
    REQUIRE(layout.endFrequency == 108.0e6);
    REQUIRE(layout.hops.size() == 2);
    REQUIRE(layout.hops[0].frequency == 95.5e6);
    REQUIRE(layout.hops[1].frequency == 100.5e6);
    REQUIRE(layout.hops[0].bands.size() == 2);
    REQUIRE(layout.hops[0].bands[0].length == 501);
}

TEST_CASE("HackRF parses sweep blocks and excludes their headers",
          "[devices][sdr][hackrf][parser]") {
    std::vector<U8> block(16384, 0);
    block[0] = 0x7f;
    block[1] = 0x7f;
    const U64 frequency = 2412000000ULL;
    for (U64 i = 0; i < 8; ++i) {
        block[2 + i] = static_cast<U8>(frequency >> (8 * i));
    }
    block[10] = 0x55;

    const auto parsed = Radio::HackRfSweepDecoder::Block::Parse(block.data(), block.size(), 16384);
    REQUIRE(parsed);
    REQUIRE(parsed->frequency == frequency);
    REQUIRE(parsed->payloadSamples == 8187);
    REQUIRE(reinterpret_cast<const U8*>(parsed->payload) == block.data() + 10);
    REQUIRE(parsed->payload[0] == 0x55);

    REQUIRE_FALSE(Radio::HackRfSweepDecoder::Block::Parse(block.data(), 16383, 16384));
    block[1] = 0x7e;
    REQUIRE_FALSE(Radio::HackRfSweepDecoder::Block::Parse(block.data(), block.size(), 16384));
}

TEST_CASE("HackRF tracks firmware cycles through dwell repeats and losses",
          "[devices][sdr][hackrf][sequence]") {
    using Event = Radio::HackRfSweepDecoder::Sequence::Event;
    Radio::HackRfSweepDecoder::Sequence sequence;
    sequence.configure({10, 20, 30}, 2);
    U64 hop = 0;
    U64 repetition = 0;
    const auto next = [&](const U64 frequency) {
        return sequence.next(frequency, hop, repetition);
    };

    REQUIRE(next(10) == Event::Waiting);
    REQUIRE(next(10) == Event::Waiting);
    REQUIRE(next(20) == Event::Waiting);
    REQUIRE(next(30) == Event::Waiting);
    REQUIRE(next(10) == Event::Capture);
    REQUIRE((hop == 0 && repetition == 0));
    REQUIRE(next(10) == Event::Capture);
    REQUIRE((hop == 0 && repetition == 1));
    REQUIRE(next(20) == Event::Capture);
    REQUIRE((hop == 1 && repetition == 0));

    REQUIRE(next(30) == Event::Invalid);
    REQUIRE(next(30) == Event::Waiting);
    REQUIRE(next(10) == Event::Capture);
    REQUIRE(next(10) == Event::Capture);

    REQUIRE(next(10) == Event::Invalid);
    REQUIRE(next(20) == Event::Waiting);
    REQUIRE(next(99) == Event::Invalid);
    REQUIRE(next(10) == Event::Waiting);

    REQUIRE(sequence.losses() == 2);
}

TEST_CASE("HackRF decoders report progress while waiting for a full firmware cycle",
          "[devices][sdr][hackrf][sequence]") {
    constexpr U64 BlockBytes = 64;
    constexpr U64 Dwell = 64;
    const auto append = [](Radio::HackRfTransferQueue::Transfer& transfer, const U64 frequency) {
        std::vector<U8> block(BlockBytes, 0);
        block[0] = 0x7f;
        block[1] = 0x7f;
        for (U64 i = 0; i < 8; ++i) {
            block[2 + i] = static_cast<U8>(frequency >> (8 * i));
        }
        transfer.bytes.insert(transfer.bytes.end(), block.begin(), block.end());
    };

    Radio::HackRfSweepDecoder decoder;
    decoder.configure({10, 20, 30}, Dwell, 4, BlockBytes, 0.0);
    U64 captures = 0;
    const auto sink = [&](const Radio::Sweep::Capture&) {
        ++captures;
        return true;
    };

    Radio::HackRfTransferQueue::Transfer waiting;
    for (const U64 frequency : {20, 30}) {
        for (U64 i = 0; i < Dwell; ++i) {
            append(waiting, frequency);
        }
    }
    REQUIRE(decoder.decode(waiting, sink));
    REQUIRE(decoder.blocks() == 2 * Dwell);
    REQUIRE(decoder.losses() == 0);
    REQUIRE(captures == 0);

    Radio::HackRfTransferQueue::Transfer first;
    for (U64 i = 0; i < Dwell; ++i) {
        append(first, 10);
    }
    REQUIRE(decoder.decode(first, sink));
    REQUIRE(decoder.blocks() == 3 * Dwell);
    REQUIRE(captures == 1);
}

TEST_CASE("HackRF transfer queues mark loss where it happened",
          "[devices][sdr][hackrf][queue]") {
    Radio::HackRfTransferQueue queue(2);
    const U8 bytes[] = {1, 2, 3};
    queue.push(bytes, 1);
    queue.push(bytes, 2);
    queue.push(bytes, 3);
    queue.push(bytes, 3);
    REQUIRE(queue.dropped() == 2);

    Radio::HackRfTransferQueue::Transfer transfer;
    REQUIRE(queue.pop(transfer, std::chrono::milliseconds(1)));
    REQUIRE(transfer.bytes.size() == 1);
    REQUIRE_FALSE(transfer.gapBefore);
    REQUIRE(queue.pop(transfer, std::chrono::milliseconds(1)));
    REQUIRE(transfer.bytes.size() == 2);
    REQUIRE_FALSE(transfer.gapBefore);
    REQUIRE_FALSE(queue.pop(transfer, std::chrono::milliseconds(1)));

    queue.push(bytes, 3);
    queue.push(bytes, 1);
    REQUIRE(queue.pop(transfer, std::chrono::milliseconds(1)));
    REQUIRE(transfer.bytes.size() == 3);
    REQUIRE(transfer.gapBefore);
    REQUIRE(queue.pop(transfer, std::chrono::milliseconds(1)));
    REQUIRE_FALSE(transfer.gapBefore);
}
