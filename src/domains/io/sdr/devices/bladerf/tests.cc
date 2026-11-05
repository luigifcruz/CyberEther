#include <catch2/catch_test_macros.hpp>

#include <array>
#include <vector>

#include "domains/io/sdr/devices/bladerf/tools/ownership.hh"
#include "domains/io/sdr/tools/samples.hh"
#include "domains/io/sdr/tools/stream_buffer.hh"
#include "scripted_radio.hh"

#include "domains/io/sdr/devices/bladerf/tools/quick_tune_budget.hh"

using namespace Jetstream;

TEST_CASE("bladeRF quick-tune budgets reuse plans and fall back before exhaustion",
          "[devices][sdr][bladerf][budget]") {
    using Decision = Radio::BladeRfQuickTuneBudget::Decision;
    Radio::BladeRfQuickTuneBudget budget(256);
    const std::vector<F64> first(200, 1.0e9);
    std::vector<F64> second(100, 2.0e9);

    REQUIRE(budget.plan(first) == Decision::Build);
    budget.built(first);
    REQUIRE(budget.used() == 200);
    REQUIRE(budget.plan(first) == Decision::Reuse);
    REQUIRE(budget.plan(second) == Decision::Fallback);
    REQUIRE(budget.remaining() == 56);

    second.resize(50);
    REQUIRE(budget.plan(second) == Decision::Build);
    budget.failed(10);
    REQUIRE(budget.used() == 210);
    REQUIRE(budget.plan(first) == Decision::Fallback);

    Radio::BladeRfQuickTuneBudget unbounded;
    REQUIRE(unbounded.plan(std::vector<F64>(1000, 1.0)) == Decision::Build);
    REQUIRE_FALSE(unbounded.remaining().has_value());
}

TEST_CASE("bladeRF ownership excludes competing stream and sweep handles",
          "[devices][sdr][bladerf][ownership]") {
    Radio::BladeRfOwnership next;
    {
        Radio::BladeRfOwnership stream;
        Radio::BladeRfOwnership otherDevice;
        REQUIRE(stream.acquire("A") == Result::SUCCESS);
        REQUIRE(next.acquire("A") == Result::ERROR);
        REQUIRE(otherDevice.acquire("B") == Result::SUCCESS);
        REQUIRE(stream.acquire("C") == Result::ERROR);
    }
    REQUIRE(next.acquire("A") == Result::SUCCESS);
}

TEST_CASE("bladeRF dual IQ conversion preserves channel order and signed values",
          "[devices][sdr][bladerf][multi-channel]") {
    const std::array<I16, 12> raw{2047, -2048, 1024, -1024, 0, 204, -100, 512,
                                 -512, 256, 2048, 0};
    std::array<CF32, 3> first;
    std::array<CF32, 3> second;
    Radio::ConvertI16Channels(raw.data(), first.data(), second.data(), 3, 2048.0f);
    REQUIRE(first[0] == CF32{2047.0f / 2048.0f, -1.0f});
    REQUIRE(second[0] == CF32{0.5f, -0.5f});
    REQUIRE(first[1] == CF32{0.0f, 204.0f / 2048.0f});
    REQUIRE(second[1] == CF32{-100.0f / 2048.0f, 0.25f});
    REQUIRE(first[2] == CF32{-0.25f, 0.125f});
    REQUIRE(second[2] == CF32{1.0f, 0.0f});
}

TEST_CASE("dual RX queues discard and overwrite complete channel pairs",
          "[devices][sdr][bladerf][multi-channel][buffer]") {
    Radio::StreamBuffer queue;
    REQUIRE(queue.resize(6, 2) == Result::SUCCESS);
    queue.reset({.frequency = 100.0e6, .sampleRate = 2.0e6, .generation = 1});
    const std::array<CF32, 8> first{0, 1, 2, 3, 4, 5, 6, 7};
    const std::array<CF32, 8> second{100, 101, 102, 103, 104, 105, 106, 107};
    Radio::StreamBuffer::Metadata metadata;
    std::array<CF32, 4> a;
    std::array<CF32, 4> b;

    SECTION("overflow keeps the latest pairs and counts each pair once") {
        REQUIRE(queue.push(1, std::array<const CF32*, 2>{first.data(), second.data()},
                           8, 0, 1));
        REQUIRE(queue.statistics().pushedSamples == 8);
        REQUIRE(queue.statistics().overwrittenSamples == 2);
        REQUIRE(queue.pop(std::array<CF32*, 2>{a.data(), b.data()}, 4, metadata) ==
                Result::SUCCESS);
        REQUIRE(a == std::array<CF32, 4>{2, 3, 4, 5});
        REQUIRE(b == std::array<CF32, 4>{102, 103, 104, 105});
    }
    SECTION("timestamp gaps discard both partial frames together") {
        REQUIRE(queue.push(1, std::array<const CF32*, 2>{first.data(), second.data()},
                           2, 0, 1));
        REQUIRE(queue.push(
            1, std::array<const CF32*, 2>{first.data() + 2, second.data() + 2}, 4, 10,
            1));
        REQUIRE(queue.pop(std::array<CF32*, 2>{a.data(), b.data()}, 4, metadata) ==
                Result::SUCCESS);
        REQUIRE(a == std::array<CF32, 4>{2, 3, 4, 5});
        REQUIRE(b == std::array<CF32, 4>{102, 103, 104, 105});
        REQUIRE(queue.statistics().discardedSamples == 2);
    }
    SECTION("retunes reject old captures for both channels") {
        REQUIRE(queue.push(1, std::array<const CF32*, 2>{first.data(), second.data()},
                           2, 0, 1));
        queue.reset({.frequency = 200.0e6, .sampleRate = 4.0e6, .generation = 2});
        REQUIRE_FALSE(queue.push(
            1, std::array<const CF32*, 2>{first.data(), second.data()}, 4, 2, 1));
        REQUIRE(queue.size() == 0);
        REQUIRE(queue.push(
            2, std::array<const CF32*, 2>{first.data() + 4, second.data() + 4}, 4, 100,
            2));
        REQUIRE(queue.pop(std::array<CF32*, 2>{a.data(), b.data()}, 4, metadata) ==
                Result::SUCCESS);
        REQUIRE(a == std::array<CF32, 4>{4, 5, 6, 7});
        REQUIRE(b == std::array<CF32, 4>{104, 105, 106, 107});
        REQUIRE(metadata.frequency == 200.0e6);
        REQUIRE(metadata.sampleRate == 4.0e6);
    }
    SECTION("explicit discontinuities cannot join partial frames") {
        REQUIRE(
            queue.push(1, std::array<const CF32*, 2>{first.data(), second.data()}, 2));
        queue.split(1);
        REQUIRE(queue.push(
            1, std::array<const CF32*, 2>{first.data() + 2, second.data() + 2}, 4));
        REQUIRE(queue.pop(std::array<CF32*, 2>{a.data(), b.data()}, 4, metadata) ==
                Result::SUCCESS);
        REQUIRE(a == std::array<CF32, 4>{2, 3, 4, 5});
        REQUIRE(b == std::array<CF32, 4>{102, 103, 104, 105});
    }
    SECTION("single-channel access cannot partially consume a pair") {
        REQUIRE_FALSE(queue.push(1, first.data(), 4));
        REQUIRE(
            queue.push(1, std::array<const CF32*, 2>{first.data(), second.data()}, 4));
        REQUIRE(queue.pop(a.data(), 4, metadata) == Result::ERROR);
        REQUIRE(queue.size() == 4);
    }
}

TEST_CASE_METHOD(
    TestRadio::ScriptedDrivers,
    "dual RX streams read one synchronized capture and apply independent gains",
    "[devices][sdr][bladerf][multi-channel][stream]") {
    TestRadio::TestScript().receiveChannels = 2;
    TestRadio::TestScript().counter = true;
    TestRadio::TestScript().readLimit = 3;
    std::unique_ptr<Radio::Stream> stream;
    REQUIRE(Radio::Stream::Open(TestRadio::Selector, stream) == Result::SUCCESS);
    Radio::Stream::Settings settings;
    settings.receiveChannels = 2;
    settings.additionalGains.resize(1);
    settings.automaticGain = false;
    settings.gain = 12.0;
    settings.additionalGains[0].automatic = false;
    settings.additionalGains[0].gain = 19.0;
    REQUIRE(stream->start(settings) == Result::SUCCESS);
    std::array<CF32, 8> a;
    std::array<CF32, 8> b;
    std::array<std::span<CF32>, 2> destinations{a, b};
    auto capture = stream->readChannels(destinations, std::chrono::milliseconds(50));
    REQUIRE(capture.status == Radio::Device::Status::Samples);
    REQUIRE(capture.count == 3);
    REQUIRE(capture.timestamp == 0);
    for (U64 i = 0; i < capture.count; ++i) {
        REQUIRE(a[i].real() == static_cast<F32>(i));
        REQUIRE(b[i] == -a[i]);
    }
    const auto generation = stream->generation();
    settings.additionalGains[0].gain = 25.0;
    REQUIRE(stream->update(settings) == Result::SUCCESS);
    REQUIRE(stream->generation() == generation);
    REQUIRE(stream->applied().gain == 12.0);
    REQUIRE(stream->applied().additionalGains[0].gain == 25.0);
    settings.additionalGains[0].automatic = true;
    REQUIRE(stream->update(settings) == Result::SUCCESS);
    REQUIRE_FALSE(stream->applied().automaticGain);
    REQUIRE(stream->applied().additionalGains[0].automatic);

    TestRadio::TestScript().failAt = "gain2:35";
    settings.additionalGains[0].gain = 35.0;
    REQUIRE(stream->update(settings) == Result::ERROR);
    REQUIRE(stream->applied().gain == 12.0);
    REQUIRE(stream->applied().additionalGains[0].gain == 25.0);
    REQUIRE(stream->settings().additionalGains[0].gain == 25.0);
    REQUIRE(stream->generation() > generation);
    capture = stream->readChannels(destinations, std::chrono::milliseconds(50));
    REQUIRE(capture.status == Radio::Device::Status::Samples);
    REQUIRE(capture.count == 3);
    for (U64 i = 0; i < capture.count; ++i) {
        REQUIRE(b[i] == -a[i]);
    }
}

TEST_CASE_METHOD(
    TestRadio::ScriptedDrivers,
    "single RX streams reject multi-channel settings and keep the legacy read path",
    "[devices][sdr][bladerf][multi-channel][compatibility]") {
    std::unique_ptr<Radio::Stream> stream;
    REQUIRE(Radio::Stream::Open(TestRadio::Selector, stream) == Result::SUCCESS);
    Radio::Stream::Settings settings;
    settings.receiveChannels = 2;
    settings.additionalGains.resize(1);
    REQUIRE(stream->start(settings) == Result::ERROR);
    settings.receiveChannels = 1;
    settings.additionalGains.clear();
    REQUIRE(stream->start(settings) == Result::SUCCESS);
    std::array<CF32, 4> a;
    std::array<std::span<CF32>, 1> destinations{a};
    REQUIRE(stream->readChannels(destinations, std::chrono::milliseconds(50)).count ==
            4);
}

TEST_CASE("multi-channel queues preserve every channel through overflow and gaps",
          "[devices][sdr][multi-channel][buffer]") {
    Radio::StreamBuffer queue;
    REQUIRE(queue.resize(6, 4) == Result::SUCCESS);
    queue.reset({.generation = 1});
    std::array<std::array<CF32, 8>, 4> input;
    std::array<std::array<CF32, 4>, 4> output;
    std::array<const CF32*, 4> sources;
    std::array<CF32*, 4> destinations;
    for (U64 channel = 0; channel < 4; ++channel) {
        for (U64 i = 0; i < 8; ++i) {
            input[channel][i] = static_cast<F32>(channel * 100 + i);
        }
        sources[channel] = input[channel].data();
        destinations[channel] = output[channel].data();
    }
    Radio::StreamBuffer::Metadata metadata;
    REQUIRE(queue.push(1, sources, 8, 0));
    REQUIRE(queue.statistics().pushedSamples == 8);
    REQUIRE(queue.statistics().overwrittenSamples == 2);
    REQUIRE(queue.pop(destinations, 4, metadata) == Result::SUCCESS);
    for (U64 channel = 0; channel < 4; ++channel) {
        REQUIRE(output[channel][0] == static_cast<F32>(channel * 100 + 2));
        REQUIRE(output[channel][3] == static_cast<F32>(channel * 100 + 5));
    }
    // Two old samples remain; a timestamp jump discards them for every channel.
    REQUIRE(queue.push(1, sources, 4, 20));
    REQUIRE(queue.pop(destinations, 4, metadata) == Result::SUCCESS);
    REQUIRE(queue.statistics().discardedSamples == 2);
    for (U64 channel = 0; channel < 4; ++channel) {
        REQUIRE(output[channel][0] == static_cast<F32>(channel * 100));
    }
    REQUIRE(queue.push(1, sources, 4));
    destinations[3] = nullptr;
    REQUIRE(queue.pop(destinations, 4, metadata) == Result::ERROR);
    REQUIRE(queue.size() == 4);
    sources[3] = nullptr;
    REQUIRE_FALSE(queue.push(1, sources, 2));
    REQUIRE(queue.size() == 4);
    REQUIRE(queue.resize(6, 0) == Result::ERROR);
    REQUIRE(queue.size() == 4);
}
