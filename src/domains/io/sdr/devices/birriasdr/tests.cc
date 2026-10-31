#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <limits>
#include <vector>

#include "domains/io/sdr/devices/birriasdr/tools/settings.hh"
#include "domains/io/sdr/tools/capture_queue.hh"
#include "domains/io/sdr_sweep/tools/sweep_plan.hh"

using namespace Jetstream;
using Radio::Device;
using Radio::BirriaSdrSettings::ReadCapabilities;

namespace {

struct Bands {
    U64 minimum = 2156000000;
    bool lowBandUnavailable = false;
    int highBandError = BIRRIASDR_OK;

    int operator()(const U32 index, uint64_t& low, uint64_t& high) const {
        if (index == 0) {
            low = minimum;
            high = 2610000000;
            return lowBandUnavailable ? BIRRIASDR_ERROR_UNSUPPORTED : BIRRIASDR_OK;
        }
        if (index == 1) {
            low = 4500000000;
            high = 6200000000;
            return highBandError;
        }
        return BIRRIASDR_ERROR_INVALID;
    }
};

}  // namespace

TEST_CASE("BirriaSDR exposes all rates and separate board-specific tuning bands",
          "[devices][sdr][birriasdr]") {
    for (const U64 minimum : {2156000000ULL, 2190000000ULL}) {
        CAPTURE(minimum);
        Device::Capabilities caps;
        REQUIRE(ReadCapabilities(Bands{.minimum = minimum}, 40e6, caps) == Result::SUCCESS);
        REQUIRE(caps.frequencyRanges.size() == 2);
        REQUIRE(caps.frequencyRanges.front().minimum == static_cast<F64>(minimum));
        REQUIRE(caps.frequencyRanges.front().maximum == 2610e6);
        REQUIRE(caps.frequencyRanges.back().minimum == 4500e6);
        REQUIRE(caps.frequencyRanges.back().maximum == 6200e6);
        REQUIRE_FALSE(Device::Range::Contains(caps.frequencyRanges, minimum - 1.0));
        REQUIRE_FALSE(Device::Range::Contains(caps.frequencyRanges, 3500e6));
        REQUIRE_FALSE(Device::Range::Contains(caps.frequencyRanges, 6200e6 + 1));
        REQUIRE(caps.fixedSampleRates() == std::vector<F64>{10e6, 20e6, 40e6, 80e6});
        REQUIRE_FALSE(Device::Range::Contains(caps.sampleRateRanges, 30e6));
        REQUIRE(caps.sweepSampleRate == 40e6);
        REQUIRE(caps.gainUnit == "steps");
    }
}

TEST_CASE("BirriaSDR sweep planning accepts the high band at every rate and rejects the gap",
          "[devices][sdr][birriasdr][sweep]") {
    Device::Capabilities caps;
    REQUIRE(ReadCapabilities(Bands{}, 40e6, caps) == Result::SUCCESS);
    using Plan = Radio::SweepPlan;
    for (const auto rate : caps.fixedSampleRates()) {
        CAPTURE(rate);
        Plan::Request request{
            .startFrequency = 5200e6,
            .endFrequency = 5300e6,
            .sampleRate = rate,
            .resolution = 100e3,
            .mode = Plan::Mode::Retune,
        };
        Plan plan;
        REQUIRE(Plan::Create(request, caps, plan) == Result::SUCCESS);
        REQUIRE(plan.sampleRate == rate);
        REQUIRE(plan.hops.size() > 1);
        for (const auto& hop : plan.hops) {
            REQUIRE(hop.frequency >= 4500e6);
            REQUIRE(hop.frequency <= 6200e6);
        }
        request.startFrequency = 3400e6;
        request.endFrequency = 3500e6;
        REQUIRE(Plan::Create(request, caps, plan) != Result::SUCCESS);
    }
}

TEST_CASE("BirriaSDR skips unavailable bands without hiding later supported bands",
          "[devices][sdr][birriasdr]") {
    Device::Capabilities caps;
    REQUIRE(ReadCapabilities(Bands{.lowBandUnavailable = true}, 40e6, caps) == Result::SUCCESS);
    REQUIRE(caps.frequencyRanges.size() == 1);
    REQUIRE_FALSE(Device::Range::Contains(caps.frequencyRanges, 2437e6));
    REQUIRE(Device::Range::Contains(caps.frequencyRanges, 5745e6));
}

TEST_CASE("BirriaSDR capability failures do not publish a partial device description",
          "[devices][sdr][birriasdr]") {
    Device::Capabilities caps;
    caps.frequencyRanges = {{100e6, 200e6}};
    caps.sampleRateRanges = {{2e6, 2e6}};
    Bands bands;
    F64 rate = 40e6;
    SECTION("transport failure after the first band") {
        bands.highBandError = BIRRIASDR_ERROR_IO;
    }
    SECTION("no reachable bands") {
        bands.lowBandUnavailable = true;
        bands.highBandError = BIRRIASDR_ERROR_UNSUPPORTED;
    }
    SECTION("invalid band bounds") {
        bands.minimum = 2700000000;
    }
    SECTION("unknown default rate") {
        rate = 2e6;
    }
    SECTION("nonfinite default rate") {
        rate = std::numeric_limits<F64>::quiet_NaN();
    }
    REQUIRE(ReadCapabilities(bands, rate, caps) == Result::ERROR);
    REQUIRE(caps.frequencyRanges.size() == 1);
    REQUIRE(caps.frequencyRanges.front().minimum == 100e6);
    REQUIRE(caps.fixedSampleRates() == std::vector<F64>{2e6});
}

TEST_CASE("BirriaSDR manual gain reaches the high-band LNA codes and full gain range",
          "[devices][sdr][birriasdr][gain]") {
    using Radio::BirriaSdrSettings::StageGains;
    const auto fourth = StageGains(4);
    REQUIRE(fourth.lna == 4);
    REQUIRE(fourth.lpf == 0);
    REQUIRE(fourth.digital == 0);
    const auto fifth = StageGains(5);
    REQUIRE(fifth.lna == 5);
    REQUIRE(fifth.lpf == 0);
    REQUIRE(fifth.digital == 0);
    const auto full = StageGains(25);
    REQUIRE(full.lna == 5);
    REQUIRE(full.lpf == 8);
    REQUIRE(full.digital == 12);
    const auto excess = StageGains(100);
    REQUIRE(excess.lna == full.lna);
    REQUIRE(excess.lpf == full.lpf);
    REQUIRE(excess.digital == full.digital);
}

TEST_CASE("BirriaSDR retunes discard every sample the library can still hold",
          "[devices][sdr][birriasdr][retune]") {
    using namespace Radio::BirriaSdrSettings;
    for (const F64 rate : {10e6, 20e6, 40e6, 80e6}) {
        CAPTURE(rate);
        const U64 packetSamples = rate <= 20e6 ? 1024 : 2048;
        const U64 backlog = (SramPackets + (InFlightTransfers + HostQueueTransfers) *
                             PacketsPerTransfer + 1) * packetSamples + TransferSamples;
        REQUIRE(HostQueueTransfers > InFlightTransfers);
        REQUIRE(StaleSamples(rate) >= backlog);

        Radio::CaptureQueue queue;
        REQUIRE(queue.resize(TransferSamples * 4) == Result::SUCCESS);
        queue.reset(1, 0, 2400e6, rate);
        queue.retune(2, 0, StaleSamples(rate), 5800e6);
        const auto fill = [](const F32 value) {
            return [value](CF32* output, U64, const U64 count) {
                std::fill(output, output + count, CF32{value, 0.0f});
            };
        };
        for (U64 pushed = 0; pushed < backlog; pushed += TransferSamples) {
            queue.push(2, std::min<U64>(TransferSamples, backlog - pushed), fill(1.0f));
        }
        for (U64 i = 0; i < 4; ++i) {
            queue.push(2, TransferSamples, fill(2.0f));
        }
        std::vector<CF32> samples(TransferSamples);
        const auto capture = queue.pop(samples, std::chrono::milliseconds(0));
        REQUIRE(capture.status == Device::Status::Samples);
        REQUIRE(capture.count > 0);
        REQUIRE(capture.frequency == 5800e6);
        for (U64 i = 0; i < capture.count; ++i) {
            REQUIRE(samples[i].real() == 2.0f);
        }
    }
}
