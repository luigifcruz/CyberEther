#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <vector>

#include <jetstream/constants.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

extern "C" {
#include <mode-s.h>
}

#include "module_impl.hh"

namespace Jetstream::Modules {

// CPR constants.
static constexpr F64 CprDlatEven = 360.0 / 60.0;
static constexpr F64 CprDlatOdd  = 360.0 / 59.0;
static constexpr F64 CprMaxVal   = 131072.0;  // 2^17
static constexpr U64 CprPairWindowMs = 10000;

static F64 cprMod(F64 a, F64 b) {
    F64 res = std::fmod(a, b);
    if (res < 0.0) {
        res += b;
    }
    return res;
}

static I32 cprNL(F64 lat) {
    if (std::abs(lat) >= 87.0) {
        return 1;
    }
    constexpr F64 nz = 15.0;
    const F64 a = 1.0 - std::cos(static_cast<F64>(JST_PI) / (2.0 * nz));
    const F64 cosLat = std::cos(std::abs(lat) * static_cast<F64>(JST_PI) / 180.0);
    const F64 tmp = a / (cosLat * cosLat);
    if (tmp >= 1.0) {
        return 1;
    }
    return static_cast<I32>(std::floor(
        2.0 * static_cast<F64>(JST_PI) / std::acos(1.0 - tmp)));
}

static bool cprGlobalDecode(F64 rawLatEven,
                            F64 rawLonEven,
                            F64 rawLatOdd,
                            F64 rawLonOdd,
                            bool mostRecentOdd,
                            F64& outLat,
                            F64& outLon) {
    const F64 latEven = rawLatEven / CprMaxVal;
    const F64 lonEven = rawLonEven / CprMaxVal;
    const F64 latOdd = rawLatOdd / CprMaxVal;
    const F64 lonOdd = rawLonOdd / CprMaxVal;

    // Compute latitude index j.
    const I32 j = static_cast<I32>(std::floor(
        59.0 * latEven - 60.0 * latOdd + 0.5));

    F64 rlatEven = CprDlatEven * (cprMod(j, 60) + latEven);
    F64 rlatOdd  = CprDlatOdd  * (cprMod(j, 59) + latOdd);

    if (rlatEven >= 270.0) {
        rlatEven -= 360.0;
    }
    if (rlatOdd >= 270.0) {
        rlatOdd -= 360.0;
    }

    // Check that both are in the same NL zone.
    const I32 nlEven = cprNL(rlatEven);
    const I32 nlOdd  = cprNL(rlatOdd);
    if (nlEven != nlOdd) {
        return false;
    }

    F64 lat, lon;
    if (mostRecentOdd) {
        lat = rlatOdd;
        const I32 ni = std::max(nlOdd - 1, 1);
        const F64 dlon = 360.0 / ni;
        const I32 m = static_cast<I32>(std::floor(
            lonEven * (nlOdd - 1) - lonOdd * nlOdd + 0.5));
        lon = dlon * (cprMod(m, ni) + lonOdd);
    } else {
        lat = rlatEven;
        const I32 ni = std::max(nlEven, 1);
        const F64 dlon = 360.0 / ni;
        const I32 m = static_cast<I32>(std::floor(
            lonEven * (nlEven - 1) - lonOdd * nlEven + 0.5));
        lon = dlon * (cprMod(m, ni) + lonEven);
    }

    if (lon >= 180.0) {
        lon -= 360.0;
    }

    // Sanity check.
    if (lat < -90.0 || lat > 90.0 || lon < -180.0 || lon > 180.0) {
        return false;
    }

    outLat = lat;
    outLon = lon;
    return true;
}

struct AdsbImplNativeCpu : public AdsbImpl,
                           public NativeCpuRuntimeContext,
                           public Scheduler::Context {
 public:
    Result create() final;
    Result destroy() final;

    Result presentInitialize() override;
    Result presentSubmit() override;
    Result computeSubmit() override;

 private:
    mode_s_t state;
    std::vector<U16> magBuf;

    static void messageCallback(mode_s_t*, struct mode_s_msg* mm);
    void updateAircraftFromMessage(const struct mode_s_msg* mm);
};

static thread_local AdsbImplNativeCpu* tls_instance = nullptr;

void AdsbImplNativeCpu::messageCallback(mode_s_t*, struct mode_s_msg* mm) {
    if (!tls_instance || !mm->crcok) {
        return;
    }
    tls_instance->updateAircraftFromMessage(mm);
}

void AdsbImplNativeCpu::updateAircraftFromMessage(const struct mode_s_msg* mm) {
    const U32 icao = (static_cast<U32>(mm->aa1) << 16) |
                     (static_cast<U32>(mm->aa2) << 8) |
                     (static_cast<U32>(mm->aa3));

    std::lock_guard<std::mutex> lock(aircraftMutex);
    auto& ac = aircraftMap[icao];
    ac.icao = icao;

    if (mm->msgtype == 17) {
        if (mm->metype >= 1 && mm->metype <= 4) {
            std::string callsign(mm->flight);
            while (!callsign.empty() && callsign.back() == ' ') {
                callsign.pop_back();
            }
            ac.callsign = callsign;
            ac.hasCallsign = true;
        } else if (mm->metype >= 9 && mm->metype <= 18) {
            if (mm->altitude > 0) {
                ac.altitude = mm->altitude;
                ac.hasAltitude = true;
            }

            // CPR position decoding.
            const auto nowNs = std::chrono::steady_clock::now().time_since_epoch();
            const U64 now = static_cast<U64>(std::chrono::duration_cast<std::chrono::milliseconds>(nowNs).count());

            if (mm->fflag) {
                // Odd frame.
                ac.rawLatOdd = mm->raw_latitude;
                ac.rawLonOdd = mm->raw_longitude;
                ac.oddTimestamp = now;
            } else {
                // Even frame.
                ac.rawLatEven = mm->raw_latitude;
                ac.rawLonEven = mm->raw_longitude;
                ac.evenTimestamp = now;
            }

            // Attempt global decode if both frames are available and
            // close enough in time.
            if (ac.evenTimestamp > 0 && ac.oddTimestamp > 0) {
                const U64 timeDiff = (ac.evenTimestamp > ac.oddTimestamp)
                    ? (ac.evenTimestamp - ac.oddTimestamp)
                    : (ac.oddTimestamp - ac.evenTimestamp);

                if (timeDiff < CprPairWindowMs) {
                    const bool mostRecentOdd = (ac.oddTimestamp >= ac.evenTimestamp);

                    F64 lat, lon;
                    if (cprGlobalDecode(static_cast<F64>(ac.rawLatEven),
                                        static_cast<F64>(ac.rawLonEven),
                                        static_cast<F64>(ac.rawLatOdd),
                                        static_cast<F64>(ac.rawLonOdd),
                                        mostRecentOdd,
                                        lat,
                                        lon)) {
                        ac.latitude = lat;
                        ac.longitude = lon;
                        ac.hasPosition = true;

                        // Append to position history track.
                        ac.track.emplace_back(lat, lon);
                        if (ac.track.size() > maxTrackPoints) {
                            ac.track.erase(ac.track.begin());
                        }
                    }
                }
            }
        } else if (mm->metype == 19) {
            if (mm->velocity > 0) {
                ac.speed = static_cast<F32>(mm->velocity);
                ac.heading = static_cast<F32>(mm->heading);
                ac.hasVelocity = true;
            }
        }
    }
}

Result AdsbImplNativeCpu::create() {
    // Create parent.

    JST_CHECK(AdsbImpl::create());

    // Validate input dtype.

    if (input.dtype() != DataType::CF32) {
        JST_ERROR("[MODULE_ADSB_NATIVE_CPU] Unsupported input data "
                  "type: {}.", input.dtype());
        return Result::ERROR;
    }

    // Initialize libmodes.

    mode_s_init(&state);
    state.fix_errors = 1;
    state.check_crc = 1;
    state.aggressive = 0;

    // Allocate magnitude buffer.

    magBuf.resize(input.size());

    return Result::SUCCESS;
}

Result AdsbImplNativeCpu::destroy() {
    magBuf.clear();
    JST_CHECK(AdsbImpl::destroy());
    return Result::SUCCESS;
}

Result AdsbImplNativeCpu::computeSubmit() {
    const CF32* iqData = reinterpret_cast<const CF32*>(input.data());
    const U64 numSamples = input.size();

    // Convert CF32 IQ to U16 magnitude in libmodes scale.
    for (U64 i = 0; i < numSamples; ++i) {
        const F32 real = iqData[i].real() * 128.0f;
        const F32 imag = iqData[i].imag() * 128.0f;
        const F32 mag = std::sqrt(real * real + imag * imag) * 360.0f;
        magBuf[i] = static_cast<U16>(std::min(mag, static_cast<F32>(std::numeric_limits<U16>::max())));
    }

    tls_instance = this;
    mode_s_detect(&state,
                  magBuf.data(),
                  static_cast<U32>(numSamples),
                  &AdsbImplNativeCpu::messageCallback);
    tls_instance = nullptr;

    // Update output tensors with current aircraft positions.
    updateOutputTensors();

    return Result::SUCCESS;
}

Result AdsbImplNativeCpu::presentInitialize() {
    return createPresent();
}

Result AdsbImplNativeCpu::presentSubmit() {
    return present();
}

JST_REGISTER_MODULE(AdsbImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
