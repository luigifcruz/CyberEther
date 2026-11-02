#ifndef JETSTREAM_DOMAINS_IO_SDR_SWEEP_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_IO_SDR_SWEEP_MODULE_IMPL_HH

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>

#include <jetstream/domains/io/sdr_sweep/module.hh>
#include <jetstream/detail/module_impl.hh>
#include <jetstream/tools/snapshot.hh>

#include "domains/io/sdr/devices/sweep.hh"
#include "domains/io/sdr/tools/stream_access.hh"
#include "domains/io/sdr_sweep/tools/sweep_plan.hh"

namespace Jetstream::Modules {

struct JETSTREAM_API SdrSweepImpl : public Module::Impl, public DynamicConfig<SdrSweep> {
 public:
    using Clock = std::chrono::steady_clock;

    struct Progress {
        U64 completedHops = 0;
        U64 totalHops = 0;
        U64 sweeps = 0;
        U64 failedHops = 0;
        U64 expiredFrames = 0;
    };

    struct MetricSnapshot {
        F32 bufferFill = 0.0f;
        F64 bufferLoss = 0.0;
        F32 sweepSpeed = 0.0f;
        F32 sweepRate = 0.0f;
    };

    ~SdrSweepImpl() override;

    Result validate() override;
    Result define() override;
    Result create() override;
    Result destroy() override;
    Result reconfigure() override;

    Progress getProgress() const;
    MetricSnapshot getMetrics(Clock::time_point now = Clock::now());
    std::vector<std::string> listAntennas() const;
    std::optional<Radio::Device::Range> getGainRange() const;
    const Radio::SweepPlan& sweepPlan() const;

 protected:
    struct JETSTREAM_API MetricState {
        void reset(Clock::time_point now, F64 span = 0.0);
        void advance(Clock::time_point now, bool occupied, bool active = true);
        void publish(Clock::time_point now, bool occupied);
        void discard(Clock::time_point now, bool occupied);

        U64 producedFrames = 0;
        U64 droppedFrames = 0;
        U64 sampledFrames = 0;
        F64 spanGHz = 0.0;
        F64 fill = 0.0;
        F64 speed = 0.0;
        F64 cadence = 0.0;
        Clock::time_point occupancyUpdated{};
        Clock::time_point reported{};
        Clock::time_point rateStarted{};
        Clock::time_point lastCompleted{};
        MetricSnapshot snapshot;
    };

    struct Frame {
        U64 sweep = 0;
        Clock::time_point started{};
        Clock::time_point completed{};
    };

    Tensor buffer;

    std::thread worker;
    std::atomic<bool> errored{false};
    std::atomic<bool> running{false};

    std::mutex slotMutex;
    std::condition_variable slotReady;
    std::vector<CF32> slot;
    std::vector<CF32> staging;
    Frame slotFrame;
    bool slotFresh = false;
    MetricState metrics;

    Tools::Snapshot<U64> completedHops{0};
    Tools::Snapshot<U64> sweeps{0};
    Tools::Snapshot<U64> expiredFrames{0};
    Tools::Snapshot<F64> sweepDuration{0.0};
    Tools::Snapshot<U64> totalHops{0};

    Result workerLoop();
    bool handleCapture(const Radio::Sweep::Capture& capture);
    bool publishFrame(Clock::time_point now);
    Result emitFrame(Clock::time_point now = Clock::now());
    void expireFrame(Clock::time_point now);
    bool frameExpired(const Frame& frame, Clock::time_point now) const;

 private:
    std::unique_ptr<Radio::Sweep> radio;
    std::vector<std::string> antennas;
    Radio::StreamAccess access;
    Radio::SweepPlan plan;
    std::optional<Radio::Device::Range> gainRange;
    Radio::SweepAssembler assembler;
    U64 instance = 0;

    Clock::time_point sweepStarted{};
    U64 sweepCounter = 0;

    Result applyPlan(const Radio::SweepPlan& next, const Radio::Sweep::Settings& settings);
    Result startWorker();
    void stopWorker();
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_IO_SDR_SWEEP_MODULE_IMPL_HH
